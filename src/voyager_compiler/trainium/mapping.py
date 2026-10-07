"""Trainium mapping legality and buffering, independent of Voyager policy."""

import math

from interstellar import loop_enum as le
from voyager_compiler.codegen.transform.bufferize.plan import KernelBufferPlan
from voyager_compiler.codegen.transform.tiling.policy import (
    VoyagerMappingPolicy,
)

from .cost import TrainiumCostModel


def extent(point, dim, level=2):
    return math.prod(point.loop_blocking(dim)[: level + 1]) * math.prod(
        point.loop_partitioning(dim)[: level + 1]
    )


from .execution import slot_bytes


class TrainiumMappingPolicy:
    search_fully_connected = True
    matrix_only_fusion = True
    speed_only = True

    def __init__(self, config, tuning=None):
        from .execution import TrainiumTuning

        self.config = config
        self.tuning = tuning or TrainiumTuning()
        if self.tuning.isa_lowering and config.name != "trainium-v3":
            raise ValueError(
                "The pinned ISA expansion contract supports Trainium2 only"
            )

    def options(self):
        from dataclasses import asdict

        return (
            dict(self._recorded_options)
            if hasattr(self, "_recorded_options")
            else asdict(self.tuning)
        )

    def partition(self, architecture, size_fn, layer, mapping):
        return None, 1

    def evaluate(
        self, runtime, architecture, layer, mapping, bank_groups, scratch_slots
    ):
        return runtime.evaluate(architecture, layer, mapping)

    def prepare_matrix(self, problem, tiler):
        from voyager_compiler.codegen.transform.tiling.traversal import (
            MappingTraversal,
        )
        from voyager_compiler.codegen.node_info import (
            is_bmm,
            is_conv2d,
            weight_transforms,
        )
        from voyager_compiler.codegen.transform.tiling.cost import (
            get_dtype_width,
        )

        anchor = problem.anchor
        batch = math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1
        transposed, repeat = weight_transforms(anchor.args[1])[1:3]
        repeated = (
            math.prod(repeat[: max(0, len(anchor.args[1].shape) - 2)])
            if repeat
            else 1
        )
        runtime = MappingTraversal(batch, batch // repeated)
        runtime.input_dtype_name = str(anchor.args[0].value.dtype).removeprefix(
            "torch."
        )
        runtime.output_dtype_name = str(anchor.value.dtype).removeprefix(
            "torch."
        )
        runtime.input_dtype_width = problem.if_bits
        runtime.weight_dtype_width = problem.fl_bits
        runtime.output_dtype_width = problem.of_bits
        runtime.weight_transposed = transposed
        runtime.has_tail = problem.has_tail
        from voyager_compiler.codegen.node_info import get_arg_value

        bias = get_arg_value(anchor, 2, "bias", None)
        runtime.bias_width = (
            get_dtype_width(bias.value.dtype) if hasattr(bias, "value") else 0
        )
        return self.prepare(
            None,
            runtime,
            int(anchor.value.shape[0]) if is_conv2d(anchor) else 1,
            node=anchor,
            constraint=problem.constraint,
        )

    def nonmatrix_cost(self, kind, node, default):
        from .cost import vector_candidate

        def cost(node, tile_sizes, shapes, tiling):
            result = vector_candidate(
                self.config, node, tile_sizes, shapes, tiling, self.tuning
            )
            return (math.inf, math.inf) if result is None else result[1:]

        return cost

    def nonmatrix_footprint(self, node, shapes, sharing, default):
        from .cost import vector_candidate
        from voyager_compiler.codegen.transform.tiling.contracts import (
            NonMatrixFootprint,
        )

        result = vector_candidate(
            self.config, node, (), shapes, (1,), self.tuning
        )
        return NonMatrixFootprint(math.inf if result is None else result[0], [])

    def nonmatrix_slot_size(self, node, slots):
        from .lowering import RECIPES, operation_name, reduction_workspace
        from voyager_compiler.codegen.node_info import get_anchor_node
        anchor = get_anchor_node(node)
        name = operation_name(anchor.target)
        reserve = self.tuning.sbuf_reserve_bytes
        if name in RECIPES:
            # The row recipe's live values are included in its footprint.
            # They replace the generic named reduction scratch, and are not
            # duplicated for every logical pipeline slot.
            return self.config.scratchpad_size
        return (self.config.scratchpad_size - reserve) // slots

    def vector_limits(self, anchor, limits):
        return limits

    def place_local_buffers(self, model, bufs):
        from .constraints import place_local_buffers

        return place_local_buffers(model, bufs, self.config, self.tuning)

    def schedule(self):
        # Preserve the shared builder's consecutive external reductions and
        # full convolution filter. Drop Voyager's L2 IC-innermost restriction.
        hints = VoyagerMappingPolicy(self.config).schedule()
        del hints["schedule_hint"]["IC"]["level2"]
        # Inner levels describe software-tile traversal, not Voyager L1 SRAMs.
        return hints

    def prepare(self, size_factory, runtime, actual_batch, **kwargs):
        node = kwargs["node"]
        conv = "conv2d" in str(node.target)
        constraint = kwargs.get("constraint")
        config = self.config
        from voyager_compiler.codegen.node_info import weight_is_ck

        runtime.weight_hbm_ck = weight_is_ck(node) != runtime.weight_transposed
        # Batches of convolution are independent tiles, as in Gemmini's plan.
        runtime.batch *= actual_batch
        # Convolution weights are shared across batches; tiled weights reload
        # through _batch_loads, while a whole retained weight is fetched once.

        def sizes(
            counts, point, level, partitioning_accum, bank_size, num_banks
        ):
            if (
                level == 2
                and constraint is not None
                and not constraint.allows(lambda d: extent(point, d, level))
            ):
                return (math.inf,) * 3
            if level != 2:
                return counts
            m = extent(point, le.OX) * extent(point, le.OY)
            n, k = extent(point, le.OC), extent(point, le.IC)
            # Converter support limits are explicit; they are not fabricated
            # hardware capacities. GEMM software tiles contain many ISA tiles.
            if conv and (n > 128 or k > 128 or m > 512):
                return (math.inf,) * 3
            a, c, b = counts
            # Early pruning uses the minimum depth; evaluate checks each
            # complete one/two-buffer plan against both physical stores.
            wpart = n if conv or not runtime.weight_transposed else k
            plan = make_plan(point, actual_batch, runtime.batch, 1)
            size = (
                plan.input_slots * slot_bytes(a, k, runtime.input_dtype_width)
                + plan.weight_slots
                * slot_bytes(b, wpart, runtime.weight_dtype_width)
                + (plan.output_slots + plan.scratch_slots)
                * slot_bytes(c, n, runtime.output_dtype_width)
            )
            # Fused bias and pointwise operands; shared allocator remains the
            # final graph-wide authority on lifetimes and capacity.
            size += (
                2 * slot_bytes(n, n, runtime.bias_width)
                if runtime.bias_width
                else 0
            )
            return size + self.tuning.sbuf_reserve_bytes, 0, 0

        return sizes, TrainiumCostModel(
            runtime,
            config,
            conv=conv,
            batch_tiles=actual_batch,
            tuning=self.tuning,
        )


def make_plan(mapping, batch_tiles=1, batch_count=None, max_depth=2):
    batch_count = batch_tiles if batch_count is None else batch_count

    def slots(dims):
        return (
            max_depth
            if batch_count
            * math.prod(mapping.loop_blockings[d][3] for d in dims)
            > 1
            else 1
        )

    return KernelBufferPlan(
        input_slots=slots((le.IC, le.OX, le.OY, le.ON)),
        weight_slots=slots((le.IC, le.OC, le.FX, le.FY)),
        output_slots=(
            max_depth
            if batch_count
            * math.prod(
                mapping.loop_blockings[d][3] for d in (le.OC, le.OX, le.OY)
            )
            > 1
            else 1
        ),
        scratch_slots=1,
        batch_tiles=batch_tiles,
    )
