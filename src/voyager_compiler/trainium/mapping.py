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


def slot_bytes(elements, partitions, bits):
    return (
        math.ceil(math.ceil(elements / min(partitions, 128)) * bits / 8 / 16)
        * 16
        * 128
    )


class TrainiumMappingPolicy:
    search_fully_connected = True
    matrix_only_fusion = True
    speed_only = True

    def __init__(self, config):
        self.config = config

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
            if constraint is not None and not constraint.allows(
                lambda d: extent(point, d, level)
            ):
                return (math.inf,) * 3
            if level != 2:
                return counts
            m = extent(point, le.OX) * extent(point, le.OY)
            n, k = extent(point, le.OC), extent(point, le.IC)
            # Converter support limits are explicit; they are not fabricated
            # hardware capacities. GEMM software tiles contain many ISA tiles.
            if m * math.ceil(n / 128) > 4096 or (
                conv and (n > 128 or k > 128 or m > 512)
            ):
                return (math.inf,) * 3
            a, c, b = counts
            wpart = n if conv or not runtime.weight_transposed else k
            plan = make_plan(point, actual_batch, runtime.batch)
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
            return size, 0, 0

        return sizes, TrainiumCostModel(
            runtime, config, conv=conv, batch_tiles=actual_batch
        )

    def bufferization(
        self, runtime, architecture, layer, mapping, bank_groups, scratch_slots
    ):
        runtime.calculate_runtime(architecture, layer, mapping)
        return runtime.plan


def make_plan(mapping, batch_tiles=1, batch_count=None):
    batch_count = batch_tiles if batch_count is None else batch_count

    def slots(dims):
        return (
            2
            if batch_count
            * math.prod(mapping.loop_blockings[d][3] for d in dims)
            > 1
            else 1
        )

    return KernelBufferPlan(
        input_slots=slots((le.IC, le.OX, le.OY, le.ON)),
        weight_slots=slots((le.IC, le.OC, le.FX, le.FY)),
        output_slots=2
        if batch_count
        * math.prod(mapping.loop_blockings[d][3] for d in (le.OC, le.OX, le.OY))
        > 1
        else 1,
        scratch_slots=1,
        batch_tiles=batch_tiles,
    )
