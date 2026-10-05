"""Gemmini constraints, independent of the Voyager storage/cost policy."""

import math
from dataclasses import dataclass, field
from .scheduling import SubmissionPolicy
from interstellar import loop_enum as le
from voyager_compiler.codegen.transform.bufferize.plan import KernelBufferPlan
from .execution import GemminiCostModel
from .constraints import matrix_storage


@dataclass(frozen=True)
class GemminiTuning:
    """Compiler choices, not hardware facts and not per-kernel tile presets.

    Two conservative wide pointwise working sets preserve the validated
    policy. Their expansion ratio and budget derive from hardware storage
    widths/capacities; changing this count is an explicit policy experiment.
    """

    separate_accumulator_banks: bool = True
    pointwise_wide_working_sets: int = 2
    submission: SubmissionPolicy = field(default_factory=SubmissionPolicy)

    def __post_init__(self):
        if type(self.separate_accumulator_banks) is not bool:
            raise TypeError("Bank separation must be boolean")
        if (
            type(self.pointwise_wide_working_sets) is not int
            or self.pointwise_wide_working_sets < 1
        ):
            raise ValueError("Pointwise working-set count must be positive")
        if not isinstance(self.submission, SubmissionPolicy):
            raise TypeError("Submission must be a SubmissionPolicy")


class GemminiMappingPolicy:
    search_fully_connected = True
    matrix_only_fusion = True
    speed_only = True

    def __init__(self, config, tuning=None):
        self.config = config
        self.tuning = tuning or GemminiTuning()
        if not isinstance(self.tuning, GemminiTuning):
            raise TypeError("Gemmini policy requires GemminiTuning")

    def partition(self, architecture, size_fn, layer, mapping):
        return None, 1

    def prepare_matrix(self, problem, tiler):
        from voyager_compiler.codegen.transform.tiling.traversal import (
            MappingTraversal,
        )
        from voyager_compiler.codegen.node_info import (
            is_bmm,
            is_conv2d,
            weight_transforms,
        )

        anchor = problem.anchor
        batch = math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1
        repeat = weight_transforms(anchor.args[1])[2]
        weight_repeat = (
            math.prod(repeat[: max(0, len(anchor.args[1].shape) - 2)])
            if repeat
            else 1
        )
        traversal = MappingTraversal(batch, batch // weight_repeat)
        return self.prepare(
            None,
            traversal,
            int(anchor.value.shape[0]) if is_conv2d(anchor) else 1,
            constraint=problem.constraint,
        )

    def evaluate(
        self, runtime, architecture, layer, mapping, bank_groups, scratch_slots
    ):
        return runtime.evaluate(architecture, layer, mapping)

    def options(self):
        from dataclasses import asdict

        return asdict(self.tuning)

    def nonmatrix_cost(self, kind, node, default):
        # Preserve the validated nonmatrix estimate while providing an explicit
        # override for a backend with different pointwise/pooling lowering.
        return default

    def nonmatrix_footprint(self, node, shapes, sharing, default):
        return default(node, shapes, self.config, sharing)

    def nonmatrix_slot_size(self, node, slots):
        config = self.config
        # Conservative bound for physical wide temporaries, not a fake SRAM.
        # The number of working sets is a documented lowering policy.
        storage_bytes = config.memory_instance("accumulator").size.value
        input_bytes = (
            config.memory_instance("scratchpad").row_bytes
            / config.pe_array_size[1]
        )
        expansion = config.accumulator_element_bytes / input_bytes
        budget = int(
            storage_bytes
            / expansion
            / self.tuning.pointwise_wide_working_sets
            / slots
        )
        return min(config.usable_scratchpad_size // slots, budget)

    def vector_limits(self, anchor, limits):
        from voyager_compiler.codegen.transform.tiling.search import (
            is_elementwise_op,
        )

        last_dim, multiple_of = limits
        return (None if is_elementwise_op(anchor) else last_dim), multiple_of

    def place_local_buffers(self, model, bufs):
        from .constraints import place_local_buffers

        return place_local_buffers(model, bufs, self.config)

    def schedule(self):
        rows, cols = self.config.pe_array_size
        # L3 reduction consecutiveness is a capability of the shared builder.
        # FX/FY remain within the software convolution tile: its DMA window
        # contains the whole filter. Inner traversal has its own WS policy.
        return {
            "schedule_hint": {
                "IC": {
                    "level0": {"order": 1, "partitioning_size": rows},
                    "level1": {"order": -1},
                    "level3": {"order": 0},
                },
                "OC": {"level0": {"order": 0, "partitioning_size": cols}},
                "OX": {"level1": {"order": 0}},
                "OY": {"level1": {"order": 1}},
                **{
                    d: {
                        "level0": {"blocking_size": 1, "partitioning_size": 1},
                        "level1": {"order": rank},
                        "level2": {"blocking_size": 1, "partitioning_size": 1},
                        "level3": {"blocking_size": 1, "partitioning_size": 1},
                    }
                    for d, rank in (("FX", 2), ("FY", 3))
                },
            }
        }

    def prepare(self, size_factory, runtime, actual_batch, **kwargs):
        # No call to Voyager's byte packing, bank merging, vector-tail or
        # scratch policy. Gemmini stores INT8 DMA operands and INT32 psums.
        batch = actual_batch
        constraint = kwargs.get("constraint")
        config = self.config

        def sizes(
            counts, point, level, partitioning_accum, bank_size, num_banks
        ):
            if constraint is not None:

                def extent(d):
                    return math.prod(point.loop_blockings[d][: level + 1]) * (
                        partitioning_accum[d]
                        if partitioning_accum is not None
                        else 1
                    )

                if not constraint.allows(extent):
                    return (math.inf,) * 3
            a, c, b = counts
            if level == 2:
                # A one-image plan is the minimum legal footprint. Runtime
                # compares supported batch factors and checks each full plan.
                plan = make_plan(point, 1, batch_tiles=batch)
                requirements = matrix_storage(
                    config, a, b, c, plan, self.tuning
                )
                return (
                    sum(
                        r.allocated_bytes
                        for r in requirements
                        if r.memory == "scratchpad"
                    ),
                    sum(
                        r.allocated_bytes
                        for r in requirements
                        if r.memory == "accumulator"
                    ),
                    0,
                )
            # Batch tiling is selected by the target buffer plan. Requiring
            # all images at L1 would reject valid one-image inner traversals
            # before the L2/runtime plan gets a chance to consider them.
            return a, c, b

        return sizes, GemminiCostModel(runtime, batch, config, self.tuning)


def make_plan(mapping, batch=1, batch_tiles=1):
    def slots(dims, copies=1):
        count = copies * math.prod(mapping.loop_blockings[d][3] for d in dims)
        return 1 if count == 1 else 2

    return KernelBufferPlan(
        input_slots=slots((le.OX, le.OY, le.IC, le.ON), batch_tiles),
        weight_slots=slots((le.OC, le.IC, le.FX, le.FY)),
        output_slots=2,
        batch_tiles=batch_tiles,
        # The shared builder uses one scratch slot. Conversion alternates
        # two physical generations of it; quantized output refs alias these
        # generations and do not allocate two further wide copies.
        accumulator_copies=2,
    )
