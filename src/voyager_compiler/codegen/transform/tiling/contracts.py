"""Internal contracts between search, resource accounting and bufferization."""

from dataclasses import dataclass
import math
from voyager_compiler.hardware_config import CapacityUnit
from .execution import ExecutionPlan


@dataclass(frozen=True)
class StorageRequirement:
    memory: str
    bytes_per_slot: int
    slots: int = 1
    alignment: int = 1

    def __post_init__(self):
        if (
            any(
                type(v) is not int
                for v in (self.bytes_per_slot, self.slots, self.alignment)
            )
            or self.bytes_per_slot < 0
            or self.slots < 1
            or self.alignment < 1
        ):
            raise ValueError(
                "Storage requirements need nonnegative bytes and positive integer slots/alignment"
            )

    @property
    def allocated_bytes(self):
        return (
            ((self.bytes_per_slot + self.alignment - 1) // self.alignment)
            * self.alignment
            * self.slots
        )


def resources_fit(config, requirements):
    demands = {}
    for item in requirements:
        demands[item.memory] = (
            demands.get(item.memory, 0) + item.allocated_bytes
        )
    for name, demand in demands.items():
        memory = config.memory_instance(name)
        if memory.size.unit != CapacityUnit.BYTES or memory.size.value is None:
            raise ValueError(
                "Byte capacity required for storage requirement: " + name
            )
        if demand > memory.size.value - memory.reserved_bytes:
            return False
    return True


@dataclass(frozen=True)
class CandidateEvaluation:
    cycles: float
    buffer_plan: object
    storage: tuple = ()
    diagnostics: tuple = ()
    execution_plan: ExecutionPlan | None = None

    def __post_init__(self):
        if math.isnan(self.cycles) or self.cycles < 0:
            raise ValueError("Estimated cycles must be nonnegative")


@dataclass(frozen=True)
class SelectedMapping:
    mapping: object
    access_list: object
    bank_groups: object
    scratch_slots: int
    evaluation: CandidateEvaluation


@dataclass(frozen=True)
class MatrixProblem:
    anchor: object
    layer: object
    out_dtype: object
    fused_specs: object
    constraint: object
    has_tail: object
    single_k_tail_extra_pass: object
    split_k_tail_extra_pass: object
    tail_keeps_shape: object
    scratch_regions: object
    outlier_rate: object
    outlier_pct: object
    out_outlier_pct: object
    boundary: object
    if_bits: object
    fl_bits: object
    of_bits: object
    if_scale_bits: object
    fl_scale_bits: object
    of_scale_bits: object


@dataclass(frozen=True)
class NonMatrixFootprint:
    slot_bytes: int
    bank_groups: object
