"""Instruction geometry shared by analytical search and NKI realization.

Policy values are not hardware capacities. Service is engine occupancy;
completion latency is modeled separately. No hardware-fitted coefficients.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainiumTuning:
    sbuf_reserve_bytes: int = 4 << 20
    dma_transpose: bool = False
    explicit_isa: bool = True
    isa_lowering: bool = True
    copy_policy: str = "scalar"
    max_buffer_depth: int = 2
    dma_rows: int = 1024
    dma_columns: int = 4096
    buffer_allocation: str = "compiler"
    min_buffer_depth: int = 1
    movement_search_budget: int = 0
    movement_search_beam: int = 2

    def __post_init__(self):
        if (
            type(self.movement_search_budget) is not int
            or self.movement_search_budget < 0
            or type(self.movement_search_beam) is not int
            or self.movement_search_beam < 1
        ):
            raise ValueError("Invalid movement search budget/beam")
        if (
            type(self.sbuf_reserve_bytes) is not int
            or self.sbuf_reserve_bytes < 0
        ):
            raise ValueError("Invalid temporary reserve")
        if type(
            self.max_buffer_depth
        ) is not int or self.max_buffer_depth not in (1, 2):
            raise ValueError("Shared builders support one or two slots")
        if any(
            type(x) is not int or x < 1
            for x in (self.dma_rows, self.dma_columns)
        ):
            raise ValueError("DMA panel limits must be positive")
        if any(
            type(value) is not bool
            for value in (
                self.dma_transpose,
                self.explicit_isa,
                self.isa_lowering,
            )
        ):
            raise TypeError("Instruction mode switches must be boolean")
        if self.buffer_allocation not in ("compiler", "legacy_logical"):
            raise ValueError("Unknown buffer allocation contract")
        if not 1 <= self.min_buffer_depth <= self.max_buffer_depth:
            raise ValueError("Invalid minimum buffer depth")
        if self.copy_policy not in ("balanced", "scalar"):
            raise ValueError("Unknown ISA copy policy")
        if self.isa_lowering and not self.explicit_isa:
            raise ValueError(
                "ISA lowering requires explicit matmul instructions"
            )


def dma_panels(rows, cols, row_boundary, *, transpose, store, tuning):
    # Transpose-on-load puts contiguous HBM columns on SBUF partitions.
    # Its free axis may span multiple 128-row instruction blocks.
    direct = transpose and not store and tuning.dma_transpose
    row_limit = tuning.dma_rows if direct else 128
    col_limit = 128 if transpose else tuning.dma_columns
    row_limit = min(row_limit, row_boundary)
    for row in range(0, rows, row_limit):
        for col in range(0, cols, col_limit):
            yield row, col, min(row_limit, rows - row), min(
                col_limit, cols - col
            ), direct


def matmul_panels(m, n, k):
    for mi in range(0, m, 512):
        for ni in range(0, n, 128):
            yield mi, ni, min(512, m - mi), min(128, n - ni), tuple(
                (ki, min(128, k - ki)) for ki in range(0, k, 128)
            )


def slot_bytes(elements, partitions, bits):
    return (
        math.ceil(
            math.ceil(elements / partitions)
            * math.ceil(partitions / min(partitions, 128))
            * bits
            / 8
            / 16
        )
        * 16
        * 128
    )


def matrix_storage(
    config,
    a,
    b,
    c,
    k,
    n,
    bits,
    weight_bits,
    out_bits,
    plan,
    tuning,
    weight_partitions=None,
):
    from voyager_compiler.codegen.transform.tiling.contracts import (
        StorageRequirement,
    )

    # Three PSUM banks: current GEMM accumulator and up to two staged transpose
    # results. This candidate certificate reserves capacity; the selected ISA
    # program subsequently fixes and validates physical bank assignment.
    bank = (
        config.memory_instance("PSUM").size.value
        // config.memory_instance("PSUM").banks
    )
    requirements = (
        StorageRequirement(
            "SBUF", slot_bytes(a, k, bits), plan.input_slots, 2048
        ),
        StorageRequirement(
            "SBUF",
            slot_bytes(b, weight_partitions or n, weight_bits),
            plan.weight_slots,
            2048,
        ),
        StorageRequirement(
            "SBUF",
            slot_bytes(c, n, out_bits),
            plan.output_slots + plan.scratch_slots,
            2048,
        ),
        StorageRequirement("SBUF", tuning.sbuf_reserve_bytes, 1, 2048),
        StorageRequirement("PSUM", bank, 3, bank),
    )
    return requirements


@dataclass(frozen=True)
class EngineEvent:
    engine: str
    service_ns: float
    dependencies: tuple = ()
    completion_delay_ns: float = 0


def schedule_events(events):
    """ASAP resource timeline: launch/latency does not monopolize an engine.

    Events depend on explicit predecessors; issue order is preserved per engine.
    This is an analytical overlap model, not a simulation of NKI's scheduler.
    """
    available, service, finishes = {}, {}, []
    for i, event in enumerate(events):
        if (
            event.service_ns < 0
            or event.completion_delay_ns < 0
            or any(d < 0 or d >= i for d in event.dependencies)
        ):
            raise ValueError("Invalid event duration or forward dependency")
        start = max(
            available.get(event.engine, 0),
            max((finishes[d] for d in event.dependencies), default=0),
        )
        active_end = start + event.service_ns
        available[event.engine] = active_end
        finishes.append(active_end + event.completion_delay_ns)
        service[event.engine] = service.get(event.engine, 0) + event.service_ns
    return max(finishes, default=0), service
