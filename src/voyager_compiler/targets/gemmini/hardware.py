"""Parameterized square INT8/INT32 weight-stationary Gemmini family."""

from voyager_compiler.hardware import (
    AcceleratorConfig,
    Bandwidth,
    BandwidthUnit,
    CapacityUnit,
    ComputationUnit,
    Connection,
    DataType,
    MemoryHierarchy,
    MemoryInstance,
    MemoryLevel,
    MemorySize,
    ParameterProvenance,
    SpatialUnrolling,
    StorageTarget,
)
from voyager_compiler.ir.plan import ExecutionOptions


class GemminiHardware(AcceleratorConfig):
    @property
    def vector_lanes(self):
        # Accumulator read/scale width; there is no general vector engine.
        return self.memory_instance("accumulator").word_bytes // 4


def hardware(
    *,
    dim=16,
    scratchpad_kib=256,
    accumulator_kib=64,
    dma_bytes_per_cycle=16,
    dram_bytes=16 << 20,
    scratchpad_banks=4,
    accumulator_banks=2,
):
    """Hardware facts, independent of numerical policy and timing calibration.

    DIM16 and DIM32 are the donor-validated family. The ISA bus packet width
    remains 16 bytes; DMA service bandwidth is a separate model parameter.
    """
    values = (
        dim,
        scratchpad_kib,
        accumulator_kib,
        dram_bytes,
        scratchpad_banks,
        accumulator_banks,
    )
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(
            "Gemmini dimensions, capacities and banks must be positive integers"
        )
    if dim < 4 or dim > 256 or dim & (dim - 1):
        raise ValueError("Gemmini DIM must be a power of two between 4 and 256")
    if not 0 < dma_bytes_per_cycle < float("inf"):
        raise ValueError("Gemmini DMA bandwidth must be finite and positive")
    for size, banks, word in (
        (scratchpad_kib << 10, scratchpad_banks, dim),
        (accumulator_kib << 10, accumulator_banks, 4 * dim),
    ):
        if size % (2 * banks * word):
            raise ValueError(
                "Gemmini memories require whole rows per bank and half context"
            )
    matrix = ComputationUnit(
        "matrix",
        (DataType("int8", 8),),
        frozenset(("matmul", "conv2d")),
        (SpatialUnrolling("IC", dim), SpatialUnrolling("OC", dim)),
    )

    def memory(name, size, banks, word, roles):
        return MemoryInstance(
            name,
            MemorySize(size, CapacityUnit.BYTES),
            frozenset(roles),
            banks=banks,
            word_bytes=word,
            allocation_alignment_bytes=word,
        )

    memories = MemoryHierarchy(
        (
            MemoryLevel(
                "on_chip",
                (
                    memory(
                        "scratchpad",
                        scratchpad_kib << 10,
                        scratchpad_banks,
                        dim,
                        (StorageTarget.ACTIVATION, StorageTarget.WEIGHT),
                    ),
                    memory(
                        "accumulator",
                        accumulator_kib << 10,
                        accumulator_banks,
                        4 * dim,
                        (StorageTarget.PSUM,),
                    ),
                ),
            ),
            MemoryLevel(
                "off_chip",
                (memory("dram", dram_bytes, 1, 16, tuple(StorageTarget)),),
            ),
        )
    )
    links = (
        Connection(
            "dram_sram",
            "dram",
            "scratchpad",
            Bandwidth(dma_bytes_per_cycle, BandwidthUnit.BYTES_PER_CYCLE),
        ),
        Connection(
            "accumulator_dram",
            "accumulator",
            "dram",
            Bandwidth(dma_bytes_per_cycle, BandwidthUnit.BYTES_PER_CYCLE),
        ),
    )
    return GemminiHardware(
        name=f"gemmini-ws{dim}",
        target_name="gemmini",
        frequency=None,
        computation_units=(matrix,),
        memory=memories,
        connections=links,
        execution=ExecutionOptions(1, False),
        provenance=(
            ParameterProvenance(
                "memory",
                "documented",
                "voyager-gemmini-dse/DSE-PROVENANCE.json",
                "Parameterized WS INT8/INT32 family; see target README for calibration scope.",
            ),
        ),
    )


def parameters(config):
    """Freeze command-relevant hardware facts with the saved target IR."""
    dim, columns = config.pe_array_size
    if dim != columns:
        raise ValueError("Gemmini command expansion requires a square array")
    sp, acc = (
        config.memory_instance(key) for key in ("scratchpad", "accumulator")
    )
    return dict(
        dim=dim,
        scratchpad_kib=sp.size.value // 1024,
        accumulator_kib=acc.size.value // 1024,
        scratchpad_banks=sp.banks,
        accumulator_banks=acc.banks,
        dma_bytes_per_cycle=config.connection_bytes_per_cycle("dram_sram"),
        dram_bytes=config.memory_instance("dram").size.value,
    )
