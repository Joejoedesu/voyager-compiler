"""The pinned 16x16 weight-stationary Gemmini storage and command interface."""

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


def hardware():
    matrix = ComputationUnit(
        "matrix",
        (DataType("int8", 8),),
        frozenset(("matmul", "conv2d")),
        (SpatialUnrolling("IC", 16), SpatialUnrolling("OC", 16)),
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
                        256 << 10,
                        4,
                        16,
                        (StorageTarget.ACTIVATION, StorageTarget.WEIGHT),
                    ),
                    memory(
                        "accumulator", 64 << 10, 2, 64, (StorageTarget.PSUM,)
                    ),
                ),
            ),
            MemoryLevel(
                "off_chip",
                (memory("dram", 16 << 20, 1, 16, tuple(StorageTarget)),),
            ),
        )
    )
    links = (
        Connection(
            "dram_sram",
            "dram",
            "scratchpad",
            Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
        ),
        Connection(
            "accumulator_dram",
            "accumulator",
            "dram",
            Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
        ),
    )
    return GemminiHardware(
        name="gemmini-ws16",
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
                "voyager-extensible/gemmini/hardware.py",
                "256 KiB scratchpad and 64 KiB accumulator; timing is an unfitted cycle model.",
            ),
        ),
    )
