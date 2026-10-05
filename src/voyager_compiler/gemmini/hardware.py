"""Lean hardware graph. Capacities are physical, not Voyager parser defaults."""

from voyager_compiler.hardware_config import (
    AcceleratorConfig,
    AccessMode,
    Bandwidth,
    BandwidthUnit,
    CapacityUnit,
    ComputationUnit,
    ComputeMode,
    Connection,
    DataType,
    MemoryHierarchy,
    MemoryInstance,
    MemoryLevel,
    MemorySize,
    MemoryPort,
    OperationTiming,
    OperandCapability,
    OperationCapability,
    ParameterProvenance,
    SpatialUnrolling,
    StorageTarget,
    TransferGeometry,
)

MEMORY_BASE = 0x80000000
MEMORY_BYTES = 16 * 1024 * 1024


def lean_config(args=None):
    """Exactly the pinned 16x16 WS instance; frequency is a time-unit convention."""
    if args is not None:
        geometry = getattr(args, "pe_array_size", None)
        if geometry is not None and tuple(geometry) != (16, 16):
            raise ValueError(
                "Pinned Gemmini Lean requires --pe_array_size 16,16"
            )
        for key, expected in (
            ("scratchpad_size", 262144),
            ("num_banks", 4),
            ("bank_width", 16),
        ):
            actual = getattr(args, key, None)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"Pinned Gemmini Lean requires {key}={expected}; select its INT8 recipe"
                )
    targets = frozenset(StorageTarget)
    sp = MemoryInstance(
        "scratchpad",
        MemorySize(262144, CapacityUnit.BYTES),
        targets,
        banks=4,
        row_bytes=16,
        buffering=2,
        reserved_bytes=0,
        ports=(MemoryPort("rw", AccessMode.READ_WRITE, scope="bank"),),
    )
    acc = MemoryInstance(
        "accumulator",
        MemorySize(65536, CapacityUnit.BYTES),
        frozenset({StorageTarget.PSUM}),
        banks=2,
        row_bytes=64,
        buffering=2,
        ports=(
            MemoryPort("read", AccessMode.READ, scope="bank"),
            MemoryPort("write", AccessMode.WRITE, scope="bank"),
        ),
    )
    dram = MemoryInstance(
        "dram",
        MemorySize(MEMORY_BYTES, CapacityUnit.BYTES),
        targets,
        word_bytes=16,
        allocation_alignment_bytes=16,
    )
    cmd = MemoryInstance(
        "instruction_queue",
        MemorySize(2, CapacityUnit.ELEMENTS),
        frozenset({StorageTarget.INDEX}),
    )
    units = (
        ComputationUnit(
            "matrix",
            spatial_unrolling=(
                SpatialUnrolling("rows", 16),
                SpatialUnrolling("cols", 16),
            ),
            modes=(
                ComputeMode(
                    "weight_stationary",
                    tuple(
                        OperationCapability(
                            op,
                            (
                                OperandCapability(
                                    "a",
                                    DataType("int8", 8),
                                    AccessMode.READ,
                                    ("scratchpad",),
                                ),
                                OperandCapability(
                                    "b",
                                    DataType("int8", 8),
                                    AccessMode.READ,
                                    ("scratchpad",),
                                ),
                                OperandCapability(
                                    "c",
                                    DataType("int32", 32),
                                    AccessMode.READ_WRITE,
                                    ("accumulator",),
                                ),
                            ),
                        )
                        for op in ("matmul", "conv2d")
                    ),
                    timing=OperationTiming(
                        startup_cycles=31,
                        issue_interval_cycles=2,
                        occupancy_cycles=4,
                    ),
                ),
            ),
        ),
    )
    rate = lambda n: Bandwidth(n, BandwidthUnit.BYTES_PER_CYCLE)
    return LeanConfig(
        name="gemmini",
        backend="gemmini",
        computation_units=units,
        memory=MemoryHierarchy(
            (
                MemoryLevel(
                    "local",
                    (sp, acc, cmd)
                    + tuple(
                        MemoryInstance(
                            name,
                            MemorySize(depth, CapacityUnit.ELEMENTS),
                            frozenset({StorageTarget.INDEX}),
                        )
                        for name, depth in (
                            ("reservation_load", 8),
                            ("reservation_execute", 16),
                            ("reservation_store", 4),
                            ("controller_load", 8),
                            ("controller_execute", 8),
                            ("controller_store", 2),
                        )
                    ),
                ),
                MemoryLevel("external", (dram,)),
            )
        ),
        connections=(
            Connection(
                "dram_dma",
                "dram",
                "scratchpad",
                rate(16),
                buffer_depth=64,
                latency_ns=8,
                target_port="rw",
                transfer_geometry=TransferGeometry(16, 64),
                service_resource="controller_load",
            ),
            Connection(
                "dram_acc",
                "dram",
                "accumulator",
                shared_with="dram_dma",
                latency_ns=None,
            ),
            Connection(
                "acc_to_dram",
                "accumulator",
                "dram",
                shared_with="dram_dma",
                bidirectional=False,
                source_port="read",
                latency_ns=None,
                service_bandwidth=rate(10),
                transfer_geometry=TransferGeometry(16, 16),
                service_resource="controller_store",
            ),
            Connection(
                "spad_array",
                "scratchpad",
                "matrix",
                rate(16),
                bidirectional=False,
                source_port="rw",
            ),
            Connection(
                "array_acc",
                "matrix",
                "accumulator",
                rate(64),
                bidirectional=False,
                target_port="write",
            ),
            Connection(
                "acc_feedback",
                "accumulator",
                "matrix",
                rate(64),
                bidirectional=False,
                source_port="read",
            ),
            Connection(
                "rocc",
                "instruction_queue",
                "matrix",
                rate(20),
                bidirectional=False,
            ),
        ),
        frequency=1.0,
        provenance=(
            ParameterProvenance(
                "connections.acc_to_dram.service_bandwidth",
                "measured",
                "results/gemmini/meta-compiler-work/dma-service/summary.json",
                notes="256 INT8 mvout commands with 4/8/16 rows: store-busy cycles 1639/3275/6553 for 16384/32768/65536 bytes. Approximately 10 bytes/cycle endpoint service, sharing the 16-byte/cycle bus. Extra service scales with rows, not command count; measured on the fixed-latency replay memory.",
            ),
            ParameterProvenance(
                "frequency",
                "assumed",
                notes="Unit conversion only; report cycles, not physical runtime",
            ),
            ParameterProvenance(
                "memory", "documented", "Gemmini/docs/interface.md"
            ),
            ParameterProvenance(
                "computation_units",
                "documented",
                "Gemmini/chipyard/generators/gemmini/src/main/scala/gemmini/Configs.scala",
                notes="leanConfig: WS, no full-width accumulator mvout, no array reads from accumulator or writes to scratchpad; bias through DMA",
            ),
            ParameterProvenance(
                "connections",
                "documented",
                "Gemmini/docs/interface.md",
                notes="CPU-free io.cmd replay; 16-byte uncached TileLink memory beat, 64 DMA requests; instruction delivery has no CPU cost",
            ),
        ),
    )


class LeanConfig(AcceleratorConfig):
    """Physical graph plus views for the shared compiler's scheduling interface.

    The four Interstellar levels below are mapping levels, not extra memories.
    L0 is the array, L1 its inner traversal, L2 explicit DMA buffers, L3 DRAM.
    ``vector_lanes`` denotes the DMA/scaler width; no vector ALU is fabricated.
    Accumulator-backed logical references use a distinct address arena. The
    converter assigns their physical rows across the full 64 KiB accumulator;
    the logical arena is bookkeeping, not a second physical allocation.
    """

    pe_array_size = property(
        lambda s: tuple(
            s.compute_unit("matrix").unrolling(d).factor
            for d in ("rows", "cols")
        )
    )
    accumulator_element_bytes = property(
        lambda s: s.memory_instance("accumulator").row_bytes
        // s.pe_array_size[1]
    )
    vector_lanes = property(lambda s: s.pe_array_size[1])
    matrix_vector_lanes = property(lambda s: s.pe_array_size[1])
    accumulator_lanes = property(lambda s: s.pe_array_size[1])
    input_buffer_size = property(lambda s: s.pe_array_size[1])
    weight_buffer_size = property(lambda s: s.pe_array_size[1])
    accum_buffer_size = property(
        lambda s: s.memory_instance("accumulator").size.value
        // s.memory_instance("accumulator").row_bytes
    )
    double_buffered_accum_buffer = property(lambda s: False)
    # The logical buffer address space concatenates two physical arenas.
    scratchpad_size = property(
        lambda s: sum(
            s.memory_instance(n).size.value
            for n in ("scratchpad", "accumulator")
        )
    )
    usable_scratchpad_size = property(
        lambda s: s.scratchpad_size - s.scratchpad_offset
    )
    scratchpad_offset = property(
        lambda s: s.pe_array_size[0] * s.memory_instance("scratchpad").row_bytes
    )
    num_banks = property(lambda s: None)
    usable_banks = property(lambda s: None)
    bank_size = property(lambda s: None)
    bank_width = property(lambda s: s.memory_instance("accumulator").row_bytes)
    bytes_per_cycle = property(
        lambda s: s.connection_bytes_per_cycle("dram_dma")
    )
    dram_bandwidth = property(lambda s: s.bytes_per_cycle)
    dram_access_latency = property(
        lambda s: s.connection("dram_dma").latency_ns
    )
    access_latency_cycles = property(
        lambda s: s.dram_access_latency * s.frequency
    )
    dram_energy_per_bit = property(lambda s: 0)
    bank_switch_cycles = property(lambda s: 0)
    kernel_launch_overhead = property(lambda s: 0)
    output_slack = property(lambda s: 0)
    spmm_row_cycles = property(lambda s: 0)
    spmm_scale_rows = property(lambda s: 0)

    def sram_bandwidth_bits(self, element_bits):
        return self.connection_bytes_per_cycle("spad_array") * 8

    def compute_bandwidth(self, unit="vector"):
        return self.pe_array_size[1]


def interstellar_memory(config):
    """Mapping-level capacities, retaining the existing analytical search.

    The target model minimizes execution time; traffic remains diagnostic.
    SP and accumulator have different bank granularities, so the target's
    size callback applies their placement policies independently rather than
    imposing one bank size on the entire Interstellar level.
    """
    sp = config.memory_instance("scratchpad")
    acc = config.memory_instance("accumulator")
    # L1 is an inner traversal capacity, not another physical memory. Allow
    # the whole legal software tile; L2 checks its actual live INT32 copies.
    return dict(
        buf_capacity_list=[
            [1, 1, 1],
            [
                sp.size.value - sp.reserved_bytes,
                acc.size.value // config.accumulator_element_bytes,
            ],
            [sp.size.value, acc.size.value],
            [config.memory_instance("dram").size.value],
        ],
        buf_access_cost_list=[[0, 0, 0], [0, 0], [0, 0], [1]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0], [0, 0], [0]],
        memory_partitions=[[0, 1, 2], [0, 1, 0], [0, 1, 0], [0, 0, 0]],
        para_count_list=[
            config.pe_array_size[0] * config.pe_array_size[1],
            1,
            1,
            1,
        ],
        bank_size_list=[None, None, None, None],
    )
