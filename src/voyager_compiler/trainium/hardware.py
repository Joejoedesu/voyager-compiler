"""One physical NeuronCore, never a device-wide bandwidth/compute budget."""

from functools import partial

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
    OperandCapability,
    OperationCapability,
    ParameterProvenance,
    SpatialUnrolling,
    StorageTarget,
)

ARCH = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/"
SOURCES = {
    2: ARCH + "trainium_inferentia2_arch.html",
    3: ARCH + "trainium2_arch.html",
}
TARGETS = {"trainium-v2": 2, "trainium-v3": 3}


def neuron_core(version=2, args=None):
    if version not in (2, 3):
        raise ValueError("Expected NeuronCore version 2 or 3")
    # Ignore Voyager parser defaults; these are distinct physical topologies.
    dtype = lambda name, bits: DataType(name, bits)
    fp32 = dtype("float32", 32)
    types = (dtype("bfloat16", 16), dtype("float16", 16), fp32)
    matrix = ComputationUnit(
        "TensorE",
        modes=(
            ComputeMode(
                "dense",
                tuple(
                    OperationCapability(
                        "matmul",
                        (
                            OperandCapability(
                                "stationary", dt, AccessMode.READ, ("SBUF",)
                            ),
                            OperandCapability(
                                "moving", dt, AccessMode.READ, ("SBUF",)
                            ),
                            OperandCapability(
                                "result", fp32, AccessMode.WRITE, ("PSUM",)
                            ),
                        ),
                    )
                    for dt in types
                ),
                (SpatialUnrolling("K", 128), SpatialUnrolling("M", 128)),
            ),
        ),
    )
    units = (matrix,) + tuple(
        ComputationUnit(
            name, supported_dtypes=types, supported_operations=frozenset(ops)
        )
        for name, ops in (
            ("VectorE", ("add", "reduce", "copy", "maxpool")),
            ("ScalarE", ("activation", "reciprocal", "sqrt")),
            ("GpSimdE", ("scalar_control",)),
            ("DMA", ("copy",)),
            ("SyncE", ("synchronize",)),
        )
    )
    data_targets = frozenset((StorageTarget.ACTIVATION, StorageTarget.WEIGHT))
    memory = MemoryHierarchy(
        (
            MemoryLevel(
                "accumulator",
                (
                    MemoryInstance(
                        "PSUM",
                        MemorySize(2 << 20, CapacityUnit.BYTES),
                        frozenset((StorageTarget.PSUM,)),
                        partitions=128,
                        banks=8,
                        single_bank_allocation=True,
                    ),
                ),
            ),
            MemoryLevel(
                "scratch",
                (
                    MemoryInstance(
                        "SBUF",
                        MemorySize(
                            (24 if version == 2 else 28) << 20,
                            CapacityUnit.BYTES,
                        ),
                        data_targets,
                        partitions=128,
                    ),
                ),
            ),
            MemoryLevel(
                "external",
                (
                    MemoryInstance(
                        "HBM",
                        MemorySize(None, CapacityUnit.BYTES),
                        data_targets,
                    ),
                ),
            ),
        )
    )
    unknown = Bandwidth(None, BandwidthUnit.GB_PER_SECOND)
    routes = (
        ("SBUF", "TensorE", False),
        ("TensorE", "PSUM", False),
        ("SBUF", "VectorE", True),
        ("PSUM", "VectorE", True),
        ("SBUF", "ScalarE", True),
        ("PSUM", "ScalarE", True),
        ("SBUF", "GpSimdE", True),
        ("HBM", "DMA", True),
        ("DMA", "SBUF", True),
    )
    return TrainiumConfig(
        name=f"trainium-v{version}",
        backend="trainium",
        frequency=2.8 if version == 2 else 2.4,
        computation_units=units,
        memory=memory,
        connections=tuple(
            Connection(
                f"{a}_{b}",
                a,
                b,
                Bandwidth(
                    272 if version == 2 else 368, BandwidthUnit.GB_PER_SECOND
                )
                if a in ("HBM", "DMA")
                else unknown,
                bidirectional=bi,
                latency_ns=1300 if a == "HBM" else None,
            )
            for a, b, bi in routes
        ),
        provenance=(
            ParameterProvenance(
                "connections.HBM_DMA.bandwidth.value",
                "documented",
                "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/deep-dives/nki-dma-bandwidth-guide.html",
                "16-engine aggregate per core, not device HBM or paper roofline normalization",
            ),
            ParameterProvenance(
                "connections.HBM_DMA.latency_ns",
                "assumed",
                "https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/deep-dives/nki-dma-bandwidth-guide.html",
                "Approximate documented cross-engine latency. Serial per-command charging is a conservative model assumption, not calibrated initiation interval.",
            ),
            ParameterProvenance(
                "frequency",
                "documented",
                SOURCES[version],
                "TensorE clock; other engines have distinct clocks.",
            ),
        )
        + tuple(
            ParameterProvenance(field, "documented", SOURCES[version])
            for field in (
                "memory.levels.scratch.instances.SBUF.size.value",
                "memory.levels.scratch.instances.SBUF.partitions",
                "memory.levels.accumulator.instances.PSUM.size.value",
                "memory.levels.accumulator.instances.PSUM.partitions",
                "memory.levels.accumulator.instances.PSUM.banks",
            )
        ),
    )


def validate_tile(
    m, n, k, *, dtype="float32", banks=1, sbuf_bytes=0, version=2
):
    """Validate ISA axes: m=stationary free, n=moving free, k=partition.

    The converter maps logical GEMM B to stationary and A to moving, so these
    m/n names are reversed from logical GEMM M/N in mapping cost reports.
    Packed FP8/sparse modes are excluded.
    """
    if version not in (2, 3) or dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError(
            "Unsupported target or precision; no implicit FP8 reinterpretation"
        )
    if any(type(x) is not int or x <= 0 for x in (m, n, k, banks)):
        raise ValueError(
            "Tile extents and bank count must be positive integers"
        )
    if m > 128 or k > 128 or n > 512 or banks > 8:
        raise ValueError(
            "TensorE requires M,K <= 128, N <= 512 and at most 8 PSUM banks"
        )
    if sbuf_bytes < 0 or sbuf_bytes > (24 if version == 2 else 28) << 20:
        raise ValueError("SBUF capacity exceeded")


def register_targets():
    from voyager_compiler.targets import (
        Target,
        register_backend,
        register_target,
    )
    from voyager_compiler.trainium.backend import TrainiumBackend

    register_backend("trainium", TrainiumBackend())
    for name, version in TARGETS.items():
        register_target(
            Target(name, "trainium", "trainium", partial(neuron_core, version))
        )


def validate_concurrent_access(version, accesses):
    """Conservative documented shared-interface checks for future scheduling.

    Each access is (engine, memory, bank). Bank is required for PSUM; the current
    lowerer relies on NKI's dependency insertion and does not schedule engines
    independently. v2 shares VectorE/GpSimdE SBUF and VectorE/ScalarE PSUM access.
    v3 permits the latter pair only when they use different PSUM banks.
    """
    config = neuron_core(version)
    for engine, memory, bank in accesses:
        if not (
            config.connected(engine, memory) or config.connected(memory, engine)
        ):
            raise ValueError(f"No direct interface: {engine}/{memory}")
        if memory == "PSUM" and (type(bank) is not int or not 0 <= bank < 8):
            raise ValueError("PSUM bank must be in [0, 8)")
    for i, (engine, memory, bank) in enumerate(accesses):
        for other, store, other_bank in accesses[i + 1 :]:
            if memory != store:
                continue
            pair = {engine, other}
            if (
                version == 2
                and memory == "SBUF"
                and pair == {"VectorE", "GpSimdE"}
            ):
                raise ValueError(
                    "NeuronCore-v2 VectorE/GpSimdE share the SBUF interface"
                )
            if (
                memory == "PSUM"
                and pair == {"VectorE", "ScalarE"}
                and (version == 2 or bank == other_bank)
            ):
                raise ValueError("VectorE/ScalarE PSUM conflict")


class TrainiumConfig(AcceleratorConfig):
    """Views needed by the shared scheduler; physical stores remain SBUF/PSUM.

    The shared scratchpad arena maps to SBUF. TensorE results use a temporary
    PSUM tile and are copied to the scheduled SBUF destination. No fake Voyager
    L1 stores or vector/PSUM streaming links are added to the hardware graph.
    Compatibility rate views initialize shared geometry helpers only. Trainium
    candidate ranking uses trainium.cost, not Voyager SRAM/stream equations.
    """

    pe_array_size = property(lambda s: (128, 128))
    vector_lanes = property(lambda s: 128)
    matrix_vector_lanes = property(lambda s: 128)
    accumulator_lanes = property(lambda s: 128)
    input_buffer_size = property(lambda s: 128)
    weight_buffer_size = property(lambda s: 128)
    accum_buffer_size = property(lambda s: 128)
    double_buffered_accum_buffer = property(lambda s: False)
    scratchpad_size = property(lambda s: s.memory_instance("SBUF").size.value)
    # Reserve 4 MiB (32 KiB per partition) for compiler and instruction temporaries.
    usable_scratchpad_size = property(lambda s: s.scratchpad_size - (4 << 20))
    scratchpad_offset = property(lambda s: 4 << 20)
    num_banks = property(lambda s: None)
    usable_banks = property(lambda s: None)
    bank_size = property(lambda s: None)
    bank_width = property(lambda s: 512)
    double_buffered_l2 = property(lambda s: True)
    num_slots = property(lambda s: 2)
    dram_size = property(lambda s: None)
    dram_bandwidth = property(lambda s: s.connection("HBM_DMA").bandwidth.value)
    bytes_per_cycle = property(lambda s: s.dram_bandwidth / s.frequency)
    dram_access_latency = property(lambda s: s.connection("HBM_DMA").latency_ns)
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
        return 128 * element_bits

    def compute_bandwidth(self, unit="vector"):
        return 512

    def connection(self, name):
        if name == "matrix_vector_stream":
            # Compatibility cost-model query; does not declare a direct link.
            return Connection(
                "cost_proxy",
                "TensorE",
                "PSUM",
                Bandwidth(512, BandwidthUnit.BYTES_PER_CYCLE),
            )
        return super().connection(name)
