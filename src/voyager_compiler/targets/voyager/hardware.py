"""Voyager physical graph factory; software buffering is compilation policy."""

from voyager_compiler.hardware import (
    AcceleratorConfig,
    Bandwidth,
    BandwidthUnit,
    CapacityUnit,
    ComputationUnit,
    Connection,
    DataType,
    FusionStage,
    ISAPipeline,
    MemoryHierarchy,
    MemoryInstance,
    MemoryLevel,
    MemorySize,
    SpatialUnrolling,
    StorageTarget,
    UnrollingRef,
)
from voyager_compiler.ir.plan import ExecutionOptions


def voyager_config(
    *,
    pe_array_size=(32, 32),
    vector_unit_width=None,
    matrix_vector_unit_width=None,
    accumulator_width=None,
    frequency=1.0,
    input_buffer_size=1024,
    weight_buffer_size=1024,
    accum_buffer_size=1024,
    double_buffered_accum_buffer=False,
    scratchpad_size=None,
    scratchpad_offset=0,
    num_banks=None,
    bank_width=None,
    double_buffered_l2=True,
    async_pipeline=True,
    dram_size=16.0,
    dram_bandwidth=64.0,
    dram_access_latency=100.0,
    dram_energy_per_bit=6.25,
):
    """Instantiate Voyager, the default profile. All Voyager defaults live here.

    Other accelerators construct AcceleratorConfig from their own typed graph;
    they do not inherit Voyager's topology or its measured timing constants.
    """
    ic, oc = UnrollingRef("matrix", "IC"), UnrollingRef("matrix", "OC")
    vl = UnrollingRef("vector", "lanes")
    # Null hardware is retained for transform()'s no-padding mode. Its
    # otherwise unused widths have a concrete fallback, without dangling refs.
    active = pe_array_size is not None

    def dim(name, factor, fallback):
        return SpatialUnrolling(
            name,
            factor if factor is not None or active else 1,
            fallback if active else None,
        )

    dtypes = tuple(DataType(f"int{bits}", bits) for bits in range(1, 33)) + (
        DataType("bfloat16", 16),
        DataType("float32", 32),
        DataType("fp8_e4m3", 8),
        DataType("fp8_e5m2", 8),
        DataType("fp8_e5m3", 8),
        DataType("posit8_1", 8),
        DataType("lut4_to_int6", 4),
    )
    matrix_ops = frozenset(
        ("conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx")
    )
    vector_ops = frozenset(
        (
            "dequantize",
            "add",
            "sub",
            "mul",
            "div",
            "exp",
            "abs",
            "relu",
            "gelu",
            "sigmoid",
            "silu",
            "tanh",
            "hardtanh",
            "layer_norm",
            "softmax",
            "quantize",
            "quantize_mx",
            "quantize_mx_outlier",
            "quantize_affine",
            "max_pool2d",
            "avg_pool2d",
            "adaptive_avg_pool2d",
            "linear",
            "matmul",
        )
    )
    units = (
        ComputationUnit(
            "matrix",
            dtypes,
            matrix_ops,
            (
                (
                    SpatialUnrolling("IC", pe_array_size[0]),
                    SpatialUnrolling("OC", pe_array_size[1]),
                )
                if active
                else ()
            ),
        ),
        ComputationUnit(
            "vector",
            dtypes,
            vector_ops,
            (
                dim("lanes", vector_unit_width, oc),
                dim("pool_channels", accumulator_width, vl),
            ),
            # Sphinx measurement: issue and fill/drain per normalization pass.
            launch_overhead_cycles=96,
        ),
        ComputationUnit(
            "matrix_vector",
            dtypes,
            frozenset(("linear", "matmul", "linear_mx", "matmul_mx")),
            (dim("lanes", matrix_vector_unit_width, oc),),
        ),
        ComputationUnit(
            "sparse_matrix",
            dtypes,
            frozenset(("spmm",)),
            (
                (
                    SpatialUnrolling("IC", fallback=ic),
                    SpatialUnrolling("OC", fallback=oc),
                )
                if active
                else ()
            ),
            # Sparse accumulator ring drain/restart (measured 7.3–7.5 cycles).
            row_overhead_cycles=8,
        ),
    )
    activation, psum, weight = (
        StorageTarget.ACTIVATION,
        StorageTarget.PSUM,
        StorageTarget.WEIGHT,
    )
    shared = frozenset(StorageTarget)
    elements, byte, rows = (
        CapacityUnit.ELEMENTS,
        CapacityUnit.BYTES,
        CapacityUnit.ROWS,
    )
    scope = (ic, oc) if active else ()
    levels = (
        MemoryLevel(
            "PE",
            tuple(
                MemoryInstance(
                    f"pe_{role.value}",
                    MemorySize(1, elements),
                    frozenset((role,)),
                    access_cost=1,
                )
                for role in (activation, psum, weight)
            ),
            scope,
        ),
        MemoryLevel(
            "L1",
            (
                MemoryInstance(
                    "input_buffer",
                    MemorySize(input_buffer_size, elements),
                    frozenset((activation,)),
                    (ic,) if active else (),
                    access_cost=10,
                ),
                MemoryInstance(
                    "accum_buffer",
                    MemorySize(accum_buffer_size, elements),
                    frozenset((psum,)),
                    (oc,) if active else (),
                    buffering=2 if double_buffered_accum_buffer else 1,
                    access_cost=10,
                ),
                MemoryInstance(
                    "weight_buffer",
                    MemorySize(weight_buffer_size, elements),
                    frozenset((weight,)),
                    (oc,) if active else (),
                    access_cost=10,
                ),
                MemoryInstance(
                    "spmm_scales",
                    MemorySize(32, rows),  # SpMMUnit.h's DoubleBuffer<32>
                    frozenset((StorageTarget.SCALE,)),
                    buffering=2,
                ),
            ),
        ),
        MemoryLevel(
            "L2",
            (
                MemoryInstance(
                    "scratchpad",
                    MemorySize(scratchpad_size, byte),
                    shared,
                    banks=num_banks,
                    word_bytes=bank_width,
                    reserved_bytes=scratchpad_offset,
                    access_cost=100,
                    bank_switch_cycles=8,  # ordered read-master round trip
                ),
            ),
        ),
        MemoryLevel(
            "DRAM",
            (
                MemoryInstance(
                    "dram",
                    MemorySize(
                        None if dram_size is None else dram_size * 1024**3, byte
                    ),
                    shared,
                    access_cost=1000,
                    energy_pj_per_bit=dram_energy_per_bit,
                ),
            ),
        ),
    )
    sram_rate = Bandwidth(
        None if active else 1,
        BandwidthUnit.ELEMENTS_PER_CYCLE,
        memory_port="scratchpad",
        unrolling=scope,
        reduction="min",
    )
    external_rate = Bandwidth(dram_bandwidth, BandwidthUnit.GB_PER_SECOND)
    register_rate = Bandwidth(1, BandwidthUnit.ELEMENTS_PER_CYCLE)
    edges = [
        Connection(
            "dram_sram",
            "dram",
            "scratchpad",
            external_rate,
            latency_ns=dram_access_latency,
        ),
        Connection("sram_vector", "scratchpad", "vector", external_rate),
        Connection(
            "sram_matrix_vector", "scratchpad", "matrix_vector", external_rate
        ),
        Connection("sram_sparse", "scratchpad", "sparse_matrix", sram_rate),
        Connection(
            "matrix_vector_stream",
            "matrix",
            "vector",
            Bandwidth(
                None,
                BandwidthUnit.ELEMENTS_PER_CYCLE,
                unrolling=(oc,) if active else (vl,),
            ),
            bidirectional=False,
            # Matrix FIFO, vector input FIFO and intermediate pipeline stages;
            # the existing model uses 24 vectors for the measured 20–40 range.
            buffer_depth=24,
        ),
        Connection("sram_scales", "scratchpad", "spmm_scales", sram_rate),
        Connection(
            "scales_sparse", "spmm_scales", "sparse_matrix", register_rate
        ),
    ]
    for key, memory, role in (
        ("input", "input_buffer", activation),
        ("accum", "accum_buffer", psum),
        ("weight", "weight_buffer", weight),
    ):
        reg = f"pe_{role.value}"
        edges.extend(
            (
                Connection(f"sram_{key}", "scratchpad", memory, sram_rate),
                Connection(f"{key}_pe", memory, reg, register_rate),
                Connection(f"{key}_matrix", reg, "matrix", register_rate),
            )
        )
    quant = (
        "quantize",
        "quantize_mx",
        "quantize_mx_outlier",
        "quantize_affine",
    )
    gemm_quant = tuple(op for op in quant if op != "quantize_mx_outlier")
    anchor = FusionStage(
        "matrix",
        ("conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"),
        "matrix_tail",
    )
    dequant = FusionStage("vector", ("dequantize",))
    pipelines = (
        (
            anchor,
            dequant,
            FusionStage(
                "vector", ("add", "sub", "mul", "div"), "constant_divisor"
            ),
            FusionStage("vector", ("exp", "abs", "relu")),
            FusionStage("vector", ("add", "mul", "div"), "constant_divisor"),
            FusionStage("vector", gemm_quant),
        ),
        (
            anchor,
            dequant,
            FusionStage(
                "vector", ("gelu", "sigmoid", "silu", "tanh", "hardtanh")
            ),
            FusionStage("vector", gemm_quant),
        ),
        (
            FusionStage("vector", ("layer_norm", "softmax")),
            FusionStage("vector", quant),
        ),
    )
    return AcceleratorConfig(
        name="voyager",
        target_name="voyager",
        execution=ExecutionOptions(
            2 if double_buffered_l2 else 1, async_pipeline
        ),
        frequency=frequency,
        computation_units=units,
        memory=MemoryHierarchy(levels),
        connections=tuple(edges),
        isa_pipelines=tuple(
            ISAPipeline(name, stages)
            for name, stages in zip(
                (
                    "matrix_elementwise",
                    "matrix_activation",
                    "reduction_quantize",
                ),
                pipelines,
            )
        ),
    )
