"""Operand domains and hidden state survive alternate scheduling policies."""

import pytest

from voyager_compiler.trainium.compiled_analysis import predict
from voyager_compiler.trainium.operand_timing import cost, key


def ins(op, pc, operands, engine="Vector"):
    return dict(
        opcode=op,
        compiler_pc=pc,
        raw_bir_id=pc,
        instruction_type="REGULAR",
        subgroup=engine,
        operands=operands,
    )


def mask(pc):
    return ins(
        "LOAD_MASK_SELECT", pc, "masks=" + ",".join(["0"] * 4 + ["32"] * 28)
    )


def shuffle(pc):
    return ins(
        "STREAM_SHUFFLE",
        pc,
        "src=bf16@0x0[1][64] dst=bf16@0x4000[1][64] channels=32",
    )


def test_mask_raw_and_overwrite_hazard_without_source_order():
    r, g = predict(
        {"instructions": [mask(0), shuffle(1), mask(2)]}, in_order=False
    )
    assert r["hidden_mask_register_dependencies"] == 3
    assert any(
        e.source == 0 and e.milestone == "result"
        for e in g.nodes[1].dependencies
    )
    assert any(
        e.source == 1 and e.milestone == "read" for e in g.nodes[2].dependencies
    )
    assert all(n.latency_ns is not None for n in g.nodes)
    with pytest.raises(ValueError, match="mask producer"):
        predict({"instructions": [shuffle(0)]})
    with pytest.raises(ValueError, match="mask configuration"):
        predict({"instructions": [ins("LOAD_MASK_SELECT", 0, "masks=0,33")]})


def test_affine_fill_register_dependency_and_overwrite():
    move = ins("MOVE", 0, "dtype=uint32 $R[61]=0xc61c0000", "GpSimd")
    select = ins(
        "TENSOR_SCALAR_AFFINE_SELECT",
        1,
        "cmp_op=AS_GTE fill_reg=$R[61] mask_base=0 "
        "mask_pattern=[-1,1,1,1][64,1,1,1] "
        "src=fp32@0x0[1][64] dst=fp32@0x4000[1][64] channels=128",
        "GpSimd",
    )
    r, g = predict(
        {
            "instructions": [
                move,
                select,
                move | {"compiler_pc": 2, "raw_bir_id": 2},
            ]
        },
        in_order=False,
    )
    assert r["hidden_mask_register_dependencies"] == 3
    assert g.nodes[1].dependencies[0].source == 0
    assert any(
        e.source == 1 and e.milestone == "read" for e in g.nodes[2].dependencies
    )
    with pytest.raises(ValueError, match="producer"):
        predict({"instructions": [select]})


def test_multi_output_reduction_uses_input_cardinality_and_operand_path():
    def reduce(width):
        return ins(
            "TENSOR_REDUCE",
            0,
            f"op=ADD dim=X src=fp32@0x0[1,2][2,{width//2}] "
            f"dst=fp32@0x4000[1][{width//2}] channels=128",
        )

    a, ga = predict({"instructions": [reduce(8)]}, context_model=True)
    b, gb = predict({"instructions": [reduce(32)]}, context_model=True)
    assert a["unknown_completion_events"] == b["unknown_completion_events"] == 0
    assert ga.nodes[0].issue_ns == gb.nodes[0].issue_ns
    assert ga.nodes[0].latency_ns < gb.nodes[0].latency_ns


def test_operand_domain_does_not_invent_strided_copy_latency():
    d = dict(
        opcode="tensor_copy",
        engine="VectorE",
        source_dtype="bfloat16",
        dtype="float32",
        source_memory="SBUF",
        destination_memory="SBUF",
        partitions=128,
        free=64,
        source_free=64,
        function="",
        source_shape=(64,),
        source_strides=(1,),
        destination_shape=(64,),
        destination_strides=(1,),
    )
    assert key(d).endswith("||0")
    assert cost(d, 64 / 0.96) is not None
    assert cost(d | {"source_strides": (5,)}, 64 / 0.96) is None
    assert cost(d | {"free": 65536}, 64 / 0.96) is None


def test_predicated_scalar_requires_predicate_geometry():
    bad = ins(
        "COPY_PREDICATED_SCALAR",
        0,
        "pred=uint8@0x0[1][32] src=0xff7fffff "
        "dst=fp32@0x4000[1][64] channels=128",
    )
    with pytest.raises(ValueError, match="predicated scalar"):
        predict({"instructions": [bad]})


def test_pool_singletons_and_binary_source_zero_are_normalized():
    seen = {}
    records = [
        ins(
            "POOL",
            0,
            "AVERAGE dim=XY scale=0.125 "
            "src=fp32@0x0[1,1,1,1][8,1,1,1] "
            "dst=fp32@0x4000[1,1,1,1][1,1,1,1] channels=128",
        ),
        ins(
            "TENSOR_TENSOR",
            1,
            "op=ADD "
            "src0=bf16@0x0[1][64] src1=bf16@0x8000[1][64] "
            "dst=bf16@0xc000[1][64] channels=4",
        ),
    ]
    predict(
        {"instructions": records},
        context_model=True,
        descriptor_observer=seen.update,
    )
    assert seen["u0_POOL"]["function"] == "add"
    assert seen["u0_POOL"]["reduction_rank"] == 1
    assert seen["u1_TENSOR_TENSOR"]["source_dtype"] == "bfloat16"
    assert seen["u1_TENSOR_TENSOR"]["opcode"] == "tensor_tensor"


def test_binary_second_input_dtype_is_a_coverage_boundary():
    binary = ins(
        "TENSOR_TENSOR",
        0,
        "op=ADD "
        "src0=bf16@0x0[1][64] src1=bf16@0x8000[1][64] "
        "dst=bf16@0xc000[1][64] channels=4",
    )
    known, _ = predict({"instructions": [binary]}, context_model=True)
    assert known["unknown_completion_events"] == 0
    _, packed_graph = predict({"instructions": [binary]}, context_model=True)
    assert packed_graph.nodes[0].issue_ns == pytest.approx(64 / 0.96)
    unknown, _ = predict(
        {
            "instructions": [
                binary
                | {
                    "operands": binary["operands"].replace(
                        "src1=bf16", "src1=fp16"
                    )
                }
            ]
        },
        context_model=True,
    )
    assert unknown["unknown_completion_events"] == 1


def test_partial_partition_accumulator_has_no_unmeasured_early_forwarding():
    producer = ins(
        "ACTIVATE",
        0,
        "EXP src=fp32@0x0[1][512] "
        "dst=bf16@0x4000[1][512] channels=4 "
        "accumulator_cmd=ZERO_ACCUMULATE",
        "Scalar",
    )
    readback = ins(
        "ACTIVATION_READ_ACCUMULATOR",
        1,
        "dst=fp32@0x8000[1][1] channels=4",
        "Scalar",
    )
    r, g = predict({"instructions": [producer, readback]}, context_model=True)
    assert r["unknown_completion_events"] == 0
    assert g.nodes[0].forward_ns == g.nodes[0].latency_ns
    assert g.nodes[1].read_ns <= g.nodes[1].latency_ns


def test_selected_activation_matches_native_implicit_bias_operand():
    from voyager_compiler.trainium.hardware import neuron_core
    from voyager_compiler.trainium.instruction_plan import Builder
    from voyager_compiler.trainium.program_analysis import analyze_selected

    hw = neuron_core(3)
    b = Builder([], hw)
    b.add("a = nl.ndarray((128,64), dtype=nl.bfloat16, buffer=nl.sbuf)")
    b.add("z = nisa.activation(nl.copy, a)")
    b.program.outputs = ("z",)
    selected = []
    result = analyze_selected(
        b.program,
        hw,
        execution_model="context",
        graph_observer=lambda g, d: selected.append((g, d)),
    )
    assert not result["unknown_completion"]
    native = ins(
        "ACTIVATE",
        0,
        "IDENTITY bias_ptr=fp32@0x8000 "
        "src=bf16@0x0[1][64] dst=bf16@0x4000[1][64] "
        "channels=128 scale=1.0 imm=0.0 accumulator_cmd=IDLE",
        "Scalar",
    )
    _, graph = predict({"instructions": [native]}, context_model=True)
    operation = next(n for n in selected[0][0].nodes if n.resource == "ScalarE")
    assert operation.latency_ns == pytest.approx(graph.nodes[0].latency_ns)
