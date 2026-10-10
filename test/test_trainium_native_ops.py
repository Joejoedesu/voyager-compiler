"""Native opcode timing, hidden state, and static-only coverage contracts."""

import pytest

from voyager_compiler.trainium.compiled_analysis import predict, fields
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.native_timing import evaluate
from voyager_compiler.trainium.native_dma import partition_bytes, payload_ns


def instruction(op, pc, operands, engine="Vector", bir=None):
    return dict(
        opcode=op,
        compiler_pc=pc,
        raw_bir_id=pc if bir is None else bir,
        instruction_type="REGULAR",
        subgroup=engine,
        operands=operands,
    )


def scan(width=256, seed="fp32@0.000000", partitions=128, stride=1):
    return instruction(
        "TENSOR_TENSOR_SCAN",
        0,
        f"src0=fp32@0x0[{stride},1][{width},1] src1=fp32@0x4000[1,1][{width},1] "
        f"dst=fp32@0x8000[1,1][{width},1] ops=MULTIPLY,ADD channels={partitions} imm={seed}",
    )


def test_scan_tensor_seed_adds_read_completion_without_changing_service():
    _, a = predict({"instructions": [scan()]})
    r, b = predict({"instructions": [scan(seed="[fp32@0xc000]")]})
    assert a.nodes[0].issue_ns == b.nodes[0].issue_ns
    assert b.nodes[0].latency_ns - a.nodes[0].latency_ns == pytest.approx(60)
    assert not r["missing_laws"]
    _, wider = predict({"instructions": [scan(width=1024)]})
    assert wider.nodes[0].issue_ns / a.nodes[0].issue_ns == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(partitions=32),
        dict(stride=2),
        dict(width=16384),
        dict(seed="[fp32@0x2000000]"),
    ],
)
def test_scan_uncharacterized_domain_has_service_but_no_fabricated_completion(
    kwargs,
):
    r, graph = predict({"instructions": [scan(**kwargs)]})
    assert graph.nodes[0].occupancy_ns > 0
    assert graph.nodes[0].latency_ns is None
    assert r["unknown_completion_events"] == 1
    assert r["missing_laws"]["native.tensor_tensor_scan.VectorE"] == 1


def producer(pc, command="ZERO_ACCUMULATE"):
    return instruction(
        "ACTIVATE",
        pc,
        "EXP src=fp32@0x0[1,1,1][256,1,1] dst=fp32@0x4000[1,1,1][256,1,1] "
        f"channels=128 accumulator_cmd={command}",
        "Scalar",
    )


def readback(pc):
    return instruction(
        "ACTIVATION_READ_ACCUMULATOR",
        pc,
        "dst=fp32@0x8000[1][1] channels=128",
        "Scalar",
    )


def test_accumulator_raw_and_war_survive_disabling_engine_order():
    r, g = predict(
        {"instructions": [producer(0), readback(1), producer(2)]},
        in_order=False,
    )
    assert r["accumulator_state_dependencies"] == 2
    assert any(
        g.nodes[e.source].name.endswith("ACTIVATE") and e.milestone == "forward"
        for e in g.nodes[1].dependencies
    )
    assert any(
        g.nodes[e.source].name.endswith("ACTIVATION_READ_ACCUMULATOR")
        and e.milestone == "read"
        for e in g.nodes[2].dependencies
    )
    assert g.nodes[0].issue_ns < g.nodes[0].forward_ns < g.nodes[0].latency_ns
    assert g.nodes[1].read_ns < g.nodes[1].latency_ns


def test_accumulator_continuation_and_invalidated_state():
    r, _ = predict(
        {"instructions": [producer(0), producer(1, "ACCUMULATE"), readback(2)]},
        in_order=False,
    )
    assert r["accumulator_state_dependencies"] == 2
    for records in (
        [readback(0)],
        [producer(0, "ACCUMULATE")],
        [producer(0), producer(1, "IDLE"), readback(2)],
    ):
        with pytest.raises(ValueError, match="producer"):
            predict({"instructions": records})


def test_ordinary_activation_does_not_use_accumulator_calibration():
    i = producer(0, "IDLE")
    assert (
        evaluate(
            i["opcode"],
            "ScalarE",
            fields(i["operands"]),
            i["operands"],
            neuron_core(3),
        )
        is None
    )


def test_register_move_has_cost_and_rejects_unmeasured_forms():
    move = instruction("MOVE", 0, "dtype=uint32 $R[61]=0xc61c0000", "GpSimd")
    _, g = predict({"instructions": [move]})
    assert g.nodes[0].resource == "GpSimdE"
    assert g.nodes[0].issue_ns == g.nodes[0].latency_ns == 68
    with pytest.raises(ValueError, match="MOVE form"):
        predict(
            {
                "instructions": [
                    dict(move, operands="dtype=uint32 $R[61]=$R[60]")
                ]
            }
        )


def test_retained_stationary_operand_needs_no_duplicate_load_or_notification():
    load = instruction(
        "LDWEIGHTS",
        0,
        "src=bfloat16@0x0[-1,0,0][128,1,1] 128*128",
        "Tensor",
        bir=0,
    )
    matmul = instruction(
        "MATMUL",
        1,
        "S[5] (Tensor)++@complete src=bfloat16@0x4000[1,0,0][512,1,1] dst=0x2000000[1,0,0][512,1,1] 128*128",
        "Tensor",
        bir=0,
    )
    retained = dict(matmul, compiler_pc=2, raw_bir_id=1)
    r, _ = predict({"instructions": [load, matmul, retained]})
    assert r["retained_stationary_matmuls"] == 1
    assert r["static_instructions"] == 3
    assert r["mapped_operations"]["nki.matmul.bfloat16"] == 2
    with pytest.raises(ValueError, match="stationary operand"):
        predict({"instructions": [retained]})


def test_cast_long_dtype_alias_is_owned_by_production_adapter():
    cast = instruction(
        "CAST",
        0,
        "src=float32@0x2000000[1][128] dst=bfloat16@0x4000[1][128] channels=128",
        "Scalar",
    )
    r, _ = predict({"instructions": [cast]})
    assert r["native_cast_alias_count"] == 1
    assert r["mapped_operations"]["nki.copy.PSUM.bfloat16.ScalarE"] == 1


def descriptor():
    return dict(
        source_num_sb_partitions=3,
        source_offset="[[0x4080], [0x1004480], [0x1804480], [0x1804880]]",
        source_steps="[[1, 262144]], [[1, 262144]], [[1, 262144]], [[1, 262144]]",
        read_shape=[[1024, 1]] * 4,
    )


def test_fragmented_dma_repeated_partition_preserves_bytes_and_busiest_engine():
    loads = partition_bytes(descriptor(), True, 4096)
    assert loads == {0: 1024, 64: 1024, 96: 2048}
    assert payload_ns(loads, 16) == 2048
    with pytest.raises(ValueError, match="byte accounting"):
        partition_bytes(descriptor(), True, 6144)
    bad = descriptor() | {
        "source_steps": "[[2, 262144]], [[1, 262144]], [[1, 262144]], [[1, 262144]]"
    }
    with pytest.raises(ValueError, match="strides"):
        partition_bytes(bad, True, 4096)


def test_new_operations_do_not_admit_observed_timing():
    with pytest.raises(ValueError, match="static"):
        predict({"instructions": [scan() | {"duration": 1}]})
