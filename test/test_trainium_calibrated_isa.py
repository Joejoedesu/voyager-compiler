"""Measured geometry selection and static compiled-ISA model boundaries."""

from dataclasses import replace

import pytest

from voyager_compiler.trainium import isa
from voyager_compiler.trainium.compiled_analysis import predict
from voyager_compiler.trainium.dependencies import compute_graph
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Builder
from voyager_compiler.trainium.program_analysis import analyze_selected


def test_short_k_changes_timing_not_instruction_expansion():
    hw = neuron_core(3)
    short = isa.matmul(512, 128, 64, 32, hw)
    full = isa.matmul(512, 128, 128, 32, hw)
    assert short.instructions == full.instructions
    assert short.implementation == full.implementation == "nki.matmul.float32"
    assert hw.timing_profile.operation(short.timing_implementation).evaluate(
        0
    ) == (1707, 2765)
    assert full.timing_implementation == full.implementation
    for m, n, k, bits in [
        (256, 128, 64, 32),
        (512, 64, 64, 32),
        (512, 128, 32, 32),
        (512, 128, 64, 16),
    ]:
        ex = isa.matmul(m, n, k, bits, hw)
        assert ex.timing_implementation == ex.implementation
    assert (
        neuron_core(2).timing_profile.operation(short.timing_implementation)
        is None
    )


def test_candidate_and_selected_plan_both_consume_short_k_law():
    hw = neuron_core(3)
    graph, _ = compute_graph(hw, 512, 128, 64, 32, True, TrainiumTuning(matmul_orientation="weights"))
    mm = [n for n in graph.nodes if n.implementation == "nki.matmul.float32"]
    assert len(mm) == 1
    assert (mm[0].occupancy_ns, mm[0].latency_ns) == (1707, 2765)
    b = Builder([], hw)
    b.add("a = nl.ndarray((64,128), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("b = nl.ndarray((64,512), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("c = nisa.nc_matmul(a,b)")
    b.program.outputs = ("c",)
    result = analyze_selected(b.program, hw)
    assert result["service_ns"]["TensorE"] == 1707
    assert result["prediction_ns"] == (
        result["service_ns"]["VectorE"]
        + 2765
        + hw.timing_profile.fixed_kernel_ns
    )
    assert not result["unknown_completion"]


def stream(pc, *, src=0, dst=4096, channels=32, dtype="fp32", stride=1):
    return dict(
        opcode="STREAM_TRANSPOSE",
        subgroup="Vector",
        compiler_pc=pc,
        raw_bir_id=pc,
        instruction_type="TRANSPOSE",
        operands=f"src={dtype}@{src:#x}[{stride},1,1,1][32,1,1,1] dst={dtype}@{dst:#x}[1,1,1,1][32,1,1,1] channels={channels}",
    )


def test_stream_context_changes_and_control_waits():
    data = {
        "instructions": [
            stream(0),
            stream(1),
            dict(
                opcode="NOP",
                subgroup="Vector",
                compiler_pc=2,
                raw_bir_id=2,
                operands="",
            ),
            stream(3),
            stream(4, src=32 * 262144, dst=64 * 262144),
            stream(5, src=32 * 262144, dst=64 * 262144),
        ]
    }
    result, graph = predict(data)
    operations = [n for n in graph.nodes if n.resource == "VectorE"]
    assert [n.occupancy_ns for n in operations] == [234, 93, 93, 234, 93]
    assert result["unsupported_instructions"] == 0
    assert (
        result["selected_timing_laws"][
            "nki.stream_transpose.float32.SBUF.VectorE.switch"
        ]
        == 2
    )
    data["instructions"].reverse()
    shuffled, _ = predict(data)
    assert shuffled["prediction_us"] == result["prediction_us"]


def test_dependent_stream_uses_ready_latency_not_just_issue_rate():
    a, b = stream(0), stream(1)
    a["operands"] += " S[4](test)++@complete"
    b["operands"] += " S[4](test)>=1"
    hw = replace(
        neuron_core(3),
        timing_profile=replace(
            neuron_core(3).timing_profile, fixed_kernel_ns=0
        ),
    )
    result, _ = predict({"instructions": [a, b]}, hardware=hw)
    assert result["prediction_us"] == (234 + 233) / 1000
    assert result["encoded_waits"] == 1


def test_other_vector_payload_resets_stream_context():
    fill = dict(
        opcode="MEMSET",
        subgroup="Vector",
        compiler_pc=1,
        raw_bir_id=1,
        operands="dst=fp32@0x4000[1,1,1,1][32,1,1,1] channels=32",
    )
    result, _ = predict({"instructions": [stream(0), fill, stream(2)]})
    assert result["selected_timing_laws"] == {
        "nki.stream_transpose.float32.SBUF.VectorE.switch": 2
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(channels=16),
        dict(dtype="bf16"),
        dict(src=0x2000000),
        dict(dst=0x2000000),
        dict(stride=2),
    ],
)
def test_uncharacterized_stream_is_explicitly_unsupported(kwargs):
    result, _ = predict({"instructions": [stream(0, **kwargs)]})
    assert result["prediction_us"] is None
    assert result["unsupported_instructions"] == 1


@pytest.mark.parametrize(
    "data",
    [
        dict(instructions=[], duration=1),
        dict(instructions=[stream(0) | {"duration": 1}]),
        dict(instructions=[], static_dma=[{"duration": 1}]),
        dict(instructions=[], dma_audit={"duration": 1}),
    ],
)
def test_profile_timing_cannot_enter_prediction(data):
    with pytest.raises(ValueError, match="static"):
        predict(data)


def test_unknown_opcode_and_unresolved_dependencies_fail():
    with pytest.raises(ValueError, match="unsupported compiled opcode"):
        predict({"instructions": [stream(0) | {"opcode": "LOAD_MASK_SELECT"}]})
    ins = stream(0)
    ins["operands"] += " S[99](unknown)>=1"
    with pytest.raises(ValueError, match="unresolved semaphore"):
        predict({"instructions": [ins]})
