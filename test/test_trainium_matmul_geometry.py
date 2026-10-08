"""Operand geometry, pipeline context, and consistent search/ISA timing."""

from dataclasses import replace
from unittest.mock import patch
import pytest
from voyager_compiler.trainium import isa
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Builder
from voyager_compiler.trainium.program_analysis import analyze_selected
from voyager_compiler.trainium.dependencies import evaluate_graph
from voyager_compiler.trainium.orientation import matrix_choice
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.codegen.transform.tiling.execution import (
    evaluate_graph as evaluate,
)


def test_moving_and_stationary_feed_overlap():
    hw = neuron_core(3)

    def cost(m, ms, ss):
        return isa.matmul(
            m,
            128,
            128,
            32,
            hw,
            moving_stride=ms,
            stationary_stride=ss,
            streaming=True,
        )

    narrow = cost(128, 1, 1)
    moving = cost(128, 16, 1)
    stationary = cost(128, 1, 16)
    assert narrow.instructions == moving.instructions == stationary.instructions
    assert (
        moving.timing_override[0]
        == stationary.timing_override[0]
        > narrow.timing_override[0]
    )
    assert cost(128, 16, 16).timing_override[0] == moving.timing_override[0]
    assert (
        cost(512, 1, 16).timing_override[0]
        == cost(512, 1, 1).timing_override[0]
    )
    assert moving.timing_override[1] > stationary.timing_override[1]


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((128, 128, 128, 16), dict(moving_stride=16, stationary_stride=1)),
        ((512, 128, 64, 32), dict(moving_stride=16, stationary_stride=1)),
        ((128, 64, 128, 32), dict(moving_stride=16, stationary_stride=1)),
        ((128, 128, 128, 32), dict(moving_stride=None, stationary_stride=1)),
        ((128, 128, 128, 32), dict(moving_stride=32, stationary_stride=1)),
    ],
)
def test_uncharacterized_geometry_keeps_existing_law(args, kwargs):
    hw = neuron_core(3)
    new = isa.matmul(*args, hw, **kwargs)
    old = isa.matmul(*args, hw)
    assert new.timing_override is None
    assert (new.tensor_cycles, new.timing_implementation) == (
        old.tensor_cycles,
        old.timing_implementation,
    )


def test_startup_completion_is_separate_from_forwarding():
    hw = neuron_core(3)
    kw = dict(moving_stride=1, stationary_stride=1)
    cold = isa.matmul(128, 128, 128, 32, hw, **kw).timing_override
    steady = isa.matmul(
        128, 128, 128, 32, hw, **kw, streaming=True
    ).timing_override
    assert cold[1] > steady[1] > steady[2]
    assert cold[0] == steady[0] == cold[2] == steady[2]


def test_actual_views_and_fresh_operands_determine_timing():
    hw = neuron_core(3)
    b = Builder([], hw)
    for line in [
        "a = nl.ndarray((128,128), dtype=nl.float32, buffer=nl.sbuf)",
        "b = nl.ndarray((128,2048), dtype=nl.float32, buffer=nl.sbuf)",
        "p = nl.arange(128)[:,None]",
        "f = nl.arange(128)[None,:]",
        "z = nl.zeros((128,128), dtype=nl.float32, buffer=nl.psum)",
        "z[...] += nisa.nc_matmul(a,b[p,f*16])",
        "z[...] += nisa.nc_matmul(a,b[p,f*16+1])",
        "c = nisa.tensor_copy(a, engine=nisa.vector_engine)",
        "z[...] += nisa.nc_matmul(c,b[p,f*16+2])",
    ]:
        b.add(line)
    seen = []

    def capture(graph):
        seen.append(graph)
        return evaluate(graph)

    with patch(
        "voyager_compiler.trainium.program_analysis.evaluate_graph", capture
    ):
        r = analyze_selected(b.program, hw)
    assert r["matmul_geometry_events"] == {"cold": 2, "steady": 1}
    mm = [n for n in seen[0].nodes if n.name.endswith("_nc_matmul")]
    assert all(n.occupancy_ns > 400 for n in mm)
    assert any(d.milestone == "forward" for d in mm[1].dependencies)
    assert any(d.milestone == "result" for d in mm[2].dependencies)


def test_choice_changes_with_input_layout_without_orientation_penalty():
    hw = neuron_core(3)
    t = TrainiumTuning(matmul_operands="reuse")
    first = matrix_choice(
        hw,
        128,
        2048,
        2048,
        32,
        False,
        t,
        output_row=True,
        weight_layout="k_partitioned",
    )
    last = matrix_choice(
        hw,
        128,
        2048,
        1024,
        32,
        False,
        t,
        input_row=True,
        weight_layout="k_partitioned",
    )
    assert first["orientation"] == "activations"
    assert last["orientation"] == "weights"
    assert len(first["candidates"]) == len(last["candidates"]) == 2


def test_compiled_operand_geometry_matches_selected_service():
    from voyager_compiler.trainium.compiled_analysis import predict

    instructions = []
    for pc, opcode in enumerate(["LDWEIGHTS", "MATMUL", "LDWEIGHTS", "MATMUL"]):
        operands = (
            "fp32_mode=LOW transpose_mode=DISABLED src=fp32@0x0[-1,0,0][128,1,1] 128*128"
            if opcode == "LDWEIGHTS"
            else "acc_flags=0 fp32_mode=LOW_HIGH src=fp32@0x1000[16,0,0][128,1,1] dst=0x2000000[1,0,0][128,1,1] 128*128"
        )
        instructions.append(
            dict(
                opcode=opcode,
                subgroup="Tensor",
                compiler_pc=pc,
                raw_bir_id=1,
                instruction_type="REGULAR",
                operands=operands,
            )
        )
    result, graph = predict({"instructions": instructions})
    node = next(n for n in graph.nodes if n.resource == "TensorE")
    expected = isa.matmul(
        128, 128, 128, 32, neuron_core(3), moving_stride=16, stationary_stride=1
    ).timing_override
    assert node.occupancy_ns == pytest.approx(expected[0])
    assert node.latency_ns == pytest.approx(expected[1])
