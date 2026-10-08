"""Search choices, conversion costs, and persisted orientation contracts."""

from dataclasses import replace
import json

import pytest

from voyager_compiler.trainium.execution import TrainiumTuning, matmul_panels
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.orientation import (
    matrix_choice,
    orientation_graph,
)
from voyager_compiler.trainium.dependencies import evaluate_graph


def test_rectangular_cover_and_hardware_limits():
    # Ragged dimensions exercise coverage, not just expected panel counts.
    import numpy as np

    for orientation in ("weights", "activations"):
        coverage = np.zeros((193, 577), dtype=int)
        work = 0
        for mi, ni, m, n, ks in matmul_panels(193, 577, 257, orientation):
            stationary, moving = (n, m) if orientation == "weights" else (m, n)
            assert stationary <= 128 and moving <= 512
            assert sum(k for _, k in ks) == 257 and max(k for _, k in ks) <= 128
            coverage[mi : mi + m, ni : ni + n] += 1
            work += m * n * sum(k for _, k in ks)
        assert (coverage == 1).all()
        assert work == 193 * 577 * 257


def test_output_consumer_changes_choice_without_fitted_rates():
    hw = neuron_core(3)
    tuning = TrainiumTuning(matmul_operands="reuse")
    generic = matrix_choice(hw, 128, 128, 128, 32, False, tuning)
    row = matrix_choice(hw, 128, 128, 128, 32, False, tuning, output_row=True)
    assert generic["orientation"] == "weights"
    assert row["orientation"] == "activations"
    assert {c["orientation"] for c in row["candidates"]} == {
        "weights",
        "activations",
    }
    for choice, output_row in ((generic, False), (row, True)):
        for candidate in choice["candidates"]:
            graph, _ = orientation_graph(
                hw,
                128,
                128,
                128,
                32,
                False,
                replace(tuning, matmul_orientation=candidate["orientation"]),
                output_row=output_row,
            )
            assert (
                candidate["prediction_ns"] == evaluate_graph(graph).duration_ns
            )


@pytest.mark.parametrize("orientation", ["weights", "activations"])
def test_forced_choice_is_preserved(orientation):
    choice = matrix_choice(
        neuron_core(3),
        128,
        512,
        256,
        32,
        False,
        TrainiumTuning(matmul_orientation=orientation),
        output_row=True,
    )
    assert choice["orientation"] == orientation
    assert len(choice["candidates"]) == 1
    assert choice["candidates"][0]["matmul_calls"] == (
        8 if orientation == "weights" else 2
    )


def test_selected_choice_mismatch_is_rejected(tmp_path):
    from test_trainium_row_regions import case
    import voyager_compiler as vc
    from voyager_compiler.codegen.transform.bufferize import (
        BufferizationOptions,
    )
    from voyager_compiler.trainium.planning import select_plan

    graph, args, _, context = case(("gemm", "add", "norm"), m=128, k=128, n=128)
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(row_regions=True),
    )
    path = tmp_path / "hardware.json"
    record = json.loads(path.read_text())
    choice = record["row_regions"][0]["selected"]["matrix_choices"][0]
    assert choice["orientation"] == "activations"
    choice["orientation"] = "weights"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="orientation mismatch"):
        select_plan(tmp_path, context=context)
