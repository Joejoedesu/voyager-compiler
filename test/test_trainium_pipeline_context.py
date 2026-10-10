from unittest.mock import patch
from dataclasses import replace
from voyager_compiler.codegen.transform.tiling.execution import (
    OperationEvent as E,
    Dependency as D,
    RepeatedGraph as G,
    evaluate_graph,
)
from voyager_compiler.trainium.calibrated_isa import (
    pipeline_timing,
    sectors,
    evaluate,
)


def test_access_sectors_saturate_not_address_span():
    assert [sectors(128, s) for s in [1, 2, 3, 4, 8, 24]] == [
        32,
        64,
        96,
        128,
        128,
        128,
    ]


def test_pipeline_uses_predicted_overlap():
    params = {
        "models": {
            "transpose_stream": {
                "completion_ns": 631,
                "issue_ns": 213,
                "max_stride": 24,
            }
        }
    }
    desc = {
        k: dict(
            transpose=True,
            dtype="float32",
            partitions=128,
            source_free=128,
            source_stride=1,
        )
        for k in ["a", "b"]
    }
    with patch(
        "voyager_compiler.trainium.calibrated_isa.parameters",
        return_value=params,
    ):
        a = E("a", "TensorE", 53, 53, 477)
        b = E("b", "TensorE", 53, 53, 477)
        independent = evaluate_graph(
            G((a, b)), event_timing=pipeline_timing(desc)
        )
        assert independent.duration_ns == 844
        dependent = evaluate_graph(
            G((a, replace(b, dependencies=(D(0),)))),
            event_timing=pipeline_timing(desc),
        )
        assert dependent.duration_ns == 954
        assert evaluate_graph(G((a, b))).duration_ns == 530


def test_dynamic_timing_cannot_rewrite_graph():
    import pytest

    with pytest.raises(ValueError):
        evaluate_graph(
            G((E("a", "TensorE", 1, 1, 1),)),
            event_timing=lambda n, t: replace(n, resource="VectorE"),
        )


def test_unknown_copy_geometry_keeps_unknown():
    d = dict(
        dtype="float32",
        partitions=128,
        opcode="tensor_copy",
        engine="ScalarE",
        free=128,
        source_memory="SBUF",
        source_stride=2,
        destination_stride=4,
    )
    assert evaluate(d, 100, None, None) is None


def test_callback_state_does_not_leak_between_candidates():
    params = {
        "models": {
            "transpose_stream": {
                "completion_ns": 631,
                "issue_ns": 213,
                "max_stride": 24,
            }
        }
    }
    desc = {
        "a": dict(
            transpose=True,
            dtype="float32",
            partitions=128,
            source_free=128,
            source_stride=1,
        )
    }
    with patch(
        "voyager_compiler.trainium.calibrated_isa.parameters",
        return_value=params,
    ):
        for _ in range(2):
            assert (
                evaluate_graph(
                    G((E("a", "TensorE", 53, 53, 477),)),
                    event_timing=pipeline_timing(desc),
                ).duration_ns
                == 477
            )


def test_identity_callback_preserves_repeated_graphs():
    import random

    rng = random.Random(82)
    for trial in range(50):
        nodes = []
        for i in range(12):
            deps = tuple(
                D(
                    j,
                    milestone=rng.choice(
                        ["result", "issue", "read", "forward"]
                    ),
                )
                for j in range(i)
                if rng.random() < 0.15
            )
            nodes.append(
                E(
                    str(i),
                    rng.choice(["TensorE", "ScalarE", "VectorE"]),
                    rng.randrange(1, 10),
                    rng.randrange(1, 10),
                    rng.randrange(10, 30),
                    dependencies=deps,
                )
            )
        graph = G(tuple(nodes), repetitions=3)
        assert evaluate_graph(graph) == evaluate_graph(
            graph, event_timing=lambda n, start: n
        )


def test_shortk_same_geometry_different_pipeline_context():
    d = dict(
        dtype="float32",
        partitions=128,
        opcode="nc_matmul",
        moving=512,
        stationary=128,
        contraction=64,
        moving_stride=1,
        stationary_stride=1,
    )
    fresh = evaluate(dict(d, streaming=False), 10, 10, None)
    dense = evaluate(dict(d, streaming=True), 10, 10, None)
    assert fresh == (1707, 2077, 1707)
    assert dense == (1707, 2765, 1707)
    timing = pipeline_timing({"a": d, "b": d})
    assert timing(E("a", "TensorE", 10, 10, 10), 0).latency_ns == 2077
    assert timing(E("b", "TensorE", 10, 10, 10), 1707).latency_ns == 2765
    assert timing(E("a", "TensorE", 10, 10, 10), 10000).latency_ns == 2077


def test_copy_interpolates_width_but_not_unknown_stride():
    d = dict(
        dtype="float32",
        partitions=128,
        opcode="tensor_copy",
        engine="ScalarE",
        free=256,
        source_memory="SBUF",
        source_stride=3,
        destination_stride=1,
    )
    assert evaluate(d, 100, 100, None) is not None
    assert evaluate(dict(d, source_stride=33), 100, 100, None) is None
