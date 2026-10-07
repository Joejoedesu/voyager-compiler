"""Finite replay checks for bounded search scoring, including buffer reuse."""

import pytest

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency as D,
    OperationEvent as E,
    RepeatedGraph as G,
    evaluate_graph,
)
from voyager_compiler.trainium.dependencies import (
    estimate_graph,
    SteadyTiming,
)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_window_preserves_reuse_dependencies_and_boundary_work(depth):
    count = 10001
    graph = G(
        (
            E("initial_load", "dma", 10, 10, 100, period=count),
            E(
                "compute",
                "tensor",
                20,
                20,
                80,
                dependencies=(D(0), D(2, depth)),
            ),
            E("copy", "vector", 10, 10, 40, dependencies=(D(1),)),
            E(
                "final_store",
                "dma",
                10,
                10,
                150,
                dependencies=(D(2),),
                period=count,
                phase=count - 1,
            ),
        ),
        count,
    )
    result = estimate_graph(graph)
    exact = evaluate_graph(graph)
    assert isinstance(result, SteadyTiming)
    assert result.sample_repetitions < 100
    assert result.duration_ns == pytest.approx(exact.duration_ns)
    assert result.service_ns == exact.service_ns
    assert result.period % depth == 0


def test_periodic_loads_and_split_reduction_keep_cadence_and_remainder():
    graph = G(
        (
            E("load", "dma", 10, 10, 100, period=4),
            E("compute", "tensor", 20, 20, 60, dependencies=(D(0), D(1, 1))),
            E(
                "combine",
                "vector",
                5,
                5,
                20,
                dependencies=(D(1), D(2, 1)),
                period=4,
                except_phase=True,
            ),
            E(
                "store",
                "dma",
                10,
                10,
                120,
                dependencies=(D(1), D(2)),
                period=4,
                phase=3,
            ),
        ),
        10003,
    )
    result = estimate_graph(graph)
    assert isinstance(result, SteadyTiming)
    exact = evaluate_graph(graph)
    assert result.duration_ns == pytest.approx(exact.duration_ns)
    assert result.service_ns == exact.service_ns
    assert result.sample_repetitions % 4 == graph.repetitions % 4


def test_interior_one_time_work_requires_finite_replay():
    graph = G(
        (
            E("work", "tensor", 1, 1, 1),
            E("interior", "tensor", 100, 100, 100, period=30000, phase=15000),
        ),
        30000,
    )
    result = estimate_graph(graph)
    assert not isinstance(result, SteadyTiming)
    assert result == evaluate_graph(graph)


def test_short_graph_uses_finite_replay():
    graph = G((E("work", "tensor", 1, 1, None),), 5)
    result = estimate_graph(graph)
    assert not isinstance(result, SteadyTiming)
    assert result.unknown_latency == ("work",)
    assert result == evaluate_graph(graph)


@pytest.mark.parametrize("rows", [64, 128])
def test_softmax_window_outlasts_transient_resource_overlap(rows):
    from voyager_compiler.trainium.hardware import neuron_core
    from voyager_compiler.trainium.execution import TrainiumTuning
    from voyager_compiler.trainium.lowering import tile_graph

    shape = (rows, 4096)
    template = tile_graph(
        neuron_core(3),
        ("softmax",),
        shape,
        (shape,),
        TrainiumTuning(isa_lowering=True),
    )
    graph = G(template.nodes, 4096)
    result = estimate_graph(graph)
    exact = evaluate_graph(graph)
    assert isinstance(result, SteadyTiming)
    assert result.sample_repetitions < graph.repetitions
    assert result.duration_ns == pytest.approx(exact.duration_ns, rel=0.005)
    assert dict(result.service_ns) == pytest.approx(dict(exact.service_ns))


def test_long_reload_cadence_preserves_boundaries_and_buffer_reuse():
    count = 16384
    graph = G(
        (
            E("identity", "dma", 10, 10, 100, period=count),
            E("weight", "dma", 10, 10, 100, period=1024),
            E("input", "dma", 10, 10, 100, dependencies=(D(4, 2),)),
            E(
                "compute",
                "tensor",
                20,
                20,
                80,
                dependencies=(D(0), D(1), D(2)),
            ),
            E("copy", "vector", 10, 10, 40, dependencies=(D(3),)),
            E("store", "dma", 10, 10, 150, dependencies=(D(4),)),
        ),
        count,
    )
    result = estimate_graph(graph)
    exact = evaluate_graph(graph)
    assert isinstance(result, SteadyTiming)
    assert result.evaluation == "startup_steady_tail_compressed_cadence"
    assert result.sample_repetitions <= count // 16
    assert result.duration_ns == pytest.approx(exact.duration_ns, rel=0.005)
    assert dict(result.service_ns) == pytest.approx(dict(exact.service_ns))
