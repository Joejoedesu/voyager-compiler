"""Slice hazards and scheduling legality; no hardware or benchmark fitting."""

from dataclasses import replace
from types import SimpleNamespace
import random
import pytest
import numpy as np
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Builder
from voyager_compiler.trainium.region_dependencies import (
    refine,
    intersects,
    footprint,
)
from voyager_compiler.trainium.issue_schedule import schedule
from voyager_compiler.trainium.physical_context import transform, identity
from voyager_compiler.codegen.transform.tiling.execution import (
    OperationEvent as E,
    Dependency as D,
    RepeatedGraph as G,
    evaluate_graph,
)


def sliced():
    b = Builder(
        [SimpleNamespace(shape=(4, 8), dtype="float32")], neuron_core(3)
    )
    for name in ("x", "out"):
        b.add(f"{name} = nl.ndarray((4,8), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x[:,0:8:2], src=a0[:,0:8:2])")
    b.add("nisa.dma_copy(dst=x[:,1:8:2], src=a0[:,1:8:2])")
    b.add(
        "nisa.tensor_copy(dst=out[:,0:8:2], src=x[:,0:8:2], engine=nisa.scalar_engine)"
    )
    return b


def test_progression_intersection_matches_enumeration():
    rng = random.Random(41)
    for _ in range(2000):
        a = range(rng.randrange(20), rng.randrange(40), rng.randrange(1, 12))
        b = range(rng.randrange(20), rng.randrange(40), rng.randrange(1, 12))
        assert intersects(a, b) == bool(set(a) & set(b))


def test_strided_read_waits_for_its_writer_not_unrelated_last_writer():
    b = sliced()
    original = b.program.record()
    p, audit = refine(b.program)
    assert p.instructions[1].dependencies == ()
    assert p.instructions[2].dependencies == (0,)
    assert b.program.record() == original
    assert audit["refined_edges_removed"] > 0


def test_partial_overwrite_preserves_old_writer_on_uncovered_elements():
    b = sliced()
    b.add("nisa.dma_copy(dst=x[:,0:4:2], src=a0[:,0:4:2])")
    b.add("nisa.tensor_copy(dst=out, src=x, engine=nisa.scalar_engine)")
    p, _ = refine(b.program)
    assert {0, 1, 3}.issubset(p.instructions[4].dependencies)
    assert 2 in p.instructions[3].dependencies  # Write-after-read.


def test_conservative_fallback_never_drops_overlapping_writer():
    p, audit = refine(sliced().program, max_view_elements=1)
    assert 0 in p.instructions[1].dependencies
    assert 1 in p.instructions[2].dependencies
    assert audit["conservative_accesses"] > 0


def test_explicit_extra_edge_is_retained():
    b = sliced()
    b.add("nisa.dma_copy(dst=out[:,1:8:2], src=a0[:,1:8:2])")
    i = b.program.instructions[3]
    b.program.instructions[3] = replace(
        i, dependencies=tuple(sorted(set(i.dependencies) | {0}))
    )
    p, _ = refine(b.program)
    assert 0 in p.instructions[3].dependencies


def test_physical_reuse_edges_remain_required():
    b = sliced()
    b.program.allocate()
    p, _ = refine(b.program)
    for before, after in b.program.reuse_edges():
        assert before in p.instructions[after].dependencies
    assert p.placements == b.program.placements


def test_scheduler_moves_ready_work_ahead_of_blocked_same_engine_work():
    g = G(
        (
            E("producer", "TensorE", 1, 1, 100),
            E("blocked", "ScalarE", 10, 10, 10, dependencies=(D(0),)),
            E("ready", "ScalarE", 10, 10, 10),
        )
    )
    scheduled, _ = schedule(g, window=4)
    names = [n.name for n in scheduled.nodes]
    assert names.index("ready") < names.index("blocked")
    serialized, _ = schedule(g, window=1)
    assert [n.name for n in serialized.nodes].index("blocked") < [
        n.name for n in serialized.nodes
    ].index("ready")
    assert (
        evaluate_graph(transform(g, "scheduled-ready")[0]).duration_ns
        < evaluate_graph(transform(g, "context-ready")[0]).duration_ns
    )


def test_scheduler_keeps_forwarding_and_physical_reuse_dependencies():
    g = G(
        (
            E("a", "TensorE", 5, 5, 100, forward_ns=5),
            E(
                "b",
                "TensorE",
                5,
                5,
                100,
                dependencies=(D(0, milestone="forward"),),
            ),
            E("overwrite", "VectorE", 1, 1, 2, dependencies=(D(1),)),
        )
    )
    result, _ = schedule(g)
    b = next(n for n in result.nodes if n.name == "b")
    assert any(
        result.nodes[d.source].name == "a" and d.milestone == "forward"
        for d in b.dependencies
    )
    o = next(n for n in result.nodes if n.name == "overwrite")
    assert any(
        result.nodes[d.source].name == "b" and d.milestone == "result"
        for d in o.dependencies
    )


def test_schedule_identity_records_policy_and_dependency_code():
    a, b = identity("scheduled-ready", reorder_window=4), identity(
        "scheduled-ready", reorder_window=16
    )
    assert a != b
    assert "region_dependencies.py_sha256" in a
    with pytest.raises(ValueError):
        schedule(G(()), window=0)


def test_footprint_fast_path_and_general_views_match_address_sets():
    from itertools import product

    base = np.indices((6, 8))
    cases = [
        tuple(base),
        tuple(base[:, ::-1, ::2]),
        tuple(base.reshape(2, 4, 12)),
        (np.arange(6), np.arange(6)),
        (np.array([0, 1, 4]), np.array([2, 3, 5])),
        tuple(base[:, :0, :]),
    ]
    for points in cases:
        f = footprint(points, (6, 8))
        actual = set(zip(*(a.ravel() for a in points)))
        bound = set(product(*f.axes))
        assert actual <= bound
        if f.exact:
            assert actual == bound
    assert footprint(tuple(base), (6, 8)).exact
    assert not footprint((np.arange(6), np.arange(6)), (6, 8)).exact
