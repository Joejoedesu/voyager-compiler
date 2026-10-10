"""Experimental, static-only physical ISA ordering and readiness models.

No native compilation or runtime observations enter candidate scoring. Engine
order is source order, not a promise of the native scheduler's chosen order.
Prefix completion represents cumulative semaphore readiness. Launch gates are
an explicitly separate hypothesis, inherited from earlier calibration cases.
"""

from collections import Counter
from dataclasses import replace
from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    OperationEvent,
    RepeatedGraph,
)

MODES = (
    "baseline",
    "ordered",
    "prefix",
    "prefix-ready",
    "primitives",
    "context",
    "context-ready",
    "window4",
    "window16",
    "window64",
    "pipeline",
    "pipeline-ready",
    "pipeline-startup",
    "pipeline-startup-ready",
)
READY_NS = {
    "DMAIssue": 893.5,
    "copy": 37.0,
    "transpose": 47.5,
    "compute": 100.0,
}


def transform(graph, mode):
    if mode not in MODES:
        raise ValueError("Unknown physical model: " + mode)
    if mode in ("baseline", "primitives", "pipeline", "pipeline-startup"):
        return graph, {}
    if graph.repetitions != 1:
        raise ValueError(
            "Physical context requires a fully expanded finite graph"
        )
    use_prefix = mode in (
        "prefix",
        "prefix-ready",
        "context",
        "context-ready",
        "pipeline-ready",
        "pipeline-startup-ready",
    )
    use_gates = mode in (
        "prefix-ready",
        "context-ready",
        "pipeline-ready",
        "pipeline-startup-ready",
    )
    nodes = []
    remap = {}
    prefix = {}
    last = {}
    previous_prefix = {}
    counts = Counter()
    horizon = (
        int(mode.removeprefix("window")) if mode.startswith("window") else 1
    )
    history = {}
    for i, n in enumerate(graph.nodes):
        dependencies = []
        cross = []
        for d in n.dependencies:
            src = graph.nodes[d.source]
            # Internal DMA dispatch->payload is one pipeline, not a handoff.
            cross_engine = src.resource != n.resource and not (
                {src.resource, n.resource} <= {"DMAIssue", "DMA"}
            )
            target = (
                prefix[d.source]
                if use_prefix and cross_engine and d.milestone == "result"
                else remap[d.source]
            )
            dep = replace(d, source=target)
            dependencies.append(dep)
            if cross_engine and d.milestone == "result":
                cross.append(dep)
        prior = history.setdefault(n.resource, [])
        if len(prior) >= horizon:
            dependencies.append(Dependency(prior[-horizon], milestone="issue"))
            counts["issue_order_edges"] += 1
        if use_gates and cross:
            kind = (
                "DMAIssue"
                if n.resource == "DMAIssue"
                else (
                    "transpose"
                    if "transpose" in n.implementation
                    else ("copy" if "copy" in n.implementation else "compute")
                )
            )
            # Exclude matmul until an independently characterized law exists.
            if (
                n.resource in ("DMAIssue", "ScalarE", "VectorE")
                or "transpose" in n.implementation
            ):
                gate = len(nodes)
                nodes.append(
                    OperationEvent(
                        f"{n.name}_ready",
                        "ReadinessGate",
                        0,
                        0,
                        READY_NS[kind],
                        dependencies=tuple(cross),
                    )
                )
                dependencies.append(Dependency(gate))
                counts["readiness_" + kind] += 1
        remap[i] = len(nodes)
        last[n.resource] = len(nodes)
        nodes.append(
            replace(n, dependencies=tuple(dict.fromkeys(dependencies)))
        )
        prior.append(remap[i])
        if use_prefix:
            edges = [Dependency(remap[i])]
            if n.resource in previous_prefix:
                edges.append(Dependency(previous_prefix[n.resource]))
            prefix[i] = len(nodes)
            previous_prefix[n.resource] = len(nodes)
            nodes.append(
                OperationEvent(
                    f"{n.name}_prefix",
                    "CompletionPrefix",
                    0,
                    0,
                    0,
                    dependencies=tuple(edges),
                )
            )
        else:
            prefix[i] = remap[i]
    return RepeatedGraph(tuple(nodes)), dict(counts)


def identity(mode):
    """Persist hypotheses and model contents with every score/cache identity."""
    import hashlib
    from pathlib import Path

    base = Path(__file__).parent
    record = dict(
        mode=mode,
        order_scope="source order / bounded source horizon; native order not available",
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if mode in (
        "primitives",
        "context",
        "context-ready",
        "pipeline",
        "pipeline-ready",
        "pipeline-startup",
        "pipeline-startup-ready",
    ):
        record["primitive_model_sha256"] = hashlib.sha256(
            (base / "isa_characterization.json").read_bytes()
        ).hexdigest()
        record["primitive_evaluator_sha256"] = hashlib.sha256(
            (base / "calibrated_isa.py").read_bytes()
        ).hexdigest()
    if mode in (
        "prefix-ready",
        "context-ready",
        "pipeline-ready",
        "pipeline-startup-ready",
    ):
        record["readiness_ns"] = dict(READY_NS)
        record["readiness_evidence"] = (
            "Frozen earlier four fused-kernel admission/handoff study; transferability hypothesis, not universal hardware constants."
        )
    return record
