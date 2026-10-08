import collections, hashlib, json, statistics
from dataclasses import replace
from pathlib import Path
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.orientation import orientation_graph
from voyager_compiler.trainium.dependencies import evaluate_graph
from voyager_compiler.codegen.transform.tiling.execution import (
    RepeatedGraph,
    Dependency,
)

root = Path("results/trainium/weight-layout-2026-10-08")
hw = neuron_core(3)
result = {}


def stats(xs):
    xs = sorted(xs)
    return dict(
        count=len(xs),
        median=statistics.median(xs),
        p10=xs[len(xs) // 10],
        p90=xs[len(xs) * 9 // 10],
    )


for orientation, mode, search in [
    ("weights", "full-auto", 4.697603892),
    ("activations", "full-k-activations", 5.282841225),
]:
    path = root / mode / "matmul_add_rmsnorm/profile.json"
    raw = path.read_bytes()
    p = json.loads(raw)
    groups = collections.defaultdict(list)
    for i in p["instruction"]:
        if (
            i.get("opcode") in ("MATMUL", "LDWEIGHTS")
            and i.get("instruction_type") == "REGULAR"
            and i["operands"].rstrip().endswith("128*128")
        ):
            groups[i["raw_bir_id"]].append(i)
    gs = sorted(groups.values(), key=lambda g: min(i["compiler_pc"] for i in g))
    assert len(gs) == (8192 if orientation == "weights" else 2048)
    assert all(len(g) == 4 for g in gs)
    spans = [
        max(i["timestamp"] + i["duration"] for i in g)
        - min(i["timestamp"] for i in g)
        for g in gs
    ]
    spacing = [
        min(i["timestamp"] for i in b) - min(i["timestamp"] for i in a)
        for a, b in zip(gs, gs[1:])
        if max(i["compiler_pc"] for i in a) + 1
        == min(i["compiler_pc"] for i in b)
    ]
    interval = statistics.median(spacing)
    latency = statistics.median(spans)
    graph, _ = orientation_graph(
        hw,
        128,
        2048,
        2048,
        32,
        False,
        TrainiumTuning(matmul_operands="reuse", matmul_orientation=orientation),
        output_row=True,
        weight_layout="k_partitioned",
    )
    baseline = evaluate_graph(graph).duration_ns

    def variant(issue=False, completion=False):
        return replace(
            graph,
            nodes=tuple(
                (
                    replace(
                        n,
                        issue_ns=interval if issue else n.issue_ns,
                        occupancy_ns=interval if issue else n.occupancy_ns,
                        latency_ns=latency if completion else n.latency_ns,
                    )
                    if n.name.startswith("matmul_")
                    else n
                )
                for n in graph.nodes
            ),
        )

    variants = {}
    for label, gr in [
        ("baseline", graph),
        ("observed_launch_spacing", variant(True)),
        ("observed_group_span", variant(False, True)),
        ("both_observed", variant(True, True)),
    ]:
        duration = evaluate_graph(gr).duration_ns
        variants[label] = dict(
            matrix_tile_us=duration / 1000,
            whole_search_ms=search + 32 * (duration - baseline) / 1e6,
        )
    # Only bound GEMM output accumulators; transpose temporaries remain unbounded.
    evictions = []
    for i, n in enumerate(graph.nodes):
        parents = [
            d.source
            for d in n.dependencies
            if graph.nodes[d.source].name.startswith("matmul_")
        ]
        if not n.implementation.startswith("nki.copy.PSUM.") or not parents:
            continue
        first = parents[0]
        while True:
            prior = [
                d.source
                for d in graph.nodes[first].dependencies
                if graph.nodes[d.source].name.startswith("matmul_")
            ]
            if not prior:
                break
            first = prior[0]
        clear = next(
            d.source
            for d in graph.nodes[first].dependencies
            if graph.nodes[d.source].name.startswith("psum_clear")
        )
        evictions.append((clear, i))
    bounded = list(graph.nodes)
    for j, (clear, _) in enumerate(evictions):
        if j >= 8:
            bounded[clear] = replace(
                bounded[clear],
                dependencies=bounded[clear].dependencies
                + (Dependency(evictions[j - 8][1]),),
            )
    variants["eight_gemm_accumulator_slots_only"] = dict(
        matrix_tile_us=evaluate_graph(RepeatedGraph(tuple(bounded))).duration_ns
        / 1000
    )
    summary = p["summary"][0]
    result[orientation] = dict(
        profile=str(path),
        profile_sha256=hashlib.sha256(raw).hexdigest(),
        main_matmul_calls=len(gs),
        observed_launch_spacing_ns=stats(spacing),
        observed_group_span_ns=stats(spans),
        tensor_active_ms=summary["tensor_engine_active_time"] * 1000,
        variants=variants,
    )
output = dict(
    scope="Read-only GEMM-first profile diagnosis. Observed launch spacing includes scheduling and operand-read effects: not an isolated hardware issue calibration. Group span is not a measured forwarding latency. Eight-slot variant constrains only GEMM accumulators, not transpose PSUM or SBUF temporaries. Whole-search substitutions preserve other stage costs; these are diagnostic oracles, not new predictions.",
    results=result,
)
(root / "ranking-trace-diagnostic.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print(json.dumps(output, indent=2))
