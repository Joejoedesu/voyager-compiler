"""Read-only timing sensitivity for the measured GEMM-first orientation pair.

Alternative dependency assumptions are diagnostics, never production calibration.
"""
from dataclasses import replace
import json
from pathlib import Path

from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.orientation import orientation_graph
from voyager_compiler.trainium.dependencies import evaluate_graph
from voyager_compiler.codegen.transform.tiling.execution import RepeatedGraph, Dependency
from trainium_neff_gap_audit import timeline


def main():
    hw = neuron_core(3)
    output = {}
    for orientation in ("weights", "activations"):
        graph, _ = orientation_graph(hw, 128, 2048, 2048, 32, False,
            TrainiumTuning(matmul_operands="reuse", matmul_orientation=orientation),
            output_row=True, weight_layout="k_partitioned")
        timing = evaluate_graph(graph)
        starts, ends, _ = timeline(graph)
        evictions = []
        for i, event in enumerate(graph.nodes):
            if not event.implementation.startswith("nki.copy.PSUM."):
                continue
            parents = [d.source for d in event.dependencies if graph.nodes[d.source].name.startswith("matmul_")]
            if not parents:
                continue
            first = parents[0]
            while True:
                earlier = [d.source for d in graph.nodes[first].dependencies if graph.nodes[d.source].name.startswith("matmul_")]
                if not earlier:
                    break
                first = earlier[0]
            clear = next(d.source for d in graph.nodes[first].dependencies if graph.nodes[d.source].name.startswith("psum_clear"))
            evictions.append((clear, i))
        assert len(evictions) == (16 if orientation == "weights" else 4)
        intervals = [(starts[c],1) for c,_ in evictions] + [(ends[e],-1) for _,e in evictions]
        live = peak = 0
        for _, delta in sorted(intervals, key=lambda x:(x[0],x[1])):
            live += delta
            peak = max(peak,live)
        assert live == 0 and peak > 0
        matmul = next(n for n in graph.nodes if n.name.startswith("matmul_"))
        forward = RepeatedGraph(tuple(replace(n,forward_ns=max(n.occupancy_ns,n.issue_ns))
            if n.name.startswith("matmul_") else n for n in graph.nodes))
        serial = list(graph.nodes)
        for (_, previous), (clear, _) in zip(evictions, evictions[1:]):
            serial[clear] = replace(serial[clear], dependencies=serial[clear].dependencies+(Dependency(previous),))
        output[orientation] = dict(duration_us=timing.duration_ns/1000,
            service_us={k:v/1000 for k,v in timing.service_ns},
            matmul_event=dict(issue_ns=matmul.issue_ns, occupancy_ns=matmul.occupancy_ns,
                             completion_ns=matmul.latency_ns, forward_ns=matmul.forward_ns),
            output_accumulators=len(evictions), peak_overlapping_output_accumulators=peak,
            forward_at_occupancy_diagnostic_us=evaluate_graph(forward).duration_ns/1000,
            one_output_accumulator_diagnostic_us=evaluate_graph(RepeatedGraph(tuple(serial))).duration_ns/1000)
    result = dict(scope="GEMM-first one 128-row tile including output conversion; counterfactual dependency assumptions only; production model unchanged", results=output)
    path = Path("results/trainium/weight-layout-2026-10-08/ranking-dependency-diagnostic.json")
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
