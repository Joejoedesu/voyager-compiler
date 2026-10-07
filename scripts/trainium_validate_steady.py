"""Compare search-window estimates with finite replay of saved matrix plans."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    OperationEvent,
    RepeatedGraph,
)
from voyager_compiler.trainium.dependencies import (
    estimate_graph,
    evaluate_graph,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 32])
    args = parser.parse_args()
    records = []
    for path in sorted(args.artifacts.glob("*/hardware.json")):
        for index, plan in enumerate(
            json.loads(path.read_text())["execution_plans"]
        ):
            raw = plan["graph"]
            nodes = tuple(
                OperationEvent(
                    **dict(
                        node,
                        dependencies=tuple(
                            Dependency(**edge) for edge in node["dependencies"]
                        ),
                    )
                )
                for node in raw["nodes"]
            )
            for scale in args.scales:
                total = raw["repetitions"] * scale
                # Increase the number of steady periods while preserving a
                # once-only prefix/drain. This is a model test, not device data.
                scaled = tuple(
                    (
                        replace(
                            node,
                            period=total,
                            phase=(
                                total - 1
                                if node.phase == raw["repetitions"] - 1
                                else node.phase
                            ),
                        )
                        if node.period == raw["repetitions"] and scale != 1
                        else node
                    )
                    for node in nodes
                )
                graph = RepeatedGraph(scaled, total)
                evaluate_graph.cache_clear()
                estimate_graph.cache_clear()
                start = perf_counter()
                estimated = estimate_graph(graph)
                estimate_seconds = perf_counter() - start
                evaluate_graph.cache_clear()
                start = perf_counter()
                exact = evaluate_graph(graph)
                exact_seconds = perf_counter() - start
                record = dict(
                    case=path.parent.name,
                    plan=index,
                    scale=scale,
                    hardware_sha256=hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    template_nodes=len(nodes),
                    repetitions=total,
                    sample_repetitions=getattr(
                        estimated, "sample_repetitions", total
                    ),
                    estimate_ns=estimated.duration_ns,
                    exact_ns=exact.duration_ns,
                    relative_error=estimated.duration_ns / exact.duration_ns
                    - 1,
                    estimate_seconds=estimate_seconds,
                    exact_seconds=exact_seconds,
                    evaluation=getattr(
                        estimated, "evaluation", "exact_finite_graph"
                    ),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    if not records:
        raise ValueError("No saved matrix execution plans found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                scope="Analytical extrapolation versus finite replay; no hardware timing validation",
                results=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
