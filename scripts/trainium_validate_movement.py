"""Detailed frozen-model versus matching NEFF/NTFF validation; never refits."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import csv
from trainium_report import summarize


def union(xs):
    end = total = 0
    for start, stop in sorted(xs):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def validate(case):
    row = summarize(case)
    if row["status"] != "pass":
        return row
    p = json.loads((case / "profile_full.json").read_text())
    ins = p["instruction"]
    plan = json.loads((case / "nki/plan.json").read_text())
    a = plan["program_analysis"]
    r = json.loads((case / "result.json").read_text())
    row["benchmark_p50_repeats_us"] = [x["p50_us"] for x in r["latencies"]]
    row["old_matrix_only_prediction_us"] = row.get("predicted_us")
    row["predicted_us"] = (
        None
        if a["whole_program_prediction_ns"] is None
        else a["whole_program_prediction_ns"] / 1000
    )
    row["timing_complete"] = a["whole_program_timing_complete"]
    row["unknown_completion"] = [
        name
        for e in r["estimates"]
        for name in (e.get("dependency_model") or {}).get(
            "unknown_completion", []
        )
    ]
    row["boundary_prediction_us"] = a["boundary_prediction_ns"] / 1000
    row["modeled_hbm_bytes"] = a["hbm"]["total_bytes"]
    row["profile_hbm_read_bytes"] = p["summary"][0]["hbm_read_bytes"]
    row["profile_hbm_write_bytes"] = p["summary"][0]["hbm_write_bytes"]
    row["hbm_difference_bytes"] = (
        row["profile_hbm_bytes"] - row["modeled_hbm_bytes"]
    )
    row["relative_hbm_error"] = (
        abs(row["hbm_difference_bytes"]) / row["profile_hbm_bytes"]
    )
    row["has_unmodeled_boundary_work"] = any(
        not b["timing_complete"] for b in a["boundaries"]
    )
    if row["predicted_us"] is not None:
        row["signed_error_percent"] = 100 * (
            row["predicted_us"] / row["nc_p50_us"] - 1
        )
        row["relative_latency_error"] = abs(row["signed_error_percent"]) / 100
        row["profile_error_percent"] = 100 * (
            row["predicted_us"] / row["profile_us"] - 1
        )
        row["actual_over_modeled"] = row["nc_p50_us"] / row["predicted_us"]
        row["predicted_useful_compute_fraction"] *= (
            row["old_matrix_only_prediction_us"] / row["predicted_us"]
        )
        row["benchmark_useful_compute_fraction"] = (
            row["predicted_useful_compute_fraction"]
            * row["predicted_us"]
            / row["nc_p50_us"]
        )
    math = [x for x in ins if x["opcode"] in ("LDWEIGHTS", "MATMUL")]
    regular = [x for x in math if x["instruction_type"] == "REGULAR"]
    trans = [x for x in math if x["instruction_type"] == "TRANSPOSE"]
    copies = [x for x in ins if x["opcode"] in ("COPY", "CAST", "MEMSET")]
    dmas = sorted(
        [x for x in ins if x["opcode"].startswith("DMA_")],
        key=lambda x: x["timestamp"],
    )
    direction = lambda x: (
        "load"
        if "src_table_index=" in x["operands"]
        else "store" if "dst_table_index=" in x["operands"] else "other"
    )
    gaps = {}
    for side in ["load", "store"]:
        group = [x for x in dmas if direction(x) == side]
        deltas = [
            b["timestamp"] - a["timestamp"] for a, b in zip(group, group[1:])
        ]
        gaps[side] = dict(
            count=len(group),
            median_issue_spacing_ns=st.median(deltas) if deltas else None,
            first_issue_ns=group[0]["timestamp"] if group else None,
            last_issue_ns=group[-1]["timestamp"] if group else None,
        )
    endmath = max((x["timestamp"] + x["duration"] for x in math), default=0)
    endcopy = max((x["timestamp"] + x["duration"] for x in copies), default=0)
    row["trace"] = dict(
        dma=gaps,
        tensor_regular_union_us=union(
            (x["timestamp"], x["timestamp"] + x["duration"]) for x in regular
        )
        / 1000,
        tensor_transpose_union_us=union(
            (x["timestamp"], x["timestamp"] + x["duration"]) for x in trans
        )
        / 1000,
        last_tensor_math_end_us=endmath / 1000,
        last_copy_end_us=endcopy / 1000,
        output_commands_after_all_math=sum(
            direction(x) == "store" and x["timestamp"] > endmath for x in dmas
        ),
        output_commands_after_all_copies=sum(
            direction(x) == "store" and x["timestamp"] > endcopy for x in dmas
        ),
        copy_opcodes=dict(
            Counter(x["label"] + ":" + x["opcode"] for x in copies)
        ),
        source_calls=a["source_calls"],
        compiled_dma_opcodes=dict(Counter(x["opcode"] for x in dmas)),
    )
    aggregates = [x for x in p["dma"] if x["aggregated"] == "yes"]
    row["trace"]["dma_payload_union_us"] = (
        union(
            (x["timestamp"], x["timestamp"] + x["duration"]) for x in aggregates
        )
        / 1000
    )
    row["trace"]["dma_trace_visible_read_bytes"] = sum(
        x["read_size"] for x in aggregates
    )
    row["trace"]["dma_trace_visible_write_bytes"] = sum(
        x["write_size"] for x in aggregates
    )
    row["trace"][
        "dma_trace_scope"
    ] = "Visible aggregated payload events can omit terminal transfers; summary counters are authoritative for total HBM bytes."
    index = {x["id"]: x for x in ins}
    ready = []
    for command in dmas:
        if direction(command) != "store":
            continue
        for pred in command.get("predecessors", []):
            producer = index.get(pred["id"])
            if (
                producer
                and producer["opcode"] in ("COPY", "CAST")
                and producer["label"] in ("Scalar", "Vector")
            ):
                end = producer["timestamp"] + producer["duration"]
                ready.append(
                    dict(
                        store=command["bir_instruction_name"],
                        producer=producer["bir_instruction_name"],
                        producer_end_ns=end,
                        store_issue_ns=command["timestamp"],
                        ready_to_issue_ns=command["timestamp"] - end,
                    )
                )
    groups = defaultdict(list)
    for op in math:
        groups[op["bir_instruction_name"]].append(op)
    spans = [
        dict(
            bir=k,
            kind=xs[0]["instruction_type"],
            span_ns=max(x["timestamp"] + x["duration"] for x in xs)
            - min(x["timestamp"] for x in xs),
            opcodes=dict(Counter(x["opcode"] for x in xs)),
        )
        for k, xs in groups.items()
    ]
    row["trace"]["tensor_groups"] = spans
    row["trace"]["output_producer_to_issue"] = ready
    row["trace"]["max_output_ready_to_issue_us"] = (
        max((x["ready_to_issue_ns"] for x in ready), default=0) / 1000
    )
    annotated = sum(
        x.get("hbm_read_bytes", 0) + x.get("hbm_write_bytes", 0) for x in ins
    )
    row["trace"]["neff_annotated_hbm_bytes"] = annotated
    assert annotated == row["profile_hbm_bytes"]
    throttles = [x for x in p.get("ham", []) if x["n"] and x["k"] < x["n"]]
    overlap = lambda events: union(
        (
            max(h["timestamp"], e["timestamp"]),
            min(h["timestamp"] + h["duration"], e["timestamp"] + e["duration"]),
        )
        for h in throttles
        for e in events
        if max(h["timestamp"], e["timestamp"])
        < min(h["timestamp"] + h["duration"], e["timestamp"] + e["duration"])
    )
    row["trace"]["throttle_visible_us"] = (
        union(
            (x["timestamp"], x["timestamp"] + x["duration"]) for x in throttles
        )
        / 1000
    )
    row["trace"]["throttle_overlap_regular_math_us"] = overlap(regular) / 1000
    row["trace"]["throttle_overlap_all_math_us"] = overlap(math) / 1000
    row["trace"][
        "throttle_scope"
    ] = "Activity-limit indications; overlap is evidence, not an additive causal delay. Empty HAM export does not certify no throttling."
    row["program_analysis"] = a
    (case / "movement_validation.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    frozen = hashlib.sha256(
        (root / "frozen_timing_profile.json").read_bytes()
    ).hexdigest()
    assert frozen == (root / "timing_profile.sha256").read_text().strip()
    for result in sorted(root.glob("*/*/result.json")):
        row = validate(result.parent)
        row["name"] = str(result.parent.relative_to(root))
        rows.append(row)
        print(
            row["name"],
            row["status"],
            row.get("predicted_us"),
            row.get("nc_p50_us"),
            row.get("hbm_difference_bytes"),
            flush=True,
        )
    passed = [
        r
        for r in rows
        if r["status"] == "pass" and r.get("predicted_us") is not None
    ]
    report = dict(
        frozen_profile_sha256=frozen,
        rows=rows,
        statistics=dict(
            cases=len(rows),
            passed=sum(r["status"] == "pass" for r in rows),
            modeled=len(passed),
            mean_absolute_percent_error=st.mean(
                r["relative_latency_error"] * 100 for r in passed
            ),
            median_absolute_percent_error=st.median(
                r["relative_latency_error"] * 100 for r in passed
            ),
            worst_absolute_percent_error=max(
                r["relative_latency_error"] * 100 for r in passed
            ),
        ),
        scope="Each prediction was frozen before application benchmarking. Device p50 and instrumented profile are separate executions; do not add their timings.",
    )
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "name",
        "problem",
        "predicted_us",
        "nc_p50_us",
        "benchmark_p50_repeats_us",
        "signed_error_percent",
        "profile_us",
        "modeled_hbm_bytes",
        "profile_hbm_bytes",
        "hbm_difference_bytes",
        "timing_complete",
        "compiled_isa_count_match",
    ]
    with (root / "validation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["statistics"]))


if __name__ == "__main__":
    main()
