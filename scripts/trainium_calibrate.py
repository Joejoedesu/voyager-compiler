"""Derive target timing parameters ONLY from isolated primitive experiments.

Application kernels are excluded. Width 32/256 DMA cases are held out. Output
includes every sample and the fitting rules, and is frozen before validation.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import numpy as np


def median(xs):
    return float(st.median(xs))


def affine(samples):
    # Nonnegative affine law, not a multiplier chosen to match an application.
    xs, ys = zip(*samples)
    if len(set(xs)) < 2:
        return max(0, median(ys) - median(xs)), 1.0
    scale, base = np.polyfit(xs, ys, 1)
    return max(0, float(base)), max(0, float(scale))


def completion_time(d, xs):
    end = max(x["timestamp"] + x["duration"] for x in xs)
    offsets = []
    for x in xs:
        for sid in re.findall(
            r"S\[(\d+)\] \([^)]*\)\+\+@complete", x["operands"]
        ):
            updates = [
                e["timestamp"]
                for e in d["semaphore_update"]
                if e["id"].startswith(f"S[{sid}] ")
                and e["value"] == 1
                and e["timestamp"] >= end
            ]
            if updates:
                offsets.append(min(updates))
    return min(offsets) if offsets else end


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("probes", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    data = {}
    evidence = {}
    for path in sorted(args.probes.glob("*/result.json")):
        row = json.loads(path.read_text())
        if row["status"] != "pass":
            continue
        folder = path.parent
        assert (
            hashlib.sha256((folder / "program.py").read_bytes()).hexdigest()
            == row["source_sha256"]
        )
        for file, expected in row["artifact_sha256"].items():
            assert (
                hashlib.sha256((folder / file).read_bytes()).hexdigest()
                == expected
            )
        d = json.loads((folder / "profile.json").read_text())
        data[row["case"]] = d
        rows.append(row)
        evidence[row["case"]] = row["artifact_sha256"]

    def select(mode, dtype=None, count=None):
        return [
            r
            for r in rows
            if r["mode"] == mode
            and (dtype is None or r["dtype"] == dtype)
            and (count is None or r["count"] == count)
        ]

    dma_samples = {"load": [], "store": []}
    spacing = {"load": [], "store": []}
    fixed = []
    for row in select("dma"):
        if row["width"] not in (128, 512):
            continue
        d = data[row["case"]]
        ins = d["instruction"]
        ag = [x for x in d["dma"] if x["aggregated"] == "yes"]
        commands = sorted(
            [x for x in ins if x["opcode"].startswith("DMA_")],
            key=lambda x: x["timestamp"],
        )
        for direction in ("load", "store"):
            cs = [
                x
                for x in commands
                if (
                    "src_table_index="
                    if direction == "load"
                    else "dst_table_index="
                )
                in x["operands"]
            ]
            if row["count"] == 16:
                spacing[direction].extend(
                    b["timestamp"] - a["timestamp"] for a, b in zip(cs, cs[1:])
                )
            if row["count"] != 1:
                continue
            for x in cs:
                sid = int(re.search(r"\bsemaphore=(\d+)", x["operands"])[1])
                candidates = [
                    a for a in ag if a["semaphore_id"].startswith(f"S[{sid}] ")
                ]
                if len(candidates) != 1:
                    continue
                a = candidates[0]
                end = a["timestamp"] + a["duration"]
                updates = [
                    e["timestamp"]
                    for e in d["semaphore_update"]
                    if e["id"].startswith(f"S[{sid}] ") and e["value"] == 16
                ]
                if not updates:
                    continue
                dma_samples[direction].append(
                    dict(
                        case=row["case"],
                        ideal_ns=a["read_size"] / 368,
                        duration_ns=a["duration"],
                        dispatch_ns=a["timestamp"] - x["timestamp"],
                        notification_ns=min(updates) - end,
                    )
                )
        waits = [
            x
            for x in ins
            if x["label"] == "GpSimd"
            and x["opcode"] == "NOP"
            and "qGpSimdDynamic" in x["operands"]
        ]
        if row["count"] == 1 and waits:
            fixed.append(
                commands[0]["timestamp"]
                + d["summary"][0]["total_time"] * 1e9
                - max(x["timestamp"] + x["duration"] for x in waits)
            )
    dma = {}
    for direction, samples in dma_samples.items():
        if len(samples) < 4:
            raise ValueError(
                f"Incomplete isolated DMA calibration: {direction}"
            )
        base, scale = affine(
            [(x["ideal_ns"], x["duration_ns"]) for x in samples]
        )
        dma[direction] = dict(
            issue_ns=median(spacing[direction]),
            dispatch_ns=median(x["dispatch_ns"] for x in samples),
            payload_floor_ns=0,
            payload_base_ns=base,
            payload_scale=scale,
            notification_ns=median(x["notification_ns"] for x in samples),
        )
    primitives = []
    primitive_samples = {}
    for dtype in ("float32", "bfloat16"):
        completion = []
        copy_completion = []
        stream = []
        stream_details = []
        vectors = {}
        for row in select("matmul", dtype, 1):
            d = data[row["case"]]
            groups = defaultdict(list)
            for x in d["instruction"]:
                if x["opcode"] in ("MATMUL", "LDWEIGHTS"):
                    groups[x["bir_instruction_name"]].append(x)
                if x["opcode"] in ("COPY", "CAST") and x["label"] == "Scalar":
                    copy_completion.append(
                        (
                            row["width"] / 1.2,
                            completion_time(d, [x]) - x["timestamp"],
                        )
                    )
            for xs in groups.values():
                span = max(x["timestamp"] + x["duration"] for x in xs) - min(
                    x["timestamp"] for x in xs
                )
                completion.append(
                    (
                        (4 if dtype == "float32" else 1) * row["width"] / 2.4,
                        completion_time(d, xs)
                        - min(x["timestamp"] for x in xs),
                    )
                )
        for row in select("ready", dtype, 16):
            d = data[row["case"]]
            groups = defaultdict(list)
            for x in d["instruction"]:
                if x["opcode"] in ("MATMUL", "LDWEIGHTS"):
                    groups[x["bir_instruction_name"]].append(x)
            starts = sorted(
                min(x["timestamp"] for x in xs) for xs in groups.values()
            )
            if len(starts) != 16:
                raise ValueError(
                    "Ready-stream probe was folded or expanded unexpectedly"
                )
            deltas = [b - a for a, b in zip(starts, starts[1:])]
            stream_details.append(
                dict(case=row["case"], group_start_deltas_ns=deltas)
            )
            # Interior spacing is a diagnostic; it is not intrinsic issue rate.
            stream.append(
                (
                    (4 if dtype == "float32" else 1) * row["width"] / 2.4,
                    median(deltas[3:-2]),
                )
            )
        if len(completion) != 2 or len(stream) != 2:
            raise ValueError(f"Incomplete matmul characterization {dtype}")
        base, scale = affine(completion)
        # A compiler-managed stream has multiple scheduling phases. Its group
        # start spacing is not an isolated engine initiation interval; retain
        # documented throughput and characterize dependent completion only.
        primitives.append(
            dict(
                implementation=f"nki.matmul.{dtype}",
                issue_floor_ns=0,
                service_scale=1,
                completion_base_ns=base,
                completion_service_scale=scale,
            )
        )
        cb, cs = affine(copy_completion)
        primitives.append(
            dict(
                implementation=f"nki.copy.PSUM.{dtype}.ScalarE",
                issue_floor_ns=0,
                service_scale=1,
                completion_base_ns=cb,
                completion_service_scale=cs,
            )
        )
        trans = []
        for row in select("transpose", dtype):
            d = data[row["case"]]
            groups = defaultdict(list)
            for x in d["instruction"]:
                if x["opcode"] in ("MATMUL", "LDWEIGHTS"):
                    groups[x["bir_instruction_name"]].append(x)
            trans.extend(
                max(x["timestamp"] + x["duration"] for x in xs)
                - min(x["timestamp"] for x in xs)
                for xs in groups.values()
            )
        if not trans:
            raise ValueError("Missing transpose samples")
        primitives.append(
            dict(
                implementation=f"nki.transpose_copy.{dtype}.ScalarE",
                issue_floor_ns=0,
                service_scale=1,
                completion_base_ns=max(0, median(trans) - 128 / 2.4),
                completion_service_scale=1,
            )
        )
        for mode, impl in [
            ("gather", f"nki.copy.SBUF.{dtype}.VectorE"),
            ("fill", f"nki.memset.{dtype}.VectorE"),
        ]:
            samples = []
            for row in select(mode, dtype, 16):
                d = data[row["case"]]
                ops = [
                    x
                    for x in d["instruction"]
                    if x["label"] == "Vector"
                    and x["opcode"]
                    in ("COPY", "COPY_PREDICATED_SCALAR", "MEMSET")
                ]
                if not ops:
                    raise ValueError(f"No realized operations for {mode}")
                samples.append(
                    (
                        max(64, row["width"]) / 0.96,
                        median(x["duration"] for x in ops),
                    )
                )
            if len(samples) != 2:
                raise ValueError(f"Missing vector calibration {mode} {dtype}")
            vectors[mode] = samples
            vb, vs = affine(samples)
            primitives.append(
                dict(
                    implementation=impl,
                    issue_floor_ns=0,
                    service_scale=1,
                    completion_base_ns=vb,
                    completion_service_scale=vs,
                )
            )
        primitive_samples[dtype] = dict(
            matmul_completion=completion,
            matmul_stream=stream,
            matmul_stream_details=stream_details,
            vector_completion=vectors,
            scalar_copy_completion=copy_completion,
            transpose_completion_ns=trans,
        )
    profile = dict(
        name="trn2-primitives-2026-10-06",
        compiler="neuronx-cc 2.22.12471 / Trainium2",
        primitives=primitives,
        **dma,
        fixed_kernel_ns=median(fixed),
        evidence="results/trainium/movement-probes/calibration.json",
    )
    artifact = dict(
        method=__doc__,
        dma_samples=dma_samples,
        dma_spacing_ns=spacing,
        fixed_overhead_samples_ns=fixed,
        primitive_samples=primitive_samples,
        artifacts=evidence,
        profile=profile,
        limitations=[
            "Measured full 128-partition tiles; partial shapes extrapolate these laws.",
            "Engine initiation uses documented throughput; compiler-managed ready streams have scheduling phases, so their observed start spacing is retained as validation evidence, not mistaken for a hardware initiation interval. Completion is characterized; vector and transpose completion exclude unisolated notification delay.",
            "No whole-kernel latency or winning mapping enters calibration.",
            "Finite backend command queues and physical-bank contention remain unspecified.",
        ],
    )
    (args.probes / "calibration.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
