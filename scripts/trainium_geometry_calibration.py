"""Reproduce the operand-feed model from isolated probes; applications excluded."""

from pathlib import Path
import hashlib, json, statistics
from voyager_compiler.trainium.hardware import neuron_core


def main():
    root = Path("results/trainium/matmul-geometry-2026-10-08")
    rows = []
    for directory in ["primitives", "validation-primitives"]:
        for p in sorted((root / directory).glob("*/result.json")):
            r = json.loads(p.read_text())
            assert r["status"] == "pass"
            for f, h in r["artifact_sha256"].items():
                assert (
                    hashlib.sha256((p.parent / f).read_bytes()).hexdigest() == h
                )
            r["case"] = p.parent.name
            r["split"] = directory
            rows.append(r)
    by = {r["case"]: r for r in rows if r["split"] == "primitives"}
    # Existing FP32 arithmetic lower bound: four TensorE cycles per moving
    # element. Round inferred regime ratio to integral FP32 feed passes.
    scale = round(
        statistics.median(
            by[f"w{m}-m16-s1"]["spacing_ns"]["median"]
            / by[f"w{m}-m1-s1"]["spacing_ns"]["median"]
            for m in [128, 512]
        )
    )
    max_full = max(
        s
        for s in [1, 2, 4, 8, 16]
        if by[f"w128-m{s}-s1"]["spacing_ns"]["median"]
        < 1.5 * by["w128-m1-s1"]["spacing_ns"]["median"]
    )
    fits = [
        by[f"w{m}-m{ms}-s{ss}"]
        for m in [128, 512]
        for ms, ss in [(1, 1), (16, 1), (1, 16)]
    ]

    def feeds(r):
        moving = (
            4 * r["width"] * (scale if r["moving_stride"] > max_full else 1)
        )
        stationary = (
            4 * 128 * (scale if r["stationary_stride"] > max_full else 1)
        )
        return moving, max(moving, stationary)

    slopes = []
    for ms in [1, 16]:
        a, b = by[f"w128-m{ms}-s1"], by[f"w512-m{ms}-s1"]
        ma, fa = feeds(a)
        mb, fb = feeds(b)
        slopes.append(
            (
                (b["span_ns"]["median"] - a["span_ns"]["median"]) * 2.4
                - (fb - fa)
            )
            / (mb - ma)
        )
    tail = round(statistics.median(slopes) * 16) / 16
    issue = (
        round(
            statistics.median(
                r["spacing_ns"]["median"] * 2.4 - feeds(r)[1] for r in fits
            )
            / 8
        )
        * 8
    )
    base = (
        round(
            statistics.median(
                r["span_ns"]["median"] * 2.4 - feeds(r)[1] - tail * feeds(r)[0]
                for r in fits
            )
            / 24
        )
        * 24
    )
    params = dict(
        moving_cycles_per_element=4,
        stationary_cycles_per_element=4,
        strided_feed_scale=scale,
        full_rate_max_stride=max_full,
        issue_overhead_cycles=issue,
        completion_base_cycles=base,
        moving_tail_scale=tail,
    )
    model = neuron_core(3).timing_profile.matmul_geometry
    assert all(getattr(model, k) == v for k, v in params.items()), params
    summaries = []
    for r in rows:
        pred = model.evaluate(
            r["width"],
            128,
            128,
            "float32",
            r["moving_stride"],
            r["stationary_stride"],
            2.4,
        )
        summaries.append(
            dict(
                case=r["case"],
                split=r["split"],
                measured_issue_ns=r["spacing_ns"]["median"],
                predicted_issue_ns=pred[0],
                issue_error_pct=100 * (pred[0] / r["spacing_ns"]["median"] - 1),
                measured_span_ns=r["span_ns"]["median"],
                predicted_span_ns=pred[1],
                span_error_pct=100 * (pred[1] / r["span_ns"]["median"] - 1),
                result_sha256=hashlib.sha256(
                    (root / r["split"] / r["case"] / "result.json").read_bytes()
                ).hexdigest(),
            )
        )
    out = dict(
        parameters=params,
        scope="FP32 K=N=128; empirical effective feed regimes, no inferred physical bank topology. Only isolated primitives used. Independent validation-primitives never used for parameter fitting. Completion applies to sustained streams; existing isolated completion retained on restart.",
        derivation="Fit scale from stride16/stride1 ratios rounded to integer feed passes; full-rate boundary from 128-wide stride sweep; moving completion tail from 128-to-512 slopes rounded to 1/16; median issue residual rounded to 8 cycles; median completion residual rounded to 24 cycles.",
        results=summaries,
    )
    (root / "calibration.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(params))
    print(
        "held-out max issue error %",
        max(
            abs(r["issue_error_pct"])
            for r in summaries
            if r["split"] == "validation-primitives"
        ),
    )


if __name__ == "__main__":
    main()
