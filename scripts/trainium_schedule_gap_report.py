"""Record plan/replay gaps and audit prior-schedule search coverage.

Reuses authenticated full-shape measurements. Prior compiled programs are scored
with the current timing model, without feeding measured timing into prediction.
This is a diagnostic comparison, not a production schedule importer.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from trainium_full_movement_report import digest
from voyager_compiler.trainium.compiled_analysis import fields, predict


def inventory(data):
    counts = Counter()
    tensor_shapes = Counter()
    tensor_seen = set()
    for ins in data["instructions"]:
        op = ins["opcode"]
        counts[op] += 1
        if op == "MATMUL":
            key = ins["raw_bir_id"]
            if key in tensor_seen:
                continue
            tensor_seen.add(key)
            transpose = ins["instruction_type"] == "TRANSPOSE"
            counts[
                "logical_tensor_transpose" if transpose else "logical_matmul"
            ] += 1
            if not transpose:
                tensors = fields(ins["operands"])
                moving = math.prod(tensors["src"][3])
                match = re.search(r"(\d+)\*(\d+)\s*$", ins["operands"])
                contracting, stationary = map(int, match.groups())
                tensor_shapes[
                    f"moving={moving},stationary={stationary},K={contracting}"
                ] += 1
        if op == "COPY":
            tensors = fields(ins["operands"])
            memory = "PSUM" if tensors["src"][1] >= 0x2000000 else "SBUF"
            counts[f"COPY_{memory}_to_SBUF"] += 1
    return dict(counts=dict(counts), logical_matmul_shapes=dict(tensor_shapes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prior-experiment", type=Path, required=True)
    args = parser.parse_args()
    out = args.results / "schedule-gap"
    out.mkdir(exist_ok=True)
    report = json.loads((args.results / "report.json").read_text())
    manifest = {
        row["case"]: row
        for row in json.loads(
            (args.results / "baseline-manifest.json").read_text()
        )
    }
    gaps, comparisons = [], []
    for row in report["results"]:
        case = row["case"]
        folder = args.results / "selected" / case
        plan_us = row["selected_predicted_us"]
        replay_us = row["compiled_static_prediction"]["prediction_us"]
        gaps.append(
            dict(
                case=case,
                plan_us=plan_us,
                compiled_replay_us=replay_us,
                measured_us=row["selected_hardware_us"],
                replay_minus_plan_us=replay_us - plan_us,
                replay_change_percent=100 * (replay_us / plan_us - 1),
                status="open: event-by-event causal attribution not established",
            )
        )
        prior_dir = args.prior_experiment / "artifacts" / case
        provenance = json.loads(
            (prior_dir / "extraction_provenance.json").read_text()
        )
        assert (
            provenance["artifact_sha256"] == manifest[case]["artifact_sha256"]
        )
        for name, sha in provenance["artifact_sha256"].items():
            assert digest(prior_dir / name) == sha, (case, name)
        prior_static = prior_dir / "compiled_static.json"
        assert digest(prior_static) == provenance["compiled_static_sha256"]
        data = json.loads(prior_static.read_text())
        prediction, _ = predict(data)
        (out / f"{case}-prior-prediction.json").write_text(
            json.dumps(prediction, indent=2) + "\n"
        )
        new_static = folder / "compiled_static.json"
        new_provenance = json.loads(
            (folder / "extraction_provenance.json").read_text()
        )
        assert digest(new_static) == new_provenance["compiled_static_sha256"]
        current_inventory = inventory(json.loads(new_static.read_text()))
        search = json.loads((folder / "movement-search.json").read_text())
        seed = search["candidates"][0]["selected_bindings"]
        candidates = []
        for index, candidate in enumerate(search["candidates"]):
            selected = candidate.get("selected_bindings", {})
            changed = {
                key: value
                for key, value in selected.items()
                if value != seed[key]
            }
            candidates.append(
                dict(
                    index=index + 1,
                    status=candidate["status"],
                    changed_bindings=changed,
                    uses_stream_transpose=any(
                        "stream_transpose" in v for v in selected.values()
                    ),
                )
            )
        origin = Path(
            json.loads((folder / "generation.json").read_text())[
                "movement_search_origin"
            ]
        )
        assert digest(folder / "model.txt") == digest(origin / "model.txt")
        comparison = dict(
            case=case,
            prior_hardware_us=row["prior_hardware_us"],
            voyager_hardware_us=row["selected_hardware_us"],
            prior_current_model_us=prediction["prediction_us"],
            voyager_compiled_model_us=replay_us,
            voyager_plan_model_us=plan_us,
            prior_faster_in_compiled_model=prediction["prediction_us"]
            < replay_us,
            prior_hbm_bytes=prediction["hbm_read_bytes"]
            + prediction["hbm_write_bytes"],
            voyager_hbm_bytes=row["measured_hbm_bytes"],
            prior_inventory=inventory(data),
            voyager_inventory=current_inventory,
            prior_missing_laws=prediction["missing_laws"],
            prior_unsupported_instructions=prediction[
                "unsupported_instructions"
            ],
            prior_static_sha256=digest(prior_static),
            voyager_static_sha256=digest(new_static),
            prior_source=str((prior_dir / "baseline.py").resolve()),
            fixed_model_sha256=digest(folder / "model.txt"),
            fixed_model_origin=str(origin),
            search_evaluated=search["evaluated"],
            search_nominal_combinations=search["nominal_combinations"],
            candidates=candidates,
            max_changed_bindings=max(
                len(c["changed_bindings"]) for c in candidates
            ),
            stream_candidates_evaluated=sum(
                c["uses_stream_transpose"] for c in candidates
            ),
            stream_choices_admitted=sum(
                "stream_transpose" in choice["name"]
                for request in search["requests"].values()
                for choice in request["choices"]
            ),
        )
        comparisons.append(comparison)
        print(
            case,
            f"prior replay={prediction['prediction_us']:.3f} us",
            f"Voyager replay={replay_us:.3f} us",
            flush=True,
        )
    (out / "plan-replay-gap.json").write_text(
        json.dumps(
            dict(
                definition="Same timing profile; different graphs/adapters. Signed difference is not an attribution to added computation.",
                new_replay_information=[
                    "encoded engine PC order",
                    "encoded semaphore completion dependencies",
                    "concrete operands and DMA descriptors",
                    "backend instruction realization",
                ],
                excluded_inputs=[
                    "measured duration",
                    "timestamp",
                    "observed wait time",
                    "application runtime",
                ],
                open_work="Match operations, compare dependencies/order and transfer representation, and quantify adapter differences before attributing the gap.",
                results=gaps,
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "comparison.json").write_text(
        json.dumps(
            dict(
                scope="Authenticated prior and Voyager compiled metadata; current timing model; no new hardware runs or timing calibration. Compiled replay acceptance is not production-schedule representability.",
                source_sha256={
                    str(p): digest(p)
                    for p in [
                        Path(__file__),
                        *Path("src/voyager_compiler/trainium").glob("*.py"),
                        Path(
                            "src/voyager_compiler/trainium/timing_trainium2.json"
                        ),
                    ]
                },
                full_report_sha256=digest(args.results / "report.json"),
                results=comparisons,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
