"""Recheck the integrated model against authenticated prior-work experiments."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from voyager_compiler.trainium.compiled_analysis import predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        hardware_measurements="authenticated reuse; no new device executions",
        cases=[],
    )
    for record in json.loads((args.experiment / "manifest.json").read_text()):
        case = record["case"]
        folder = args.experiment / "artifacts" / case
        for name, sha in record["artifact_sha256"].items():
            assert digest(folder / name) == sha, (case, name)
        provenance = json.loads(
            (folder / "extraction_provenance.json").read_text()
        )
        assert (
            digest(folder / "compiled_static.json")
            == provenance["compiled_static_sha256"]
        )
        data = json.loads((folder / "compiled_static.json").read_text())
        expected = json.loads(
            (
                args.experiment / "followup/results" / case / "extended.json"
            ).read_text()
        )
        actual, _ = predict(data)
        for name in (
            "prediction_us",
            "hbm_read_bytes",
            "hbm_write_bytes",
            "encoded_waits",
            "unsupported_instructions",
            "unknown_completion_events",
        ):
            assert actual[name] == expected[name], (
                case,
                name,
                actual[name],
                expected[name],
            )
        assert (
            actual["selected_timing_laws"]
            == expected["selected_extension_laws"]
        )
        random.Random(29).shuffle(data["instructions"])
        shuffled, _ = predict(data)
        assert actual == shuffled, case
        assert actual["hbm_read_bytes"] == data["dma_audit"]["read_bytes"]
        assert actual["hbm_write_bytes"] == data["dma_audit"]["write_bytes"]
        report["cases"].append(
            dict(
                case=case,
                prediction_us=actual["prediction_us"],
                hardware_us=record["prior_us"],
                error_percent=100
                * (actual["prediction_us"] / record["prior_us"] - 1),
                exact_experiment_match=True,
                input_order_invariance=True,
                static_input_sha256=provenance["compiled_static_sha256"],
                result=actual,
            )
        )
        print(case, actual["prediction_us"], "PASS", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
