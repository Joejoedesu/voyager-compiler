"""Authenticate operand-policy experiments and replay timing-free native ISA."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from trainium_full_movement_report import (
    authenticate,
    digest,
    measured,
    reference_shapes,
)
from trainium_schedule_gap_report import inventory
from voyager_compiler.trainium.compiled_analysis import predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--prior-experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location(
        "static_extract", args.prior_experiment / "extract.py"
    )
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)
    prior = {
        r["case"]: r
        for r in json.loads(
            (args.baseline / "baseline-manifest.json").read_text()
        )
    }
    rows = []
    for root in args.roots:
        for result_path in sorted(root.glob("*/result.json")):
            folder = result_path.parent
            result = authenticate(folder)
            seed = authenticate(args.baseline / "seed" / folder.name)
            assert (
                result["reference_sha256"] == seed["reference_sha256"]
            ), folder
            for key in (
                "compiler_version",
                "compiler_flags",
                "warmup",
                "iterations",
                "repeats",
                "timing_scope",
            ):
                assert result[key] == seed[key], (folder, key)
            target = folder / "profile.json"
            if not target.exists():
                command = [
                    "/opt/aws/neuron/bin/neuron-profile",
                    "view",
                    "-n",
                    str((folder / "file.neff").resolve()),
                    "-s",
                    str((folder / "profile.ntff").resolve()),
                    "--output-format",
                    "json",
                    "--output-file",
                    str(target.resolve()),
                ]
                with (folder / "extraction.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=log, check=True)
            raw = target.read_bytes()
            profile = json.loads(raw)
            origin = Path(profile["profile_info"][0]["ntff_filename"])
            assert (
                digest(origin) == result["artifact_sha256"]["profile.ntff"]
            ), folder
            data = extraction.static_metadata(profile)
            static_path = folder / "compiled_static.json"
            static_path.write_text(json.dumps(data, indent=2) + "\n")
            prediction, _ = predict(data)
            (folder / "compiled_prediction.json").write_text(
                json.dumps(prediction, indent=2) + "\n"
            )
            (folder / "extraction_provenance.json").write_text(
                json.dumps(
                    dict(
                        artifact_sha256=result["artifact_sha256"],
                        profile_uncompressed_sha256=hashlib.sha256(
                            raw
                        ).hexdigest(),
                        compiled_static_sha256=digest(static_path),
                        extraction_script_sha256=digest(
                            args.prior_experiment / "extract.py"
                        ),
                        instruction_field_allowlist=extraction.INSTRUCTION_FIELDS,
                        dma_field_allowlist=extraction.DMA_FIELDS,
                        ordering="compiled engine PC and semaphore dependencies only",
                    ),
                    indent=2,
                )
                + "\n"
            )
            selection = json.loads((folder / "selection.json").read_text())
            hardware = json.loads((folder / "hardware.json").read_text())
            plan_us = (
                selection["program_analysis"]["whole_program_prediction_ns"]
                / 1000
            )
            hbm_bytes = (
                prediction["hbm_read_bytes"] + prediction["hbm_write_bytes"]
            )
            measured_bytes = (
                data["dma_audit"]["read_bytes"]
                + data["dma_audit"]["write_bytes"]
            )
            assert hbm_bytes == measured_bytes, folder
            assert (
                selection["program_analysis"]["selected_instruction_analysis"][
                    "hbm_bytes"
                ]
                == measured_bytes
            ), folder
            row = dict(
                case=folder.name,
                variant=root.name,
                directory=str(folder.resolve()),
                reference_arrays=reference_shapes(
                    Path(result.get("reference_path", folder / "reference.npz"))
                ),
                reference_sha256=result["reference_sha256"],
                compiler_flags=result["compiler_flags"],
                compiler_version=result["compiler_version"],
                runtime_core_visibility=result.get("visible_cores"),
                seed_runtime_core_visibility=seed.get("visible_cores"),
                dge_notifications=result.get("dge_notifications"),
                tuning=json.loads((folder / "compilation.json").read_text())[
                    "compiler"
                ]["policy"],
                software_tiles=[
                    e["software_tile"] for e in hardware["estimates"]
                ],
                buffer_depths=[
                    e["buffer_depth"] for e in hardware["estimates"]
                ],
                prior_hardware_us=prior[folder.name]["prior_us"],
                seed_hardware_us=measured(seed),
                hardware_us=measured(result),
                hardware_p50_repeats_us=[
                    x["p50_us"] for x in result["latencies"]
                ],
                speedup_over_seed=measured(seed) / measured(result),
                latency_reduction_percent=100
                * (1 - measured(result) / measured(seed)),
                plan_us=plan_us,
                compiled_replay_us=prediction["prediction_us"],
                plan_error_percent=100 * (plan_us / measured(result) - 1),
                replay_error_percent=100
                * (prediction["prediction_us"] / measured(result) - 1),
                hbm_bytes=hbm_bytes,
                hardware_correctness=result["hardware"],
                spill_save_bytes=profile["summary"][0].get("spill_save_bytes"),
                spill_reload_bytes=profile["summary"][0].get(
                    "spill_reload_bytes"
                ),
                inventory=inventory(data),
                stats=selection["stats"],
                compiled_prediction=prediction,
                artifact_sha256=result["artifact_sha256"],
                program_sha256=result["program_sha256"],
            )
            rows.append(row)
            print(
                folder.name,
                root.name,
                f'{row["hardware_us"]:.0f} us; {row["latency_reduction_percent"]:.2f}% faster',
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            dict(
                scope="Fresh hardware correctness and median of three device p50s; same reference inputs, pinned compiler and timing protocol as authenticated seed. No timing calibration; CPU simulator not run.",
                exclusions="BMM-softmax deferred. LayerNorm/maxpool unchanged and not remeasured. Whole-region matrix/reduction fusion remains unimplemented.",
                source_sha256={
                    str(p): digest(p)
                    for p in [
                        Path(__file__),
                        Path(
                            "src/voyager_compiler/trainium/compiled_analysis.py"
                        ),
                        Path(
                            "src/voyager_compiler/trainium/timing_trainium2.json"
                        ),
                    ]
                },
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
