"""Run the pinned Voyager commands against this checkout, without rebaselining.

Usage: python validation/run_references.py OUTPUT [--only CASE] [--jobs 2]
The output directory must not exist. References are only read.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFY = (
    "Results match",
    "WARNING: output verification failed:",
    "Mismatched elements:",
    "Greatest absolute difference:",
    "Greatest relative difference:",
    "Skipping output verification",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_pin(root, pin):
    path = root / pin["path"]
    if digest(path) != pin["sha256"]:
        raise ValueError(f"sealed reference identity changed: {path}")
    return path


def source_hashes():
    paths = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=REPO,
        )
        .decode()
        .split("\0")
    )
    return {
        p: digest(REPO / p)
        for p in sorted(set(paths))
        if p and (REPO / p).is_file()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=REPO.parent / "reference-point"
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--dependency-path", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached models/datasets exclusively",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output.resolve()
    reference = args.reference.resolve()
    if output.is_relative_to(reference):
        parser.error(
            "write candidate results outside the sealed reference tree"
        )
    pins = json.loads(
        (REPO / "validation/reference-baselines.json").read_text()
    )["voyager"]
    baseline_path = verify_pin(reference.parent, pins["baseline"])
    commands_path = verify_pin(reference.parent, pins["commands"])
    image_manifest = (
        verify_pin(reference.parent, pins["imagenet_inputs"])
        if "imagenet_inputs" in pins
        else None
    )
    verify_pin(REPO, pins["suite"])
    if "artifact_checksums" in pins:
        checksums = verify_pin(reference.parent, pins["artifact_checksums"])
        for line in checksums.read_text().splitlines():
            expected_digest, name = line.split("  ", 1)
            if digest(checksums.parent / name) != expected_digest:
                raise ValueError(f"sealed reference artifact changed: {name}")
    replacements = {}
    for label, pin in pins.get("artifact_replacements", {}).items():
        manifest_path = verify_pin(reference.parent, pin)
        manifest = json.loads(manifest_path.read_text())
        if (
            label != manifest["case"]
            or digest(manifest_path.parent / "model.txt")
            != manifest["model_sha256"]
        ):
            raise ValueError(f"invalid approved replacement: {label}")
        replacements[label] = manifest
    baseline = json.loads(baseline_path.read_text())
    suite = [
        s.strip()
        for s in (REPO / "validation/voyager-suite.txt")
        .read_text()
        .splitlines()
        if s.strip() and not s.startswith("#")
    ]
    if set(args.only) - set(suite):
        parser.error(f"unknown suite cases: {set(args.only) - set(suite)}")
    selected = [s for s in suite if not args.only or s in args.only]
    expected = {c["label"]: c for c in baseline["cases"]}
    commands = {}
    for line in commands_path.read_text().splitlines():
        command = shlex.split(line)
        index = command.index("--model_output_dir") + 1
        label = command[index].removeprefix("<run_dir>/")
        command[:2] = [sys.executable, str(REPO / "validation/run_case.py")]
        command[index] = str(output / label)
        # The pinned frontend used this explicit approximation before PTQ.
        # Native RMSNorm is the new compiler's default and must stay so.
        if (
            label.startswith(("llama_", "vit/"))
            and "--numerics_recipe" not in command
        ):
            command += ["--numerics_recipe", "voyager_reference_20260927"]
        policy = pins.get("search_policy")
        if policy is not None:
            if (
                "--runtime_tolerance" not in command
                or float(command[command.index("--runtime_tolerance") + 1])
                != policy["runtime_tolerance"]
            ):
                raise ValueError(
                    f"{label}: command does not specify the pinned search policy"
                )
        if image_manifest and label in (
            "resnet18/INT8/16x16",
            "vit/MXNF4/64x64",
        ):
            command += ["--imagenet-manifest", str(image_manifest)]
        commands[label] = command
    output.mkdir(parents=True, exist_ok=False)
    dependency_paths = [str(p.resolve()) for p in args.dependency_path]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (str(REPO / "src"), str(REPO), *dependency_paths)
        ),
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(output / "matplotlib"),
    )
    env.update(HF_UPDATE_DOWNLOAD_COUNTS="0", HF_HUB_DISABLE_TELEMETRY="1")
    cache_source = None
    if args.offline:
        # Reuse existing Arrow/metadata files, with all new locks/cache writes
        # inside this run. Do not require write access to the user's HF cache.
        cache_source = Path(
            os.environ.get(
                "HF_DATASETS_CACHE", Path.home() / ".cache/huggingface/datasets"
            )
        ).resolve()
        cache = output / "dataset-cache"
        cache.mkdir()
        for dataset in ("glue", "wikitext"):
            for source in sorted((cache_source / dataset).rglob("*")):
                if source.is_file() and source.suffix in (".arrow", ".json"):
                    destination = cache / source.relative_to(cache_source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.symlink_to(source.resolve())
        env.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_CACHE=str(cache),
        )
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,torch,voyager_compiler; print(json.dumps(dict(compiler=voyager_compiler.__file__,torch=torch.__version__)))",
        ],
        cwd=REPO,
        env=env,
        text=True,
    )
    imported = json.loads(probe.strip().splitlines()[-1])
    if not Path(imported["compiler"]).resolve().is_relative_to(REPO / "src"):
        raise RuntimeError(f"Wrong compiler imported: {imported}")
    write_json(
        output / "invocation.json",
        dict(
            checkout=str(REPO),
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            environment=imported,
            python=sys.executable,
            selected=selected,
            dependency_paths=dependency_paths,
            offline=args.offline,
            full_gate=selected == suite,
            baseline_sha256=digest(baseline_path),
            search_policy=pins.get("search_policy"),
            imagenet_inputs=pins.get("imagenet_inputs"),
            dataset_cache_source=str(cache_source) if cache_source else None,
            suite_sha256=digest(REPO / "validation/voyager-suite.txt"),
            commands={s: commands[s] for s in selected},
            artifact_replacements=replacements,
        ),
    )
    (output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=REPO)
    )
    sources = source_hashes()
    write_json(output / "source-sha256.json", sources)

    def run(label):
        folder = output / label
        folder.mkdir(parents=True)
        start = time.monotonic()
        with (folder / "run.log").open("w") as log:
            code = subprocess.run(
                commands[label],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        log = (folder / "run.log").read_text(errors="replace")
        verification = [
            line.strip() for line in log.splitlines() if line.startswith(VERIFY)
        ]
        failure = []
        if code:
            failure = re.findall(r"got (\w+Error\(.+\))\s*$", log, re.MULTILINE)
            if not failure:
                failure = re.findall(r"^[\w.]*Error: .+$", log, re.MULTILINE)[
                    -1:
                ]
            failure = list(
                dict.fromkeys(re.sub(r"\bs\d+\b", "s?", s) for s in failure)
            )
        status = (
            "error"
            if code
            else (
                "numeric_drift"
                if "WARNING: output verification failed:" in log
                else "ok"
            )
        )
        model = folder / "model.txt"
        actual_hash = digest(model) if model.is_file() else None
        expected_hash = baseline["model_sha256"][
            f"{baseline['run']}/{label}/model.txt"
        ]
        if label in replacements:
            expected_hash = replacements[label]["model_sha256"]
        checks = dict(
            model_hash=actual_hash == expected_hash,
            status=status == expected[label]["status"],
            verification=verification == expected[label]["verification"],
            failure=failure == expected[label]["failure"],
            compiled=code == 0,
        )
        result = dict(
            label=label,
            exit_code=code,
            status=status,
            elapsed_seconds=time.monotonic() - start,
            checks=checks,
            passed=all(checks.values()),
            model_sha256=actual_hash,
            expected_model_sha256=expected_hash,
            verification=verification,
            failure=failure,
        )
        write_json(folder / "comparison.json", result)
        print(
            f"{label}: {'PASS' if result['passed'] else 'FAIL'} {checks}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(run, selected))
    source_unchanged = source_hashes() == sources
    passed = all(r["passed"] for r in results) and source_unchanged
    write_json(
        output / "comparison.json",
        dict(
            passed=passed,
            full_gate=selected == suite,
            cases=results,
            expected_cases=len(suite),
            selected_cases=len(selected),
            source_unchanged=source_unchanged,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
