"""Run pinned VCS replay; chain network segments using actual hardware outputs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_matches(record, replay, output, gemmini):
    manifest = json.loads((replay / 'memory.json').read_text())
    inputs = [(replay / r[key]).resolve() for r in manifest['regions']
              for key in ('input', 'expected') if key in r]
    if any(not p.exists() for p in inputs):
        return False
    current_inputs = {str(p): digest(p) for p in inputs}
    files = {
        "command_sha256": replay / "commands.jsonl",
        "manifest_sha256": replay / "memory.json",
        "simulator_sha256": gemmini / "build/vcs/simv",
        "build_manifest_sha256": gemmini / "build/manifest.json",
        "versions_sha256": gemmini / "docs/versions.json",
    }
    return (
        record.get("pass")
        and record.get("input_file_sha256")
        and record['input_file_sha256'] == current_inputs
        and record.get("outputs")
        and all(record.get(k) == digest(p) for k, p in files.items())
        and all(
            Path(p).exists() and digest(p) == h
            for p, h in record["input_file_sha256"].items()
        )
        and all(
            (output / p).exists() and digest(output / p) == v["sha256"]
            for p, v in record["outputs"].items()
        )
    )


def validate(replay, output, gemmini, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    if resume and (output / "result.json").exists():
        old = json.loads((output / "result.json").read_text())
        if record_matches(old, replay, output, gemmini):
            print(
                str(output),
                "VERIFIED_REUSE",
                old["measured_cycles"],
                flush=True,
            )
            return old
    with (output / "runner.log").open("w") as log:
        subprocess.run(
            [
                "bash",
                str(gemmini / "scripts/run.sh"),
                str(replay),
                "--out",
                str(output),
                "--timeout",
                "100000000",
                "--host-timeout",
                "3600",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = json.loads((output / "result.json").read_text())
    print(
        str(output),
        result["pass"],
        result.get("measured_cycles"),
        result.get("error"),
        flush=True,
    )
    if not result["pass"]:
        raise RuntimeError(str(output / "result.json"))
    return result


def network(root, gemmini, resume=False):
    manifest = json.loads((root / "network.json").read_text())
    state = root / "state"
    state.mkdir(exist_ok=True)
    results = []
    for segment in manifest["segments"]:
        replay = root / segment["path"]
        output = root / "vcs" / replay.name
        for name in segment["chained_inputs"]:
            if not (state / (name + ".bin")).exists():
                raise ValueError("Missing actual hardware output: " + name)
        result = validate(replay, output, gemmini, resume)
        for name in segment["outputs"]:
            shutil.copyfile(output / (name + ".bin"), state / (name + ".bin"))
        results.append(dict(segment=segment["path"], result=result))
        (root / "validation.json").write_text(
            json.dumps(dict(complete=False, segments=results), indent=2) + "\n"
        )
    host = []
    for op in manifest.get("host_outputs", []):
        value = np.fromfile(state / (op["input"] + ".bin"), dtype="i1").astype(
            np.float32
        ) * np.float32(op["scale"])
        expected = np.fromfile(op["expected"], dtype="<f4")
        np.testing.assert_array_equal(value[: expected.size], expected)
        value.tofile(state / (op["output"] + ".f32.bin"))
        host.append(
            dict(
                **op,
                match=True,
                sha256=digest(state / (op["output"] + ".f32.bin")),
            )
        )
    logical = manifest.get("logical_output")
    if logical:
        physical_name = manifest["outputs"][0]
        value = np.fromfile(state / (physical_name + ".f32.bin"), dtype="<f4")
        count = int(np.prod(logical["shape"]))
        expected = np.fromfile(
            Path(manifest["source"]).parent / logical["reference"], dtype="<f4"
        )
        np.testing.assert_array_equal(value[:count], expected)
        value[:count].tofile(state / "output.f32.bin")
    report = dict(
        complete=True,
        pass_=True,
        segments=results,
        host_outputs=host,
        logical_output=logical,
        measured_cycles=sum(r["result"]["measured_cycles"] for r in results),
        timing_scope="Sum of independent CPU-free replay segments; excludes host staging and output decode",
    )
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


def benchmarks(root, gemmini, jobs, compare_only=False):
    cases = [f"gemm{i}" for i in range(6)] + [f"conv{i}" for i in range(3)]

    def run(case):
        if compare_only:
            record = json.loads((root / case / "vcs/result.json").read_text())
            if not record_matches(
                record, root / case / "replay", root / case / "vcs", gemmini
            ):
                raise ValueError("Stale candidate evidence: " + case)
            return case, record
        return case, validate(
            root / case / "replay", root / case / "vcs", gemmini
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = dict(pool.map(run, cases))
    comparison = []
    for case, result in results.items():
        baselines = {
            variant: json.loads(
                (
                    gemmini
                    / "results/autocomp"
                    / case
                    / variant
                    / "result.json"
                ).read_text()
            )
            for variant in ("exo_unoptimized", "exo_optimized")
        }
        original = gemmini / "examples/autocomp" / case
        candidate = root / case / "replay"
        for variant, baseline in baselines.items():
            if not baseline["pass"]:
                raise ValueError("Failed Exo baseline")
            for key in (
                "hardware_config",
                "memory_config",
                "simulator_sha256",
                "versions_sha256",
                "build_manifest_sha256",
            ):
                if result[key] != baseline[key]:
                    raise ValueError(case + ": incomparable " + key)
            if (
                baseline["command_sha256"]
                != digest(original / (variant + ".jsonl"))
                or baseline["manifest_sha256"]
                != digest(original / "memory.json")
                or any(
                    not Path(p).exists() or digest(p) != h
                    for p, h in baseline["input_file_sha256"].items()
                )
            ):
                raise ValueError("Stale Exo evidence: " + case + "/" + variant)
        old_memory = json.loads((original / "memory.json").read_text())
        new_memory = json.loads((candidate / "memory.json").read_text())
        lowering = json.loads((candidate / "lowering.json").read_text())
        # Layout copies were independently checked by the converter. Compare
        # the bytes consumed by the hardware, excluding the unpacked duplicate.
        layout_sources = {
            x["source"] for x in lowering["input_layouts"].values()
        }
        for key in ("input", "expected"):
            old_hashes = sorted(
                digest(original / r[key])
                for r in old_memory["regions"]
                if key in r
            )
            new_hashes = sorted(
                digest(candidate / r[key])
                for r in new_memory["regions"]
                if key in r and r["name"] not in layout_sources
            )
            if old_hashes != new_hashes:
                raise ValueError(case + ": different " + key + " bytes")
        cycles = result["measured_cycles"]
        base = baselines["exo_unoptimized"]["measured_cycles"]
        opt = baselines["exo_optimized"]["measured_cycles"]
        comparison.append(
            dict(
                case=case,
                cycles=cycles,
                exo_unoptimized=base,
                exo_optimized=opt,
                speedup_vs_unoptimized=base / cycles,
                speedup_vs_optimized=opt / cycles,
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument(
        "--gemmini",
        type=Path,
        default=Path("/home/zhouhua/Research/ML/Gemmini"),
    )
    p.add_argument("--network", action="store_true")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compare-only", action="store_true")
    a = p.parse_args()
    if a.network:
        network(a.root.resolve(), a.gemmini, a.resume)
    else:
        benchmarks(a.root.resolve(), a.gemmini, a.jobs, a.compare_only)
