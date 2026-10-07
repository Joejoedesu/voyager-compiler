"""Check generated NKI on Trainium and retain evidence; simulation is opt-in."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback
import sys
import subprocess
import shutil
import tempfile
import time
import importlib.metadata
import numpy as np
import neuronxcc.nki as nki


def check(actual, expected, atol=5e-4, rtol=5e-4):
    if isinstance(actual, (tuple, list)):
        assert len(actual) == 1
        actual = actual[0]
    actual = np.asarray(actual).astype(np.float32)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return dict(correct=True, max_abs_error=error, atol=atol, rtol=rtol)


def main():
    os.environ.setdefault("NEURON_RT_ENABLE_DGE_NOTIFICATIONS", "1")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent)
        + ":/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/trainium/run",
    )
    p.add_argument("--case")
    p.add_argument("--reference-root", type=Path)
    simulation = p.add_mutually_exclusive_group()
    simulation.add_argument(
        "--simulate",
        action="store_true",
        help="Also run the official NKI CPU correctness simulator",
    )
    simulation.add_argument(
        "--skip-simulator",
        action="store_true",
        help="Hardware-only validation (the default)",
    )
    p.add_argument(
        "--save-compiler-artifacts",
        action="store_true",
        help="Retain pinned SDK compiler diagnostics in a fresh per-run directory",
    )
    p.add_argument("--wait-device-pid", type=int)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--reuse-compiled",
        action="store_true",
        help="Reuse file.neff authenticated by compiled_binary.json; still check device output",
    )
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    reference_root = (
        args.reference_root.resolve() if args.reference_root else None
    )
    roots = sorted(args.artifacts.resolve().glob("*/nki/program.py"))
    if args.case:
        roots = [p for p in roots if p.parent.parent.name == args.case]
    if not roots:
        raise SystemExit("No generated kernels found")
    if len(roots) > 1:
        # The pinned runtime can invalidate descriptors between separately
        # compiled kernels. Isolate cases while retaining the same device lock.
        codes = [
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--case",
                    source.parent.parent.name,
                ]
            ).returncode
            for source in roots
        ]
        raise SystemExit(int(any(codes)))

    def serialize_device(kernel, save_neff=None):
        # Pinned SDK boundary: allow CPU tracing/compilation to proceed while
        # serializing all actual Neuron execution and benchmark measurements.
        import fcntl
        import time

        execute = kernel.execute_neff

        def locked(*values, **kwargs):
            kernel._voyager_execution = (values, kwargs)
            if save_neff is not None:
                # Preserve the exact binary used by the correctness run before
                # the pinned SDK removes its temporary compilation directory.
                neff = Path(values[0] if values else kwargs["neff"])
                if neff.resolve() != save_neff.resolve():
                    shutil.copy2(neff, save_neff)
                if (save_neff.parent / "reference.npz").exists():
                    # Retain a resumable compilation before device execution or
                    # profiling can fail. Large native builds need not repeat.
                    binary_hash = hashlib.sha256()
                    with save_neff.open("rb") as f:
                        for chunk in iter(lambda: f.read(8 << 20), b""):
                            binary_hash.update(chunk)
                    artifact_hashes = {
                        filename: result[key]
                        for filename, key in (
                            ("nki/program.py", "program_sha256"),
                            ("instructions.json", "instructions_sha256"),
                            ("hardware.json", "hardware_sha256"),
                            ("model.txt", "model_sha256"),
                            ("reference.npz", "reference_sha256"),
                        )
                    }
                    artifact_hashes["file.neff"] = binary_hash.hexdigest()
                    (save_neff.parent / "compiled_binary.json").write_text(
                        json.dumps(
                            {
                                "compiler_version": result["compiler_version"],
                                "compiler_flags": result["compiler_flags"],
                                "artifact_sha256": artifact_hashes,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                replay_kwargs = dict(kwargs)
                replay_values = (save_neff.name, *values[1:]) if values else ()
                if not values:
                    replay_kwargs["neff"] = save_neff.name
                kernel._voyager_execution = (replay_values, replay_kwargs)
            with open("/tmp/voyager-trainium-device.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if args.wait_device_pid:
                    while Path(f"/proc/{args.wait_device_pid}").exists():
                        time.sleep(1)
                return execute(*values, **kwargs)

        kernel.execute_neff = locked
        return kernel

    failed = False
    for source in roots:
        root = source.parent.parent
        plan = json.loads((root / "nki/plan.json").read_text())
        semantic_outputs = plan.get("abi", {}).get("semantic_output_count", 1)

        def semantic(value):
            return (
                value[:semantic_outputs]
                if isinstance(value, (list, tuple))
                else value
            )

        result = dict(
            case=root.name,
            program_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            compiler_flags="--target=trn2 --lnc=1",
            warmup=10,
            iterations=100,
            timing_scope="NKI benchmark nc_latency; device execution, excludes host invocation",
            visible_cores=os.environ.get("NEURON_RT_VISIBLE_CORES"),
            dge_notifications=os.environ.get(
                "NEURON_RT_ENABLE_DGE_NOTIFICATIONS"
            ),
            repeats=args.repeats,
            compiler_version=importlib.metadata.version("neuronx-cc"),
            model_sha256=hashlib.sha256(
                (root / "model.txt").read_bytes()
            ).hexdigest(),
            hardware_sha256=hashlib.sha256(
                (root / "hardware.json").read_bytes()
            ).hexdigest(),
            instructions_sha256=(
                hashlib.sha256(
                    (root / "instructions.json").read_bytes()
                ).hexdigest()
                if (root / "instructions.json").exists()
                else None
            ),
        )
        os.chdir(root)
        try:
            spec = importlib.util.spec_from_file_location(
                "generated_" + root.name, source
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            reference = (
                reference_root / root.name if reference_root else root
            ) / "reference.npz"
            result["reference_sha256"] = hashlib.sha256(
                reference.read_bytes()
            ).hexdigest()
            with np.load(reference) as data:
                inputs = [data[f"a{i}"] for i in range(len(data.files) - 1)]
                expected = data["expected"]
            generation = json.loads((root / "generation.json").read_text())
            tolerance = generation.get("tolerance", dict(atol=5e-4, rtol=5e-4))
            if generation.get("input_dtype") == "bfloat16":
                import ml_dtypes

                inputs = [x.astype(ml_dtypes.bfloat16) for x in inputs]
            result["simulation"] = (
                {
                    "status": "not_run",
                    "reason": "Hardware-only validation; device output is checked independently",
                }
                if not args.simulate
                else check(
                    semantic(nki.simulate_kernel(module.kernel, *inputs)),
                    expected,
                    **tolerance,
                )
            )
            compile_options = {}
            if args.save_compiler_artifacts:
                artifacts_dir = tempfile.mkdtemp(prefix="compiler-", dir=root)
                compile_options["artifacts_dir"] = artifacts_dir
                result["compiler_artifacts"] = artifacts_dir
            run = nki.baremetal(
                module.kernel,
                additional_compile_opt=result["compiler_flags"],
                **compile_options,
            )
            if args.reuse_compiled:
                cached = json.loads((root / "compiled_binary.json").read_text())
                if any(
                    cached[key] != result[key]
                    for key in ("compiler_version", "compiler_flags")
                ):
                    raise ValueError(
                        "Cached binary compiler configuration differs"
                    )
                required = {
                    "nki/program.py",
                    "instructions.json",
                    "hardware.json",
                    "model.txt",
                    "reference.npz",
                    "file.neff",
                }
                if set(cached["artifact_sha256"]) != required:
                    raise ValueError("Cached binary manifest is incomplete")
                for filename, expected_hash in cached[
                    "artifact_sha256"
                ].items():
                    h = hashlib.sha256()
                    with (root / filename).open("rb") as f:
                        for chunk in iter(lambda: f.read(8 << 20), b""):
                            h.update(chunk)
                    if h.hexdigest() != expected_hash:
                        raise ValueError(
                            f"Cached binary no longer matches {filename}"
                        )
                # Pinned SDK boundary: trace the unchanged kernel to recover its
                # IR/argument ABI, then execute the authenticated existing NEFF.
                run._compile = lambda ir: str(root / "file.neff")
                result["compiled_binary_reused"] = cached
            serialize_device(run, save_neff=root / "file.neff")
            started = time.perf_counter()
            result["hardware"] = check(
                semantic(run(*inputs)), expected, **tolerance
            )
            result["compile_and_correctness_seconds"] = (
                time.perf_counter() - started
            )
            pending = root / "execution_progress.json"
            pending.write_text(
                json.dumps(
                    {**result, "status": "correctness_pass_benchmark_pending"},
                    indent=2,
                )
                + "\n"
            )
            benchmark = nki.benchmark(
                module.kernel,
                warmup=10,
                iters=100,
                additional_compile_opt=result["compiler_flags"],
                save_neff_name="file.neff",
                save_trace_name="profile.ntff",
            )
            serialize_device(benchmark)
            repeats = []
            result["latencies"] = repeats
            for repeat in range(args.repeats):
                # Reuse the validated binary with its original IR and bindings.
                # A bare filename is required by the pinned profiler collector.
                values, kwargs = run._voyager_execution
                benchmark.execute_neff(*values, **kwargs)
                latency = benchmark.benchmark_result.nc_latency
                repeats.append(
                    {
                        f"p{q}_us": float(latency.get_latency_percentile(q))
                        for q in (0, 50, 90, 99, 100)
                    }
                )
                pending.write_text(
                    json.dumps(
                        {
                            **result,
                            "status": "correctness_pass_benchmark_pending",
                        },
                        indent=2,
                    )
                    + "\n"
                )
            result["benchmark_compilation"] = (
                "Validated NEFF reused for all timing repeats; no benchmark recompilation"
            )
            hardware = json.loads((root / "hardware.json").read_text())
            result["estimates"] = hardware["estimates"]
            result["status"] = "pass"
        except Exception:
            result["status"] = "fail"
            result["error"] = traceback.format_exc()
            failed = True
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        result["artifact_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("file.neff", "profile.ntff")
            if (root / name).is_file()
        }
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (root / "execution_progress.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
