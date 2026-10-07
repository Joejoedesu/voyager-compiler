"""Matched device replay of unchanged local prior-work kernels.

Only import preambles and documented argument/output views are adapted. Source
hashes, input hashes, correctness, benchmark scope and NEFF/NTFF are retained.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
import subprocess
import importlib.metadata
import numpy as np
import neuronxcc.nki as nki

SOURCES = {
    "maxpool": "autocomp/sols/trn-advanced-nki1/2_maxpool_ref.py",
    "layernorm": "autocomp/sols/trn-tutorial-nki1/1_layernorm_ref.py",
    "matmul_add_rmsnorm": "AccelOpt/NKIBench/kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py",
    "add_rmsnorm_matmul": "AccelOpt/NKIBench/kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py",
    "swiglu": "AccelOpt/NKIBench/kernels/swiglu_M4096_N3072_K1024_0.py",
    "bmm_softmax": "AccelOpt/NKIBench/kernels/bmm_softmax_B16_K64_M4096_N4096_0.py",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="*", default=list(SOURCES))
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    inputs_root = args.inputs.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(args.cases) > 1:
        codes = [
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--inputs",
                    str(inputs_root),
                    "--output",
                    str(output),
                    "--repeats",
                    str(args.repeats),
                    "--cases",
                    name,
                ]
            ).returncode
            for name in args.cases
        ]
        raise SystemExit(int(any(codes)))

    def serialize_device(kernel):
        import fcntl

        execute = kernel.execute_neff

        def locked(*values, **kwargs):
            kernel._voyager_execution = (values, kwargs)
            with open("/tmp/voyager-trainium-device.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return execute(*values, **kwargs)

        kernel.execute_neff = locked
        return kernel

    os.environ.setdefault("NEURON_RT_ENABLE_DGE_NOTIFICATIONS", "1")
    os.environ["PATH"] = (
        "/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    for name in args.cases:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        os.chdir(folder)
        path = Path("/home/ubuntu/ML/prior_works") / SOURCES[name]
        code = path.read_text()
        data_path = inputs_root / name / "reference.npz"
        record = dict(
            case=name,
            source=str(path),
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),
            reference_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
            flags="--target=trn2 --lnc=1",
            warmup=10,
            iterations=100,
            repeats=args.repeats,
            timing_scope="NKI nc_latency, one core, device execution excluding host invocation",
        )
        try:
            with np.load(data_path) as data:
                values = [data[f"a{i}"] for i in range(len(data.files) - 1)]
                expected = data["expected"]
            if name in ("maxpool", "layernorm"):
                code = (
                    "import math\nimport numpy as np\nimport neuronxcc.nki as nki\nimport neuronxcc.nki.language as nl\n"
                    + code
                )
            emitted = folder / "baseline.py"
            emitted.write_text(code)
            spec = importlib.util.spec_from_file_location(
                "prior_" + name, emitted
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            function = (
                module.solution
                if name in ("maxpool", "layernorm")
                else module.kernel
            )
            if name == "maxpool":
                values = [values[0], 3]
            elif name == "layernorm":
                values = [values[0], 1e-5, values[1], values[2]]
            elif name in ("matmul_add_rmsnorm", "add_rmsnorm_matmul"):
                values = [values[0], values[1], 1e-5, values[2], values[3]]
            elif name == "swiglu":
                # Reference order is x, up, down, gate; its documented input views follow.
                values = [
                    values[0].reshape(8, 4, 128, 8, 128),
                    values[2].reshape(8, 128, 3072),
                    values[3].reshape(24, 128, 1024),
                    values[1].reshape(8, 128, 3072),
                ]
            actual = serialize_device(
                nki.baremetal(function, additional_compile_opt=record["flags"])
            )(*values)
            if isinstance(actual, (tuple, list)):
                actual = actual[0]
            actual = np.asarray(actual).reshape(expected.shape)
            delta = actual - expected
            record["max_abs_error"] = float(np.max(np.abs(delta)))
            record["relative_l2_error"] = float(
                np.linalg.norm(delta.ravel())
                / max(np.linalg.norm(expected.ravel()), 1e-30)
            )
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
            record["correct"] = True
            benchmark = nki.benchmark(
                function,
                warmup=10,
                iters=100,
                additional_compile_opt=record["flags"],
                save_neff_name="file.neff",
                save_trace_name="profile.ntff",
            )
            serialize_device(benchmark)
            record["latencies"] = []
            for repeat in range(args.repeats):
                if repeat == 0:
                    benchmark(*values)
                else:
                    replay_values, replay_kwargs = benchmark._voyager_execution
                    benchmark.execute_neff(*replay_values, **replay_kwargs)
                latency = benchmark.benchmark_result.nc_latency
                record["latencies"].append(
                    {
                        f"p{q}_us": float(latency.get_latency_percentile(q))
                        for q in (0, 50, 90, 99, 100)
                    }
                )
            record["benchmark_compilation"] = (
                "One NEFF reused for all timing repeats"
            )
            record["status"] = "pass"
        except Exception:
            record["status"] = "fail"
            record["error"] = traceback.format_exc()
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == record["source_sha256"]
        )
        record["compiler_version"] = importlib.metadata.version("neuronx-cc")
        record["artifact_sha256"] = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("baseline.py", "file.neff", "profile.ntff")
            if (folder / name).is_file()
        }
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
