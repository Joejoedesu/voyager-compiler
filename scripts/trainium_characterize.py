"""Isolated NKI primitive characterization; never fits whole-kernel runtimes.

Uses the installed SDK, saves source, NEFF, NTFF and decoded instructions.
Run serially on one core using the NKI environment.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import neuronxcc.nki as nki


def source(mode, count, width, dtype):
    header = [
        "import neuronxcc.nki as nki",
        "import neuronxcc.nki.language as nl",
        "import neuronxcc.nki.isa as nisa",
        "@nki.compiler.skip_middle_end_transformations",
        "@nki.jit",
        "def kernel(a, b):",
    ]
    lines = []

    def emit(s):
        lines.append("    " + s)

    emit(
        f'out = nl.ndarray(({count if mode in ("dma", "matmul", "ready", "fill", "gather") else 1}, 128, {width}), dtype=nl.{dtype}, buffer=nl.shared_hbm)'
    )
    emit("p = nl.arange(128)[:, None]")
    emit(f"f = nl.arange({width})[None, :]")
    if mode == "fill":
        for i in range(count):
            emit(
                f"y{i} = nisa.memset((128, {width}), value={i}, dtype=nl.{dtype}, engine=nisa.vector_engine)"
            )
            emit(f"nisa.dma_copy(dst=out[{i}, p, f], src=y{i})")
    elif mode == "gather":
        for i in range(count):
            emit(
                f"x{i} = nl.ndarray((128, {width*2}), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
            emit(
                f"nisa.dma_copy(dst=x{i}, src=b[{i}, p, nl.arange({width*2})[None, :]])"
            )
            emit(
                f"y{i} = nl.ndarray((128, {width}), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
            emit(
                f"y{i}[...] = nisa.tensor_copy(x{i}[p, f*2], engine=nisa.vector_engine)"
            )
            emit(f"nisa.dma_copy(dst=out[{i}, p, f], src=y{i})")
    elif mode == "ready":
        emit(
            f"x = nl.ndarray((128, {count*width}), dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
        emit(
            f"w = nl.ndarray((128, {count*128}), dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
        emit(
            f"nisa.dma_copy(dst=x, src=b[p, nl.arange({count*width})[None, :]])"
        )
        emit(f"nisa.dma_copy(dst=w, src=a[p, nl.arange({count*128})[None, :]])")
        for i in range(count):
            emit(
                f"z{i} = nisa.nc_matmul(w[p, {i*128}+nl.arange(128)[None, :]], x[p, {i*width}+f])"
            )
            emit(
                f"y{i} = nl.ndarray((128, {width}), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
            emit(
                f"y{i}[...] = nisa.tensor_copy(z{i}, engine=nisa.scalar_engine)"
            )
            emit(f"nisa.dma_copy(dst=out[{i}, p, f], src=y{i})")
    elif mode in ("dma", "matmul", "accumulate"):
        for i in range(count):
            emit(
                f"x{i} = nl.ndarray((128, {width}), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
            emit(f"nisa.dma_copy(dst=x{i}, src=b[{i}, p, f])")
            if mode != "dma":
                emit(
                    f"w{i} = nl.ndarray((128, 128), dtype=nl.{dtype}, buffer=nl.sbuf)"
                )
                emit(
                    f"nisa.dma_copy(dst=w{i}, src=a[{i}, p, nl.arange(128)[None, :]])"
                )
        if mode == "accumulate":
            emit(
                f"acc = nl.zeros((128, {width}), dtype=nl.float32, buffer=nl.psum)"
            )
        for i in range(count):
            if mode == "dma":
                emit(f"nisa.dma_copy(dst=out[{i}, p, f], src=x{i})")
            else:
                if mode == "matmul":
                    emit(f"acc{i} = nisa.nc_matmul(w{i}, x{i})")
                else:
                    emit(f"acc += nisa.nc_matmul(w{i}, x{i})")
                if mode == "matmul" or i == count - 1:
                    emit(
                        f"y{i} = nl.ndarray((128, {width}), dtype=nl.{dtype}, buffer=nl.sbuf)"
                    )
                    emit(
                        f'y{i}[...] = nisa.tensor_copy({"acc"+str(i) if mode=="matmul" else "acc"}, engine=nisa.scalar_engine)'
                    )
                    emit(
                        f'nisa.dma_copy(dst=out[{i if mode=="matmul" else 0}, p, f], src=y{i})'
                    )
    else:
        emit(f"x = nl.ndarray((128, 128), dtype=nl.{dtype}, buffer=nl.sbuf)")
        emit("nisa.dma_copy(dst=x, src=b[0, p, f])")
        for i in range(count):
            if mode == "transpose":
                emit(
                    f'z{i} = nisa.nc_transpose({"x" if i==0 else "y"+str(i-1)}, engine=nisa.tensor_engine)'
                )
            emit(
                f"y{i} = nl.ndarray((128, 128), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
            emit(
                f'y{i}[...] = nisa.tensor_copy({"z"+str(i) if mode=="transpose" else ("x" if i==0 else "y"+str(i-1))}, engine=nisa.scalar_engine)'
            )
        emit(f"nisa.dma_copy(dst=out[0, p, f], src=y{count-1})")
    emit("return out")
    return "\n".join(header + lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--case", nargs="*")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NEURON_RT_ENABLE_DGE_NOTIFICATIONS", "1")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent)
        + ":/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    jobs = []
    for dtype in ["float32", "bfloat16"]:
        for width in [128, 512]:
            for count in [1, 16]:
                for mode in ["dma", "matmul"]:
                    jobs.append((mode, count, width, dtype))
            jobs.append(("accumulate", 16, width, dtype))
            jobs.append(("ready", 16, width, dtype))
            jobs.append(("fill", 16, width, dtype))
            jobs.append(("gather", 16, width, dtype))
        for count in [1, 8, 16]:
            jobs.append(("transpose", count, 128, dtype))
        jobs.append(("copy", 16, 128, dtype))
    for width in [32, 256]:
        jobs.append(("dma", 16, width, "float32"))
    records = []
    for mode, count, width, dtype in jobs:
        name = f"{mode}_{dtype}_w{width}_n{count}"
        if args.case and name not in args.case:
            continue
        folder = root / name
        folder.mkdir(exist_ok=True)
        os.chdir(folder)
        code = source(mode, count, width, dtype)
        path = folder / "program.py"
        path.write_text(code)
        record = dict(
            case=name,
            mode=mode,
            count=count,
            width=width,
            dtype=dtype,
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),
            flags="--target=trn2 --lnc=1",
        )
        try:
            rng = np.random.default_rng(701)
            a = (rng.standard_normal((count, 128, 128)) * 0.1).astype("float32")
            b = (
                rng.standard_normal(
                    (count, 128, width * (2 if mode == "gather" else 1))
                )
                * 0.1
            ).astype("float32")
            if dtype == "bfloat16":
                import ml_dtypes

                a = a.astype(ml_dtypes.bfloat16)
                b = b.astype(ml_dtypes.bfloat16)
            aa, bb = a.astype("float32"), b.astype("float32")
            if mode in ("matmul", "ready"):
                expected = np.swapaxes(aa, 1, 2) @ bb
            elif mode == "accumulate":
                expected = np.sum(
                    np.swapaxes(aa, 1, 2) @ bb, axis=0, keepdims=True
                )
            elif mode == "transpose":
                expected = bb[:1].transpose(0, 2, 1) if count % 2 else bb[:1]
            elif mode == "copy":
                expected = bb[:1]
            elif mode == "fill":
                expected = np.broadcast_to(
                    np.arange(count, dtype="float32")[:, None, None],
                    (count, 128, width),
                )
            elif mode == "gather":
                expected = bb[:, :, ::2]
            else:
                expected = bb
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if mode == "ready":
                a = np.ascontiguousarray(a.transpose(1, 0, 2).reshape(128, -1))
                b = np.ascontiguousarray(b.transpose(1, 0, 2).reshape(128, -1))
            actual = nki.baremetal(
                module.kernel, additional_compile_opt=record["flags"]
            )(a, b)
            tol = 0.03 if dtype == "bfloat16" else 0.0005
            np.testing.assert_allclose(
                np.asarray(actual).astype("float32"),
                expected,
                atol=tol,
                rtol=tol,
            )
            record["max_abs_error"] = float(
                np.max(np.abs(np.asarray(actual).astype("float32") - expected))
            )
            bench = nki.benchmark(
                module.kernel,
                warmup=10,
                iters=100,
                additional_compile_opt=record["flags"],
                save_neff_name="file.neff",
                save_trace_name="profile.ntff",
            )
            record["p50_us"] = []
            for _ in range(args.repeats):
                bench(a, b)
                record["p50_us"].append(
                    float(
                        bench.benchmark_result.nc_latency.get_latency_percentile(
                            50
                        )
                    )
                )
            cmd = [
                "/opt/aws/neuron/bin/neuron-profile",
                "view",
                "-n",
                "file.neff",
                "-s",
                "profile.ntff",
                "--output-format",
                "json",
                "--output-file",
                "profile.json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            (folder / "decode.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            record["artifact_sha256"] = {
                f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                for f in ["file.neff", "profile.ntff"]
            }
            record["status"] = "pass"
        except Exception:
            record["status"] = "fail"
            record["error"] = traceback.format_exc()
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps(record), flush=True)
    all_records = [
        json.loads(p.read_text()) for p in sorted(root.glob("*/result.json"))
    ]
    (root / "results.json").write_text(json.dumps(all_records, indent=2) + "\n")
    if any(r["status"] == "fail" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
