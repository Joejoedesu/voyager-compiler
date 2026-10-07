"""Isolated VectorE instruction characterization, with direct SBUF placement."""

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
import neuronxcc.nki as nki


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--operation", choices=("binary", "scalar", "copy_scalar"))
    p.add_argument("--width", type=int)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.operation is None:
        for op in ("binary", "scalar"):
            for width in (128, 512, 2048, 8192):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output",
                        str(root),
                        "--operation",
                        op,
                        "--width",
                        str(width),
                    ],
                    check=True,
                )
        return
    root = root / f"{a.operation}-{a.width}"
    root.mkdir(exist_ok=True)
    os.chdir(root)
    os.environ["PATH"] = (
        str(Path(sys.executable).parent)
        + ":/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    n = a.width
    code = [
        "import neuronxcc.nki as nki",
        "import neuronxcc.nki.language as nl",
        "import neuronxcc.nki.isa as nisa",
        "import neuronxcc.nki.compiler as ncc",
        "@nki.compiler.skip_middle_end_transformations",
        "@nki.jit",
        "def kernel(a,b):",
    ]

    def line(s):
        code.append("    " + s)

    for i, name in enumerate(("x", "y", "z")):
        line(
            f"{name} = nl.ndarray((128,{n}),dtype=nl.float32,buffer=ncc.sbuf.alloc(lambda idx,pdim_size,fdim_size: (0,{i*n*4})))"
        )
    line("nisa.dma_copy(dst=x,src=a)")
    line("nisa.dma_copy(dst=y,src=b)")
    for i in range(16):
        src, dst = ("x", "z") if i % 2 == 0 else ("z", "x")
        call = (
            f"nisa.tensor_copy({src},engine=nisa.scalar_engine)"
            if a.operation == "copy_scalar"
            else (
                f"nisa.tensor_tensor({src},y,op=nl.multiply,engine=nisa.vector_engine)"
                if a.operation == "binary"
                else f"nisa.tensor_scalar({src},op0=nl.multiply,operand0=0.99,engine=nisa.vector_engine)"
            )
        )
        line(f"{dst}[...] = {call}")
    line(f"out = nl.ndarray((128,{n}),dtype=nl.float32,buffer=nl.shared_hbm)")
    line("nisa.dma_copy(dst=out,src=x)")
    line("return out")
    path = root / "program.py"
    path.write_text("\n".join(code) + "\n")
    spec = importlib.util.spec_from_file_location("probe", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    x = np.random.default_rng(29).normal(size=(128, n)).astype(np.float32)
    y = np.full_like(x, 0.99)
    np.savez(root / "reference.npz", a=x, b=y)

    def lock(kernel):
        import fcntl

        execute = kernel.execute_neff

        def locked(*args, **kw):
            with open("/tmp/voyager-trainium-device.lock", "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                return execute(*args, **kw)

        kernel.execute_neff = locked
        return kernel

    flags = "--target=trn2 --lnc=1"
    actual = lock(nki.baremetal(m.kernel, additional_compile_opt=flags))(x, y)
    np.testing.assert_allclose(
        actual,
        x if a.operation == "copy_scalar" else x * np.float32(0.99) ** 16,
        atol=1e-5,
        rtol=1e-5,
    )
    benchmark = lock(
        nki.benchmark(
            m.kernel,
            warmup=10,
            iters=100,
            additional_compile_opt=flags,
            save_neff_name="file.neff",
            save_trace_name="profile.ntff",
        )
    )
    benchmark(x, y)
    subprocess.run(
        [
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
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    data = json.loads((root / "profile.json").read_text())
    opcode = {
        "binary": "TENSOR_TENSOR",
        "scalar": "TENSOR_SCALAR",
        "copy_scalar": "COPY",
    }[a.operation]
    durations = [
        i["duration"] for i in data["instruction"] if i["opcode"] == opcode
    ]
    assert len(durations) == 16, durations
    record = dict(
        operation=a.operation,
        width=n,
        partitions=128,
        dtype="float32",
        status="pass",
        compiler=importlib.metadata.version("neuronx-cc"),
        flags=flags,
        iterations=100,
        warmup=10,
        instruction_durations_ns=durations,
        median_instruction_ns=statistics.median(durations),
        artifact_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "program.py",
                "reference.npz",
                "file.neff",
                "profile.ntff",
            )
        },
    )
    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
