"""Direct-address, matched FP32 matmul geometry probes on real Trainium2.

Vary only moving width and free-axis operand strides. No kernel timing fitting.
"""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import numpy as np
import neuronxcc.nki as nki


def source(width, moving_stride, stationary_stride, steps=16, groups=8):
    afree = 128 * stationary_stride + steps + groups
    bfree = width * moving_stride + steps + groups
    boff = ((afree * 4 + 63) // 64) * 64
    yoff = ((boff + bfree * 4 + 63) // 64) * 64
    lines = [
        "import neuronxcc.nki as nki",
        "import neuronxcc.nki.language as nl",
        "import neuronxcc.nki.isa as nisa",
        "import neuronxcc.nki.compiler as ncc",
        "@nki.compiler.skip_middle_end_transformations",
        "@nki.jit",
        "def kernel(a,b):",
    ]

    def put(s):
        lines.append("    " + s)

    for name, free, offset in [
        ("aa", afree, 0),
        ("bb", bfree, boff),
        ("yy", width * groups, yoff),
    ]:
        put(
            f"{name}=nl.ndarray((128,{free}),dtype=nl.float32,buffer=ncc.sbuf.alloc(lambda idx,pdim_size,fdim_size:(0,{offset})))"
        )
    put("nisa.dma_copy(dst=aa,src=a)")
    put("nisa.dma_copy(dst=bb,src=b)")
    put("p=nl.arange(128)[:,None]")
    put("n=nl.arange(128)[None,:]")
    put(f"m=nl.arange({width})[None,:]")
    for g in range(groups):
        put(
            f"z{g}=nl.ndarray((128,{width}),dtype=nl.float32,buffer=ncc.psum.alloc(lambda idx,pdim_size,fdim_size:({g},0,0)))"
        )
        put(
            f"z{g}[...]=nisa.memset((128,{width}),value=0,dtype=nl.float32,engine=nisa.vector_engine)"
        )
        for step in range(steps):
            put(
                f"z{g}[...] += nisa.nc_matmul(aa[p,n*{stationary_stride}+{step+g}],bb[p,m*{moving_stride}+{step+g}])"
            )
        put(
            f"yy[p,{g*width}+m]=nisa.tensor_copy(z{g},engine=nisa.scalar_engine)"
        )
    put(
        f"out=nl.ndarray((128,{width*groups}),dtype=nl.float32,buffer=nl.shared_hbm)"
    )
    put("nisa.dma_copy(dst=out,src=yy)")
    put("return out")
    return "\n".join(lines) + "\n", afree, bfree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--moving-stride", type=int, default=1)
    ap.add_argument("--stationary-stride", type=int, default=1)
    args = ap.parse_args()
    case = (
        args.output.resolve()
        / f"w{args.width}-m{args.moving_stride}-s{args.stationary_stride}"
    )
    case.mkdir(parents=True, exist_ok=True)
    if (case / "result.json").exists():
        print("existing", case, flush=True)
        return
    os.chdir(case)
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
    os.environ["PATH"] = (
        str(Path(sys.executable).parent)
        + ":/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:"
        + os.environ["PATH"]
    )
    code, afree, bfree = source(
        args.width, args.moving_stride, args.stationary_stride
    )
    Path("program.py").write_text(code)
    spec = importlib.util.spec_from_file_location("probe", case / "program.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(928)
    a = rng.normal(0, 0.05, (128, afree)).astype("float32")
    b = rng.normal(0, 0.05, (128, bfree)).astype("float32")
    expected = np.concatenate(
        [
            sum(
                a[:, np.arange(128) * args.stationary_stride + i + g].T
                @ b[:, np.arange(args.width) * args.moving_stride + i + g]
                for i in range(16)
            )
            for g in range(8)
        ],
        axis=1,
    )
    np.savez("reference.npz", a=a, b=b, expected=expected)

    def lock(kernel, save=False):
        execute = kernel.execute_neff

        def run(*values, **kwargs):
            if save:
                neff = Path(values[0] if values else kwargs["neff"])
                if neff.resolve() != Path("file.neff").resolve():
                    shutil.copy2(neff, "file.neff")
                kernel.replay_values = (
                    ("file.neff", *values[1:]) if values else ()
                )
                kernel.replay_kwargs = dict(kwargs)
                if not values:
                    kernel.replay_kwargs["neff"] = "file.neff"
            with open("/tmp/voyager-trainium-device.lock", "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                return execute(*values, **kwargs)

        kernel.execute_neff = run
        return kernel

    flags = "--target=trn2 --lnc=1"
    run = lock(
        nki.baremetal(
            mod.kernel, additional_compile_opt=flags, save_neff_name="file.neff"
        ),
        True,
    )
    actual = np.asarray(run(a, b))
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
    np.save("actual.npy", actual)
    bench = lock(
        nki.benchmark(
            mod.kernel,
            warmup=10,
            iters=100,
            additional_compile_opt=flags,
            save_neff_name="file.neff",
            save_trace_name="profile.ntff",
        )
    )
    times = []
    for _ in range(3):
        bench.execute_neff(*run.replay_values, **run.replay_kwargs)
        times.append(
            float(bench.benchmark_result.nc_latency.get_latency_percentile(50))
        )
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
    data = json.loads(Path("profile.json").read_text())
    groups = defaultdict(list)
    for i in data["instruction"]:
        if i["opcode"] in ("MATMUL", "LDWEIGHTS"):
            groups[i["raw_bir_id"]].append(i)
    gs = sorted(groups.values(), key=lambda g: min(i["compiler_pc"] for i in g))
    assert len(gs) == 128 and all(len(g) == 4 for g in gs)
    spacing = [
        min(i["timestamp"] for i in y) - min(i["timestamp"] for i in x)
        for x, y in zip(gs, gs[1:])
        if max(i["compiler_pc"] for i in x) + 1
        == min(i["compiler_pc"] for i in y)
    ]
    spans = [
        max(i["timestamp"] + i["duration"] for i in g)
        - min(i["timestamp"] for i in g)
        for g in gs
    ]

    def stat(xs):
        xs = sorted(xs)
        return dict(
            count=len(xs),
            median=statistics.median(xs),
            p10=xs[len(xs) // 10],
            p90=xs[len(xs) * 9 // 10],
        )

    record = dict(
        status="pass",
        width=args.width,
        moving_stride=args.moving_stride,
        stationary_stride=args.stationary_stride,
        steps=16,
        groups=8,
        dtype="float32",
        flags=flags,
        p50_us=times,
        spacing_ns=stat(spacing),
        span_ns=stat(spans),
        max_abs_error=float(np.max(np.abs(actual - expected))),
        operand_example=[i["operands"] for i in gs[16]],
        artifact_sha256={
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in [
                "program.py",
                "file.neff",
                "profile.ntff",
                "profile.json",
                "reference.npz",
                "actual.npy",
            ]
        },
    )
    Path("result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
