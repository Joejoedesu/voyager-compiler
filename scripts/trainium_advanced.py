"""Generate the six reviewed workloads through Voyager's shared compiler."""

import argparse
import hashlib
import json
from pathlib import Path
import traceback
import time
import numpy as np
import torch
import torch.nn.functional as F
import voyager_compiler as vc
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.trainium.hardware import neuron_core

PRIOR_ROOT = Path("/home/ubuntu/ML/prior_works")
CASES = {
    "maxpool": (
        "autocomp/sols/trn-advanced-nki1/2_maxpool_ref.py",
        (4096, 4096),
    ),
    "layernorm": (
        "autocomp/sols/trn-tutorial-nki1/1_layernorm_ref.py",
        (4096, 8192),
    ),
    "matmul_add_rmsnorm": (
        "AccelOpt/NKIBench/kernels/matmul_add_rmsnorm_M4096_N2048_K2048_0.py",
        (4096, 2048, 2048),
    ),
    "add_rmsnorm_matmul": (
        "AccelOpt/NKIBench/kernels/add_rmsnorm_matmul_M4096_N2048_K1024_0.py",
        (4096, 2048, 1024),
    ),
    "swiglu": (
        "AccelOpt/NKIBench/kernels/swiglu_M4096_N3072_K1024_0.py",
        (4096, 3072, 1024),
    ),
    "bmm_softmax": (
        "AccelOpt/NKIBench/kernels/bmm_softmax_B16_K64_M4096_N4096_0.py",
        (16, 4096, 4096, 64),
    ),
}


class Workload(torch.nn.Module):
    def __init__(self, name, pool_padding=0):
        super().__init__()
        self.name = name
        self.pool_padding = pool_padding

    def forward(self, *a):
        if self.name == "maxpool":
            return F.max_pool2d(
                a[0][None, None], 3, stride=1, padding=self.pool_padding
            )[0, 0]
        if self.name == "layernorm":
            return F.layer_norm(a[0], (a[0].shape[-1],), a[1], a[2], 1e-5)
        if self.name == "matmul_add_rmsnorm":
            y = a[0] @ a[1] + a[2]
            return F.rms_norm(y, (y.shape[-1],), a[3], 1e-5)
        if self.name == "add_rmsnorm_matmul":
            y = a[0] + a[2]
            return F.rms_norm(y, (y.shape[-1],), a[3], 1e-5) @ a[1]
        if self.name == "swiglu":
            return (F.silu(a[0] @ a[1]) * (a[0] @ a[2])) @ a[3]
        return torch.softmax(a[0] @ a[1], dim=-1)


def inputs(name, small, override=None):
    shape = CASES[name][1]
    if small:
        shape = (
            (32, 194)
            if name == "maxpool"
            else (
                (128, 256)
                if name == "layernorm"
                else (
                    (2, 128, 128, 128)
                    if name == "bmm_softmax"
                    else (128, 128, 128)
                )
            )
        )
    if override is not None:
        shape = tuple(override)
    rand = lambda *s: torch.randn(*s)
    if name == "maxpool":
        return (rand(*shape),)
    if name == "layernorm":
        return (rand(*shape), rand(shape[-1]), rand(shape[-1]))
    if name == "bmm_softmax":
        b, m, n, k = shape
        return (rand(b, m, k), rand(b, k, n))
    m, n, k = shape
    if name == "swiglu":
        return (
            rand(m, k) * 0.1,
            rand(k, n) * 0.1,
            rand(k, n) * 0.1,
            rand(n, k) * 0.1,
        )
    return (
        rand(m, k),
        rand(k, n),
        rand(m, n if name == "matmul_add_rmsnorm" else k),
        rand(n if name == "matmul_add_rmsnorm" else k),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--temporary-buffer-depth", type=int, default=1,
        help="SBUF slot-pool multiplier; 1 preserves existing placement, >1 rotates bounded slots (strict ISA only)",
    )
    p.add_argument(
        "--strict-realization", action=argparse.BooleanOptionalAction, default=True,
        help="Fix physical ISA buffers in Voyager (default); --no-strict-realization lets NKI allocate them",
    )
    p.add_argument("--cases", nargs="*", default=list(CASES))
    p.add_argument("--small", action="store_true")
    p.add_argument("--pointwise-fusion", action="store_true")
    p.add_argument("--row-regions", action="store_true", help="Compose row-independent operations using the shared bufferizer")
    p.add_argument("--matmul-orientation", choices=("auto", "weights", "activations"), default="auto")
    p.add_argument(
        "--matmul-operands",
        choices=("staged", "direct", "reuse"),
        default="staged",
    )
    p.add_argument("--references-only", action="store_true")
    p.add_argument("--tile", type=int, nargs=3)
    p.add_argument("--shape", type=int, nargs="+")
    p.add_argument("--pool-padding", type=int, choices=(0, 1), default=0)
    p.add_argument("--matmul-weight-layout", choices=("auto", "generic", "k_partitioned"), default="auto", help="Whole invariant row-region weight layout; auto searches both")
    a = p.parse_args()
    if a.shape and (
        len(a.cases) != 1
        or len(a.shape) != len(CASES[a.cases[0]][1])
        or min(a.shape) < 1
    ):
        p.error("--shape requires one case and its positive shape dimensions")
    torch.set_num_threads(2)
    torch.manual_seed(29)
    a.output.mkdir(parents=True, exist_ok=True)
    inventory = {
        name: dict(
            source=str(PRIOR_ROOT / source),
            sha256=hashlib.sha256(
                (PRIOR_ROOT / source).read_bytes()
            ).hexdigest(),
            full_shape=shape,
        )
        for name, (source, shape) in CASES.items()
    }
    (a.output / "shortlist.json").write_text(json.dumps(inventory, indent=2))
    for name in a.cases:
        root = a.output.resolve() / name
        root.mkdir(parents=True, exist_ok=True)
        old_error = root / "generation_error.txt"
        if old_error.exists():
            old_error.rename(
                root / f"generation_error.previous-{time.time_ns()}.txt"
            )
        try:
            x = inputs(name, a.small, a.shape)
            module = Workload(name, a.pool_padding)
            with torch.no_grad():
                expected = module(*x)
                if a.references_only:
                    np.savez(
                        root / "reference.npz",
                        **{f"a{i}": v.numpy() for i, v in enumerate(x)},
                        expected=expected.numpy(),
                    )
                    (root / "generation.json").write_text(
                        json.dumps(
                            dict(
                                case=name,
                                small=a.small,
                                input_dtype="float32",
                                tolerance=dict(atol=1e-3, rtol=1e-3),
                            ),
                            indent=2,
                        )
                    )
                    print("REFERENCES", name, flush=True)
                    continue
                graph = vc.export_model(module, x)
                from voyager_compiler.trainium.execution import TrainiumTuning
                from voyager_compiler.trainium.mapping import (
                    TrainiumMappingPolicy,
                )

                hw = neuron_core(3)
                context = CompilerContext.resolve(
                    hw,
                    TrainiumMappingPolicy(
                        hw,
                        TrainiumTuning(
                            matmul_operands=a.matmul_operands,
                            matmul_orientation=a.matmul_orientation,
                            matmul_weight_layout=a.matmul_weight_layout,
                            strict_realization=a.strict_realization,
                            temporary_buffer_depth=a.temporary_buffer_depth,
                            pointwise_fusion=a.pointwise_fusion,
                        ),
                    ),
                )
                if a.tile:
                    from dataclasses import replace
                    from interstellar import loop_enum as le
                    from voyager_compiler.codegen.transform.tiling.tiler import (
                        TileConstraint,
                    )

                    prepare = context.policy.prepare_matrix
                    constraint = TileConstraint(
                        exact=tuple(zip((le.OX, le.OC, le.IC), a.tile))
                    )
                    context.policy.prepare_matrix = (
                        lambda problem, tiler: prepare(
                            replace(
                                problem,
                                constraint=constraint.merged(
                                    problem.constraint
                                ),
                            ),
                            tiler,
                        )
                    )
                vc.transform(graph, x, context=context)
                (root / "transformed.txt").write_text(str(graph.graph))
                vc.compile(
                    graph,
                    x,
                    context=context,
                    output_dir=root,
                    dump_tensors=False,
                    bufferization_options=BufferizationOptions(row_regions=a.row_regions),
                )
                torch.testing.assert_close(
                    graph(*x), expected, atol=1e-3, rtol=1e-3
                )
                context.realize(root)
            if name == "maxpool":
                # Channel=1: the collateral NHWC ABI is a shape-only view of this 2-D result.
                plan = json.loads((root / "nki/plan.json").read_text())
                expected = expected.reshape(plan["outputs"][0]["shape"])
            np.savez(
                root / "reference.npz",
                **{f"a{i}": v.numpy() for i, v in enumerate(x)},
                expected=expected.numpy(),
            )
            (root / "generation.json").write_text(
                json.dumps(
                    dict(
                        case=name,
                        input_dtype="float32",
                        small=a.small,
                        shape=a.shape,
                        pool_padding=a.pool_padding,
                        row_regions=a.row_regions,
                        diagnostic_tile=a.tile,
                        prior_work=inventory[name],
                        bufferized_correct=True,
                        tolerance=dict(atol=1e-3, rtol=1e-3),
                    ),
                    indent=2,
                )
            )
            print("GENERATED", name, flush=True)
        except Exception:
            error = traceback.format_exc()
            (root / "generation_error.txt").write_text(error)
            print("FAILED", name, error, flush=True)


if __name__ == "__main__":
    main()
