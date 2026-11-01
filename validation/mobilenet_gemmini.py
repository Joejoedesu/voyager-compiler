"""Prepare numerical checks and queue-ready replay; never download assets.

PYTHONPATH=src:test python validation/mobilenet_gemmini.py OUT --case model \
    --weights /path/to/mobilenet_v2.pth --image-size 224
Run the emitted replay through vcs-sim-queue separately. Random inputs establish
execution consistency, not dataset accuracy. Each case requires a fresh output.
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.gemmini.emission.replay import write_replay
from voyager_compiler.targets.gemmini.target import TARGET
from voyager_compiler.workloads.mobilenet_v2 import WORKLOAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--case", choices=("dw1", "dw2", "block", "model"), required=True
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--input", type=Path, help="Preprocessed NCHW float32 .npy"
    )
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--quantization", choices=("default", "local"), default="local"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(9)
    target, options = TARGET, {}
    if args.case == "model":
        workload = WORKLOAD
        options = dict(weights=args.weights, image_size=args.image_size)
        if args.weights:
            digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
            workload = replace(workload, assets=((str(args.weights), digest),))
    elif args.case == "block":
        from torchvision.models.mobilenetv2 import InvertedResidual

        model = torch.nn.Sequential(
            InvertedResidual(8, 8, 1, 6),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        ).eval()
        x = torch.randn(1, 8, 9, 9)
        workload = WorkloadSpec(
            "inverted-residual", lambda _: model, lambda m, o: ((x,), {})
        )
    else:
        stride = int(args.case[-1])

        class Depthwise(torch.nn.Module):
            def forward(self, x, w, b):
                return torch.ops.gemmini.depthwise_conv2d.default(
                    x, w, b, stride, 1, 0.125
                )

        inputs = (
            torch.randint(-8, 8, (2, 9, 11, 3), dtype=torch.int8),
            torch.randint(-4, 4, (3, 1, 3, 3), dtype=torch.int8),
            torch.tensor([-11, 0, 9], dtype=torch.int32),
        )
        workload = WorkloadSpec(
            args.case, lambda _: Depthwise(), lambda m, o: (inputs, {})
        )
        recipe = next(
            r
            for r in TARGET.implementations
            if r.name == "gemmini.depthwise_conv2d.hardware_loop"
        )
        target = replace(
            TARGET,
            implementations=(
                replace(
                    recipe,
                    candidates=lambda n, c: (
                        {"tile": (2, 3), "mode": "hardware_loop"},
                    ),
                ),
            ),
        )
    if args.input or args.calibration:
        import numpy as np

        if args.input:
            options["input"] = torch.from_numpy(
                np.load(args.input, allow_pickle=False).copy()
            )
        if args.calibration:
            options["calibration"] = [
                ((torch.from_numpy(np.load(p, allow_pickle=False).copy()),), {})
                for p in args.calibration
            ]
        paths = [*([args.input] if args.input else []), *args.calibration]
        workload = replace(
            workload,
            assets=workload.assets
            + tuple(
                (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
                for p in dict.fromkeys(paths)
            ),
        )
    options["gemmini_local_domains"] = args.quantization == "local"
    result = compile_workload(
        workload, target, args.output, options=options, dump_tensors=False
    )
    outputs, _ = run(result, reference=result.reference, rtol=0, atol=0)
    commands = args.output / "gemmini/commands.json"
    write_replay(result, args.output / "replay", commands)
    stream = json.loads(commands.read_text())
    summary = dict(
        case=args.case,
        quantization=args.quantization,
        host_exact=True,
        endpoint_validated=False,
        implementations=dict(
            Counter(b["implementation"] for b in stream["bodies"])
        ),
        comparison="independent converted graph where quantized; RTL replay compares all allocated bytes",
    )
    if args.case in ("block", "model"):
        scale = result.bundle["provenance"]["gemmini_quantization"]["outputs"][
            0
        ]["scale"]
        reference = result.original_reference.flatten().float()
        actual = tree_flatten(outputs)[0][0].flatten().float() * scale
        summary["probe_accuracy"] = dict(
            cosine=torch.nn.functional.cosine_similarity(
                actual, reference, dim=0
            ).item(),
            nrmse=(
                (actual - reference).norm() / reference.norm().clamp_min(1e-12)
            ).item(),
            scope=(
                "single supplied input; not dataset accuracy"
                if args.input
                else "single synthetic input; not ImageNet accuracy"
            ),
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
