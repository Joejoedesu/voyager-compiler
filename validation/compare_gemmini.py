"""Local donor comparison and replay preparation; no downloads or model assets.

Run with --donor PATH to compare the independent legacy numerical frontend.
Generated replay uses only compiled inputs/parameters, never reference outputs.
RTL invocation is separate: Gemmini/scripts/run.sh CASE/replay --out CASE/rtl.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def vision(case):
    import torch
    from torchvision import models

    torch.manual_seed(21)
    if case in ("resnet18",):
        model = getattr(models, case)(weights=None).eval()
    else:

        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 17, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(17)
                self.relu = torch.nn.ReLU()
                self.relu2 = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(3, 2, 1)
                self.avg = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(17, 9)
                self.proj = torch.nn.Conv2d(17, 17, 1)

            def forward(self, x):
                y = self.relu(self.bn(self.conv(x)))
                y = self.relu2(y + self.proj(y))
                return self.fc(torch.flatten(self.avg(self.pool(y)), 1))

        model = Vision().eval()
    from voyager_compiler.export_utils import get_conv_bn_layers

    model = torch.ao.quantization.fuse_modules(
        model, get_conv_bn_layers(model), inplace=False
    )
    return model, (
        (
            torch.randn(1, 3, 32, 32)
            if case.startswith("resnet")
            else torch.randn(1, 3, 8, 8)
        ),
    )


def worker(args):
    import numpy as np
    import torch
    from torch.utils._pytree import tree_flatten

    torch.set_num_threads(4)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.case in ("vision", "resnet18"):
        model, inputs = vision(args.case)
    else:
        from voyager_compiler.targets.gemmini.target import TARGET

        torch.manual_seed(9)
        if args.case.startswith("matmul"):

            class Matrix(torch.nn.Module):
                def forward(self, a, b, d):
                    return torch.ops.gemmini.matmul.default(a, b, d, 0.125)

            model = Matrix()
            inputs = (
                torch.randint(-8, 8, (19, 35), dtype=torch.int8),
                torch.randint(-8, 8, (35, 21), dtype=torch.int8),
                torch.arange(-10, 11, dtype=torch.int32),
            )
        elif args.case == "conv_split":

            class Conv(torch.nn.Module):
                def forward(self, x, w, b):
                    return torch.ops.gemmini.conv2d.default(
                        x, w, b, 2, 1, 0.125
                    )

            model = Conv()
            inputs = (
                torch.randint(-3, 4, (2, 9, 9, 19), dtype=torch.int8),
                torch.randint(-3, 4, (3, 3, 19, 21), dtype=torch.int8),
                torch.arange(-10, 11, dtype=torch.int32),
            )
        else:
            raise ValueError(args.case)
    if args.worker == "donor":
        from voyager_compiler.gemmini.frontend import emit_fx
        from voyager_compiler.gemmini.hardware import lean_config

        graph = torch.fx.symbolic_trace(model)
        emit_fx(
            graph, inputs, output_dir=out / "compiler", config=lean_config()
        )
        outputs = graph.meta["gemmini_quantized_output"]
        evidence = dict(
            kind="donor_integer_reference", compiler="voyager-gemmini"
        )
    else:
        from dataclasses import replace

        from voyager_compiler.extensions import WorkloadSpec
        from voyager_compiler.pipeline import compile_workload
        from voyager_compiler.runtime import run
        from voyager_compiler.targets.gemmini.emission.replay import (
            write_replay,
        )
        from voyager_compiler.targets.gemmini.target import TARGET

        target = TARGET
        if args.case.startswith("matmul"):
            mode = args.case.removeprefix("matmul_")
            spec = next(
                r
                for r in TARGET.implementations
                if r.name == f"gemmini.matmul.{mode}"
            )
            spec = replace(
                spec,
                candidates=lambda n, c: ({"tile": (16, 16, 16), "mode": mode},),
            )
            target = replace(TARGET, implementations=(spec,))
        if args.case == "conv_split":
            spec = next(
                r
                for r in TARGET.implementations
                if r.name == "gemmini.conv2d.hardware_loop"
            )
            spec = replace(
                spec,
                candidates=lambda n, c: (
                    {"tile": (2, 3, 16, 16), "mode": "hardware_loop"},
                ),
            )
            target = replace(TARGET, implementations=(spec,))
        result = compile_workload(
            WorkloadSpec(args.case, lambda _: model, lambda m, o: (inputs, {})),
            target,
            out / "compiler",
            dump_tensors=False,
        )
        outputs, evidence = run(
            result, reference=result.reference, rtol=0, atol=0
        )
        evidence.update(
            choices={
                k: dict(
                    implementation=v.implementation,
                    parameters=dict(v.parameters),
                    latency=v.cost.latency,
                )
                for k, v in result.program.plan.choices.items()
            },
            dram_bytes=result.program.plan.allocation.dram_bytes,
            quantization="shared converted-graph reference checked before bufferization",
        )
        try:
            write_replay(
                result, out / "replay", out / "compiler/gemmini/commands.json"
            )
        except ValueError as error:
            if "16 MiB" not in str(error):
                raise
            evidence["replay_limit"] = str(error)
    leaves, _ = tree_flatten(outputs)
    np.savez(
        out / "outputs.npz",
        **{str(i): x.detach().cpu().numpy() for i, x in enumerate(leaves)},
    )
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--worker", choices=("current", "donor"))
    parser.add_argument(
        "--case",
        default="vision",
        choices=(
            "vision",
            "resnet18",
            "matmul_explicit",
            "matmul_hardware_loop",
            "conv_split",
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "vision",
            "resnet18",
            "matmul_explicit",
            "matmul_hardware_loop",
            "conv_split",
        ],
    )
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    import numpy as np

    repo = Path(__file__).resolve().parents[1]
    donor = (args.donor or repo.with_name("voyager-gemmini")).resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for case in args.cases:
        modes = (
            ("current", "donor")
            if args.donor and case in ("vision", "resnet18")
            else ("current",)
        )
        for mode in modes:
            checkout = repo if mode == "current" else donor
            env = dict(
                os.environ,
                PYTHONPATH=str(checkout / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                HF_HUB_OFFLINE="1",
                HF_DATASETS_OFFLINE="1",
                MPLCONFIGDIR=str(out / "matplotlib"),
            )
            with (out / f"{case}-{mode}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(out / f"{case}-{mode}"),
                        "--worker",
                        mode,
                        "--case",
                        case,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if "donor" in modes:
            current = np.load(out / f"{case}-current/outputs.npz")
            original = np.load(out / f"{case}-donor/outputs.npz")
            equal = current.files == original.files and all(
                np.array_equal(current[k], original[k]) for k in current.files
            )
            if not equal:
                raise AssertionError(f"{case}: donor numerical mismatch")
            results[case] = dict(donor_outputs_exact=True)
        else:
            results[case] = dict(
                host_outputs_exact=True, rtl_replay_prepared=True
            )
        (out / "comparison.json").write_text(
            json.dumps(
                dict(
                    passed=len(results) == len(args.cases),
                    expected_cases=args.cases,
                    cases=results,
                    scope="Host/reference and selected-plan checks; RTL results recorded separately",
                ),
                indent=2,
            )
            + "\n"
        )
        print(case, "PASS", flush=True)


if __name__ == "__main__":
    main()
