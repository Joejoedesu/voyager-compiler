"""Generate fresh Voyager collateral and independent PyTorch references."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import voyager_compiler as vc
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.converter import convert
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy
from voyager_compiler.trainium.execution import TrainiumTuning


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class AddRelu(torch.nn.Module):
    def forward(self, a, b):
        return torch.relu(a + b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/trainium/run",
    )
    parser.add_argument("--dma-transpose", action="store_true")
    parser.add_argument("--legacy-isa", action="store_true")
    parser.add_argument(
        "--isa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Explicit NKI ISA and panel result storage (default); --no-isa replays the language baseline",
    )
    parser.add_argument("--buffer-depth", type=int, default=2)
    parser.add_argument(
        "--force-buffer-depth",
        type=int,
        choices=(1, 2),
        help="Diagnostic: compare identical tiles at a specified logical depth",
    )
    parser.add_argument(
        "--copy-policy", choices=("balanced", "scalar"), default="scalar"
    )
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"))
    parser.add_argument(
        "--tile",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="Diagnostic fixed software tile; still uses shared search",
    )
    args = parser.parse_args()
    if args.legacy_isa and args.isa:
        parser.error(
            "--legacy-isa requires --no-isa for a language-baseline comparison"
        )
    torch.set_num_threads(2)
    torch.manual_seed(29)
    cases = [
        ("gemm128", 128, 128, 128),
        ("split_k", 128, 128, 256),
        ("gemm256", 256, 256, 256),
        ("gemm512", 512, 256, 512),
    ]
    if args.shape:
        cases = [("gemm_custom", *args.shape)]
    jobs = [
        (name, Matmul(), (torch.randn(m, k), torch.randn(k, n)))
        for name, m, n, k in cases
    ]
    jobs.append(
        (
            "add_relu",
            AddRelu(),
            (torch.randn(128, 128), torch.randn(128, 128)),
        )
    )
    if args.shape:
        jobs = jobs[:1]
    for name, module, inputs in jobs:
        inputs = tuple(x.to(getattr(torch, args.dtype)) for x in inputs)
        atol, rtol = (0.05, 0.02) if args.dtype == "bfloat16" else (5e-4, 5e-4)
        if args.cases and name not in args.cases:
            continue
        root = args.output.resolve() / name
        root.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            expected = module(*inputs)
            graph = vc.export_model(module, inputs)
            config = neuron_core(3)
            context = CompilerContext.resolve(
                config,
                TrainiumMappingPolicy(
                    config,
                    TrainiumTuning(
                        dma_transpose=args.dma_transpose,
                        explicit_isa=not args.legacy_isa,
                        isa_lowering=args.isa,
                        copy_policy=args.copy_policy,
                        max_buffer_depth=args.force_buffer_depth
                        or args.buffer_depth,
                        min_buffer_depth=args.force_buffer_depth or 1,
                    ),
                ),
            )
            if args.tile:
                from dataclasses import replace
                from interstellar import loop_enum as le
                from voyager_compiler.codegen.transform.tiling.tiler import (
                    TileConstraint,
                )

                prepare = context.policy.prepare_matrix
                constraint = TileConstraint(
                    exact=tuple(zip((le.OX, le.OC, le.IC), args.tile))
                )
                context.policy.prepare_matrix = lambda problem, tiler: prepare(
                    replace(
                        problem,
                        constraint=constraint.merged(problem.constraint),
                    ),
                    tiler,
                )
            vc.transform(graph, inputs, context=context)
            vc.compile(
                graph,
                inputs,
                context=context,
                output_dir=root,
                dump_tensors=False,
            )
            CompilerContext.from_artifacts(root, config).realize(root)
            torch.testing.assert_close(
                graph(*inputs), expected, atol=atol, rtol=rtol
            )
        plan = json.loads((root / "nki/plan.json").read_text())
        estimates = json.loads((root / "hardware.json").read_text())[
            "estimates"
        ]
        if estimates and all(d % 128 == 0 for x in inputs for d in x.shape):
            assert (
                sum(e["dma_commands"] for e in estimates)
                == plan["stats"]["isa_dma_panels"]
            )
            assert (
                sum(e["tensor_instructions"] for e in estimates)
                == plan["stats"]["tensor_instructions"]
            )
        assert len(plan["arguments"]) == len(
            inputs
        ), "Unexpected parameter arguments"
        np.savez(
            root / "reference.npz",
            **{f"a{i}": x.float().numpy() for i, x in enumerate(inputs)},
            expected=expected.float().numpy(),
        )
        second = root / "reconverted"
        convert(root, second, "trainium-v3")
        assert (second / "program.py").read_bytes() == (
            root / "nki/program.py"
        ).read_bytes()
        (root / "generation.json").write_text(
            json.dumps(
                dict(
                    case=name,
                    seed=29,
                    torch=torch.__version__,
                    diagnostic_tile=args.tile,
                    input_dtype=args.dtype,
                    tolerance=dict(atol=atol, rtol=rtol),
                    compiler=str(Path(vc.__file__).resolve()),
                    target="trainium-v3",
                    bufferized_correct=True,
                    reconversion_identical=True,
                    program_sha256=hashlib.sha256(
                        (root / "nki/program.py").read_bytes()
                    ).hexdigest(),
                ),
                indent=2,
            )
        )
        print("GENERATED", name, flush=True)


if __name__ == "__main__":
    main()
