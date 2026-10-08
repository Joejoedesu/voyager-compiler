"""Generate held-out operation orderings through the shared row-region pass."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import voyager_compiler as vc
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy


class Chain(torch.nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x, w, residual, gamma):
        for op in self.order:
            if op == "gemm":
                x = x @ w
            elif op == "add":
                x = x + residual
            else:
                x = F.rms_norm(x, (x.shape[-1],), gamma, 1e-5)
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matmul-orientation", choices=("auto", "weights", "activations"), default="auto")
    parser.add_argument("--matmul-weight-layout", choices=("auto", "generic", "k_partitioned"), default="auto", help="Whole invariant row-region weight layout; auto searches both")
    parser.add_argument("--shape", type=int, nargs=3, default=(192,256,128), metavar=("M", "N", "K"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    m, n, k = args.shape
    for order in itertools.permutations(("gemm", "add", "norm")):
        torch.manual_seed(72)
        root = args.output.resolve() / "_".join(order)
        root.mkdir(parents=True, exist_ok=True)
        add_width = n if order.index("gemm") < order.index("add") else k
        norm_width = n if order.index("gemm") < order.index("norm") else k
        x = (
            torch.randn(m, k),
            torch.randn(k, n),
            torch.randn(m, add_width),
            torch.randn(norm_width),
        )
        model = Chain(order)
        expected = model(*x)
        hw = neuron_core(3)
        context = CompilerContext.resolve(
            hw,
            TrainiumMappingPolicy(hw, TrainiumTuning(matmul_operands="reuse", matmul_orientation=args.matmul_orientation, matmul_weight_layout=args.matmul_weight_layout)),
        )
        graph = vc.export_model(model, x)
        vc.transform(graph, x, context=context)
        vc.compile(
            graph,
            x,
            context=context,
            output_dir=root,
            dump_tensors=False,
            bufferization_options=BufferizationOptions(row_regions=True),
        )
        torch.testing.assert_close(graph(*x), expected, atol=1e-3, rtol=1e-3)
        context.realize(root)
        np.savez(
            root / "reference.npz",
            **{f"a{i}": v.numpy() for i, v in enumerate(x)},
            expected=expected.numpy(),
        )
        (root / "generation.json").write_text(
            json.dumps(
                dict(
                    input_dtype="float32",
                    shape=[m, n, k],
                    order=order,
                    row_regions=True,
                    held_out=True,
                    bufferized_correct=True,
                    tolerance=dict(atol=1e-3, rtol=1e-3),
                ),
                indent=2,
            )
        )
        print("GENERATED", root.name, flush=True)


if __name__ == "__main__":
    main()
