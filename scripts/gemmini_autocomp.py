"""Compile AutoComp kernels through the same transform/compile flow as Voyager."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from voyager_compiler import compile, export_model, transform
from voyager_compiler.gemmini.collateral import convert
from voyager_compiler.gemmini.hardware import lean_config
from voyager_compiler.gemmini.mapping import GemminiMappingPolicy, GemminiTuning
from voyager_compiler.gemmini.scheduling import SubmissionPolicy
from voyager_compiler.quantization.fake_quantize import get_quantization_map
from voyager_compiler.targets import get_backend


class Kernel(torch.nn.Module):
    def __init__(self, weight, bias=None, conv=False):
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float() if bias is not None else None)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("qmap", get_quantization_map("int8"))
        self.conv = conv

    def forward(self, x):
        y = (
            torch.nn.functional.conv2d(x, self.weight, self.bias)
            if self.conv
            else x @ self.weight
        )
        return torch.ops.quantized_ops.quantize(
            y, self.scale, qmap=self.qmap, rounding="nearest_even_int8"
        )


def generate(
    gemmini,
    output,
    names,
    *,
    interstellar_cost_tradeoff=False,
    runtime_tolerance=None,
    tuning=None,
):
    torch.set_num_threads(4)
    for name in names:
        source = gemmini / "examples/autocomp" / name
        meta = json.loads((source / "benchmark.json").read_text())

        def tensor(n, shape, dtype="i1"):
            return torch.from_numpy(
                np.fromfile(source / (n + ".bin"), dtype=dtype)
                .reshape(shape)
                .copy()
            )

        conv = meta["kind"] != "gemm"
        if conv:
            h, c = meta["dimensions"]
            a = (
                tensor("inp", (4, h + 2, h + 2, c))
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
            )
            b = tensor("weights", (3, 3, c, c)).permute(3, 2, 0, 1).contiguous()
            d = tensor("bias", (c,), "<i4")
        else:
            m, k, n = meta["dimensions"]
            a = tensor("A", (m, k)).float()
            b = tensor("B", (k, n))
            d = None
        module = Kernel(b, d, conv).eval()
        expected = module(a).detach().clone()
        graph = export_model(module, (a,))
        for node in graph.graph.nodes:
            if (
                node.op == "placeholder"
                or (node.op == "get_attr" and node.target == "weight")
                or node.target is torch.ops.quantized_ops.quantize.default
            ):
                node.meta["dtype"] = "int8"
            elif node.op == "get_attr" and node.target == "bias":
                node.meta["dtype"] = "int32"
        config = lean_config()
        policy = GemminiMappingPolicy(config, tuning)
        transform(
            graph,
            (a,),
            config=config,
            patterns=get_backend("gemmini").fusion_patterns(config),
            layout_policy="systolic",
        )
        root = output / name
        compile(
            graph,
            (a,),
            config=config,
            output_dir=root,
            interstellar_cost_tradeoff=interstellar_cost_tradeoff,
            runtime_tolerance=runtime_tolerance,
            mapping_policy=policy,
        )
        (root / "search-policy.json").write_text(
            json.dumps(
                dict(
                    interstellar_cost_tradeoff=False,
                    requested_interstellar_cost_tradeoff=interstellar_cost_tradeoff,
                    runtime_tolerance=(
                        0.02 if runtime_tolerance is None else runtime_tolerance
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
        # Reference checks compare the executable bufferized graph with the
        # original PyTorch operator, not a second hardware schedule.
        torch.testing.assert_close(graph(a), expected, rtol=0, atol=0)
        print(name, "BUFFERIZED_REFERENCE_PASS", flush=True)
        print(convert(root), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gemmini",
        type=Path,
        default=Path("/home/zhouhua/Research/ML/Gemmini"),
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/gemmini/autocomp")
    )
    p.add_argument(
        "--only",
        nargs="+",
        default=[f"gemm{i}" for i in range(6)] + [f"conv{i}" for i in range(3)],
    )
    p.add_argument(
        "--interstellar-cost-tradeoff",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--runtime-tolerance", type=float, default=None)
    p.add_argument(
        "--submission-quanta",
        type=int,
        nargs=3,
        metavar=("EX", "LD", "ST"),
        default=(32, 4, 4),
    )
    p.add_argument(
        "--separate-accumulator-banks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--pointwise-wide-working-sets", type=int, default=2)
    a = p.parse_args()
    generate(
        a.gemmini,
        a.output,
        a.only,
        interstellar_cost_tradeoff=a.interstellar_cost_tradeoff,
        runtime_tolerance=a.runtime_tolerance,
        tuning=GemminiTuning(
            separate_accumulator_banks=a.separate_accumulator_banks,
            pointwise_wide_working_sets=a.pointwise_wide_working_sets,
            submission=SubmissionPolicy(*a.submission_quanta),
        ),
    )
