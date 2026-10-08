"""Row-axis composition, cross-operation local storage, and safe rejection."""

import itertools
import json

import pytest
import torch
import torch.nn.functional as F

import voyager_compiler as vc
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.codegen.transform.bufferize.row_regions import (
    analyze_row_region,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy
from voyager_compiler.trainium.execution import TrainiumTuning


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
            elif op == "norm":
                x = F.rms_norm(x, (x.shape[-1],), gamma, 1e-5)
        return x


def case(order, m=192, k=128, n=256):
    add_width = n if order.index("gemm") < order.index("add") else k
    norm_width = n if order.index("gemm") < order.index("norm") else k
    torch.manual_seed(72)
    args = (
        torch.randn(m, k),
        torch.randn(k, n),
        torch.randn(m, add_width),
        torch.randn(norm_width),
    )
    model = Chain(order)
    graph = vc.export_model(model, args)
    hw = neuron_core(3)
    context = CompilerContext.resolve(
        hw, TrainiumMappingPolicy(hw, TrainiumTuning(matmul_operands="reuse"))
    )
    vc.transform(graph, args, context=context)
    return graph, args, model(*args), context


@pytest.mark.parametrize(
    "order", list(itertools.permutations(("gemm", "add", "norm")))
)
def test_every_order_keeps_internal_edges_local(order, tmp_path):
    graph, args, expected, context = case(order)
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(row_regions=True),
    )
    torch.testing.assert_close(graph(*args), expected, rtol=1e-3, atol=1e-3)
    record = json.loads((tmp_path / "hardware.json").read_text())[
        "row_regions"
    ][0]
    assert record["status"] == "selected"
    assert record["selected"]["tile_rows"] <= 128
    assert record["rows"] // record["selected"]["tile_rows"] >= 2
    instructions = json.loads((tmp_path / "instructions.json").read_text())
    assert len(instructions["outputs"]) == 1  # no HBM intermediate workspace
    selected = json.loads((tmp_path / "selection.json").read_text())[
        "program_analysis"
    ]["selected_instruction_analysis"]
    expected_boundary = (
        sum(x.numel() * x.element_size() for x in args)
        + expected.numel() * expected.element_size()
    )
    # Explicit transpose identity is a genuine one-time HBM constant.
    assert selected["hbm_bytes"] == expected_boundary + 65536


def test_operand_roles_not_coincident_extent():
    graph, _, _, _ = case(("add", "norm", "gemm"), m=128, k=128, n=128)
    nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
    region = analyze_row_region(nodes)
    matrix = nodes[-1]
    assert matrix.args[1] not in region.varying
    assert len(region.varying) == 2


def test_reduction_across_rows_is_rejected():
    class Model(torch.nn.Module):
        def forward(self, x, w):
            return F.rms_norm(x, x.shape, eps=1e-5) @ w

    args = (torch.randn(32, 64), torch.randn(64, 128))
    graph = vc.export_model(Model(), args)
    from voyager_compiler.shape_prop import ShapeProp

    ShapeProp(graph).propagate(*args)
    nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
    with pytest.raises(ValueError, match="streaming row axis"):
        analyze_row_region(nodes)


def test_external_consumer_rejects_region():
    graph, _, _, _ = case(("add", "norm", "gemm"))
    nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
    output = next(n for n in graph.graph.nodes if n.op == "output")
    output.args = ((nodes[-1], nodes[0]),)
    with pytest.raises(ValueError, match="outside the region"):
        analyze_row_region(nodes)


def test_invariant_weight_capacity_is_charged():
    graph, _, _, context = case(("add", "norm", "gemm"), m=128, k=4096, n=4096)
    region = analyze_row_region(
        [n for n in graph.graph.nodes if n.op == "call_function"]
    )
    result = context.policy.row_region_candidate(region, 1)
    assert not result["legal"]
    assert "SBUF" in result["reason"]


def test_pointwise_rule_is_not_a_residual_pattern(tmp_path):
    graph, args, _, context = case(("add", "norm", "gemm"))
    multiply = next(
        n for n in graph.graph.nodes if n.target == torch.ops.aten.add.Tensor
    )
    multiply.target = torch.ops.aten.mul.Tensor
    graph.recompile()
    expected = F.rms_norm(args[0] * args[2], (128,), args[3], 1e-5) @ args[1]
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(row_regions=True),
    )
    torch.testing.assert_close(graph(*args), expected, atol=1e-3, rtol=1e-3)
    selected = json.loads((tmp_path / "hardware.json").read_text())[
        "row_regions"
    ][0]
    assert selected["status"] == "selected"
    assert "aten.mul.Tensor" in selected["operations"]
