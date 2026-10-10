"""Whole-reduction regions: semantic correctness, residency, and honest fallback."""

import json
import pytest
import torch
import torch.nn.functional as F
import voyager_compiler as vc
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.codegen.transform.bufferize.stream_regions import (
    analyze_stream_region,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy
from voyager_compiler.trainium.execution import TrainiumTuning


class Chain(torch.nn.Module):
    def forward(self, x, w, u, v):
        return (F.silu(x @ w) * (x @ u)) @ v


class BMM(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, w):
        return torch.softmax(x @ w, self.dim)


def prepare(module, args):
    hw = neuron_core(3)
    context = CompilerContext.resolve(
        hw,
        TrainiumMappingPolicy(
            hw,
            TrainiumTuning(
                matmul_operands="reuse", matmul_weight_layout="generic"
            ),
        ),
    )
    graph = vc.export_model(module, args)
    vc.transform(graph, args, context=context)
    return graph, context


def compile_check(module, args, tmp_path, force_stage=False):
    graph, context = prepare(module, args)
    if force_stage:
        evaluate = context.policy.stream_region_candidate
        context.policy.stream_region_candidate = lambda r, m, s: (
            evaluate(r, m, s)
            if s == "stage"
            else dict(legal=False, reason="test-only stage probe")
        )
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(stream_regions=True, stream_region_search="compact"),
    )
    torch.testing.assert_close(
        graph(*args), module(*args), rtol=1e-3, atol=1e-3
    )
    hw = json.loads((tmp_path / "hardware.json").read_text())
    record = hw["stream_regions"][0]
    assert record["status"] == "selected"
    program = json.loads((tmp_path / "instructions.json").read_text())
    assert len(program["outputs"]) == 1
    return record


@pytest.mark.parametrize("stage", [False, True])
def test_multigemms_local_and_stage_storage_reuse(tmp_path, stage):
    torch.manual_seed(4)
    args = tuple(
        torch.randn(*s) * 0.1
        for s in ((256, 128), (128, 256), (128, 256), (256, 128))
    )
    record = compile_check(Chain(), args, tmp_path, stage)
    assert record["selected"]["weight_residency"] == (
        "stage" if stage else "resident"
    )
    boundary = sum(x.numel() * 4 for x in args) + 256 * 128 * 4
    actual = record["selected"]["hbm_bytes"]
    if stage:
        assert actual > boundary
    else:
        assert actual == boundary


@pytest.mark.parametrize("dim", [-1, 2])
@pytest.mark.parametrize("broadcast", [False, True])
def test_batch_and_positive_softmax_axes(tmp_path, dim, broadcast):
    args = (
        torch.randn(2, 256, 64),
        torch.randn(1 if broadcast else 2, 64, 256),
    )
    record = compile_check(BMM(dim), args, tmp_path)
    assert record["batch"] == [2]
    assert (
        record["selected"]["hbm_bytes"]
        == sum(x.numel() * 4 for x in args) + 2 * 256 * 256 * 4
    )


@pytest.mark.parametrize("dim", [0, 1])
def test_reduction_over_streamed_axis_rejected(dim):
    args = (torch.randn(2, 128, 64), torch.randn(2, 64, 128))
    graph = vc.export_model(BMM(dim), args)
    from voyager_compiler.shape_prop import ShapeProp

    ShapeProp(graph).propagate(*args)
    with pytest.raises(ValueError, match="complete feature axis"):
        analyze_stream_region(
            [n for n in graph.graph.nodes if n.op == "call_function"]
        )


def test_external_consumer_rejected():
    graph, _ = prepare(
        BMM(), (torch.randn(2, 128, 64), torch.randn(2, 64, 128))
    )
    nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
    next(n for n in graph.graph.nodes if n.op == "output").args = (
        (nodes[0], nodes[-1]),
    )
    with pytest.raises(ValueError, match="outside the region"):
        analyze_stream_region(nodes)


def test_complete_feature_capacity_rejected():
    graph, context = prepare(
        BMM(), (torch.randn(1, 16, 64), torch.randn(1, 64, 65536))
    )
    from voyager_compiler.codegen.transform.bufferize.stream_regions import (
        elide_contraction_padding,
    )

    elide_contraction_padding(graph)
    region = analyze_stream_region(
        [n for n in graph.graph.nodes if n.op == "call_function"]
    )
    c = context.policy.stream_region_candidate(region, 16, "resident")
    assert not c["legal"] and "SBUF" in c["reason"]


def test_flags_are_opt_in():
    assert not BufferizationOptions().stream_regions
    with pytest.raises(ValueError):
        BufferizationOptions(stream_regions=True, row_regions=True)
    with pytest.raises(ValueError):
        BufferizationOptions(stream_regions=True, flow="resident")


def test_partial_batch_broadcast_reload_accounting(tmp_path):
    args = (torch.randn(2, 3, 128, 128), torch.randn(1, 3, 128, 128))
    record = compile_check(BMM(), args, tmp_path)
    transfers = record["selected"]["transfers"]
    assert [t["count"] for t in transfers] == [6, 6, 6]


def test_failed_region_restores_preparation_padding():
    from voyager_compiler.codegen.transform.bufferize.stream_regions import (
        plan_stream_regions,
    )
    from types import SimpleNamespace

    graph, context = prepare(
        BMM(), (torch.randn(2, 128, 64), torch.randn(2, 64, 128))
    )
    before = str(graph.graph)
    context.policy.stream_region_candidate = lambda *a: dict(
        legal=False, reason="test capacity rejection"
    )
    plan_stream_regions(graph, SimpleNamespace(mapping_policy=context.policy))
    assert str(graph.graph) == before
    assert graph.meta["stream_regions"][0]["status"] == "fallback"


@pytest.mark.parametrize("gemm_first", [False, True])
def test_normalization_uses_the_same_region_contract(tmp_path, gemm_first):
    class NormChain(torch.nn.Module):
        def forward(self, x, w, r, g):
            if gemm_first:
                return F.rms_norm(x @ w + r, (256,), g, 1e-5)
            return F.rms_norm(x + r, (128,), g, 1e-5) @ w

    args = (
        torch.randn(256, 128),
        torch.randn(128, 256),
        torch.randn(256, 256 if gemm_first else 128),
        torch.randn(256 if gemm_first else 128),
    )
    compile_check(NormChain(), args, tmp_path)


def test_internal_softmax_variant(tmp_path):
    class InternalSoftmax(torch.nn.Module):
        def forward(self, x, w):
            return torch.ops.aten._softmax.default(x @ w, 2, False)

    compile_check(
        InternalSoftmax(),
        (torch.randn(2, 128, 128), torch.randn(2, 128, 256)),
        tmp_path,
    )


def test_explicit_unsupported_layout_is_not_silently_changed():
    from dataclasses import replace
    from voyager_compiler.trainium.stream_regions import candidate

    args = (torch.randn(2, 128, 128), torch.randn(2, 128, 256))
    graph, context = prepare(BMM(), args)
    region = analyze_stream_region(
        [n for n in graph.graph.nodes if n.op == "call_function"]
    )
    result = candidate(
        context.policy.config,
        replace(context.policy.tuning, matmul_weight_layout="k_partitioned"),
        region,
        128,
        "resident",
    )
    assert not result["legal"] and result["constraint_kind"] == "lowering"
