"""New structural region semantics and rewrite contracts."""

import json
import pytest
import torch
import voyager_compiler as vc
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.codegen.transform.bufferize.block_regions import (
    discover,
    elide_contraction_padding,
)
from test_trainium_stream_regions import BMM, Chain, prepare
from voyager_compiler.trainium.lowering import (
    Step,
    RECIPES,
    recipe_for,
    expression,
)
from voyager_compiler.trainium.execution import TrainiumTuning


def compile_region(module, args, path, **overrides):
    graph, context = prepare(module, args)
    elide_contraction_padding(graph)
    regions = discover(graph)
    assert len(regions) == 1
    spec = dict(
        kind=regions[0].kind,
        rows=32,
        features=64,
        outputs=32,
        storage="sbuf",
        traversal="row",
        slots=1,
    )
    spec.update(overrides)
    graph.meta["block_region_choices"] = [spec]
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(block_regions=True),
    )
    torch.testing.assert_close(
        graph(*args), module(*args), atol=1e-3, rtol=1e-3
    )
    context.realize(path)
    return json.loads((path / "selection.json").read_text())


@pytest.mark.parametrize("storage", ["sbuf", "hbm", "recompute"])
@pytest.mark.parametrize("slots", [1, 2])
def test_chunked_softmax(tmp_path, storage, slots):
    torch.manual_seed(5)
    args = (torch.randn(2, 64, 32) * 0.1, torch.randn(1, 32, 256) * 0.1)
    compile_region(BMM(), args, tmp_path, storage=storage, slots=slots)


@pytest.mark.parametrize(
    "storage,traversal", [("sbuf", "row"), ("hbm", "row"), ("hbm", "feature")]
)
def test_gated_stage_boundary(tmp_path, storage, traversal):
    torch.manual_seed(8)
    args = tuple(
        torch.randn(*s) * 0.1
        for s in ((64, 32), (32, 256), (32, 256), (256, 128))
    )
    compile_region(
        Chain(), args, tmp_path, storage=storage, traversal=traversal
    )


def test_centered_default_recipe_is_identical():
    assert recipe_for("layer_norm", TrainiumTuning()) == RECIPES["layer_norm"]


@pytest.mark.parametrize(
    "algorithm", ["centered", "moments", "shifted_moments"]
)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("square_engine", ["vector", "scalar"])
def test_layernorm_recipe_semantics(algorithm, fused, square_engine):
    torch.manual_seed(7)
    x = torch.randn(11, 259)
    w = torch.randn(259)
    b = torch.randn(259)
    eps = 1e-5
    values = dict(input=x, weight=w, bias=b)
    for step in recipe_for(
        "layer_norm",
        TrainiumTuning(
            layernorm_algorithm=algorithm,
            layernorm_fused=fused,
            layernorm_square_engine=square_engine,
        ),
    ):
        a = [values[k] for k in step.inputs]
        kind = step.instruction
        if kind == "first":
            v = a[0][:, :1].clone()
        elif kind == "reduce":
            v = a[0].sum(-1, keepdim=True)
        elif kind == "binary":
            v = a[0] * a[1]
        elif kind in ("scalar", "parameter"):
            other = 1 / x.shape[-1] if step.scalar else a[1]
            v = {
                "multiply": torch.mul,
                "add": torch.add,
                "subtract": torch.sub,
            }[step.op](a[0], other)
        elif kind == "scale_epsilon":
            v = a[0] / x.shape[-1] + eps
        elif kind == "difference_epsilon":
            v = a[0] - a[1] + eps
        elif kind == "center_scale":
            v = (a[0] - a[1]) * a[2]
        elif kind == "activation":
            v = torch.square(a[0]) if step.op == "square" else torch.rsqrt(a[0])
        else:
            raise AssertionError(kind)
        values[step.name] = v
    torch.testing.assert_close(
        values["result"],
        torch.nn.functional.layer_norm(x, (259,), w, b, eps),
        atol=1e-5,
        rtol=1e-5,
    )


def test_activation_panel_pointwise_keeps_orientation():
    from voyager_compiler.trainium.planning import (
        InstructionPlanner,
        PanelValue,
    )
    from types import SimpleNamespace

    planner = object.__new__(InstructionPlanner)
    planner.builder = SimpleNamespace(program=SimpleNamespace(tensors={}))
    planner.tmp = lambda expression: expression
    panel = PanelValue(32, 128, [(0, 0, 32, 128, "payload")], True)
    result = planner.map_panels(lambda value: f"{value} * 1.0", panel)
    assert result.row_major
    assert result.panels == [(0, 0, 32, 128, "payload * 1.0")]


def test_gated_double_buffering_is_explicitly_rejected(tmp_path):
    args = tuple(
        torch.randn(*s) * 0.1
        for s in ((64, 32), (32, 256), (32, 256), (256, 128))
    )
    with pytest.raises(ValueError, match="one buffer slot"):
        compile_region(Chain(), args, tmp_path, slots=2)


def test_unsupported_nondivisor_is_rejected(tmp_path):
    args = (torch.randn(1, 64, 32), torch.randn(1, 32, 256))
    with pytest.raises(ValueError, match="exact row/feature divisors"):
        compile_region(BMM(), args, tmp_path, features=100)


def test_single_score_block_uses_whole_row_softmax(tmp_path):
    args = (torch.randn(2, 64, 32) * 0.1, torch.randn(1, 32, 256) * 0.1)
    compile_region(BMM(), args, tmp_path, features=256)
    model = (tmp_path / "model.txt").read_text()
    assert "aten::softmax" in model
    assert "aten::amax" not in model
