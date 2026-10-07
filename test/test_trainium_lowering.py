"""Early ISA expansion must preserve semantics and inform the shared planner."""
import torch
import voyager_compiler as vc
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.lowering import RECIPES, reduction_graph, reduction_workspace
from voyager_compiler.codegen.transform.tiling.execution import evaluate_graph


def test_rmsnorm_is_not_mean_centered():
    x = torch.full((2, 128), 3.0)
    w = torch.ones(128)
    actual = torch.ops.quantized_ops.rms_norm.default(x, [128], w, 1e-5)
    torch.testing.assert_close(actual, torch.nn.functional.rms_norm(x, (128,), w, 1e-5))
    assert actual.min() > .99


def test_early_silu_expansion_is_visible_before_bufferization():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)
    x = torch.randn(128, 128)
    graph = vc.export_model(Model(), (x,))
    vc.transform(graph, (x,), context=CompilerContext.resolve(neuron_core(3)))
    targets = [n.target for n in graph.graph.nodes]
    assert torch.ops.aten.silu.default not in targets
    assert torch.ops.aten.sigmoid.default in targets
    assert torch.ops.aten.mul.Tensor in targets
    torch.testing.assert_close(graph(x), Model()(x))


def test_early_pool_region_preserves_window_edges():
    from voyager_compiler.trainium.lowering import prepare_graph
    from voyager_compiler.shape_prop import ShapeProp, fake_like
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.max_pool2d(x, 3, 1)
    x = torch.randn(1, 1, 31, 37)
    graph = vc.export_model(Model(), (x,))
    ShapeProp(graph).propagate(fake_like(x))
    prepare_graph(graph, neuron_core(3))
    assert sum(n.target == torch.ops.aten.max_pool2d.default for n in graph.graph.nodes) == 1
    assert not any(n.target == torch.ops.aten.maximum.default for n in graph.graph.nodes)
    torch.testing.assert_close(graph(x), Model()(x))


def test_reduction_contracts_are_hardware_owned_and_have_dependencies():
    hw = neuron_core(3)
    for name, recipe in RECIPES.items():
        impl = hw.operation_implementation(f'nki.{name}.float32')
        impl.validate(hw)
        assert tuple(s.name for s in impl.steps) == tuple(s.name for s in recipe)
        graph = reduction_graph(hw, name, 128, 256)
        result = evaluate_graph(graph)
        assert result.duration_ns > 0
        assert result.unknown_latency  # No fabricated completion calibration.
        assert len(graph.nodes[-1].dependencies) > 0
        assert reduction_workspace(name, 8192, 2) > reduction_workspace(name, 256, 2)


def test_fp32_binary_service_reads_both_sbuf_operands():
    hardware=neuron_core(3)
    graph = reduction_graph(hardware, 'rms_norm', 128, 256)
    occupancy,_=hardware.timing_profile.operation('nki.binary.float32.VectorE').evaluate(512/.96)
    assert graph.nodes[0].occupancy_ns == occupancy


def test_row_reduction_keeps_ragged_feature_extent():
    class Model(torch.nn.Module):
        def forward(self, x, w, b):
            return torch.nn.functional.layer_norm(x,(257,),w,b)
    inputs=(torch.randn(129,257),torch.randn(257),torch.randn(257))
    graph=vc.export_model(Model(),inputs)
    vc.transform(graph,inputs,context=CompilerContext.resolve(neuron_core(3)))
    assert not any(n.target == torch.ops.aten.pad.default for n in graph.graph.nodes)
    torch.testing.assert_close(graph(*inputs),Model()(*inputs))
