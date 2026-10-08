"""Trainium execution contracts, capacity rejection and collateral restoration."""

from dataclasses import replace
from types import SimpleNamespace
import json
import math
import pytest
import torch
from interstellar import MappingPoint, loop_enum as le
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.codegen.transform.tiling.contracts import resources_fit
from voyager_compiler.codegen.transform.tiling.traversal import MappingTraversal
from voyager_compiler.trainium.cost import TrainiumCostModel, dma_service
from voyager_compiler.trainium.execution import (
    EngineEvent,
    TrainiumTuning,
    dma_panels,
    matmul_panels,
    matrix_storage,
    schedule_events,
    slot_bytes,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy, make_plan


def point(m=512, n=128, k=512, outer_m=1, outer_n=2, outer_k=1):
    blocks = [[1] * 4 for _ in range(le.NUM)]
    parts = [[1] * 4 for _ in range(le.NUM)]
    orders = [[6] * 4 for _ in range(le.NUM)]
    for d, v, outer in (
        (le.OX, m, outer_m),
        (le.OC, n, outer_n),
        (le.IC, k, outer_k),
    ):
        blocks[d][2] = v
        blocks[d][3] = outer
    for rank, d in enumerate((le.IC, le.OC, le.OX)):
        orders[d][3] = rank
    return MappingPoint(orders, blocks, parts)


def estimator(tuning=None):
    traversal = MappingTraversal()
    for name, value in dict(
        input_dtype_width=32,
        weight_dtype_width=32,
        output_dtype_width=32,
        weight_hbm_ck=True,
        weight_transposed=False,
        has_tail=False,
        bias_width=0,
    ).items():
        setattr(traversal, name, value)
    return TrainiumCostModel(traversal, neuron_core(3), tuning=tuning)


def test_dma_service_separates_completion_from_engine_occupancy():
    config = neuron_core(3)
    one = dma_service(config, 128, 128, 32)
    four = dma_service(config, 128, 512, 32)
    assert four.commands == 4
    assert four.payload_ns == pytest.approx(4 * one.payload_ns)
    assert four.startup_ns == one.startup_ns == 1300
    assert four.dma_ns == four.payload_ns
    # Eight consecutive partitions per engine, not round-robin assignment.
    narrow = dma_service(config, 8, 128, 32)
    assert narrow.payload_ns == one.payload_ns


def test_dma_coalescing_and_optional_transpose_have_distinct_geometry():
    staged = list(
        dma_panels(
            512, 128, 512, transpose=True, store=False, tuning=TrainiumTuning()
        )
    )
    direct = list(
        dma_panels(
            512,
            128,
            512,
            transpose=True,
            store=False,
            tuning=TrainiumTuning(dma_transpose=True),
        )
    )
    assert len(staged) == 4 and len(direct) == 1
    assert all(p[-1] for p in direct)
    coalesced = list(
        dma_panels(
            128,
            1024,
            128,
            transpose=False,
            store=False,
            tuning=TrainiumTuning(),
        )
    )
    assert len(coalesced) == 1
    assert sum(p[2] * p[3] for p in staged) == sum(p[2] * p[3] for p in direct)


def test_completion_delay_allows_independent_engine_issue():
    duration, services = schedule_events(
        [
            EngineEvent("DMA", 10, completion_delay_ns=100),
            EngineEvent("DMA", 10),
            EngineEvent("TensorE", 20, (0,)),
        ]
    )
    assert duration == 130
    assert services == {"DMA": 20, "TensorE": 20}
    with pytest.raises(ValueError):
        schedule_events([EngineEvent("DMA", 1, (0,))])


def test_large_software_tile_has_legal_instruction_panels():
    panels = list(matmul_panels(1024, 1024, 256))
    assert (
        sum(mm * nn * sum(kk for _, kk in ks) for _, _, mm, nn, ks in panels)
        == 1024 * 1024 * 256
    )
    assert all(
        mm <= 512 and nn <= 128 and all(kk <= 128 for _, kk in ks)
        for _, _, mm, nn, ks in panels
    )
    # This exceeded the retired M*ceil(N/128)<=4096 software restriction.
    rc = estimator()
    assert math.isfinite(
        rc.calculate_runtime(
            None,
            SimpleNamespace(hstd=1, wstd=1),
            point(1024, 1024, 256, outer_n=1),
        )
    )


def test_sbuf_and_psum_are_both_checked():
    config = neuron_core(3)
    plan = make_plan(point())
    requirements = matrix_storage(
        config,
        512 * 512,
        512 * 128,
        512 * 128,
        512,
        128,
        32,
        32,
        32,
        plan,
        TrainiumTuning(),
    )
    assert resources_fit(config, requirements)
    assert (
        sum(r.allocated_bytes for r in requirements if r.memory == "PSUM")
        == 3 * (2 << 20) // 8
    )
    for name in ("SBUF", "PSUM"):
        small = replace(
            config,
            memory=replace(
                config.memory,
                levels=tuple(
                    replace(
                        level,
                        instances=tuple(
                            (
                                replace(mem, size=replace(mem.size, value=1024))
                                if mem.name == name
                                else mem
                            )
                            for mem in level.instances
                        ),
                    )
                    for level in config.memory.levels
                ),
            ),
        )
        assert not resources_fit(small, requirements)
    assert slot_bytes(1, 1, 16) == 2048


def test_buffer_depth_is_a_capacity_constrained_candidate():
    mapping = point()
    assert make_plan(mapping, max_depth=1).output_slots == 1
    assert make_plan(mapping, max_depth=2).output_slots == 2
    layer = SimpleNamespace(hstd=1, wstd=1)
    rc = estimator()
    selected = rc.evaluate(None, layer, mapping)
    single = estimator(TrainiumTuning(max_buffer_depth=1)).evaluate(
        None, layer, mapping
    )
    assert selected.cycles <= single.cycles
    saved = dict(selected.diagnostics)["estimate"].copy()
    rc.calculate_runtime(None, layer, point(128, 128, 128))
    assert dict(selected.diagnostics)["estimate"] == saved


def test_bias_storage_is_not_lost_after_early_pruning():
    layer = SimpleNamespace(hstd=1, wstd=1)
    rc = estimator()
    plain = rc.evaluate(None, layer, point())
    rc.shared.bias_width = 32
    bias = rc.evaluate(None, layer, point())
    assert sum(r.allocated_bytes for r in bias.storage) > sum(
        r.allocated_bytes for r in plain.storage
    )


def test_context_is_speed_only_and_restores_instruction_policy(tmp_path):
    config = neuron_core(3)
    context = CompilerContext.resolve(
        config,
        TrainiumMappingPolicy(
            config, TrainiumTuning(dma_transpose=True, max_buffer_depth=1)
        ),
        runtime_tolerance=100,
    )
    assert not context.cost_tradeoff
    context.write(tmp_path)
    restored = CompilerContext.from_artifacts(tmp_path, config)
    assert restored.record() == context.record()


@pytest.mark.parametrize(
    "isa_lowering,strict_realization,temporary_buffer_depth",
    [(False, True, 1), (True, True, 1), (True, False, 1), (True, True, 4)],
)
def test_compile_reconvert_and_instruction_counts(
    tmp_path, isa_lowering, strict_realization, temporary_buffer_depth
):
    import voyager_compiler as vc
    from voyager_compiler.trainium.converter import convert

    class Matmul(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    torch.manual_seed(5)
    inputs = (torch.randn(128, 128), torch.randn(128, 128))
    graph = vc.export_model(Matmul(), inputs)
    config = neuron_core(3)
    context = CompilerContext.resolve(
        config,
        TrainiumMappingPolicy(
            config, TrainiumTuning(
                isa_lowering=isa_lowering,
                strict_realization=strict_realization,
                temporary_buffer_depth=temporary_buffer_depth,
            )
        ),
    )
    vc.transform(graph, inputs, context=context)
    vc.compile(graph, inputs, context=context, output_dir=tmp_path)
    torch.testing.assert_close(graph(*inputs), inputs[0] @ inputs[1])
    plan = convert(tmp_path)
    estimate = json.loads((tmp_path / "hardware.json").read_text())[
        "estimates"
    ][0]
    assert plan["stats"]["isa_dma_panels"] + int(not strict_realization) == estimate["dma_commands"]
    assert (
        plan["stats"]["tensor_instructions"] == estimate["tensor_instructions"]
    )
    if isa_lowering:
        source = (tmp_path / "nki/program.py").read_text()
        assert plan["strict_realization"] == strict_realization
        assert plan["temporary_buffering"]["depth"] == temporary_buffer_depth
        restored = CompilerContext.from_artifacts(tmp_path, config)
        assert restored.policy.tuning.temporary_buffer_depth == temporary_buffer_depth
        assert plan["program_analysis"]["physical_addresses_enforced"] == strict_realization
        assert ("ncc.sbuf.alloc" in source) == strict_realization
        assert plan["high_level_operations"] == {}
        assert "nisa.dma_copy" in source
        assert ("is_transpose=True" if strict_realization else "nisa.nc_transpose") in source
        assert plan["stats"]["panel_result_bindings"] == 1
        assert estimate["shared_constant_bytes"] == (65536 if strict_realization else 16384)
        for key, count in estimate["expanded_isa"].items():
            assert plan["expanded_isa"][key] == count
    convert(tmp_path, tmp_path / "second")
    assert (tmp_path / "second/program.py").read_bytes() == (
        tmp_path / "nki/program.py"
    ).read_bytes()
    with pytest.raises(ValueError):
        convert(tmp_path, tmp_path / "wrong", "trainium-v2")


@pytest.mark.parametrize("outer_k", [1, 2])
def test_single_slot_async_retirement_preserves_outputs(tmp_path, outer_k):
    """Changing one-slot operands must retire before the next submit/reload."""
    import voyager_compiler as vc
    from voyager_compiler.codegen.transform.tiling.tiler import TileConstraint
    from voyager_compiler.trainium.converter import convert

    class Matmul(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    torch.manual_seed(9)
    inputs = (torch.randn(128, 128 * outer_k), torch.randn(128 * outer_k, 256))
    config = neuron_core(3)
    policy = TrainiumMappingPolicy(config, TrainiumTuning(max_buffer_depth=1))
    prepare = policy.prepare_matrix
    constraint = TileConstraint(
        exact=((le.OX, 128), (le.OC, 128), (le.IC, 128))
    )
    policy.prepare_matrix = lambda problem, tiler: prepare(
        replace(problem, constraint=constraint), tiler
    )
    context = CompilerContext.resolve(config, policy)
    graph = vc.export_model(Matmul(), inputs)
    vc.transform(graph, inputs, context=context)
    vc.compile(graph, inputs, context=context, output_dir=tmp_path)
    torch.testing.assert_close(
        graph(*inputs), inputs[0] @ inputs[1], atol=5e-4, rtol=5e-4
    )
    # Converter independently checks all semaphore credits while specializing.
    convert(tmp_path)


def test_nonmatrix_speed_only_ignores_traffic_and_tolerance(monkeypatch):
    from voyager_compiler.codegen.transform.tiling import search
    from voyager_compiler.codegen.transform.tiling.contracts import (
        NonMatrixFootprint,
    )

    node = SimpleNamespace(target="add", op="call_function")
    monkeypatch.setattr(search, "get_anchor_node", lambda _: node)
    monkeypatch.setattr(
        search,
        "get_valid_tiling",
        lambda *a, **kw: [((128,), (1,)), ((64,), (2,))],
    )
    policy = TrainiumMappingPolicy(neuron_core(3))
    policy.nonmatrix_footprint = lambda *a: NonMatrixFootprint(1024, [])
    policy.nonmatrix_cost = lambda kind, node, default: default
    cost = lambda node, tile, shapes, tiling: (
        (100, 1e30) if tile == (128,) else (101, -1e30)
    )
    args = (node, (128,), lambda *a: {}, policy.config)
    assert search._search_tiling(
        *args, cost_fn=cost, tolerance=100, policy=policy
    )[0] == (128,)
    policy.speed_only = False
    assert search._search_tiling(
        *args, cost_fn=cost, tolerance=0.02, policy=policy
    )[0] == (64,)


def test_isa_fp32_expansion_does_not_double_count_pipeline_work():
    from voyager_compiler.trainium import isa

    fp32 = isa.matmul(128, 128, 128, 32)
    bf16 = isa.matmul(128, 128, 128, 16)
    assert dict(fp32.instructions) == {"LDWEIGHTS": 2, "MATMUL_REGULAR": 2}
    assert fp32.tensor_cycles == 4 * bf16.tensor_cycles
    with pytest.raises(ValueError):
        TrainiumTuning(isa_lowering=True, explicit_isa=False)


@pytest.mark.parametrize("operand_policy", ["staged", "direct", "reuse"])
@pytest.mark.parametrize("strict_realization", [True, False])
def test_isa_large_panel_result_matches_search_expansion(
    tmp_path, operand_policy, strict_realization
):
    import voyager_compiler as vc
    from voyager_compiler.codegen.transform.tiling.tiler import TileConstraint
    from voyager_compiler.trainium.converter import convert

    class Matmul(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    torch.manual_seed(17)
    inputs = (torch.randn(1024, 256), torch.randn(256, 1024))
    config = neuron_core(3)
    policy = TrainiumMappingPolicy(
        config,
        TrainiumTuning(isa_lowering=True, matmul_operands=operand_policy, strict_realization=strict_realization),
    )
    prepare = policy.prepare_matrix
    constraint = TileConstraint(
        exact=((le.OX, 1024), (le.OC, 1024), (le.IC, 256))
    )
    policy.prepare_matrix = lambda problem, tiler: prepare(
        replace(problem, constraint=constraint), tiler
    )
    context = CompilerContext.resolve(config, policy)
    graph = vc.export_model(Matmul(), inputs)
    vc.transform(graph, inputs, context=context)
    vc.compile(graph, inputs, context=context, output_dir=tmp_path)
    plan = convert(tmp_path)
    source = (tmp_path / "nki/program.py").read_text()
    estimate = json.loads((tmp_path / "hardware.json").read_text())[
        "estimates"
    ][0]
    assert plan["stats"]["tensor_instructions"] == 32
    assert plan["expanded_isa"]["MATMUL_REGULAR"] == 64
    assert plan["stats"].get("reused_weight_panels", 0) == (
        16 if operand_policy == "reuse" else 0
    )
    restored = CompilerContext.from_artifacts(tmp_path, config)
    assert restored.policy.tuning.matmul_operands == operand_policy
    assert restored.policy.tuning.strict_realization == strict_realization
    torch.testing.assert_close(
        graph(*inputs), inputs[0] @ inputs[1], atol=5e-4, rtol=5e-4
    )
    assert (
        plan["expanded_isa"]["MATMUL_TRANSPOSE"]
        == estimate["expanded_isa"]["MATMUL_TRANSPOSE"]
    )
    assert plan["high_level_operations"] == {}
    # Original whole-result allocations may still be declared by the shared
    # buffer plan, but the generated body must not assemble them.
    selected = json.loads((tmp_path / "instructions.json").read_text())
    assert all(
        selected["tensors"][name]["shape"] != [128, 8192]
        for name in selected["placements"]
        if name.startswith("t")
    )


def test_older_trainium_policy_records_remain_restorable(tmp_path):
    from voyager_compiler.trainium.backend import TrainiumBackend
    from dataclasses import asdict

    options = asdict(TrainiumTuning())
    del options["isa_lowering"]
    del options["copy_policy"]
    policy = TrainiumBackend().restore_mapping_policy(neuron_core(3), options)
    assert policy.options() == options
    assert not policy.tuning.isa_lowering
    from voyager_compiler.trainium.converter import convert

    context = CompilerContext.resolve(neuron_core(3), policy)
    with pytest.raises(
        ValueError, match="requires explicit isa_lowering=False"
    ):
        convert(tmp_path, context=context)


def test_copy_policy_has_matching_engine_resources():
    hw = neuron_core(3)
    balanced = TrainiumTuning(isa_lowering=True, copy_policy="balanced")
    scalar = replace(balanced, copy_policy="scalar")
    load = dma_service(hw, 128, 128, 32, tuning=balanced)
    store = dma_service(hw, 128, 128, 32, tuning=balanced, store=True)
    scalar_store = dma_service(hw, 128, 128, 32, tuning=scalar, store=True)
    assert load.scalar_ns > 0 and load.vector_ns == 0
    assert store.vector_ns > 0 and store.scalar_ns == 0
    assert scalar_store.scalar_ns > 0 and scalar_store.vector_ns == 0


def test_candidate_cache_preserves_diagnostics_and_exact_evaluation():
    model = estimator(TrainiumTuning(max_buffer_depth=1))
    layer = SimpleNamespace(hstd=1, wstd=1)
    mapping = point(m=128, n=128, k=128, outer_m=4, outer_n=2)
    first = model.calculate_runtime(None, layer, mapping)
    expected = json.loads(json.dumps(model.estimate))
    assert model.calculate_runtime(None, layer, mapping) == first
    assert model._runtime_cache_hits == 1
    assert json.loads(json.dumps(model.estimate)) == expected
    model.evaluate(None, layer, mapping)
    assert (
        model._runtime_cache_hits == 1
    )  # Final exact replay is a separate key.
    model.shared.has_tail = True
    modified = model.calculate_runtime(None, layer, mapping)
    fresh = estimator(TrainiumTuning(max_buffer_depth=1))
    fresh.shared.has_tail = True
    assert modified == fresh.calculate_runtime(None, layer, mapping)
    assert model.estimate == fresh.estimate


@pytest.mark.parametrize("transposed", [False, True])
def test_reused_weight_completion_and_capacity_are_explicit(transposed):
    from voyager_compiler.trainium.dependencies import compute_graph

    hw = neuron_core(3)
    direct, _ = compute_graph(
        hw,
        1024,
        256,
        256,
        32,
        transposed,
        TrainiumTuning(matmul_operands="direct"),
    )
    reused, _ = compute_graph(
        hw,
        1024,
        256,
        256,
        32,
        transposed,
        TrainiumTuning(matmul_operands="reuse"),
    )
    matmuls = [n for n in reused.nodes if n.name.startswith("matmul_")]
    assert len(matmuls) == 8
    transposes = [n for n in reused.nodes if n.name.startswith("transpose_")]
    assert len(transposes) == (0 if transposed else 4)
    assert sum(n.name.startswith("transpose_") for n in direct.nodes) == (
        0 if transposed else 8
    )
    if not transposed:
        # Each cached weight completion has consumers in both M panels.
        dependencies = [d.source for n in matmuls for d in n.dependencies]
        weight_ready = [
            i for i, n in enumerate(reused.nodes) if n.name.startswith("copy_")
        ]
        assert len(weight_ready) == 4
        assert all(dependencies.count(i) == 2 for i in weight_ready)
    plan = make_plan(point())
    staged_storage = matrix_storage(
        hw,
        1024 * 256,
        256 * 256,
        1024 * 256,
        256,
        256,
        32,
        32,
        32,
        plan,
        TrainiumTuning(),
    )
    reuse_storage = matrix_storage(
        hw,
        1024 * 256,
        256 * 256,
        1024 * 256,
        256,
        256,
        32,
        32,
        32,
        plan,
        TrainiumTuning(matmul_operands="reuse"),
    )
    assert reuse_storage[:-1] == staged_storage
    assert reuse_storage[-1].bytes_per_slot == 256 * 256 * 4


def test_capacity_rejection_has_no_finite_traffic_score():
    cost = estimator(TrainiumTuning(matmul_operands="reuse"))
    mapping = point(m=4096, n=4096, k=4096)
    layer = SimpleNamespace(hstd=1, wstd=1)
    assert math.isinf(cost.calculate_runtime(None, layer, mapping))
    assert math.isinf(cost.calculate_memory_cost(None, layer, mapping))


@pytest.mark.parametrize("fuse", [False, True])
def test_shared_pointwise_fusion_removes_only_internal_materialization(
    tmp_path, fuse
):
    import voyager_compiler as vc
    from google.protobuf import text_format
    from voyager_compiler.codegen import voyager_ir_pb2 as ir
    from voyager_compiler.trainium.converter import convert

    class Gate(torch.nn.Module):
        def forward(self, x, y):
            return torch.nn.functional.silu(x) * y

    torch.manual_seed(23)
    inputs = (torch.randn(128, 256), torch.randn(128, 256))
    config = neuron_core(3)
    context = CompilerContext.resolve(
        config,
        TrainiumMappingPolicy(config, TrainiumTuning(pointwise_fusion=fuse)),
    )
    graph = vc.export_model(Gate(), inputs)
    vc.transform(graph, inputs, context=context)
    vc.compile(graph, inputs, context=context, output_dir=tmp_path)
    torch.testing.assert_close(graph(*inputs), Gate()(*inputs))
    convert(tmp_path)
    model = text_format.Parse((tmp_path / "model.txt").read_text(), ir.Model())
    allocations = [
        out.tensor_box
        for op in model.ops
        if op.WhichOneof("op_type") == "prim"
        and op.prim.target == "voyager::alloc"
        for out in op.outputs
        if out.tensor_box.memory.level == ir.MEMORY_LEVEL_DRAM
    ]
    # Sigmoid and the first multiply need no HBM objects when fused. Both
    # external operands and the final result remain part of the same ABI.
    assert len(allocations) == (1 if fuse else 3)
    restored = CompilerContext.from_artifacts(tmp_path, config)
    assert restored.policy.tuning.pointwise_fusion == fuse
