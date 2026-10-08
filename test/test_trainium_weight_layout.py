"""Invariant payload reuse, layout capacity, and conservative fallback."""

from dataclasses import replace
import json

import pytest
import torch

import voyager_compiler as vc
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.codegen.transform.bufferize.row_regions import (
    analyze_row_region,
)
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy
from voyager_compiler.trainium.row_regions import candidate


@pytest.mark.parametrize("orientation", ["weights", "activations"])
def test_invariant_loaded_once_and_used_across_row_tiles(orientation, tmp_path):
    from test_trainium_row_regions import case

    graph, args, expected, base = case(
        ("gemm", "add", "norm"), m=192, k=256, n=512
    )
    hw = base.hardware
    context = CompilerContext.resolve(
        hw,
        TrainiumMappingPolicy(
            hw,
            replace(
                base.policy.tuning,
                matmul_orientation=orientation,
                matmul_weight_layout="k_partitioned",
            ),
        ),
    )
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(row_regions=True),
    )
    torch.testing.assert_close(graph(*args), expected, rtol=1e-3, atol=1e-3)
    result = json.loads((tmp_path / "selection.json").read_text())
    assert len(result["k_partitioned_weight_buffers"]) == 1
    weight = result["k_partitioned_weight_buffers"][0]
    copies = [
        e
        for e in result["events"]
        if e["kind"] == "copy" and e["dst"] == weight
    ]
    assert len(copies) == 1 and copies[0]["sizes"] == [256, 512]
    assert result["stats"]["k_weight_dma_panels"] == 2
    assert len(result["matrix_choices"]) == 2  # two 96-row consumers, one load
    assert all(
        c["weight_layout"] == "k_partitioned"
        and c["orientation"] == orientation
        for c in result["matrix_choices"]
    )
    record = json.loads((tmp_path / "hardware.json").read_text())[
        "row_regions"
    ][0]["selected"]
    assert record["weight_contract"]["storage_bytes"] == 256 * 512 * 4
    assert record["weight_contract"]["partition_axis"] == "K"
    assert (
        record["hbm_bytes"]
        == sum(x.numel() * 4 for x in args) + expected.numel() * 4
    )


def test_partition_padding_can_reject_new_layout_without_rejecting_old():
    from test_trainium_row_regions import Chain
    from voyager_compiler.shape_prop import ShapeProp

    # Analyze before target padding to exercise unequal physical footprints.
    args = (
        torch.randn(128, 192),
        torch.randn(192, 128),
        torch.randn(128, 128),
        torch.randn(128),
    )
    graph = vc.export_model(Chain(("gemm", "add", "norm")), args)
    ShapeProp(graph).propagate(*args)
    region = analyze_row_region(
        [n for n in graph.graph.nodes if n.op == "call_function"]
    )
    hw = neuron_core(3)
    base = candidate(hw, TrainiumTuning(matmul_operands="reuse"), region, 128)
    trials = {c["weight_layout"]: c for c in base["layout_candidates"]}
    assert (
        trials["k_partitioned"]["sbuf_bytes"] - trials["generic"]["sbuf_bytes"]
        == 32768
    )
    tuning = TrainiumTuning(
        matmul_operands="reuse",
        sbuf_reserve_bytes=(4 << 20)
        + hw.scratchpad_size
        - trials["generic"]["sbuf_bytes"]
        - 16384,
    )
    limited = candidate(hw, tuning, region, 128)
    assert limited["legal"] and limited["weight_layout"] == "generic"
    rejected = next(
        c
        for c in limited["layout_candidates"]
        if c["weight_layout"] == "k_partitioned"
    )
    assert not rejected["legal"] and "SBUF" in rejected["reason"]


def test_generic_mode_preserves_existing_realization(tmp_path):
    from test_trainium_row_regions import case

    graph, args, _, base = case(("gemm", "add", "norm"), m=128, k=128, n=128)
    context = CompilerContext.resolve(
        base.hardware,
        TrainiumMappingPolicy(
            base.hardware,
            replace(base.policy.tuning, matmul_weight_layout="generic"),
        ),
    )
    vc.compile(
        graph,
        args,
        context=context,
        output_dir=tmp_path,
        dump_tensors=False,
        bufferization_options=BufferizationOptions(row_regions=True),
    )
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert not selection["k_partitioned_weight_buffers"]
    assert all(
        c["weight_layout"] == "generic" for c in selection["matrix_choices"]
    )
