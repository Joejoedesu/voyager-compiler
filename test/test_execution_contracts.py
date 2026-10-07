"""Cross-target operation contracts and dependency-sensitive candidate timing."""

from dataclasses import asdict, replace
import json
import pytest

from voyager_compiler.hardware_config import OperationTiming
from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency as D,
    OperationEvent as E,
    RepeatedGraph as G,
    evaluate_graph,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.dependencies import compute_graph
from voyager_compiler.trainium.execution import TrainiumTuning


def test_independent_issue_is_not_dependent_completion():
    independent = G((E("op", "engine", 200, 200, 800),), 10)
    dependent = G(
        (E("op", "engine", 200, 200, 800, dependencies=(D(0, 1),)),), 10
    )
    assert evaluate_graph(independent).duration_ns == 2600
    assert evaluate_graph(dependent).duration_ns == 8000


def test_two_reusable_contexts_expose_independent_chains():
    def chain(depth):
        return G(
            (
                E(
                    "transpose",
                    "tensor",
                    100,
                    100,
                    400,
                    dependencies=(D(3, depth),),
                ),
                E("copy", "scalar", 100, 100, 200, dependencies=(D(0),)),
                E("matmul", "tensor", 200, 200, 800, dependencies=(D(1),)),
                E("evict", "scalar", 100, 100, 200, dependencies=(D(2),)),
            ),
            10,
        )

    single, double = (evaluate_graph(chain(d)) for d in (1, 2))
    assert single.duration_ns == 16000
    assert (
        double.duration_ns == 8200
    )  # ASAP issues the two contexts in bursts
    assert single.service_ns == double.service_ns
    assert (
        double.iteration_finishes_ns[-1] - double.iteration_finishes_ns[-3]
    ) / 2 == 800


def test_periodic_load_reuse_and_final_store():
    graph = G(
        (
            E("load", "dma", 10, 10, 100, period=4),
            E("compute", "tensor", 20, 20, 30, dependencies=(D(0), D(1, 1))),
            E(
                "store",
                "dma",
                10,
                10,
                100,
                dependencies=(D(1),),
                period=4,
                phase=3,
            ),
        ),
        4,
    )
    result = evaluate_graph(graph)
    assert dict(result.service_ns) == {"dma": 20, "tensor": 80}
    assert result.duration_ns == 320


def test_forwarding_and_read_retirement_have_distinct_milestones():
    for milestone, expected in (
        ("result", 500),
        ("forward", 340),
        ("read", 320),
    ):
        result = evaluate_graph(
            G(
                (
                    E(
                        "producer",
                        "tensor",
                        10,
                        10,
                        200,
                        read_ns=20,
                        forward_ns=40,
                    ),
                    E(
                        "consumer",
                        "copy",
                        10,
                        10,
                        300,
                        dependencies=(D(0, milestone=milestone),),
                    ),
                )
            )
        )
        # The producer itself still completes at 200 even if a consumer can
        # use an earlier milestone.
        assert result.duration_ns == max(200, expected)
        assert result.iteration_finishes_ns == (max(200, expected),)


def test_unknown_latency_is_reported_and_invalid_graph_is_rejected():
    result = evaluate_graph(G((E("uncharacterized", "tensor", 20, 20),)))
    assert result.unknown_latency == ("uncharacterized",)
    with pytest.raises(ValueError, match="topologically"):
        G((E("bad", "x", 1, 1, dependencies=(D(0),)),))
    with pytest.raises(ValueError):
        E("bad", "x", float("nan"), 1)
    with pytest.raises(ValueError):
        G((), 0)


def test_implementation_signatures_and_dataflow_are_validated():
    hardware = neuron_core(3)
    original = hardware.operation_implementation(
        "nki.transpose_copy.float32.ScalarE"
    )
    assert dict(original.instruction_counts()) == {
        "LDWEIGHTS": 1,
        "MATMUL_TRANSPOSE": 1,
        "COPY_SCALAR": 1,
    }

    def install(implementation):
        return replace(hardware, operation_implementations=(implementation,))

    with pytest.raises(ValueError, match="supported operand signature"):
        install(
            replace(
                original,
                steps=(
                    replace(original.steps[0], operation="not_supported"),
                    *original.steps[1:],
                ),
            )
        )
    with pytest.raises(ValueError, match="before it is produced"):
        install(replace(original, steps=tuple(reversed(original.steps))))
    with pytest.raises(ValueError):
        install(
            replace(
                original,
                values=tuple(
                    replace(v, memory="HBM") if v.name == "psum" else v
                    for v in original.values
                ),
            )
        )
    assert json.loads(json.dumps(asdict(original)))[
        "applicability"
    ].startswith("Trainium2")


def test_hardware_timing_changes_chain_without_changing_expansion():
    hardware = neuron_core(3)
    tensor = hardware.compute_unit("TensorE")
    modes = tuple(
        (
            replace(mode, timing=OperationTiming(latency_cycles=4800))
            if mode.name == "dense"
            else mode
        )
        for mode in tensor.modes
    )
    slower = replace(
        hardware,
        computation_units=tuple(
            replace(unit, modes=modes) if unit.name == "TensorE" else unit
            for unit in hardware.computation_units
        ),
    )
    args = (128, 128, 256, 32, True, TrainiumTuning())
    before, before_counts = compute_graph(hardware, *args)
    after, after_counts = compute_graph(slower, *args)
    assert before_counts == after_counts
    assert (
        evaluate_graph(after).duration_ns
        > evaluate_graph(before).duration_ns + 1000
    )


def test_realization_rejects_modified_selected_graph(tmp_path):
    # Use the existing compile/reconvert test's real artifacts, not a synthetic
    # converter mock, to check a corrupted dependency before source publication.
    from test_trainium import test_compile_reconvert_and_instruction_counts
    from voyager_compiler.trainium.converter import convert

    test_compile_reconvert_and_instruction_counts(tmp_path, True)
    record = json.loads((tmp_path / "hardware.json").read_text())
    plan = record["execution_plans"][0]
    event = next(
        n for n in plan["graph"]["nodes"] if n["name"].startswith("matmul_")
    )
    event["dependencies"] = []
    (tmp_path / "hardware.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="dependency templates"):
        convert(tmp_path, tmp_path / "tampered")
    assert not (tmp_path / "tampered/program.py").exists()


def test_multiple_matrix_regions_keep_distinct_selected_plans(tmp_path):
    import torch
    import voyager_compiler as vc
    from voyager_compiler.compilation import CompilerContext

    class Chain(torch.nn.Module):
        def forward(self, a, b, c):
            return (a @ b) @ c

    inputs = tuple(torch.randn(128, 128) for _ in range(3))
    graph = vc.export_model(Chain(), inputs)
    context = CompilerContext.resolve(neuron_core(3))
    vc.transform(graph, inputs, context=context)
    vc.compile(graph, inputs, context=context, output_dir=tmp_path)
    torch.testing.assert_close(graph(*inputs), Chain()(*inputs))
    manifest = context.realize(tmp_path)
    assert manifest["dependency_audit"]["compute_regions"] == 2
