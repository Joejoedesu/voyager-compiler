"""Movement contracts: independent issue, readiness, boundaries and allocation."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
from voyager_compiler.trainium.timing import (
    PrimitiveTiming,
    DmaTiming,
    TrainiumTimings,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.movement import TransferPanel, transfer_graph
from voyager_compiler.trainium.program_analysis import analyze_program
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.codegen.transform.tiling.execution import evaluate_graph
from test_trainium import estimator, point


def test_dma_requests_pipeline_but_completion_delays_consumers():
    law = DmaTiming(100, 200, 0, 1, 300, payload_base_ns=200)
    hw = replace(
        neuron_core(3), timing_profile=TrainiumTimings(load=law, store=law)
    )
    panels = tuple(
        TransferPanel(0, i * 128, 128, 128, 32, False, False, False, 50)
        for i in range(3)
    )
    timing = evaluate_graph(transfer_graph(hw, panels))
    assert dict(timing.service_ns) == {"DMA": 150, "DMAIssue": 300}
    assert (
        timing.duration_ns == 950
    )  # Last issue 200 + dispatch 200 + completion 250 + notify 300; occupancy remains 50.
    assert not timing.unknown_latency
    assert sum(p.bytes for p in panels) == 3 * 65536


def test_timing_law_does_not_charge_dependent_latency_as_throughput():
    law = PrimitiveTiming("matmul", 10, 1, 300, 2)
    assert law.evaluate(100) == (100, 500)
    for kwargs in [dict(issue_ns=-1), dict(notification_ns=float("nan"))]:
        with pytest.raises(ValueError):
            DmaTiming(
                **(
                    dict(
                        issue_ns=1,
                        dispatch_ns=2,
                        payload_floor_ns=0,
                        payload_scale=1,
                        notification_ns=3,
                    )
                    | kwargs
                )
            )


def test_logical_buffer_depth_has_no_unrealized_speed_credit():
    costs = []
    for depth in (1, 2):
        model = estimator(
            TrainiumTuning(min_buffer_depth=depth, max_buffer_depth=depth)
        )
        model.calculate_runtime(
            None, SimpleNamespace(hstd=1, wstd=1), point()
        )
        costs.append(model.estimate["predicted_ns"])
        assert model.estimate["buffer_depth"] == depth
        assert not model.estimate["buffer_depth_speed_credit"]
    assert costs[0] == costs[1]


def test_boundary_traffic_and_source_audit():
    c = SimpleNamespace(
        tuning=TrainiumTuning(),
        vector_records=[],
        hardware=neuron_core(3),
        stats=dict(tensor_instructions=1, isa_transposes=1, isa_dma_panels=2),
        transfer_records=[dict(read_bytes=65536, write_bytes=65536)],
        boundary_records=[
            dict(
                dtype="float32",
                output_shape=[128, 128],
                read_bytes=32768,
                write_bytes=65536,
                implementation="isa.rectangular_pad_slice",
                panels=[
                    dict(
                        rows=128,
                        columns=128,
                        valid_rows=64,
                        valid_columns=128,
                        fill=True,
                    )
                ],
            )
        ],
    )
    source = "nisa.nc_matmul(a,b)\nnisa.nc_transpose(a)\nnisa.dma_copy(a,b)\nnisa.dma_copy(b,a)"
    record = dict(
        estimates=[
            dict(
                predicted_ns=20000,
                fixed_kernel_ns=c.hardware.timing_profile.fixed_kernel_ns,
                dependency_model=dict(timing_complete=True),
            )
        ]
    )
    result = analyze_program(source, c, record)
    assert result["hbm"]["total_bytes"] == 65536 * 3 + 32768 + 16384
    assert result["boundary_prediction_ns"] > 0
    assert result["whole_program_timing_complete"]
    assert not result["physical_slot_depth_enforced"]
    with pytest.raises(ValueError, match="DMA source"):
        analyze_program(source.rsplit("\n", 1)[0], c, record)


def test_unknown_boundary_expansion_is_not_marked_complete():
    c = SimpleNamespace(
        tuning=TrainiumTuning(),
        vector_records=[],
        hardware=neuron_core(3),
        stats=dict(tensor_instructions=0, isa_transposes=0, isa_dma_panels=0),
        transfer_records=[],
        boundary_records=[
            dict(
                dtype="float32",
                output_shape=[8, 8],
                read_bytes=256,
                write_bytes=256,
            )
        ],
    )
    result = analyze_program(
        "", c, dict(estimates=[dict(predicted_ns=100, dependency_model=None)])
    )
    assert not result["whole_program_timing_complete"]
    assert result["boundaries"][0]["unmodeled"]


@pytest.mark.parametrize("bias_bits", [16, 32])
@pytest.mark.parametrize("has_tail", [False, True])
def test_bias_width_does_not_multiply_epilogue_operation_count(
    bias_bits, has_tail
):
    from voyager_compiler.trainium.cost import Service
    from voyager_compiler.trainium.dependencies import kernel_plan
    from voyager_compiler.codegen.transform.tiling.execution import (
        RepeatedGraph,
    )

    plan = kernel_plan(
        neuron_core(3),
        RepeatedGraph(()),
        [],
        Service(),
        blocks=1,
        outputs=1,
        buffers=SimpleNamespace(input_slots=1, weight_slots=1, output_slots=1),
        output_pass=100,
        has_tail=has_tail,
        bias=bias_bits,
        counts={},
    )
    epilogue = next(
        node for node in plan.graph.nodes if node.name == "epilogue"
    )
    assert epilogue.occupancy_ns == 100 * (1 + int(has_tail))
    assert (
        epilogue.latency_ns is None
    )  # Detailed epilogue timing remains uncharacterized.
