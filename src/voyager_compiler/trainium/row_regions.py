"""Resource checks and conservative composed-row candidate estimates.

Uses the existing instruction timing laws. This first policy serializes stage
estimates and boundary transfers; selected ISA analysis remains the final model
audit. No measured kernel latency is an input to this search.
"""

import math

from voyager_compiler.codegen.node_info import reduction_scratch
from voyager_compiler.codegen.transform.tiling.execution import RepeatedGraph

from .cost import dma_service
from .dependencies import compute_graph, evaluate_graph
from .execution import slot_bytes
from .lowering import RECIPES, operation_name, reduction_workspace, tile_graph
from .movement import transfer_graph


def candidate(config, tuning, region, rows):
    layouts = (
        ("generic", "k_partitioned")
        if tuning.matmul_weight_layout == "auto"
        else (tuning.matmul_weight_layout,)
    )
    trials = [
        dict(
            weight_layout=layout,
            **_candidate(config, tuning, region, rows, layout),
        )
        for layout in layouts
    ]
    legal = [
        c
        for c in trials
        if c.get("legal") and math.isfinite(c["prediction_ns"])
    ]
    if not legal:
        return dict(trials[0], layout_candidates=trials)
    selected = min(legal, key=lambda c: c["prediction_ns"])
    return dict(selected, layout_candidates=trials)


def _candidate(config, tuning, region, rows, weight_layout):
    if weight_layout == "k_partitioned" and not tuning.isa_lowering:
        return dict(
            legal=False,
            reason="K-partitioned weight loading requires ISA lowering",
        )
    if rows > 128:
        return dict(
            legal=False,
            reason="row reduction ISA layout supports at most 128 rows",
        )
    if any(
        n.value.dtype.__str__() != "torch.float32"
        for n in (*region.inputs, *region.nodes)
    ):
        return dict(
            legal=False,
            reason="initial row-region timing/layout contract is FP32",
        )
    for op in region.nodes:
        name = operation_name(op.target)
        if name not in {
            "matmul",
            "rms_norm",
            "layer_norm",
            "add",
            "sub",
            "mul",
            "maximum",
            "relu",
            "sigmoid",
            "exp",
            "tanh",
            "clone",
        }:
            return dict(
                legal=False,
                reason=f"no Trainium row-region lowering contract for {name}",
            )
        if name not in {"matmul", "rms_norm", "layer_norm"} and any(
            tuple(a.shape) != tuple(op.shape) for a in op.all_input_nodes
        ):
            return dict(
                legal=False,
                reason="row-layout pointwise tensor broadcasting is not yet supported",
            )
    shapes = {n: region.shape(n, rows) for n in (*region.inputs, *region.nodes)}

    # Match the planner's row-layout closure: norms impose a row layout on
    # their input/result, and same-shape pointwise operations preserve it.
    row_values = set()
    for op in region.nodes:
        if operation_name(op.target) in RECIPES:
            row_values.update((op, *op.all_input_nodes))
    changed = True
    while changed:
        changed = False
        for op in region.nodes:
            if operation_name(op.target) in {"matmul", "mm", *RECIPES}:
                continue
            group = {op, *op.all_input_nodes}
            if group & row_values and not group <= row_values:
                row_values.update(group)
                changed = True
    matrix = next(
        n for n in region.nodes if operation_name(n.target) in ("matmul", "mm")
    )
    weight = matrix.args[1]
    if weight_layout == "k_partitioned" and any(
        weight in n.all_input_nodes for n in region.nodes if n is not matrix
    ):
        return dict(
            legal=False,
            reason="K-partitioned weight must have only the matrix consumer inside its region",
        )

    def storage(shape, row=False):
        generic = slot_bytes(math.prod(shape), shape[-1] if shape else 1, 32)
        # Row layouts reserve all physical partitions, even for a 96-row tile.
        row_bytes = (
            128 * math.ceil(shape[-1] * 4 / 16) * 16
            if row and len(shape) == 2
            else 0
        )
        return max(generic, row_bytes)

    # One boundary slot per input/output, one local destination per operation,
    # and named reduction scratch. Conservative sum (no assumed local reuse).
    allocated = sum(
        storage(s, n in region.nodes or n in region.varying)
        for n, s in shapes.items()
    )
    if weight_layout == "k_partitioned":
        k, n = shapes[weight]
        allocated += slot_bytes(k * n, k, 32) - storage(shapes[weight])
    extra = 0
    for op in region.nodes:
        allocated += sum(
            storage(s, True)
            for _, s, _ in reduction_scratch(
                op, shapes[op], config.vector_lanes
            )
        )
        name = operation_name(op.target)
        if name in RECIPES:
            extra = max(
                extra,
                reduction_workspace(
                    name, shapes[op][-1], len(op.all_input_nodes) - 1
                ),
            )
    allocated += max(tuning.sbuf_reserve_bytes, extra)
    if allocated > config.scratchpad_size:
        return dict(
            legal=False,
            reason="whole invariant operands and local tiles exceed SBUF",
            sbuf_bytes=allocated,
        )
    repeats = region.rows // rows
    traffic = 0
    boundary_ns = 0
    for operand in (*region.inputs, region.nodes[-1]):
        shape = shapes[operand]
        store = operand is region.nodes[-1]
        count = repeats if store or operand in region.varying else 1
        dma = dma_service(
            config,
            math.prod(shape[:-1]),
            shape[-1],
            32,
            transpose=(
                operand not in row_values and operand is not weight
                if weight_layout == "k_partitioned"
                else operand not in row_values
            ),
            store=store,
            tuning=tuning,
        )
        boundary_ns += (
            count
            * evaluate_graph(transfer_graph(config, dma.panels)).duration_ns
        )
        traffic += count * dma.bytes
    matrix_choices = []
    compute_ns = 0
    stages = []
    for op in region.nodes:
        name = operation_name(op.target)
        if name in ("matmul", "mm"):
            graph, _ = compute_graph(
                config,
                rows,
                shapes[op][-1],
                op.args[0].shape[-1],
                32,
                False,
                tuning,
                input_row=op.args[0] in row_values,
                output_row=op in row_values,
                weight_layout=weight_layout,
            )
            from .orientation import matrix_choice

            matrix_choices.append(
                matrix_choice(
                    config,
                    rows,
                    shapes[op][-1],
                    op.args[0].shape[-1],
                    32,
                    False,
                    tuning,
                    input_row=op.args[0] in row_values,
                    output_row=op in row_values,
                    weight_layout=weight_layout,
                )
            )
        else:
            graph = tile_graph(
                config,
                (name,),
                shapes[op],
                tuple(shapes[a] for a in op.all_input_nodes),
                tuning,
            )
            # Internal edges stay local. Strip boundary-DMA nodes from the
            # template, preserving each compute dependency through removed
            # nodes. Physical layout conversions are still priced below.
            from dataclasses import replace
            from voyager_compiler.codegen.transform.tiling.execution import (
                Dependency,
            )

            nodes, remap = [], {}
            for i, n in enumerate(graph.nodes):
                deps = set()
                for d in n.dependencies:
                    if d.distance == 0:
                        deps.update(remap.get(d.source, ()))
                if n.resource in ("DMA", "DMAIssue"):
                    remap[i] = deps
                else:
                    remap[i] = {len(nodes)}
                    nodes.append(
                        replace(
                            n,
                            dependencies=tuple(
                                Dependency(j) for j in sorted(deps)
                            ),
                        )
                    )
            graph = RepeatedGraph(tuple(nodes))
        duration = evaluate_graph(graph).duration_ns
        if name in RECIPES:
            # Shared intermediates use the general matrix layout; a row
            # reduction requires the existing entry/exit conversion templates.
            from .lowering import layout_graph

            for shape in (shapes[op.args[0]], shapes[op]):
                duration += (
                    math.ceil(shape[-1] / 128)
                    * evaluate_graph(
                        layout_graph(config, rows, min(128, shape[-1]))
                    ).duration_ns
                )
        compute_ns += repeats * duration
        stages.append(dict(operation=name, per_tile_ns=duration))
    return dict(
        legal=True,
        sbuf_bytes=allocated,
        hbm_bytes=traffic,
        prediction_ns=boundary_ns
        + compute_ns
        + config.timing_profile.fixed_kernel_ns,
        stages=stages,
        matrix_choices=matrix_choices,
        weight_contract=dict(
            logical_shape=list(shapes[weight]),
            partition_axis="K" if weight_layout == "k_partitioned" else "N",
            load=(
                "direct_dma"
                if weight_layout == "k_partitioned"
                else "existing_transpose_load"
            ),
            lifetime=(
                "whole invariant region; one shared load; no per-row conversion"
                if weight_layout == "k_partitioned"
                else "existing per-GEMM operand conversion"
            ),
            storage_bytes=slot_bytes(
                math.prod(shapes[weight]),
                shapes[weight][0 if weight_layout == "k_partitioned" else 1],
                32,
            ),
        ),
        model_scope="serial boundary and stage templates; final physical reuse checked in selected ISA",
    )
