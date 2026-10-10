"""Conservative cost/capacity contract for independent-axis stream regions.

Search enumerates row divisors and invariant residency before bufferization.
Costs reuse the existing ISA timing laws, serializing stage and boundary costs.
This is not a claim that the compact estimate models native scheduling exactly.
"""

from dataclasses import replace
import math

from voyager_compiler.codegen.transform.bufferize.stream_regions import (
    MATRIX,
    SOFTMAX,
    local_reduction_scratch,
)
from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    RepeatedGraph,
)
from .cost import dma_service
from .dependencies import compute_graph, evaluate_graph
from .execution import slot_bytes
from .lowering import (
    RECIPES,
    operation_name,
    reduction_workspace,
    tile_graph,
    layout_graph,
)
from .movement import transfer_graph


def candidate(config, tuning, region, rows, residency):
    if tuning.matmul_weight_layout == "k_partitioned":
        return dict(
            legal=False,
            reason="stream regions currently realize generic RHS layout only",
            constraint_kind="lowering",
        )
    if residency not in ("resident", "stage"):
        return dict(legal=False, reason="unknown invariant residency policy")
    if rows > 128:
        return dict(
            legal=False,
            reason="initial stream-region policy limits local rows to 128",
            constraint_kind="policy",
        )
    if any(
        str(n.value.dtype) != "torch.float32"
        for n in (*region.inputs, *region.nodes)
    ):
        return dict(
            legal=False, reason="initial stream-region ISA contract is FP32"
        )
    if residency == "stage" and any(
        r.matrix_rhs and r.batch for r in region.roles.values()
    ):
        return dict(
            legal=False,
            reason="stage-loaded batched RHS needs an explicit batch subview contract",
        )
    names = {
        op: (
            "matmul"
            if op.target in MATRIX
            else (
                "softmax" if op.target in SOFTMAX else operation_name(op.target)
            )
        )
        for op in region.nodes
    }
    supported = {
        "matmul",
        *RECIPES,
        "add",
        "sub",
        "mul",
        "maximum",
        "relu",
        "sigmoid",
        "exp",
        "tanh",
        "clone",
    }
    if any(n not in supported for n in names.values()):
        return dict(
            legal=False,
            reason="operation lacks a Trainium stream-region ISA contract",
        )
    shapes = {n: region.shape(n, rows) for n in (*region.inputs, *region.nodes)}
    row_values = set()
    for op, name in names.items():
        if name in RECIPES:
            row_values.update((op, *op.all_input_nodes))
    changed = True
    while changed:
        changed = False
        for op, name in names.items():
            if name == "matmul" or name in RECIPES:
                continue
            group = {op, *op.all_input_nodes}
            if group & row_values and not group <= row_values:
                row_values.update(group)
                changed = True

    def storage(shape, row=False):
        generic = slot_bytes(math.prod(shape), shape[-1], 32)
        return max(
            generic,
            (
                128 * math.ceil(shape[-1] * 4 / 16) * 16
                if row and len(shape) == 2
                else 0
            ),
        )

    staged = {
        n
        for n in region.inputs
        if residency == "stage" and region.roles[n].matrix_rhs
    }
    allocated = sum(
        storage(s, n in row_values)
        for n, s in shapes.items()
        if n not in staged
    )
    if staged:
        allocated += max(storage(shapes[n]) for n in staged)
    extra = 0
    for op, name in names.items():
        allocated += sum(
            storage(s, True)
            for _, s, _ in local_reduction_scratch(
                op, shapes[op], config.vector_lanes
            )
        )
        if name in RECIPES:
            extra = max(
                extra,
                reduction_workspace(
                    name, shapes[op][-1], len(op.all_input_nodes) - 1, tuning=tuning
                ),
            )
    allocated += max(tuning.sbuf_reserve_bytes, extra)
    if allocated > config.scratchpad_size:
        return dict(
            legal=False,
            reason="explicit local buffers plus temporary reserve exceed SBUF",
            sbuf_bytes=allocated,
        )
    batches = math.prod(region.batch)
    tiles = region.rows // rows
    repeats = batches * tiles
    traffic = 0
    boundary_ns = 0
    transfers = []
    for n in (*region.inputs, region.nodes[-1]):
        shape = shapes[n]
        store = n is region.nodes[-1]
        if store or region.roles[n].row:
            count = repeats
        elif n in staged:
            count = repeats * sum(
                op.target in MATRIX and op.args[1] is n for op in region.nodes
            )
        else:
            # One retained slot is reloaded whenever its batch coordinates
            # change. Broadcast outer axes can revisit earlier coordinates.
            varying = [i for i, x in enumerate(region.roles[n].batch) if x != 1]
            count = (
                math.prod(region.batch[: max(varying) + 1]) if varying else 1
            )
        dma = dma_service(
            config,
            math.prod(shape[:-1]),
            shape[-1],
            32,
            transpose=n not in row_values,
            store=store,
            tuning=tuning,
        )
        ns = evaluate_graph(transfer_graph(config, dma.panels)).duration_ns
        boundary_ns += count * ns
        traffic += count * dma.bytes
        transfers.append(
            dict(
                operand=n.name,
                store=store,
                count=count,
                bytes_per_transfer=dma.bytes,
            )
        )
    stages = []
    compute_ns = 0
    for op, name in names.items():
        if name == "matmul":
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
                weight_layout="generic",
            )
        else:
            graph = tile_graph(
                config,
                (name,),
                shapes[op],
                tuple(shapes[a] for a in op.all_input_nodes),
                tuning,
            )
            nodes = []
            remap = {}
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
        transfers=transfers,
        batch_count=batches,
        row_tiles=tiles,
        weight_layout="generic",
        model_scope="serial boundary and stage templates; selected physical ISA remains a separate estimate",
    )
