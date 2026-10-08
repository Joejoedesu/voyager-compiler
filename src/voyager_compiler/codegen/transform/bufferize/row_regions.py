"""Compose row-separable operations before introducing DRAM intermediates.

This is a shared bufferization algorithm, not an ISA fusion pattern. A region
keeps each reduction axis whole and streams a common independent row axis.
The initial contract covers 2-D pointwise/norm DAGs and one 2-D matrix product.
An invariant matrix operand must fit locally. Unsupported regions retain the
ordinary per-kernel path; no slicing across a reduction is guessed.
"""

from dataclasses import dataclass
import math

import torch
from torch.fx import Graph, GraphModule, map_arg

from voyager_compiler.codegen.node_info import (
    get_arg_value,
    is_elementwise_op,
    reduction_op,
    reduction_scratch,
)
from voyager_compiler.codegen.subgraph import create_and_insert_subgraph
from voyager_compiler.codegen.transform.tiling.search import get_valid_tiling

_MATRIX = {torch.ops.aten.matmul.default, torch.ops.aten.mm.default}
_NORMS = {torch.ops.aten.rms_norm.default, torch.ops.aten.layer_norm.default}


@dataclass(frozen=True)
class RowRegion:
    nodes: tuple
    inputs: tuple
    rows: int
    varying: tuple

    def shape(self, node, tile_rows):
        shape = tuple(node.shape)
        if node in self.nodes or node in self.varying:
            return (tile_rows, *shape[1:])
        return shape


def analyze_row_region(nodes):
    """Prove row independence using operand roles, broadcasting and axes.

    In particular a GEMM RHS is invariant even when K happens to equal M;
    matching extents alone never establishes an axis correspondence.
    """
    nodes = tuple(nodes)
    if len(nodes) < 2:
        raise ValueError("region needs at least two operations")
    if sum(n.target in _MATRIX for n in nodes) != 1:
        raise ValueError(
            "initial row-region contract requires one matrix product"
        )
    rows = int(nodes[-1].shape[0])
    roles = {}
    for n in nodes:
        if n.op != "call_function" or len(n.shape) != 2 or n.shape[0] != rows:
            raise ValueError(
                "operations must preserve a common independent row axis"
            )
        if n is not nodes[-1] and any(u not in nodes for u in n.users):
            raise ValueError("intermediate has a consumer outside the region")
        if n.target in _MATRIX:
            a, b = n.args[:2]
            if len(a.shape) != 2 or len(b.shape) != 2 or b in nodes:
                raise ValueError(
                    "matrix RHS must be an external invariant 2-D tensor"
                )
            operands = [(a, True), (b, False)]
        elif n.target in _NORMS:
            normalized = tuple(get_arg_value(n, 1, "normalized_shape"))
            if normalized != (n.shape[-1],):
                raise ValueError(
                    "normalization reduces the streaming row axis"
                )
            operands = [(a, a is n.args[0]) for a in n.all_input_nodes]
        elif is_elementwise_op(n):
            operands = []
            for a in n.all_input_nodes:
                shape = tuple(a.shape)
                if len(shape) > 2 or (
                    len(shape) == 2 and shape[0] not in (1, rows)
                ):
                    raise ValueError("unsupported pointwise broadcasting")
                operands.append((a, len(shape) == 2 and shape[0] == rows))
        else:
            raise ValueError(f"no row-axis rule for {n.target}")
        for a, varying in operands:
            if a in roles and roles[a] != varying:
                raise ValueError(
                    "operand has conflicting invariant and streaming roles"
                )
            roles[a] = varying
    inputs = tuple(
        dict.fromkeys(
            a for n in nodes for a in n.all_input_nodes if a not in nodes
        )
    )
    return RowRegion(nodes, inputs, rows, tuple(a for a in inputs if roles[a]))


def plan_row_regions(model, tiler):
    """Shared exact-divisor enumeration; target prices and checks each tile.

    Region discovery does not match application names or operation orderings.
    Target support is explicit through ``row_region_candidate``. Conservative
    candidate estimates and rejections are retained for later ISA comparison.
    """
    evaluate = getattr(tiler.mapping_policy, "row_region_candidate", None)
    if evaluate is None:
        raise NotImplementedError(
            "target has no row-region resource/cost contract"
        )
    records = model.meta.setdefault("row_regions", [])
    runs, run = [], []
    for n in model.graph.nodes:
        eligible = n.op == "call_function" and (
            n.target in _MATRIX or n.target in _NORMS or is_elementwise_op(n)
        )
        if eligible:
            run.append(n)
        elif n.op not in ("placeholder", "get_attr"):
            if run:
                runs.append(run)
                run = []
    if run:
        runs.append(run)
    for nodes in runs:
        if len(nodes) < 2:
            continue
        record = {"operations": [str(n.target) for n in nodes]}
        records.append(record)
        try:
            region = analyze_row_region(nodes)
        except ValueError as exc:
            record.update(status="fallback", reason=str(exc))
            continue
        candidates = []
        for tile, _ in get_valid_tiling((region.rows,), exhaustive=True):
            result = evaluate(region, tile[0])
            candidates.append(dict(tile_rows=tile[0], **result))
        record["candidates"] = candidates
        legal = [
            c
            for c in candidates
            if c.get("legal") and math.isfinite(c["prediction_ns"])
        ]
        if not legal:
            record.update(
                status="fallback",
                reason="no legal whole-reduction row tile fits",
            )
            continue
        selected = min(
            legal, key=lambda c: (c["prediction_ns"], -c["tile_rows"])
        )
        record.update(
            status="selected",
            selected=selected,
            rows=region.rows,
            boundary_inputs=[n.name for n in region.inputs],
            streaming_inputs=[n.name for n in region.varying],
            internal_edges="SRAM",
            reduction_axes="whole",
        )
        grouped = create_and_insert_subgraph(list(nodes), model)
        grouped.meta["row_region"] = dict(
            tile_rows=selected["tile_rows"],
            rows=region.rows,
            varying=tuple(
                n in region.varying for n in grouped.all_input_nodes
            ),
        )
    model.graph.lint()
    model.recompile()


def build_row_region(node, *, tiler):
    """Use the existing pipeline scheduler with explicit local DPS edges."""
    from .pipeline import build_pipelined_buffers, _single_pass_kernel
    from .utils import _InputSpec, _OutputSpec, _ScratchSpec

    plan = node.meta["row_region"]
    rows = plan["tile_rows"]
    sub = node.meta["submodule"]
    graph, env, shapes = Graph(), {}, {}
    sources = list(node.all_input_nodes)
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    in_specs = []
    for ph, source, varying in zip(placeholders, sources, plan["varying"]):
        shape = (rows, *source.shape[1:]) if varying else tuple(source.shape)
        env[ph] = graph.placeholder(ph.name)
        shapes[ph] = shape
        in_specs.append(
            _InputSpec(
                shape,
                (0, None) if varying else (None,) * len(shape),
                (False,) * len(shape),
                num_slots=1,
            )
        )
    scratch_specs = []
    scratch_args = []
    # Scratch parameters are placeholders (allocated once outside the loop).
    operations = [n for n in sub.graph.nodes if n.op == "call_function"]
    for op in operations:
        if op.op != "call_function":
            continue
        shape = (rows, *op.shape[1:])
        shapes[op] = shape
        allocations = (
            [] if op is operations[-1] else [("result", shape, op.value.dtype)]
        )
        allocations += reduction_scratch(op, shape, tiler.config.vector_lanes)
        slots = {}
        for name, size, dtype in allocations:
            slots[name] = graph.placeholder(f"{op.name}_{name}_storage")
            scratch_specs.append(_ScratchSpec(size, dtype))
        scratch_args.append((op, slots))
    for op, slots in scratch_args:
        args = map_arg(op.args, lambda n: env[n])
        kwargs = dict(map_arg(op.kwargs, lambda n: env[n]))
        kwargs.update({k: v for k, v in slots.items() if k != "result"})
        computed = graph.call_function(
            reduction_op(op) or op.target, args, kwargs
        )
        if "result" in slots:
            graph.call_function(
                torch.ops.voyager.insert.default, (computed, slots["result"])
            )
        env[op] = slots.get("result", computed)
    output = next(n for n in sub.graph.nodes if n.op == "output")
    graph.output(map_arg(output.args[0], lambda n: env[n]))
    compute = GraphModule({}, graph)
    out_shape = tuple(node.shape)
    gm = build_pipelined_buffers(
        _single_pass_kernel(compute, 1, len(scratch_specs)),
        (plan["rows"] // rows,),
        in_specs,
        [
            _OutputSpec(
                out_shape,
                (rows, *out_shape[1:]),
                (0, None),
                node.value.dtype,
                num_slots=1,
            )
        ],
        tuple(n.value.clone() for n in sources),
        scratch_specs=scratch_specs,
        num_slots=1,
    )
    return gm
