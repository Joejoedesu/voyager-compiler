"""Regions over independent batch/row axes with complete local reductions.

The contract is expressed in operand roles, not workload names. Matrix K and
normalization/softmax feature axes stay whole; an output becomes available only
after its reduction completes. Internal results live in explicit SRAM scratch.
Invariant matrix inputs may be resident or loaded at each consuming stage.
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

MATRIX = {
    torch.ops.aten.matmul.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
}
NORMS = {torch.ops.aten.rms_norm.default, torch.ops.aten.layer_norm.default}
SOFTMAX = {torch.ops.aten.softmax.int, torch.ops.aten._softmax.default}


@dataclass(frozen=True)
class InputRole:
    row: bool
    matrix_rhs: bool
    batch: tuple


@dataclass(frozen=True)
class StreamRegion:
    nodes: tuple
    inputs: tuple
    rows: int
    batch: tuple
    roles: dict

    def shape(self, node, rows):
        """Local compute shape after selecting one independent batch item."""
        shape = tuple(node.shape)
        if node in self.nodes or self.roles[node].row:
            return (rows, shape[-1])
        return shape[-2:] if self.roles[node].matrix_rhs else shape


def analyze_stream_region(nodes):
    nodes = tuple(nodes)
    if len(nodes) < 2 or not any(n.target in MATRIX for n in nodes):
        raise ValueError(
            "region needs a matrix operation and another operation"
        )
    output_shape = tuple(nodes[-1].shape)
    if len(output_shape) < 2:
        raise ValueError("region output needs batch/row/feature axes")
    batch, rows = output_shape[:-2], output_shape[-2]
    roles = {}
    for n in nodes:
        if n.op != "call_function" or tuple(n.shape[:-1]) != (*batch, rows):
            raise ValueError(
                "operations must preserve independent batch and row axes"
            )
        if n is not nodes[-1] and any(u not in nodes for u in n.users):
            raise ValueError("intermediate has a consumer outside the region")
        if n.target in MATRIX:
            a, b = n.args[:2]
            if b in nodes or len(b.shape) < 2:
                raise ValueError(
                    "matrix RHS must be an external invariant tensor"
                )
            if a.shape[-1] != b.shape[-2] or b.shape[-1] != n.shape[-1]:
                raise ValueError("matrix contraction/output axes disagree")
            operands = [(a, True, False), (b, False, True)]
        elif n.target in SOFTMAX:
            dim = get_arg_value(n, 1, "dim")
            if dim % len(n.shape) != len(n.shape) - 1:
                raise ValueError(
                    "softmax must reduce the complete feature axis"
                )
            if n.target is torch.ops.aten._softmax.default and get_arg_value(
                n, 2, "half_to_float"
            ):
                raise ValueError(
                    "dtype-changing softmax is not in this contract"
                )
            operands = [(n.args[0], True, False)]
        elif n.target in NORMS:
            if tuple(get_arg_value(n, 1, "normalized_shape")) != (n.shape[-1],):
                raise ValueError(
                    "normalization must reduce the complete feature axis"
                )
            operands = [(a, a is n.args[0], False) for a in n.all_input_nodes]
        elif is_elementwise_op(n):
            operands = []
            for a in n.all_input_nodes:
                if tuple(a.shape) != tuple(n.shape):
                    raise ValueError(
                        "pointwise tensor broadcasting is not yet supported"
                    )
                operands.append((a, True, False))
        else:
            raise ValueError(f"no independent-axis rule for {n.target}")
        for a, row, rhs in operands:
            if a in nodes:
                continue
            abatch = tuple(a.shape[:-2]) if len(a.shape) >= 2 else ()
            if row and tuple(a.shape[:-1]) != (*batch, rows):
                raise ValueError(
                    "streaming operand has incompatible batch/row axes"
                )
            if abatch and (
                len(abatch) != len(batch)
                or any(x not in (1, y) for x, y in zip(abatch, batch))
            ):
                raise ValueError("unsupported batch broadcasting")
            role = InputRole(row, rhs, abatch)
            if a in roles and roles[a] != role:
                raise ValueError(
                    "operand has conflicting matrix/streaming roles"
                )
            roles[a] = role
    inputs = tuple(
        dict.fromkeys(
            a for n in nodes for a in n.all_input_nodes if a not in nodes
        )
    )
    return StreamRegion(nodes, inputs, rows, batch, roles)


def elide_contraction_padding(model, nodes=None, transactional=False):
    """Cancel matched zero-extension of a matrix contraction, algebraically.

    The shared preparation may pad K to a generic array width. The stream
    builder retains the entire logical K, and target ISA subdivision handles
    the last panel. Removing paired zeros avoids an HBM padding workspace.
    Output-axis padding and unmatched/nonnull padding are deliberately retained.
    """
    records = []
    originals = []
    pads_used = []

    def original(n, axis):
        if n.target is not torch.ops.aten.pad.default:
            return None
        if get_arg_value(
            n, 2, "mode", "constant"
        ) != "constant" or get_arg_value(n, 3, "value", 0) not in (0, None):
            return None
        pads = list(get_arg_value(n, 1, "pad"))
        rank = len(n.shape)
        if len(pads) > 2 * rank:
            return None
        pads += [0] * (2 * rank - len(pads))
        pair = 2 * (rank - 1 - axis)
        if (
            any(v for i, v in enumerate(pads) if i != pair + 1)
            or pads[pair + 1] < 0
        ):
            return None
        return n.args[0]

    for n in list(model.graph.nodes) if nodes is None else nodes:
        if n.op != "call_function" or n.target not in MATRIX:
            continue
        a, b = n.args[:2]
        aa = original(a, len(a.shape) - 1)
        bb = original(b, len(b.shape) - 2)
        if aa is None or bb is None or aa.shape[-1] != bb.shape[-2]:
            continue
        originals.append((n, n.args))
        pads_used.extend((a, b))
        n.args = (aa, bb, *n.args[2:])
        records.append(
            dict(operation=n.name, padded_k=a.shape[-1], logical_k=aa.shape[-1])
        )

    def finish(commit):
        if not commit:
            for n, args in originals:
                n.args = args
            return
        for pad in dict.fromkeys(pads_used):
            if not pad.users:
                model.graph.erase_node(pad)
        model.meta.setdefault("stream_contraction_padding", []).extend(records)

    if transactional:
        return finish
    finish(True)


def plan_stream_regions(model, tiler, *, discovery_only=False):
    evaluate = getattr(tiler.mapping_policy, "stream_region_candidate", None)
    if evaluate is None:
        raise NotImplementedError(
            "target has no stream-region resource/cost contract"
        )
    runs = []
    run = []
    for n in model.graph.nodes:
        eligible = n.op == "call_function" and (
            n.target in MATRIX | NORMS | SOFTMAX or is_elementwise_op(n)
        )
        if eligible:
            run.append(n)
        elif n.op not in ("placeholder", "get_attr"):
            if run:
                runs.append(run)
                run = []
    if run:
        runs.append(run)
    records = model.meta.setdefault("stream_regions", [])
    for nodes in runs:
        if len(nodes) < 2:
            continue
        record = {"operations": [str(n.target) for n in nodes]}
        records.append(record)
        finish_padding = elide_contraction_padding(
            model, nodes, transactional=True
        )
        try:
            region = analyze_stream_region(nodes)
        except ValueError as e:
            finish_padding(False)
            record.update(status="fallback", reason=str(e))
            continue
        candidates = []
        for tile, _ in get_valid_tiling((region.rows,), exhaustive=True):
            for residency in ("resident", "stage"):
                candidates.append(
                    dict(
                        tile_rows=tile[0],
                        weight_residency=residency,
                        **evaluate(region, tile[0], residency),
                    )
                )
        record["candidates"] = candidates
        forced = model.meta.get("stream_region_choices")
        legal = [c for c in candidates if c.get("legal") and (
            forced is not None or discovery_only or math.isfinite(c["prediction_ns"])
        )]
        if not legal:
            finish_padding(False)
            record.update(
                status="fallback",
                reason="no legal whole-reduction stream tile fits",
            )
            continue
        if forced is not None:
            binding = forced[len(records) - 1]
            matches = [c for c in legal if binding is not None and all(
                c[k] == binding[k] for k in ("tile_rows", "weight_residency")
            )]
            if len(matches) != 1:
                raise ValueError("Expanded search selected a missing/illegal region candidate")
            selected = matches[0]
        elif discovery_only:
            # Only collect legality/choices; do not rank with the compact cost.
            selected = legal[0]
        else:
            selected = min(
                legal,
                key=lambda c: (
                    c["prediction_ns"],
                    -c["tile_rows"],
                    c["weight_residency"],
                ),
            )
        record.update(
            status="selected",
            selected=selected,
            rows=region.rows,
            batch=list(region.batch),
            internal_edges="SRAM",
            reduction_axes="whole",
            boundary_inputs=[n.name for n in region.inputs],
        )
        finish_padding(True)
        grouped = create_and_insert_subgraph(list(nodes), model)
        grouped.meta["stream_region"] = dict(
            tile_rows=selected["tile_rows"],
            rows=region.rows,
            batch=region.batch,
            weight_residency=selected["weight_residency"],
            roles=tuple(region.roles[n] for n in grouped.all_input_nodes),
        )
    model.graph.lint()
    model.recompile()


def local_reduction_scratch(op, shape, lanes):
    # Keep original rank while resolving a positive reduction-axis index, then
    # remove the independent batch dimensions for the local compute tile.
    if op.target is torch.ops.aten._softmax.default:
        return [
            (name, (shape[0], lanes), op.value.dtype) for name in ("max", "sum")
        ]
    full = (*op.shape[:-2], *shape)
    return [
        (name, size[-2:], dtype)
        for name, size, dtype in reduction_scratch(op, full, lanes)
    ]


def build_stream_region(node, *, tiler):
    from .pipeline import build_pipelined_buffers, _single_pass_kernel
    from .utils import _InputSpec, _OutputSpec, _ScratchSpec

    plan = node.meta["stream_region"]
    rows = plan["tile_rows"]
    batch = tuple(plan["batch"])
    nb = len(batch)
    sub = node.meta["submodule"]
    sources = list(node.all_input_nodes)
    placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
    graph = Graph()
    env = {}
    shapes = {}
    in_specs = []
    staged = {}
    for ph, source, role in zip(placeholders, sources, plan["roles"]):
        arg = graph.placeholder(ph.name)
        shape = tuple(source.shape)
        stage = role.matrix_rhs and plan["weight_residency"] == "stage"
        if stage:
            if role.batch:
                raise ValueError(
                    "stage-loaded batched weights are not yet supported"
                )
            in_specs.append(None)
            staged[ph] = arg
            shapes[ph] = shape[-2:]
            continue
        if role.row or role.matrix_rhs:
            has_batch = bool(role.batch)
            prefix = (1,) * len(role.batch)
            tile = (*prefix, rows if role.row else shape[-2], shape[-1])
            maps = (
                (*range(nb), nb if role.row else None, None)
                if has_batch
                else (nb if role.row else None, None)
            )
            bcast = (
                tuple(x == 1 and y != 1 for x, y in zip(role.batch, batch))
                + (False, False)
                if has_batch
                else (False, False)
            )
            in_specs.append(_InputSpec(tile, maps, bcast, num_slots=1))
            local = (rows if role.row else shape[-2], shape[-1])
            shapes[ph] = local
            env[ph] = (
                graph.call_function(
                    torch.ops.aten.reshape.default, (arg, local)
                )
                if has_batch
                else arg
            )
        else:
            in_specs.append(
                _InputSpec(
                    shape,
                    (None,) * len(shape),
                    (False,) * len(shape),
                    num_slots=1,
                )
            )
            env[ph] = arg
            shapes[ph] = shape
    scratch = []
    weight_storage = None
    if staged:
        size = max(math.prod(shapes[ph]) for ph in staged)
        weight_storage = graph.placeholder("stage_weight_storage")
        scratch.append(_ScratchSpec((size,), sources[0].value.dtype))
    operations = [n for n in sub.graph.nodes if n.op == "call_function"]
    destinations = {}
    for op in operations:
        shape = (rows, op.shape[-1])
        shapes[op] = shape
        allocs = (
            [] if op is operations[-1] else [("result", shape, op.value.dtype)]
        )
        allocs += local_reduction_scratch(op, shape, tiler.config.vector_lanes)
        slots = {}
        for name, size, dtype in allocs:
            slots[name] = graph.placeholder(f"{op.name}_{name}_storage")
            scratch.append(_ScratchSpec(size, dtype))
        destinations[op] = slots
    for op in operations:
        if op.target in MATRIX and op.args[1] in staged:
            ph = op.args[1]
            shape = shapes[ph]
            elements = math.prod(shape)
            view = graph.call_function(
                torch.ops.voyager.subview.default,
                (weight_storage, [0], [elements], [1]),
            )
            view = graph.call_function(
                torch.ops.aten.reshape.default, (view, shape)
            )
            sem = graph.call_function(
                torch.ops.voyager.zeros.default, ([], torch.int64)
            )
            graph.call_function(
                torch.ops.voyager.async_copy.default,
                (staged[ph], view, [], list(shape), sem, []),
            )
            graph.call_function(torch.ops.voyager.async_wait.default, (sem,))
            env[ph] = view
        args = map_arg(op.args, lambda n: env[n])
        kwargs = dict(map_arg(op.kwargs, lambda n: env[n]))
        slots = destinations[op]
        target = reduction_op(op) or op.target
        if op.target in MATRIX:
            target = torch.ops.aten.matmul.default
        if op.target in SOFTMAX:
            target = torch.ops.quantized_ops.softmax.default
            args = (env[op.args[0]], -1)
            kwargs.pop("dim", None)
            kwargs.pop("half_to_float", None)
        kwargs.update({k: v for k, v in slots.items() if k != "result"})
        computed = graph.call_function(target, args, kwargs)
        if "result" in slots:
            graph.call_function(
                torch.ops.voyager.insert.default, (computed, slots["result"])
            )
        env[op] = slots.get("result", computed)
    result = env[operations[-1]]
    tile_shape = (*((1,) * nb), rows, node.shape[-1])
    if nb:
        result = graph.call_function(
            torch.ops.aten.reshape.default, (result, tile_shape)
        )
    graph.output(result)
    compute = GraphModule({}, graph)
    return build_pipelined_buffers(
        _single_pass_kernel(compute, 1, len(scratch)),
        (*batch, plan["rows"] // rows),
        in_specs,
        [
            _OutputSpec(
                tuple(node.shape),
                tile_shape,
                (*range(nb), nb, None),
                node.value.dtype,
                num_slots=1,
            )
        ],
        tuple(n.value.clone() for n in sources),
        scratch_specs=scratch,
        num_slots=1,
    )
