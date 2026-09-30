"""SRAM placement proposals, scheduled before per-kernel DMA construction.

Each region keeps the existing ISA operations and execution order. Distinct
live storage objects occupy disjoint whole banks: a conservative placement
that satisfies every member's simultaneous operand accesses. An infeasible
extension closes the last feasible region; unsupported kernels use the normal
bufferizer. No SRAM-to-SRAM DMA or new instruction fusion is invented here.
"""

import math
from dataclasses import replace

import torch
from torch.fx import Graph, GraphModule

from voyager_compiler.codegen.node_info import (
    _pair,
    bound_operands,
    csr_quantize_node,
    get_anchor_node,
    get_arg_value,
    is_conv2d,
    is_elementwise_op,
    is_gemm_op,
    is_nop,
    is_pooling,
    reduction_scratch,
)
from voyager_compiler.codegen.subgraph import (
    copy_graph_module,
    replace_node_with_graph_module,
)
from voyager_compiler.codegen.transform.bufferize.memory_planning import (
    MemoryPlanningError,
    _buffer_identity,
    _buffer_lifetimes,
    _greedy_best_fit,
    _timestamps,
    plan_memory,
)
from voyager_compiler.codegen.transform.bufferize.ops import MemoryLevel
from voyager_compiler.codegen.transform.bufferize.utils import (
    _collect_codebook_nodes,
    _passed_whole,
)
from voyager_compiler.shape_prop import set_node_value

voyager = torch.ops.voyager


def clone_graph(model):
    """Copy graph structure/metadata, sharing immutable parameters and modules."""
    graph, env = Graph(), {}
    for node in model.graph.nodes:
        new = graph.node_copy(node, env.__getitem__)
        env[node] = new
        if hasattr(node, "value"):
            set_node_value(new, node.value)
    result = GraphModule(model, graph)
    result.meta = dict(model.meta)
    return result


def _tensor(node):
    return isinstance(getattr(node, "value", None), torch.Tensor)


def _kernel_owned(value, sub):
    """An in-place ISA tail may write an intermediate made inside that ISA."""
    while isinstance(value, torch.fx.Node) and value.graph is sub.graph:
        if value.op != "call_function":
            return False
        schema = getattr(value.target, "_schema", None)
        if is_nop(value) or (
            schema and any(r.alias_info for r in schema.returns)
        ):
            value = value.args[0] if value.args else None
            continue
        return schema is not None
    return False


def _view_reason(node):
    source = node.args[0]
    if (
        not node.value.is_contiguous()
        or not source.value.is_contiguous()
        or node.value.numel() != source.value.numel()
        or node.value.storage_offset() != source.value.storage_offset()
    ):
        return "view requires a layout or offset transformation"
    # Copying a view into a region must not detach a later write from the
    # original storage. Follow alias users until any external write is found.
    todo, seen = [node], set()
    while todo:
        alias = todo.pop()
        if alias in seen:
            continue
        seen.add(alias)
        for user in alias.users:
            schema = getattr(user.target, "_schema", None)
            if not schema:
                continue
            for index, arg in enumerate(schema.arguments):
                if arg.alias_info and arg.alias_info.is_write:
                    value = get_arg_value(user, index, arg.name)
                    if any(
                        v is alias
                        for v in torch.utils._pytree.tree_leaves(value)
                    ):
                        return "view aliases an external mutation"
            if any(r.alias_info for r in schema.returns):
                todo.append(user)
    return None


def _reason(node, tiler):
    """Check supported access semantics, independently of streaming tiling."""
    if not _tensor(node):
        return "non-tensor or multiple-output kernel"
    if node.value.numel() == 1:
        return "scalar result uses the original scalar path"
    anchor = get_anchor_node(node)
    if anchor is None:
        return "missing ISA analysis"
    if any(not _tensor(n) for n in node.all_input_nodes):
        return "dynamic operand"
    if is_nop(node):
        return _view_reason(node)
    sub = node.meta.get("submodule")
    ops = sub.graph.nodes if isinstance(sub, GraphModule) else (node,)
    for op in ops:
        schema = getattr(op.target, "_schema", None)
        if schema:
            for index, arg in enumerate(schema.arguments):
                if arg.alias_info and arg.alias_info.is_write:
                    value = get_arg_value(op, index, arg.name)
                    if not isinstance(sub, GraphModule) or not _kernel_owned(
                        value, sub
                    ):
                        return "mutable external operand"
    if (
        csr_quantize_node(node) is not None
        or anchor.kwargs.get("A_indptr") is not None
    ):
        return "sparse kernel requires its specialized bufferizer"
    if reduction_scratch(
        node, tuple(node.value.shape), tiler.config.vector_lanes
    ):
        return (
            "explicit reduction workspace requires its specialized bufferizer"
        )
    if is_conv2d(anchor) or is_pooling(anchor):
        position = 4 if is_conv2d(anchor) else 3
        padding = get_arg_value(anchor, position, "padding", 0)
        padded = any(padding) if isinstance(padding, (tuple, list)) else padding
        if padded and is_pooling(anchor):
            return "pool padding requires a halo transfer"
        if padded:
            # Matrix InputController and DwCUnit generate boundary zeros from
            # an unpadded SRAM operand. Keep their native geometry rather than
            # asking the DMA builder to materialize a halo.
            ph, pw = _pair(padding)
            if _pair(get_arg_value(anchor, 5, "dilation", 1)) != (1, 1):
                return "resident hardware padding does not support dilation"
            if ph != pw or not 0 <= ph <= 3:
                return (
                    "resident hardware padding requires symmetric padding <= 3"
                )
            sh, sw = _pair(get_arg_value(anchor, 3, "stride", 1))
            if sh != sw:
                return (
                    "resident hardware padding requires equal spatial strides"
                )
            groups = get_arg_value(anchor, 6, "groups", 1)
            shape = anchor.args[0].value.shape
            nhwc = anchor.meta.get("transposed", False)
            channels = shape[-1] if nhwc else shape[1]
            if groups != 1 and (
                groups != channels
                or tuple(anchor.args[1].value.shape) != (channels, 1, 3, 3)
            ):
                return "resident hardware padding requires dense or 3x3 depthwise convolution"
            h, w = shape[1:3] if nhwc else shape[-2:]
            if groups != 1 and (
                sh >= 7 or (h + 2 * ph - 2) % sh or (w + 2 * pw - 2) % sw
            ):
                return "resident depthwise padding geometry is unsupported"
            code = anchor.kwargs.get("input_code")
            if code is not None:
                return "resident hardware padding with input codebooks is unsupported"
    if not (
        is_gemm_op(anchor) or is_pooling(anchor) or is_elementwise_op(anchor)
    ):
        return "unsupported whole-tensor ISA/layout"
    # Scheduling a DRAM sweep is not evidence that SRAM retention is
    # infeasible. First propose storage; schedule under that assumption below.
    return None


def _build_region(model, nodes, region_id, strategy, streamed=()):
    """Build a provisional region without modifying the source graph."""
    from voyager_compiler.codegen.transform.bufferize.bufferization import (
        annotate_tensor_spaces,
    )

    graph, env, resident = Graph(), {}, {}
    members = set(nodes)
    inputs = list(
        dict.fromkeys(
            n for op in nodes for n in op.all_input_nodes if n not in members
        )
    )
    outputs = [n for n in nodes if any(u not in members for u in n.users)]
    codebooks = _collect_codebook_nodes(model)
    # Preloads and boundary stores can lie outside an individual kernel's
    # compute run. Emission therefore treats the entire region as one layer.
    scope = (region_id, get_anchor_node(nodes[0]).target)

    def stamp(new, source=None, value=None):
        if source is not None:
            new.meta = dict(source.meta)
            value = getattr(source, "value", value)
        if value is not None:
            set_node_value(new, value)
        # Addresses belong to this placement, not to the source or a trial.
        for key in (
            "memory",
            "scratchpad",
            "slot_count",
            "slot_stride",
            "bank_group_stride",
            "space",
        ):
            new.meta.pop(key, None)
        new.meta["sram_region"] = region_id
        new.meta["sram_lowered"] = True
        new.meta["scope"] = scope
        return new

    def call(target, args, value=None):
        return stamp(graph.call_function(target, args), value=value)

    def alloc(source, space):
        result = call(
            voyager.alloc.default,
            (list(source.value.shape), source.value.dtype, int(space)),
            source.value,
        )
        if "dtype" in source.meta:
            result.meta["dtype"] = source.meta["dtype"]
        if space == MemoryLevel.SRAM:
            # Distinct live allocations never share banks. Reuse after last
            # access is handled by the existing lifetime allocator.
            result.meta["bank_group"] = result.name
        return result

    def transfer(source, dest, value):
        sem = call(
            voyager.zeros.default,
            ([1], torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )
        shape = list(value.shape)
        call(
            voyager.async_copy.default,
            (source, dest, [0] * len(shape), shape, sem),
        )
        call(voyager.async_wait.default, (sem,))

    for source in inputs:
        env[source] = stamp(graph.placeholder(source.name), source)
        env[source].meta["resident_preload"] = (
            strategy == "preload" and source.op == "get_attr"
        )
        if source in streamed:
            env[source].meta["resident_stream_weight"] = True

    def load(source):
        if source in resident:
            return resident[source]
        if source in streamed:
            resident[source] = env[source]
            return env[source]
        if not _tensor(source) or _passed_whole(source, codebooks):
            return env[source]
        buffer = alloc(source, MemoryLevel.SRAM)
        transfer(env[source], buffer, source.value)
        resident[source] = buffer
        return buffer

    if strategy == "preload":
        for source in inputs:
            if source.op == "get_attr":
                load(source)

    for node in nodes:
        for source in node.all_input_nodes:
            if source not in members:
                load(source)
        # Destination exists while the instruction reads its operands. Placing
        # its alloc after compute would incorrectly permit input/output aliasing.
        dest = None if is_nop(node) else alloc(node, MemoryLevel.SRAM)
        result = stamp(
            graph.node_copy(node, lambda n: resident.get(n, env.get(n))), node
        )
        result.meta["resident_kernel"] = node.name
        anchor = get_anchor_node(node)
        result.meta.update(anchor.meta.get("tiling", {}))
        # The matrix cost model must price the disjoint-bank placement, rather
        # than the old kernel-local shared-bank partition.
        groups = result.meta.get("bank_groups")
        if groups and all(
            isinstance(g, (set, frozenset, tuple, list)) for g in groups
        ):
            result.meta["bank_groups"] = tuple((r,) for g in groups for r in g)
        if is_nop(node):
            resident[node] = result
        else:
            call(voyager.insert.default, (result, dest))
            resident[node] = dest
        env[node] = result

    boundary = []
    for node in outputs:
        dest = alloc(node, MemoryLevel.DRAM)
        transfer(resident[node], dest, node.value)
        boundary.append(dest)
    graph.output(tuple(boundary))
    region = GraphModule(model, graph)
    region.meta["streamed_parameters"] = [n.name for n in streamed]
    annotate_tensor_spaces(region, validate_compute=not streamed)
    return region, inputs, outputs


def _clear_placement(region):
    # A proposal is allocated before and after scheduling. Addresses from its
    # first trial cannot survive insertion of reduction/staging buffers.
    for sub in region.modules():
        if isinstance(sub, GraphModule):
            for n in sub.graph.nodes:
                for key in (
                    "memory",
                    "scratchpad",
                    "slot_count",
                    "slot_stride",
                    "bank_group_stride",
                ):
                    n.meta.pop(key, None)


def _fit(region, config):
    _clear_placement(region)
    # Distinguish byte-capacity failure from a bank-granularity restriction.
    roots = _buffer_identity(region)
    bufs = _buffer_lifetimes(region, roots, _timestamps(region), config)
    _, packed = _greedy_best_fit(
        [(r, b.size, b.def_t, b.last_t, 1) for r, b in bufs.items()]
    )
    if packed + config.scratchpad_offset > config.scratchpad_size:
        return None, "SRAM capacity"
    try:
        plan = plan_memory(region, config)
    except MemoryPlanningError:
        return None, "bank allocation"
    return plan.scratchpad_bytes, None


def _schedule_region(region, tiler, cache):
    """Lower matrix kernels against a provisional, lifetime-checked placement.

    Boundary transfers use the same tile windows as their first/last compute.
    Matrix kernels reuse the normal reduction/tail builders, with tile windows
    replacing DMA slots. All live allocations (including unrelated branches)
    constrain the search; final allocation checks the resulting scratch too.
    """
    from voyager_compiler.codegen.transform.bufferize.bufferization import (
        annotate_tensor_spaces,
        propagate_logical_dtypes,
    )
    from voyager_compiler.codegen.transform.bufferize.pipeline import (
        build_conv2d,
        build_gemm,
        build_pointwise,
        build_pool,
    )

    region = copy_graph_module(region)
    for node in region.graph.nodes:
        if node.op == "call_module":
            node.meta["submodule"] = region.get_submodule(node.target)
    order = _timestamps(region)
    buffers = _buffer_lifetimes(
        region, _buffer_identity(region), order, tiler.config
    )

    def boundary_copy(buffer, incoming):
        return next(
            (
                n
                for n in region.graph.nodes
                if n.target is voyager.async_copy.default
                and n.args[1 if incoming else 0] is buffer
                and (not incoming or n.args[0].op == "placeholder")
            ),
            None,
        )

    def erase_transfer(copy):
        sem = copy.args[4]
        for u in list(sem.users):
            if u.target is voyager.async_wait.default:
                region.graph.erase_node(u)
        region.graph.erase_node(copy)
        if not sem.users:
            region.graph.erase_node(sem)

    for node in list(region.graph.nodes):
        if "resident_kernel" not in node.meta or is_nop(node):
            continue
        insert = next(
            u for u in node.users if u.target is voyager.insert.default
        )
        dest = insert.args[1]
        outgoing = boundary_copy(dest, False)
        incoming = {}
        operands = list(node.all_input_nodes)
        for i, operand in enumerate(operands):
            copy = boundary_copy(operand, True)
            if copy is None or copy.args[0].meta.get("resident_preload"):
                continue
            # Only the first consumer can replace the whole load. Later users
            # keep using the returned full SRAM allocation, without reloading.
            if all(
                u is copy or order.get(u, float("inf")) >= order[node]
                for u in operand.users
            ):
                incoming[i] = copy
        if (
            not is_gemm_op(get_anchor_node(node))
            and not incoming
            and outgoing is None
        ):
            continue
        node.meta["resident_ingress"] = tuple(incoming)
        node.meta["resident_egress"] = outgoing is not None
        # Quantized storage widths, store padding, and whole-bank rounding
        # come from the allocator, rather than the simulation tensor dtype.
        bank = tiler.config.bank_size or 1
        held = sum(
            math.ceil(b.size / bank) * bank
            for b in buffers.values()
            if b.def_t <= order[node] <= b.last_t
        )
        streaming = any(
            n.meta.get("resident_stream_weight") for n in node.all_input_nodes
        )
        constrained = replace(
            tiler, resident_bytes=held, stream_weights=streaming
        )
        key = (
            node.meta["resident_kernel"],
            held,
            streaming,
            tuple(incoming),
            outgoing is not None,
        )
        if key not in cache:
            anchor = get_anchor_node(node)
            builder = (
                build_conv2d
                if is_conv2d(anchor)
                else build_gemm
                if is_gemm_op(anchor)
                else build_pool
                if is_pooling(anchor)
                else build_pointwise
            )
            kwargs = {"async_pipeline": False} if is_gemm_op(anchor) else {}
            try:
                kernel = builder(node, tiler=constrained, num_slots=1, **kwargs)
            except RuntimeError as exc:
                if "no tiling fits on chip" not in str(exc):
                    raise
                return (
                    None,
                    "no resident schedule found within the live SRAM budget",
                )
            if kernel is None:
                return None, "unsupported resident kernel builder"
            phs = [p for p in kernel.graph.nodes if p.op == "placeholder"]
            sub = node.meta.get("submodule")
            ops = sub.graph.nodes if isinstance(sub, GraphModule) else (node,)
            propagate_logical_dtypes(
                kernel,
                {
                    p: n.meta.get("dtype")
                    for p, n in zip(phs, node.all_input_nodes)
                },
                {
                    n.target: n.meta["dtype"]
                    for n in ops
                    if n.meta.get("dtype") is not None
                },
            )
            cache[key] = kernel
        kernel = cache[key]
        if (
            not is_gemm_op(get_anchor_node(node))
            and kernel.meta.get("resident_steps") == 1
        ):
            # No overlap exists with one tile; preserve the simpler whole op.
            continue
        selected = kernel.meta.get("resident_ingress", ())
        for i in selected:
            node.replace_input_with(operands[i], incoming[i].args[0])
        region.graph.erase_node(insert)
        remap = {}
        result = replace_node_with_graph_module(
            region, node, kernel, propagate=False, value_remap=remap
        )
        output = result[0]
        for i, retained in zip(selected, result[1:]):
            erase_transfer(incoming[i])
            operands[i].replace_all_uses_with(retained)
            region.graph.erase_node(operands[i])
        if outgoing is not None:
            external = outgoing.args[1]
            erase_transfer(outgoing)
            external.replace_all_uses_with(result[-1])
            region.graph.erase_node(external)
        dest.replace_all_uses_with(output)
        region.graph.erase_node(dest)
        region.graph.erase_node(node)
        for old, new in remap.items():
            if old.op == "placeholder" or new is None:
                continue
            if hasattr(old, "value"):
                set_node_value(new, old.value)
            new.meta.update(
                sram_lowered=True,
                sram_region=node.meta["sram_region"],
                scope=node.meta["scope"],
            )
            if new.target is voyager.alloc.default:
                new.meta["bank_group"] = new.name
    region.graph.lint()
    region.recompile()
    annotate_tensor_spaces(region)
    return region, None


def _splice(model, nodes, built):
    region, inputs, outputs = built
    _clear_placement(region)
    mapping = dict(
        zip((n for n in region.graph.nodes if n.op == "placeholder"), inputs)
    )
    graph = model.graph
    with graph.inserting_before(nodes[-1]):
        for node in region.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                for old, new in zip(outputs, node.args[0]):
                    old.replace_all_uses_with(mapping[new])
                continue
            new = graph.node_copy(node, mapping.__getitem__)
            if node.op == "get_attr":
                # A scheduled region contains loop/cond bodies of its own.
                value = getattr(region, node.target)
                name = f"{node.meta.get('sram_region', 'resident')}_{node.name}"
                while hasattr(model, name):
                    name += "_"
                setattr(
                    model,
                    name,
                    copy_graph_module(value)
                    if isinstance(value, GraphModule)
                    else value,
                )
                new.target = name
            mapping[node] = new
            if hasattr(node, "value"):
                set_node_value(new, node.value)
            for key in (
                "memory",
                "scratchpad",
                "slot_count",
                "slot_stride",
                "bank_group_stride",
            ):
                new.meta.pop(key, None)
    for node in reversed(nodes):
        graph.erase_node(node)


def plan_resident_regions(model, tiler, parameter_loading="on_demand"):
    """Greedily grow connected regions in graph order and publish valid ones."""
    accepted, rejected, kernel_cache = [], [], {}
    config = tiler.config
    if config.scratchpad_size is None:
        model.meta["sram_regions"] = []
        model.meta["sram_fallbacks"] = [
            dict(reason="unspecified SRAM capacity")
        ]
        return
    pending, best, best_size = [], None, 0

    def attempt(nodes, streamed=()):
        built = _build_region(
            model, nodes, f"sram_{len(accepted)}", parameter_loading, streamed
        )
        size, reason = _fit(built[0], config)
        if reason:
            return built, size, reason
        scheduled, reason = _schedule_region(built[0], tiler, kernel_cache)
        if reason:
            return built, None, reason
        built = (scheduled, *built[1:])
        size, reason = _fit(scheduled, config)
        return built, size, reason

    def trial(nodes):
        result = attempt(nodes)
        if not result[2] or parameter_loading != "on_demand":
            return result
        # Bounded retry: keep activations pinned, but stage ordinary matrix
        # weights a tile at a time. A parameter used in any other role must
        # retain its ordinary allocation (MX/sparse roles are not generalized).
        weights = {}
        for n in nodes:
            anchor = get_anchor_node(n)
            if (
                anchor is not None
                and is_gemm_op(anchor)
                and anchor.kwargs.get("weight_scale") is None
            ):
                bound = bound_operands(n, n.meta.get("submodule"))
                weight = bound.get(anchor.args[1], anchor.args[1])
                if weight.op == "get_attr":
                    weights[n] = weight
        streamed = [
            w
            for w in dict.fromkeys(weights.values())
            if all(
                w not in n.all_input_nodes or weights.get(n) is w for n in nodes
            )
        ]
        return attempt(nodes, streamed) if streamed else result

    def finish():
        nonlocal pending, best
        # A view-only segment already costs no transfers on the original path.
        # Keep its aliases rather than introducing a pointless load/store pair.
        if best is not None and any(not is_nop(n) for n in pending):
            name = f"sram_{len(accepted)}"
            accepted.append(
                dict(
                    id=name,
                    nodes=[n.name for n in pending],
                    parameter_loading=parameter_loading,
                    sram_bytes=best_size,
                    streamed_parameters=best[0].meta.get(
                        "streamed_parameters", []
                    ),
                )
            )
            _splice(model, pending, best)
        pending, best = [], None

    for node in list(model.graph.nodes):
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":
            finish()
            break
        reason = _reason(node, tiler)
        if reason:
            finish()
            rejected.append(dict(node=node.name, reason=reason))
            continue
        if pending and not any(i in pending for i in node.all_input_nodes):
            # Sibling branches may join later. Keep them together when they
            # share an input, without moving unrelated work into a region.
            shared = {i for n in pending for i in n.all_input_nodes}
            if not shared.intersection(node.all_input_nodes):
                finish()
        candidate = pending + [node]
        built, size, reason = trial(candidate)
        if reason and pending:
            rejected.append(
                dict(node=node.name, reason=reason, action="split region")
            )
            finish()
            candidate = [node]
            built, size, reason = trial(candidate)
        if reason:
            rejected.append(
                dict(node=node.name, reason=reason, action="per_kernel")
            )
        else:
            pending, best, best_size = candidate, built, size
    finish()
    model.meta["sram_regions"] = accepted
    model.meta["sram_fallbacks"] = rejected
    model.graph.lint()
    model.recompile()
