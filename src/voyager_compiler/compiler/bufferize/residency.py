"""Whole-tensor kernels and proven SRAM store/load elimination.

This is a shared builder/refinement over the same alloc/copy/insert operations
as tiled kernels. It does not introduce another execution or allocation IR.
"""

import copy
import operator

import torch
from torch.fx import Graph, GraphModule
from torch.fx.node import map_arg

from voyager_compiler.codegen.node_info import quant_param_arg_nodes
from voyager_compiler.shape_prop import ShapeProp, fake_like

from .ops import MemoryLevel


def build_whole(node, context, parameters=None):
    """Single synchronous invocation with explicit whole-buffer transfers."""
    graph, module, bindings = Graph(), torch.nn.Module(), {}
    ops = torch.ops.voyager
    codebooks = quant_param_arg_nodes(node)
    immediate = set((parameters or {}).get("immediate_operands", ()))
    inputs = node.all_input_nodes

    def alloc(value, space, name):
        return graph.create_node(
            "call_function",
            ops.alloc.default,
            (list(value.shape), value.dtype, int(space)),
            {},
            name,
        )

    def copy_whole(source, destination, value):
        semaphore = graph.call_function(ops.zeros.default, ([], torch.int64))
        transfer = graph.call_function(
            ops.async_copy.default,
            (
                source,
                destination,
                [0] * value.ndim,
                list(value.shape),
                semaphore,
            ),
        )
        wait = graph.call_function(ops.async_wait.default, (semaphore,))
        transfer.meta["whole_copy"] = True
        wait.meta["whole_copy_wait"] = True

    placeholders = {n: graph.placeholder(n.name) for n in inputs}
    for source, placeholder in placeholders.items():
        value = source.value
        if (
            source in codebooks
            or source.name in immediate
            or not isinstance(value, torch.Tensor)
            or value.numel() == 1
        ):
            bindings[source] = placeholder
            continue
        destination = alloc(value, MemoryLevel.SRAM, f"{source.name}_tile")
        copy_whole(placeholder, destination, value)
        bindings[source] = destination

    args = map_arg(node.args, bindings.__getitem__)
    kwargs = map_arg(dict(node.kwargs), bindings.__getitem__)
    values = (
        list(node.value)
        if isinstance(node.value, (list, tuple))
        else [node.value]
    )
    # Native output ports write during the invocation. Reserve their buffers
    # before it so the allocator cannot reuse a dying input for an output.
    tiles = [
        alloc(value, MemoryLevel.SRAM, f"result_{index}_tile")
        for index, value in enumerate(values)
    ]
    if node.op == "call_module":
        module.add_module("kernel", copy.deepcopy(node.meta["submodule"]))
        computed = graph.call_module("kernel", args, kwargs)
    else:
        computed = graph.call_function(node.target, args, kwargs)
    computed.meta.update(node.meta)
    result = []
    for index, (value, tile) in enumerate(zip(values, tiles)):
        output = (
            graph.call_function(operator.getitem, (computed, index))
            if len(values) > 1
            else computed
        )
        graph.call_function(ops.insert.default, (output, tile))
        destination = alloc(value, MemoryLevel.DRAM, f"result_{index}")
        copy_whole(tile, destination, value)
        result.append(destination)
    graph.output(tuple(result))
    gm = GraphModule(module, graph)
    ShapeProp(gm).propagate(*(fake_like(n.value) for n in inputs))
    return gm


def merge_resident(model, *, limit=None):
    """Elide matching complete-buffer store/load pairs in sequential regions.

    Only copies introduced with an immediate completion wait qualify. Partial
    windows, loops, host views and externally visible stores remain explicit.
    The caller must allocate the resulting graph before accepting it.
    """
    graph, ops, stores, merged = model.graph, torch.ops.voyager, {}, 0
    for transfer in list(graph.nodes):
        if (
            not transfer.meta.get("whole_copy")
            or transfer.target != ops.async_copy.default
        ):
            continue
        source, destination, _, shape, semaphore = transfer.args[:5]
        if (
            source.meta.get("space") == "Scratchpad"
            and destination.meta.get("space") == "DRAM"
            and destination.target == ops.alloc.default
        ):
            # A store into a view remains observable through its backing even
            # when the view itself has no other users (e.g. a concat window).
            stores[destination] = (source, transfer)
            continue
        if (
            source not in stores
            or destination.meta.get("space") != "Scratchpad"
        ):
            continue
        if limit is not None and merged >= limit:
            break
        resident, store = stores[source]
        if tuple(resident.value.shape) != tuple(shape):
            continue
        if resident.meta.get("memory_instance") != destination.meta.get(
            "memory_instance"
        ):
            continue
        if resident.meta.get("buffer_layout_key") != destination.meta.get(
            "buffer_layout_key"
        ):
            continue
        # The producer's completion precedes this load in program order.
        waits = [u for u in semaphore.users if u.meta.get("whole_copy_wait")]
        if len(waits) != 1 or set(semaphore.users) != {transfer, waits[0]}:
            continue
        for user in list(destination.users):
            if user is not transfer:
                user.replace_input_with(destination, resident)
        graph.erase_node(waits[0])
        graph.erase_node(transfer)
        graph.erase_node(semaphore)
        if not destination.users:
            graph.erase_node(destination)
        merged += 1
    # A store is redundant only when all consumers were replaced, and the
    # graph output or a host view no longer refers to its DRAM destination.
    for destination, (_, transfer) in stores.items():
        if set(destination.users) != {transfer}:
            continue
        semaphore = transfer.args[4]
        waits = [u for u in semaphore.users if u.meta.get("whole_copy_wait")]
        if len(waits) != 1 or set(semaphore.users) != {transfer, waits[0]}:
            continue
        graph.erase_node(waits[0])
        graph.erase_node(transfer)
        graph.erase_node(semaphore)
        graph.erase_node(destination)
    graph.lint()
    model.recompile()
    return merged


def refine_residency(program, config, *, layout=None):
    """Accept SRAM merging only after expanded physical placement succeeds.

    Reduce the merge budget on overflow, retaining explicit transfers for the
    rest. Every accepted graph is checked by the ordinary allocator.
    """
    from voyager_compiler.compiler.allocate import (
        MemoryCapacityError,
        plan_memory,
    )

    model = program.graph
    tensors = (*model.parameters(), *model.buffers())
    memo = {id(t): t for t in tensors}
    original = copy.deepcopy(model, memo)
    limit = None
    while True:
        if layout is not None:
            layout(model)
        merged = merge_resident(model, limit=limit)
        if layout is not None:
            layout(model)
        try:
            plan_memory(model, config)
        except MemoryCapacityError:
            if not merged:
                raise
            restored = copy.deepcopy(original, memo.copy())
            model.__dict__.update(restored.__dict__)
            model.graph = restored.graph
            model.recompile()
            limit = merged // 2
            continue
        program.validation["resident_merges"] = merged
        return
