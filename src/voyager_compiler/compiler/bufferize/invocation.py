"""Native instruction bodies with explicit external effects and private fit."""

from dataclasses import replace

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from voyager_compiler.ir.plan import BufferAllocation
from voyager_compiler.shape_prop import ShapeProp, fake_like


def build_native(node, invocation):
    """A synchronous macro performs its declared DMA and private execution.

    The original operation is its executable host semantics. Native resources
    are capacity checked before freeze and recorded separately from addresses.
    """
    if node.op != "call_function" or invocation.allocation_owner != "native":
        raise ValueError(
            "native builder requires a semantic call and native allocation contract"
        )
    graph = Graph()
    bindings = {n: graph.placeholder(n.name) for n in node.all_input_nodes}
    value = node.value
    if not isinstance(value, torch.Tensor):
        raise ValueError("native template currently requires one tensor result")
    destination = graph.call_function(
        torch.ops.voyager.alloc.default, (list(value.shape), value.dtype, 3)
    )
    compute = graph.call_function(
        node.target,
        map_arg(node.args, bindings.__getitem__),
        map_arg(dict(node.kwargs), bindings.__getitem__),
    )
    compute.meta.update(node.meta)
    compute.meta["target_invoke"] = invocation
    graph.call_function(
        torch.ops.voyager.insert.default, (compute, destination)
    )
    graph.output((destination,))
    module = GraphModule(torch.nn.Module(), graph)
    ShapeProp(module).propagate(*(fake_like(n.value) for n in bindings))
    return module


def finalize_invocations(program, config, allocation):
    """Resolve effects to final graph names and check private capacities.

    Hardware scheduling never licenses unaccounted memory. Native reservations
    require exclusive use of their arenas during the invocation. Compiler
    allocations may use those arenas before or after their private lifetime.
    """
    buffers = list(allocation.buffers)
    from voyager_compiler.compiler.allocate import (
        MemoryCapacityError,
        _buffer_identity,
        _buffer_lifetimes,
        _timestamps,
    )

    order = _timestamps(program.graph)
    live_buffers = _buffer_lifetimes(
        program.graph, _buffer_identity(program.graph), order, config
    )
    for scope, module in program.graph.named_modules():
        if not isinstance(module, GraphModule):
            continue
        qualified = lambda n: f"{scope}/{n.name}"
        for node in module.graph.nodes:
            invocation = node.meta.get("target_invoke")
            if invocation is None:
                continue
            stores = [
                u.args[1]
                for u in node.users
                if u.target == torch.ops.voyager.insert.default
            ]
            if not stores:
                # Tuple and fused regions expose getitems before their stores.
                stores = [
                    u.args[1]
                    for handle in node.users
                    for u in handle.users
                    if u.target == torch.ops.voyager.insert.default
                ]
            schema = getattr(node.target, "_schema", None)
            if schema is not None:
                for index, argument in enumerate(schema.arguments):
                    if (
                        argument.alias_info is not None
                        and argument.alias_info.is_write
                    ):
                        value = (
                            node.args[index]
                            if index < len(node.args)
                            else node.kwargs.get(argument.name)
                        )
                        if isinstance(value, Node):
                            stores.append(value)
            node.meta["target_invoke"] = replace(
                invocation,
                operands=tuple(qualified(n) for n in node.all_input_nodes),
                reads=tuple(qualified(n) for n in node.all_input_nodes),
                writes=tuple(dict.fromkeys(qualified(n) for n in stores)),
            )
            demand = {}
            for requirement in invocation.native_requirements:
                demand[requirement.space] = (
                    demand.get(requirement.space, 0)
                    + requirement.bytes * requirement.slots
                )
            for name, size in demand.items():
                memory = config.memory_instance(name)
                if any(
                    root.meta.get("memory_instance", "scratchpad") == name
                    and buf.def_t <= order[node] <= buf.last_t
                    for root, buf in live_buffers.items()
                ):
                    raise MemoryCapacityError(
                        f"{node.name}: native and live compiler allocations overlap in {name}",
                        sites=(
                            (node.meta["scope"][0],)
                            if "scope" in node.meta
                            else ()
                        ),
                    )
                if (
                    memory.size.unit.value != "bytes"
                    or memory.size.value is None
                    or size > memory.size.value
                ):
                    raise ValueError(
                        f"{node.name}: native {name} demand {size} exceeds capacity"
                    )
                buffers.append(
                    BufferAllocation(
                        f"{qualified(node)}/{name}",
                        "Local",
                        None,
                        size,
                        owner="native",
                        memory_instance=name,
                    )
                )
    return replace(allocation, buffers=tuple(buffers))
