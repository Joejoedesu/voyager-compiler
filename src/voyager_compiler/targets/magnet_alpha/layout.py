"""Alpha layouts expressed on explicit buffers before common allocation."""

import operator

import torch

from voyager_compiler.ir.buffer import TargetInvoke
from voyager_compiler.ir.plan import BufferComponent
from voyager_compiler.shape_prop import fake_like

from .graph import shuffle_copy_movements


def region_nodes(node):
    return (
        tuple(node.meta["submodule"].graph.nodes)
        if node.op == "call_module"
        else (node,)
    )


def operand_bindings(node):
    if node.op != "call_module":
        return {}
    placeholders = [n for n in region_nodes(node) if n.op == "placeholder"]
    return dict(zip(placeholders, node.args))


def output_buffers(node):
    ops = torch.ops.voyager
    direct = [u.args[1] for u in node.users if u.target == ops.insert.default]
    if direct:
        return direct
    handles = sorted(
        (u for u in node.users if u.target == operator.getitem),
        key=lambda n: n.args[1],
    )
    return [
        u.args[1]
        for h in handles
        for u in h.users
        if u.target == ops.insert.default
    ]


def prepare_shuffles(model):
    """Expose secondary storage and in-place copies in the executable graph."""
    graph, ops = model.graph, torch.ops.voyager
    shuffles = {}
    for node in list(graph.nodes):
        if node.target == torch.ops.magnet_alpha.channel_shuffle.default:
            outputs = output_buffers(node)
            if len(outputs) != 1:
                raise ValueError(
                    f"{node.name}: shuffle requires one primary buffer"
                )
            primary = outputs[0]
            with graph.inserting_before(node):
                secondary = graph.create_node(
                    "call_function",
                    ops.alloc.default,
                    (list(node.value.shape), node.value.dtype, 2),
                    {},
                    f"{node.name}_secondary",
                )
            secondary.meta.update(
                space="Scratchpad",
                memory_instance="global_activation",
                dtype="int8",
            )
            secondary.value = fake_like(node.value)
            secondary.shape = node.value.shape
            logical = node.meta["scope"][0]
            shuffles[logical] = (node, primary, secondary)
            node.target = (
                torch.ops.magnet_alpha.channel_shuffle_buffered.default
            )
            node.args = (node.args[0], node.args[1], secondary, node.args[2])
            node.meta["target_invoke"] = TargetInvoke(
                "magnet-alpha.whole",
                "channel_shuffle",
                (),
                (),
                (),
                ("nmp",),
                scheduling_owner="hardware",
            )
            node.meta["target_layout"] = dict(
                kind="alpha.shuffle",
                secondary=secondary.name,
                primary=primary.name,
                channel_pitch=int(node.value.shape[1]),
            )
        elif node.target == torch.ops.magnet_alpha.channel_shuffle_copy.default:
            facts = node.meta["magnet_alpha"]
            parent, primary, secondary = shuffles[facts["logical_parent"]]
            movements = shuffle_copy_movements(int(node.value.shape[1]) // 32)
            movement = movements[int(facts["copy_index"]) - 1]
            (destination,) = output_buffers(node)
            # The physical copy changes a block in the same primary backing.
            # Keep every later read/store tied to that backing, including spills.
            destination.replace_all_uses_with(primary)
            if not destination.users:
                graph.erase_node(destination)
            node.target = (
                torch.ops.magnet_alpha.channel_shuffle_copy_buffered.default
            )
            node.args = (primary, secondary)
            node.meta["target_invoke"] = TargetInvoke(
                "magnet-alpha.whole",
                "copy",
                (),
                (),
                (),
                ("nmp",),
                scheduling_owner="hardware",
            )
            node.meta["target_layout"] = dict(
                kind="alpha.shuffle_copy",
                primary=primary.name,
                secondary=secondary.name,
                source_output=movement.source_output,
                source_entry=movement.source_entry,
                destination_entry=movement.destination_entry,
                block_channels=32,
                channel_pitch=int(node.value.shape[1]),
            )
    graph.lint()
    model.recompile()


def packed_components(model):
    """Bind fine-scale buffers to their payload's encoded physical backing."""
    from voyager_compiler.compiler.allocate import _buffer_identity, _walk

    identity = _buffer_identity(model)
    for node in _walk(model):
        node.meta.pop("physical_component", None)

    def bind(scale, payload):
        scale, payload = identity.get(scale, scale), identity.get(
            payload, payload
        )
        if (
            scale.meta.get("space") != "Scratchpad"
            or payload.meta.get("space") != "Scratchpad"
        ):
            return
        part = BufferComponent(
            "/" + payload.name, "alpha.vsq_embedded", "fine_scale"
        )
        previous = scale.meta.get("physical_component")
        if previous is not None and previous != part:
            raise ValueError(
                f"{scale.name}: fine scale has competing physical backings"
            )
        scale.meta["physical_component"] = part
        payload.meta["target_layout"] = dict(
            kind="alpha.vsq_embedded", block_values=32
        )

    for call in model.graph.nodes:
        bindings = operand_bindings(call)
        for inner in region_nodes(call):
            if inner.target == torch.ops.magnet_alpha.conv2d_vsq.default:
                for scale_name, index in (
                    ("input_scale", 0),
                    ("weight_scale", 1),
                ):
                    scale = inner.kwargs.get(scale_name)
                    if scale is not None:
                        bind(
                            bindings.get(scale, scale),
                            bindings.get(inner.args[index], inner.args[index]),
                        )
            if inner.target == torch.ops.magnet_alpha.pvs_quantize.default:
                outputs = output_buffers(call)
                if len(outputs) != 3:
                    raise ValueError(
                        f"{call.name}: PVS needs coarse, fine and payload outputs"
                    )
                bind(outputs[1], outputs[2])
