"""Alpha's inference boundary and kernels in the shared compiler."""

import torch

from voyager_compiler.compiler.bufferize.residency import build_whole
from voyager_compiler.extensions import (
    ImplementationSpec,
    OperatorSemantics,
    RuntimeModel,
    TargetSpec,
    registry,
)

from . import (  # noqa: F401 — register Torch operator schemas
    operators,
    quant_ops,
)
from .codegen import emit, encode
from .hardware import MagnetAlphaHardwareConfig


class AlphaModel(RuntimeModel):
    """Whole-region lowering; device timing is not yet characterized."""

    def prepare(self, node):
        return None


def legalize(prepared, config, options):
    from voyager_compiler.shape_prop import ShapeProp

    from .graph import apply_graph_rewrites, apply_memory_rewrites
    from .quantization import apply_quantization_rewrites

    prepared.graph.meta["target"] = "magnet-alpha"
    ShapeProp(prepared.graph).propagate(*prepared.args)
    apply_graph_rewrites(prepared.graph)
    apply_quantization_rewrites(prepared.graph)
    ShapeProp(prepared.graph).propagate(*prepared.args)
    with torch.no_grad():
        prepared.reference = prepared.graph(*prepared.args, **prepared.kwargs)
    from .padding import pad_matrix_op_dimensions

    pad_matrix_op_dimensions(prepared.graph, hardware_config=config)
    ShapeProp(prepared.graph).propagate(*prepared.args)
    apply_memory_rewrites(prepared.graph, config)
    from voyager_compiler.codegen.transform.rewrites import (
        extract_input_preprocessor,
    )

    prepared.graph, preprocessor = extract_input_preprocessor(
        prepared.graph, additional_ops=(torch.ops.alpha_quant.quantize.default,)
    )
    prepared.args = (preprocessor(prepared.args[0]), *prepared.args[1:])
    prepared.provenance["input_preprocessor"] = str(preprocessor.graph)
    for node in prepared.graph.graph.nodes:
        if node.op == "get_attr" and (
            node.meta.get("magnet_alpha_instruction_operand")
            or node.meta.get("magnet_alpha_semantic_operand")
            or node.meta.get("compiler_only")
        ):
            node.meta["instruction_constant"] = True
    from .implementations import form_execution_regions

    form_execution_regions(prepared.graph)


OPERATOR_NAMES = tuple(
    name
    for name in torch._C._dispatch_get_all_op_names()
    if name.startswith(("magnet_alpha::", "alpha_quant::"))
)
OPERATORS = tuple(
    (
        getattr(
            getattr(torch.ops, name.split("::")[0]), name.split("::")[1]
        ).default,
        OperatorSemantics(compute=name != "alpha_quant::dequantize"),
    )
    for name in sorted(OPERATOR_NAMES)
)


def refine(program, config):
    from voyager_compiler.compiler.allocate import _buffer_identity, _walk
    from voyager_compiler.compiler.bufferize.residency import refine_residency

    identities = _buffer_identity(program.graph)
    for node in _walk(program.graph):
        if (
            node.meta.get("space") == "Scratchpad"
            and node.target == torch.ops.voyager.alloc.default
        ):
            node.meta["memory_instance"] = "global_activation"
        if node.target == torch.ops.voyager.async_copy.default:
            source, destination = node.args[:2]
            source = identities.get(source, source)
            destination = identities.get(destination, destination)
            if (
                source.op == "get_attr"
                and destination.meta.get("space") == "Scratchpad"
            ):
                destination.meta["memory_instance"] = "global_weight"
    from .layout import prepare_shuffles

    prepare_shuffles(program.graph)
    refine_residency(program, config, layout=bank_layouts)
    from .schedule import select_schedules

    select_schedules(program, config)


def bank_layouts(model):
    from dataclasses import replace

    from voyager_compiler.compiler.allocate import _buffer_identity, _walk
    from voyager_compiler.compiler.search.lowering import node_dtype_bits
    from voyager_compiler.ir.plan import BankLayout

    identities = _buffer_identity(model)
    from .layout import (
        operand_bindings,
        output_buffers,
        packed_components,
        region_nodes,
    )

    # NMP weight ports address global activation memory. Declare that choice
    # before the ordinary allocator, including when the logical value is a parameter.
    nmp_buffers = set()
    for call in model.graph.nodes:
        if (
            call.meta.get("target_invoke") is not None
            and call.meta["target_invoke"].opcode == "depthwise_conv2d"
        ):
            bindings = operand_bindings(call)
            conv = next(
                n
                for n in region_nodes(call)
                if n.target
                == torch.ops.magnet_alpha.conv2d_int8_accumulator.default
            )
            weight = identities.get(
                bindings[conv.args[1]], bindings[conv.args[1]]
            )
            weight.meta["memory_instance"] = "global_activation"
            nmp_buffers.update(
                identities.get(bindings[n], bindings[n]) for n in conv.args[:2]
            )
            nmp_buffers.add(output_buffers(call)[-1])
    packed_components(model)
    for node in _walk(model):
        if (
            node.target == torch.ops.voyager.alloc.default
            and node.meta.get("space") == "Scratchpad"
        ):
            activation = node.meta["memory_instance"] == "global_activation"
            width = node_dtype_bits(node)
            shape = tuple(node.value.shape)
            cluster = 1 if not activation or width <= 4 else 2
            if activation and len(shape) == 4 and width == 8 and shape[1] == 16:
                cluster = 1
            if node in nmp_buffers and len(shape) == 4 and width == 8:
                cluster = 2
                if shape[1] != 1:
                    shape = (shape[0], ((shape[1] + 31) // 32) * 32, *shape[2:])
                else:
                    shape = (((shape[0] + 31) // 32) * 32, *shape[1:])
            node.meta["physical_shape"] = shape
            node.meta["buffer_layout_key"] = (
                shape,
                cluster,
                (
                    "alpha.vsq_embedded"
                    if node.meta.get("physical_component") is not None
                    else "payload"
                ),
            )
            node.meta["bank_layout"] = BankLayout(
                cluster, (0, 1) if activation else ()
            )
    for node in model.graph.nodes:
        if "target_invoke" not in node.meta and not str(node.target).startswith(
            "magnet_alpha."
        ):
            continue
        stores = [
            u.args[1]
            for u in node.users
            if u.target == torch.ops.voyager.insert.default
        ]
        stores += [
            u.args[1]
            for handle in node.users
            for u in handle.users
            if u.target == torch.ops.voyager.insert.default
        ]
        operands = {
            identities.get(n, n) for n in (*node.all_input_nodes, *stores)
        }
        operands = {n for n in operands if "bank_layout" in n.meta}
        for operand in operands:
            peers = {
                n.name
                for n in operands
                if n is not operand
                and n.meta["memory_instance"] == operand.meta["memory_instance"]
            }
            layout = operand.meta["bank_layout"]
            operand.meta["bank_layout"] = replace(
                layout,
                avoid_buffers=tuple(sorted(set(layout.avoid_buffers) | peers)),
            )


def build_alpha(node, context, parameters):
    immediate = tuple(
        n.name
        for n in node.all_input_nodes
        if n.meta.get("magnet_alpha_instruction_operand")
        or n.meta.get("magnet_alpha_semantic_operand")
        or n.meta.get("compiler_only")
    )
    return build_whole(node, context, {"immediate_operands": immediate})


from .quantization_flow import POLICY as QUANTIZATION_POLICY

TARGET = registry.register(
    TargetSpec(
        name="magnet-alpha",
        hardware=MagnetAlphaHardwareConfig,
        model=AlphaModel,
        implementations=(
            ImplementationSpec(
                "magnet-alpha.whole",
                lambda node, anchor: True,
                build_alpha,
                description="Whole tensor kernel with explicit shared buffer movement",
            ),
        ),
        quantization=QUANTIZATION_POLICY,
        legalize=legalize,
        operators=OPERATORS,
        refine=refine,
        encode=encode,
        emit=emit,
    )
)
