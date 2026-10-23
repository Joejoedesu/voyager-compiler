# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gemmini rewrites that preserve shared PT2E quantization semantics."""

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.fake_quantize import FusedAmaxObsFakeQuantize
from voyager_compiler.quantization_contracts import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
    PreparedQuantizationContract,
)
from voyager_compiler.quantization_regions import (
    _QUANTIZATION_DOMAIN_ALIAS,
    _finalize_quantization_regions,
    _prepare_quantization_regions,
)

_QUANTIZE = torch.ops.quantized_ops.quantize.default
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_RELU = torch.ops.aten.relu_.default
_CODE_RELU = torch.ops.aten.relu.default
_MAX_POOL = torch.ops.aten.max_pool2d.default
_ADAPTIVE_AVG_POOL = torch.ops.aten.adaptive_avg_pool2d.default
_FLATTEN = torch.ops.aten.flatten.using_ints
_ADD_TARGETS = (torch.ops.aten.add_.Tensor,)
_LINEAR = torch.ops.aten.linear.default
_NORM_TARGETS = (
    torch.ops.aten.layer_norm.default,
    torch.ops.quantized_ops.layer_norm.default,
    torch.ops.aten.softmax.int,
)
_CONV2D_TARGETS = (
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
)


def _materialize_global_average_domain_boundaries(
    model: GraphModule,
) -> GraphModule:
    """Keep shared calibration while retaining explicit average Q/DQ."""
    average_regions = {
        node.meta.get(QUANTIZATION_REGION)
        for node in model.graph.nodes
        if node.op == "call_function"
        and node.target == _ADAPTIVE_AVG_POOL
        and node.meta.get(QUANTIZATION_REGION) is not None
    }
    if not average_regions:
        return model

    modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.meta.get(QUANTIZATION_REGION) not in average_regions:
            continue
        if node.op == "call_module" and node.meta.pop(
            _QUANTIZATION_DOMAIN_ALIAS,
            False,
        ):
            module = modules.get(str(node.target))
            if not isinstance(module, FusedAmaxObsFakeQuantize):
                raise TypeError(
                    "Gemmini average-domain alias must be an amax observer"
                )
            module.enable_fake_quant()
            continue
        if node.op == "call_function" and node.target in (
            _ADAPTIVE_AVG_POOL,
            _FLATTEN,
        ):
            node.meta.pop(QUANTIZATION_REGION, None)
            node.meta.pop(QUANTIZATION_DOMAIN_DTYPE, None)
            node.meta.pop(PHYSICAL_QUANTIZATION_CONTRACT, None)
    return model


def prepare_quantization(
    model: GraphModule,
    region_qspec: object | None = None,
    hardware_config: object | None = None,
    operator_traits: object | None = None,
) -> GraphModule:
    """Apply Gemmini's prepared quantization-domain transfer policy."""
    del hardware_config
    _prepare_quantization_regions(
        model,
        region_qspec,
        operator_traits,
    )
    return _materialize_global_average_domain_boundaries(model)


def finalize_quantization_calibration(model: GraphModule) -> GraphModule:
    """Finalize shared observer state across Gemmini prepared domains."""
    _finalize_quantization_regions(model)
    _align_relu_observer_domains(model)
    # A paired post-ReLU observer can also be a shared-domain entry (for
    # example before stem MaxPool or global average). Propagate its raised
    # maximum back through that existing region without changing structure.
    _finalize_quantization_regions(model)
    return model


def _prepared_int8_observer(
    node: object,
    modules: dict[str, torch.nn.Module],
) -> FusedAmaxObsFakeQuantize | None:
    if not isinstance(node, Node) or node.op != "call_module":
        return None
    module = modules.get(str(node.target))
    if not isinstance(module, FusedAmaxObsFakeQuantize):
        return None
    contract = PreparedQuantizationContract.from_observer(module)
    if (
        contract.dtype != "int8"
        or getattr(contract.qscheme, "value", contract.qscheme)
        != "per_tensor_symmetric"
        or contract.axes
        or contract.block_size is not None
        or contract.rounding != "nearest"
        or contract.quant_min != -128
        or contract.quant_max != 127
    ):
        return None
    return module


def _align_relu_observer_domains(model: GraphModule) -> int:
    """Calibrate each helper-owned ReLU round trip in one INT8 domain.

    A stock Gemmini Conv/Add helper provides one output requantization plus
    ReLU. It cannot reproduce two independently scaled INT8 saturations.
    Preserve both prepared observers as semantic boundaries, but finalize
    their compatible per-tensor INT8 qparams from their joint calibrated
    maximum. Conversion then retains an explicit, auditable equal-scale
    Q/DQ round trip that target lowering may erase exactly.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    aligned = 0
    for activation in model.graph.nodes:
        if activation.op != "call_function" or activation.target != _RELU:
            continue
        before = _node_input(activation, 0, "input")
        users = tuple(activation.users)
        if (
            not isinstance(before, Node)
            or tuple(before.users) != (activation,)
            or len(users) != 1
        ):
            continue
        after = users[0]
        before_module = _prepared_int8_observer(before, modules)
        after_module = _prepared_int8_observer(after, modules)
        if before_module is None or after_module is None:
            continue
        semantic_producer = _node_input(before, 0, "input")
        if not isinstance(semantic_producer, Node) or not (
            _is_conv2d(semantic_producer)
            or (
                semantic_producer.op == "call_function"
                and semantic_producer.target in _ADD_TARGETS
            )
        ):
            continue
        if not all(
            bool(flag[0].item())
            for module in (before_module, after_module)
            for flag in (module.observer_enabled, module.fake_quant_enabled)
        ):
            continue
        if PreparedQuantizationContract.from_observer(
            before_module
        ) != PreparedQuantizationContract.from_observer(after_module):
            continue
        if (
            before_module.amax_history.numel() == 0
            or after_module.amax_history.numel() == 0
        ):
            continue
        maxima = torch.stack(
            (
                torch.amax(before_module.amax_history),
                torch.amax(after_module.amax_history),
            )
        )
        if not torch.isfinite(maxima).all():
            continue
        group_amax = torch.amax(maxima)
        for module in {
            id(before_module): before_module,
            id(after_module): after_module,
        }.values():
            module.amax_history[0].copy_(group_amax)
            module.calculate_qparams()
        if not torch.equal(before_module.scale, after_module.scale):
            raise RuntimeError(
                f"{activation.name}: Gemmini ReLU scales did not align"
            )
        aligned += 1
    model.meta["gemmini_relu_domain_alignment_count"] = aligned
    return aligned


def _is_conv2d(node: Node) -> bool:
    return node.op == "call_function" and node.target in _CONV2D_TARGETS


def _only_user(node: Node, target: object) -> Node | None:
    users = tuple(node.users)
    if len(users) != 1:
        return None
    user = users[0]
    if user.op != "call_function" or user.target != target:
        return None
    return user


def _only_user_in(node: Node, targets: tuple[object, ...]) -> Node | None:
    users = tuple(node.users)
    if len(users) != 1:
        return None
    user = users[0]
    if user.op != "call_function" or user.target not in targets:
        return None
    return user


def _resolve_get_attr(model: GraphModule, node: object) -> object | None:
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    value: object = model
    for field in str(node.target).split("."):
        if not hasattr(value, field):
            return None
        value = getattr(value, field)
    return value


def _is_scalar_symmetric_int8_quantize(
    model: GraphModule,
    node: Node,
) -> bool:
    """Return whether Q has the exact code-domain ReLU preconditions."""
    if node.op != "call_function" or node.target != _QUANTIZE:
        return False
    scale = _resolve_get_attr(model, _node_input(node, 1, "scale"))
    zero_point = _node_input(node, 2, "zero_point")
    axes = _node_input(node, 3, "axes")
    block_size = _node_input(node, 4, "block_size")
    qmap = _resolve_get_attr(model, _node_input(node, 5, "qmap"))
    output_code = _node_input(node, 6, "output_code")
    rounding = _node_input(node, 7, "rounding")
    quant_min = _node_input(node, 8, "quant_min")
    quant_max = _node_input(node, 9, "quant_max")
    if not isinstance(scale, torch.Tensor) or scale.numel() != 1:
        return False
    if scale.dtype != torch.float32 or not torch.isfinite(scale).all():
        return False
    if float(scale.item()) <= 0.0:
        return False
    if (
        zero_point is not None
        or axes is not None
        or block_size is not None
        or output_code is not None
    ):
        return False
    if not isinstance(qmap, torch.Tensor) or qmap.numel() != 2**16:
        return False
    values = torch.arange(
        2**16,
        dtype=torch.int16,
        device=qmap.device,
    ).view(torch.bfloat16)
    expected_qmap = torch.clamp(torch.round(values), -128.0, 127.0)
    if qmap.dtype != expected_qmap.dtype or not torch.equal(
        qmap.view(torch.int16), expected_qmap.view(torch.int16)
    ):
        return False
    return (
        rounding in (None, "nearest")
        and quant_min == -128.0
        and quant_max == 127.0
    )


def _quantize_scales_are_identical(
    model: GraphModule,
    left: Node,
    right: Node,
) -> bool:
    """Require one physical scale before erasing an INT8 Q/DQ boundary."""
    left_scale = _resolve_get_attr(model, _node_input(left, 1, "scale"))
    right_scale = _resolve_get_attr(model, _node_input(right, 1, "scale"))
    return (
        isinstance(left_scale, torch.Tensor)
        and isinstance(right_scale, torch.Tensor)
        and left_scale.dtype == right_scale.dtype
        and left_scale.shape == right_scale.shape
        and torch.equal(left_scale, right_scale)
    )


def _qdq_contracts_are_identical(
    model: GraphModule,
    quantize: Node,
    dequantize: Node,
) -> bool:
    """Require DQ to decode the exact canonical INT8 domain emitted by Q."""
    quantize_scale = _resolve_get_attr(
        model,
        _node_input(quantize, 1, "scale"),
    )
    dequantize_scale = _resolve_get_attr(
        model,
        _node_input(dequantize, 1, "scale"),
    )
    scale_matches = (
        isinstance(quantize_scale, torch.Tensor)
        and isinstance(dequantize_scale, torch.Tensor)
        and quantize_scale.dtype == dequantize_scale.dtype
        and quantize_scale.shape == dequantize_scale.shape
        and torch.equal(quantize_scale, dequantize_scale)
    )
    return (
        scale_matches
        and all(
            _node_input(quantize, index, name)
            == _node_input(dequantize, index, dequantize_name)
            for index, name, dequantize_name in (
                (2, "zero_point", "zero_point"),
                (3, "axes", "axes"),
                (4, "block_size", "block_size"),
            )
        )
        and all(
            _node_input(dequantize, index, name) is None
            for index, name in (
                (5, "input_qmap"),
                (6, "output_qmap"),
            )
        )
    )


def _clone_with_input(
    graph,
    template: Node,
    old_input: Node,
    new_input: Node,
) -> Node:
    return graph.node_copy(
        template,
        lambda node: new_input if node is old_input else node,
    )


def _replace_relu_epilogue(
    model: GraphModule,
    *,
    real_input: Node,
    intermediate_quantize: Node,
    intermediate_dequantize: Node,
    activation: Node,
    pooling: Node | None,
    final_quantize: Node,
) -> None:
    """Replace a float ReLU storage round-trip with code-value ReLU."""
    graph = model.graph
    epilogue_input = pooling or activation
    with graph.inserting_before(final_quantize):
        output_quantize = _clone_with_input(
            graph,
            final_quantize,
            epilogue_input,
            real_input,
        )
        code_relu = graph.call_function(_CODE_RELU, args=(output_quantize,))
        code_pool = None
        if pooling is not None:
            code_pool = _clone_with_input(
                graph,
                pooling,
                activation,
                code_relu,
            )

    output_quantize.meta = dict(final_quantize.meta)
    code_relu.meta = dict(intermediate_quantize.meta)
    replacement = code_relu
    if code_pool is not None:
        code_pool.meta = dict(final_quantize.meta)
        replacement = code_pool
    else:
        code_relu.meta = dict(final_quantize.meta)

    final_quantize.replace_all_uses_with(replacement)
    for node in (
        final_quantize,
        pooling,
        activation,
        intermediate_dequantize,
        intermediate_quantize,
    ):
        if node is not None:
            graph.erase_node(node)


def _legalize_quantized_value_relu(model: GraphModule) -> int:
    """Move representable Conv/Add ReLU epilogues into output-code space.

    For one positive scalar symmetric int8 domain, ``ReLU(Q(x))`` equals
    ``Q(ReLU(x))`` as a value-domain identity. Gemmini's helpers implement
    this latter boundary with one output requantization followed by ReLU.
    Reuse the already calibrated final-Q operands only when its scale is
    bit-identical to the intermediate Q/DQ scale. Otherwise the first INT8
    saturation is observable and the single helper requantization cannot
    represent it. Unsupported or branched forms are left unchanged for later
    fail-closed handling.
    """
    changed = 0
    for activation in tuple(model.graph.nodes):
        if activation.op != "call_function" or activation.target != _RELU:
            continue
        intermediate_dequantize = _node_input(activation, 0, "input")
        if (
            isinstance(intermediate_dequantize, Node)
            and intermediate_dequantize.op == "call_function"
            and intermediate_dequantize.target == _QUANTIZE
            and _is_scalar_symmetric_int8_quantize(
                model, intermediate_dequantize
            )
        ):
            producer = _node_input(intermediate_dequantize, 0, "input")
            if (
                isinstance(producer, Node)
                and producer.op == "call_function"
                and producer.target in _ADD_TARGETS
                and tuple(intermediate_dequantize.users) == (activation,)
            ):
                activation.target = _CODE_RELU
                activation.meta = dict(intermediate_dequantize.meta)
                changed += 1
            continue
        if (
            not isinstance(intermediate_dequantize, Node)
            or intermediate_dequantize.op != "call_function"
            or intermediate_dequantize.target != _DEQUANTIZE
            or tuple(intermediate_dequantize.users) != (activation,)
        ):
            continue
        intermediate_quantize = _node_input(intermediate_dequantize, 0, "input")
        if (
            not isinstance(intermediate_quantize, Node)
            or intermediate_quantize.op != "call_function"
            or intermediate_quantize.target != _QUANTIZE
            or tuple(intermediate_quantize.users) != (intermediate_dequantize,)
        ):
            continue
        if not _is_scalar_symmetric_int8_quantize(
            model,
            intermediate_quantize,
        ):
            continue
        if not _qdq_contracts_are_identical(
            model,
            intermediate_quantize,
            intermediate_dequantize,
        ):
            continue
        real_input = _node_input(intermediate_quantize, 0, "input")
        if not isinstance(real_input, Node):
            continue
        semantic_producer = real_input
        if (
            real_input.op == "call_function"
            and real_input.target == _DEQUANTIZE
        ):
            semantic_producer = _node_input(real_input, 0, "input")
        if not isinstance(semantic_producer, Node):
            continue
        if not _is_conv2d(semantic_producer) and not (
            semantic_producer.op == "call_function"
            and semantic_producer.target in _ADD_TARGETS
        ):
            continue
        pooling = _only_user(activation, _MAX_POOL)
        epilogue_input = pooling or activation
        final_quantize = _only_user(epilogue_input, _QUANTIZE)
        if final_quantize is None:
            continue
        if not _is_scalar_symmetric_int8_quantize(model, final_quantize):
            continue
        if not _quantize_scales_are_identical(
            model,
            intermediate_quantize,
            final_quantize,
        ):
            continue
        _replace_relu_epilogue(
            model,
            real_input=real_input,
            intermediate_quantize=intermediate_quantize,
            intermediate_dequantize=intermediate_dequantize,
            activation=activation,
            pooling=pooling,
            final_quantize=final_quantize,
        )
        changed += 1
    if changed:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()
    return changed


def _conv_region_members(convolution: Node) -> list[Node] | None:
    """Return one maximal helper region rooted at an ordinary Conv2d."""
    accumulator_dequantize = _only_user(convolution, _DEQUANTIZE)
    if accumulator_dequantize is None:
        return None
    output_quantize = _only_user(accumulator_dequantize, _QUANTIZE)
    if output_quantize is None:
        return None

    members = [convolution, accumulator_dequantize, output_quantize]
    input_value = _node_input(convolution, 0, "input")
    if (
        isinstance(input_value, Node)
        and input_value.op == "call_function"
        and input_value.target == _QUANTIZE
        and tuple(input_value.users) == (convolution,)
    ):
        members.insert(0, input_value)

    activation = _only_user(output_quantize, _CODE_RELU)
    if activation is not None:
        members.append(activation)
        pooling = _only_user(activation, _MAX_POOL)
        if pooling is not None:
            members.append(pooling)
        return members

    terminal_dequantize = _only_user(output_quantize, _DEQUANTIZE)
    if terminal_dequantize is None:
        return members
    activation = _only_user(terminal_dequantize, _RELU)
    if activation is None:
        return members
    pooling = _only_user(activation, _MAX_POOL)
    epilogue_input = pooling or activation
    final_quantize = _only_user(epilogue_input, _QUANTIZE)
    if final_quantize is None:
        return members
    # This float-value Q/DQ/ReLU/Q form is not a single helper operation.
    # Keep it unfused so later lowering cannot erase a semantic boundary.
    return None


def _add_region_members(addition: Node) -> list[Node] | None:
    """Return one helper-representable residual Add plus ReLU region."""
    operands = (
        _node_input(addition, 0, "input"),
        _node_input(addition, 1, "other"),
    )
    if any(
        not isinstance(operand, Node)
        or operand.op != "call_function"
        or operand.target != _DEQUANTIZE
        or tuple(operand.users) != (addition,)
        for operand in operands
    ):
        return None
    output_quantize = _only_user(addition, _QUANTIZE)
    if output_quantize is None:
        return None
    code_activation = _only_user(output_quantize, _CODE_RELU)
    if code_activation is not None:
        return [
            operands[0],
            operands[1],
            addition,
            output_quantize,
            code_activation,
        ]
    output_dequantize = _only_user(output_quantize, _DEQUANTIZE)
    if output_dequantize is None:
        return None
    activation = _only_user(output_dequantize, _RELU)
    if activation is None:
        return None
    final_quantize = _only_user(activation, _QUANTIZE)
    if final_quantize is None:
        return None
    # The extra float-value round trip is not implemented by one helper.
    return None


def _fuse_final_residual_boundary(model: GraphModule) -> None:
    """Group final Add/Q/ReLU codes and expose their existing DQ scale.

    A terminal residual block can execute ReLU directly in its Add-output code
    domain. It has no later Conv consumer, so PT2E may omit the DQ that other
    residual paths expose during qmap lowering. The residual helper produces
    the same ReLU qcodes. Insert only their matching shared DQ before global
    average pooling, reusing every qparam from the existing Add-output Q.
    """
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))
    fused_count = 0
    for addition in tuple(graph.nodes):
        if (
            addition.op != "call_function"
            or addition.target not in _ADD_TARGETS
        ):
            continue
        output_quantize = _only_user(addition, _QUANTIZE)
        if output_quantize is None:
            continue
        if not _is_scalar_symmetric_int8_quantize(model, output_quantize):
            continue
        existing_dequantize = _only_user(output_quantize, _DEQUANTIZE)
        if (
            existing_dequantize is not None
            and not _qdq_contracts_are_identical(
                model,
                output_quantize,
                existing_dequantize,
            )
        ):
            continue
        activation_input = existing_dequantize or output_quantize
        activation = _only_user_in(
            activation_input,
            (_RELU, _CODE_RELU),
        )
        if activation is None:
            continue
        average = _only_user(activation, _ADAPTIVE_AVG_POOL)
        if average is None:
            continue
        scale = _node_input(output_quantize, 1, "scale")
        zero_point = _node_input(output_quantize, 2, "zero_point")
        axes = _node_input(output_quantize, 3, "axes")
        block_size = _node_input(output_quantize, 4, "block_size")
        qmap = _node_input(output_quantize, 5, "qmap")
        if not isinstance(scale, Node) or not isinstance(qmap, Node):
            continue
        if existing_dequantize is None:
            with graph.inserting_before(average):
                average_dequantize = graph.call_function(
                    _DEQUANTIZE,
                    args=(
                        activation,
                        scale,
                        zero_point,
                        axes,
                        block_size,
                        qmap,
                        None,
                    ),
                )
            average_dequantize.meta = dict(activation.meta)
            members = [addition, output_quantize, activation]
        else:
            with graph.inserting_before(average):
                code_activation = graph.call_function(
                    _CODE_RELU,
                    args=(output_quantize,),
                )
                average_dequantize = graph.node_copy(
                    existing_dequantize,
                    lambda node: (
                        code_activation if node is output_quantize else node
                    ),
                )
            code_activation.meta = dict(output_quantize.meta)
            average_dequantize.meta = dict(existing_dequantize.meta)
            members = [addition, output_quantize, code_activation]
        average.replace_input_with(activation, average_dequantize)
        if existing_dequantize is not None:
            graph.erase_node(activation)
            graph.erase_node(existing_dequantize)
        terminal_meta = dict(output_quantize.meta)
        outer = _create_and_insert_subgraph(
            members,
            model,
            named_modules,
        )
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        graph.lint()
        graph.eliminate_dead_code()
        model.recompile()


def _average_region_members(average: Node) -> list[Node] | None:
    """Return the shared DQ/Average/Flatten/Q helper boundary."""
    input_dequantize = _node_input(average, 0, "input")
    if (
        not isinstance(input_dequantize, Node)
        or input_dequantize.op != "call_function"
        or input_dequantize.target != _DEQUANTIZE
        or tuple(input_dequantize.users) != (average,)
    ):
        return None
    flatten = _only_user(average, _FLATTEN)
    if flatten is None:
        return None
    output_quantize = _only_user(flatten, _QUANTIZE)
    if output_quantize is None:
        return None
    return [input_dequantize, average, flatten, output_quantize]


def _norm_region_members(normalization: Node) -> list[Node] | None:
    """Return the shared DQ/normalization/Q binding for tiled_norm_auto."""
    input_dequantize = _node_input(normalization, 0, "input")
    if (
        not isinstance(input_dequantize, Node)
        or input_dequantize.op != "call_function"
        or input_dequantize.target != _DEQUANTIZE
        or tuple(input_dequantize.users) != (normalization,)
    ):
        return None
    output_quantize = _only_user(normalization, _QUANTIZE)
    if output_quantize is None:
        return None
    return [input_dequantize, normalization, output_quantize]


def _linear_region_members(linear: Node) -> list[Node] | None:
    """Return the shared Q/Linear/DQ/Q region for ``tiled_matmul_auto``."""
    accumulator_dequantize = _only_user(linear, _DEQUANTIZE)
    if accumulator_dequantize is None:
        return None
    output_quantize = _only_user(accumulator_dequantize, _QUANTIZE)
    if output_quantize is None:
        return None
    input_quantize = _node_input(linear, 0, "input")
    if (
        isinstance(input_quantize, Node)
        and input_quantize.op == "call_function"
        and input_quantize.target == _QUANTIZE
        and tuple(input_quantize.users) == (linear,)
    ):
        return [
            input_quantize,
            linear,
            accumulator_dequantize,
            output_quantize,
        ]
    if isinstance(input_quantize, Node) and input_quantize.op == "call_module":
        return [linear, accumulator_dequantize, output_quantize]
    return None


def _node_input(node: Node, index: int, name: str):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name)


def _fuse_shared_conv_regions(model: GraphModule) -> None:
    """Capture helper-representable shared Conv2d regions before tiling.

    This is a target-owned *node grouping* rule, not a new operator or
    lowering.  Prefer the longer helper forms ``Q -> Conv2d -> DQ -> Q ->
    ReLU -> MaxPool`` and ``Q -> Conv2d -> DQ -> Q -> ReLU``.  Remaining
    convolutions retain the four-member shared form.  Longest-pattern-first
    selection ensures every original semantic node belongs to at most one
    target grouping.
    """
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    named_modules = dict(model.named_modules(remove_duplicate=False))
    convolutions = tuple(node for node in model.graph.nodes if _is_conv2d(node))
    fused_count = 0
    for convolution in convolutions:
        if convolution not in model.graph.nodes:
            continue
        region = _conv_region_members(convolution)
        if region is None:
            continue
        terminal_meta = dict(region[-1].meta)
        outer = _create_and_insert_subgraph(
            region,
            model,
            named_modules,
        )
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()


def _fuse_shared_residual_regions(model: GraphModule) -> None:
    """Group shared residual Add/ReLU semantics for ``tiled_resadd_auto``."""
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    named_modules = dict(model.named_modules(remove_duplicate=False))
    additions = tuple(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target in _ADD_TARGETS
    )
    fused_count = 0
    for addition in additions:
        if addition not in model.graph.nodes:
            continue
        region = _add_region_members(addition)
        if region is None:
            continue
        terminal_meta = dict(region[-1].meta)
        outer = _create_and_insert_subgraph(region, model, named_modules)
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()


def _fuse_shared_average_regions(model: GraphModule) -> None:
    """Group the shared classifier global-average boundary."""
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    named_modules = dict(model.named_modules(remove_duplicate=False))
    averages = tuple(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == _ADAPTIVE_AVG_POOL
    )
    fused_count = 0
    for average in averages:
        if average not in model.graph.nodes:
            continue
        region = _average_region_members(average)
        if region is None:
            continue
        terminal_meta = dict(region[-1].meta)
        outer = _create_and_insert_subgraph(region, model, named_modules)
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()


def _fuse_shared_norm_regions(model: GraphModule) -> None:
    """Create a structural binding point for shared normalization semantics."""
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    named_modules = dict(model.named_modules(remove_duplicate=False))
    normalizations = tuple(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target in _NORM_TARGETS
    )
    fused_count = 0
    for normalization in normalizations:
        if normalization not in model.graph.nodes:
            continue
        region = _norm_region_members(normalization)
        if region is None:
            continue
        terminal_meta = dict(region[-1].meta)
        outer = _create_and_insert_subgraph(region, model, named_modules)
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()


def _fuse_shared_linear_regions(model: GraphModule) -> None:
    """Group shared Q/Linear/DQ/Q semantics for ``tiled_matmul_auto``."""
    from voyager_compiler.codegen.mapping import _create_and_insert_subgraph
    from voyager_compiler.pt2e_utils import update_submod_user_meta

    named_modules = dict(model.named_modules(remove_duplicate=False))
    linears = tuple(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == _LINEAR
    )
    fused_count = 0
    for linear in linears:
        if linear not in model.graph.nodes:
            continue
        region = _linear_region_members(linear)
        if region is None:
            continue
        terminal_meta = dict(region[-1].meta)
        outer = _create_and_insert_subgraph(region, model, named_modules)
        outer.meta.update(terminal_meta)
        update_submod_user_meta(model, outer, named_modules)
        fused_count += 1
    if fused_count:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()


def apply_quantization_rewrites(model: GraphModule) -> GraphModule:
    """Expose and fuse shared Conv2d regions before generic physical passes."""
    _legalize_quantized_value_relu(model)
    _fuse_final_residual_boundary(model)
    _fuse_shared_residual_regions(model)
    _fuse_shared_average_regions(model)
    _fuse_shared_conv_regions(model)
    _fuse_shared_linear_regions(model)
    _fuse_shared_norm_regions(model)
    return model


__all__ = [
    "apply_quantization_rewrites",
    "finalize_quantization_calibration",
    "prepare_quantization",
]
