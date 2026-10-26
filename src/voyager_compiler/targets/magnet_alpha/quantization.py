# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha quantization preparation and target-lowering orchestration."""

import copy
import logging
import math
import operator
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.compiler.analysis import is_address_only_view
from voyager_compiler.ir.graph import (
    QUANTIZATION_INPUT_REGION,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    PreparedDomainFormatConstraint,
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainRuleSet,
    PreparedDomainTransfer,
    PreparedDomainTransferRule,
    PreparedQuantizationContract,
    propagate_prepared_domain_values,
    set_physical_quantization_contract,
)
from voyager_compiler.targets.magnet_alpha.boundary_lowering import (
    lower_magnet_alpha_internal_scalar_int8_qdq,
    lower_magnet_alpha_physical_int8_pvs_bridges,
    lower_magnet_alpha_physical_int8_relu_qdq,
    lower_magnet_alpha_pvs_outputs,
)
from voyager_compiler.targets.magnet_alpha.domains import (
    _QUANTIZATION_DOMAIN_ALIAS,
    _QUANTIZATION_DOMAIN_ENTRY,
    _finalize_quantization_regions,
    _move_output_observers_after_epilogues,
    _prepare_quantization_regions,
    _prepare_vsq_join_regions,
)
from voyager_compiler.targets.magnet_alpha.fixed_point import (
    alpha_w8_accumulator_real_bound,
)
from voyager_compiler.targets.magnet_alpha.fx import (
    fetch_attr,
    get_arg_value,
    get_identity_slice_source,
    get_leading_channel_slice_source,
)
from voyager_compiler.targets.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    RESIDUAL_ADD_TARGETS,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    node_shape as _node_shape,
)
from voyager_compiler.targets.magnet_alpha.movement_lowering import (
    lower_magnet_alpha_integer_data_movement,
)
from voyager_compiler.targets.magnet_alpha.nmp_fake_epilogue import (
    MagnetAlphaNMPFakeEpilogue,
)
from voyager_compiler.targets.magnet_alpha.nmp_lowering import (
    lower_magnet_alpha_depthwise_fixed_point,
)
from voyager_compiler.targets.magnet_alpha.observers import (
    FusedAmaxObsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.targets.magnet_alpha.pe_fake_epilogue import (
    MagnetAlphaPEFakeEpilogue,
    _observer_scale_endpoints,
)
from voyager_compiler.targets.magnet_alpha.pe_lowering import (
    finalize_magnet_alpha_vsq_operands,
    fold_magnet_alpha_physical_int8_relus,
    lower_magnet_alpha_fixed_point_epilogues,
    lower_magnet_alpha_vsq_convolutions,
)
from voyager_compiler.targets.magnet_alpha.pool_lowering import (
    lower_magnet_alpha_adaptive_average_pool2d,
)
from voyager_compiler.targets.magnet_alpha.residual_lowering import (
    lower_magnet_alpha_saturating_adds,
)

logger = logging.getLogger(__name__)


def _unique_module_name(model: GraphModule, base: str) -> str:
    name = base
    suffix = 0
    while hasattr(model, name):
        suffix += 1
        name = f"{base}_{suffix}"
    return name


def _prepared_output_observer(node: Node, modules) -> bool:
    if node.op != "call_module" or not isinstance(node.target, str):
        return False
    module = modules.get(node.target)
    if not isinstance(module, FusedAmaxObsFakeQuantize):
        return False
    if module.dtype == "int8":
        return True
    return (
        module.dtype == "int4"
        and module.block_size == ALPHA_VSQ_BLOCK_VALUES
        and module.scale_dtype == "uint8"
    )


def _bind_magnet_alpha_residual_pe_output_domains(
    model: GraphModule,
) -> GraphModule:
    """Make a residual PE produce the add's calibrated INT8 code domain.

    A prepared residual frontier normally contains two observers: the PE's
    output observer and the add-region entry observer.  Leaving both active
    converts that edge to ``Qadd(DQold(PPU(..., Sold)))``.  Alpha cannot run
    that rescale between a PE accumulator and its fused EADD, so make the PE
    observer the region entry and retain the old entry only as a calibration
    alias.  This happens before calibration and therefore never edits frozen
    qparams after the fact.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    int8_contract = PhysicalQuantizationContract("int8")
    rebound = 0

    for add in model.graph.nodes:
        if add.target not in RESIDUAL_ADD_TARGETS:
            continue
        region = add.meta.get(QUANTIZATION_INPUT_REGION)
        if region is None:
            region = add.meta.get(QUANTIZATION_REGION)
        if region is None:
            continue
        for entry in (
            get_arg_value(add, 0, "input"),
            get_arg_value(add, 1, "other"),
        ):
            if (
                not isinstance(entry, Node)
                or not entry.meta.get(_QUANTIZATION_DOMAIN_ENTRY)
                or entry.meta.get(QUANTIZATION_REGION) != region
                or tuple(entry.users) != (add,)
                or entry.op != "call_module"
                or not entry.args
            ):
                continue
            entry_module = modules.get(str(entry.target))
            source_observer = entry.args[0]
            if (
                not isinstance(entry_module, FusedAmaxObsFakeQuantize)
                or entry_module.dtype != "int8"
                or entry_module.is_per_channel
                or entry_module.block_size is not None
                or not isinstance(source_observer, Node)
                or source_observer.op != "call_module"
                or tuple(source_observer.users) != (entry,)
                or not source_observer.args
            ):
                continue
            source_module = modules.get(str(source_observer.target))
            if isinstance(source_module, MagnetAlphaPEFakeEpilogue):
                set_physical_quantization_contract(entry, int8_contract)
                rebound += 1
                continue
            fixed_node = source_observer.args[0]
            fixed_module = (
                modules.get(str(fixed_node.target))
                if isinstance(fixed_node, Node)
                and fixed_node.op == "call_module"
                else None
            )
            if not isinstance(
                source_module,
                FusedAmaxObsFakeQuantize,
            ) or not isinstance(fixed_module, MagnetAlphaPEFakeEpilogue):
                continue
            if PreparedQuantizationContract.from_observer(
                source_module
            ) != PreparedQuantizationContract.from_observer(entry_module):
                raise ValueError(
                    f"{add.name}: residual PE and add entry use incompatible "
                    "prepared INT8 domains"
                )

            source_observer.meta[QUANTIZATION_REGION] = region
            source_observer.meta[_QUANTIZATION_DOMAIN_ENTRY] = True
            source_observer.meta.pop(_QUANTIZATION_DOMAIN_ALIAS, None)
            set_physical_quantization_contract(source_observer, int8_contract)

            entry.meta.pop(_QUANTIZATION_DOMAIN_ENTRY, None)
            entry.meta[_QUANTIZATION_DOMAIN_ALIAS] = True
            set_physical_quantization_contract(entry, int8_contract)
            entry_module.disable_fake_quant()
            entry_module.enable_observer()
            rebound += 1

    if rebound:
        model.graph.lint()
        model.recompile()
    logger.info(
        "Bound %d Magnet Alpha residual PE outputs to add domains",
        rebound,
    )
    return model


def _pe_boundary_view_source(node: Node, current: Node) -> Node | None:
    source = get_identity_slice_source(node)
    if source is None:
        source = get_leading_channel_slice_source(node)
    if source is None and is_address_only_view(node, current):
        candidate = get_arg_value(node, 0, "input")
        source = candidate if isinstance(candidate, Node) else None
    return source


def _prepared_pe_boundary(conv: Node, modules):
    """Find one explicit PE output domain and its optional ReLU."""
    current = conv
    views = []
    visited = set()
    while current not in visited and len(current.users) == 1:
        visited.add(current)
        user = next(iter(current.users))
        if _prepared_output_observer(user, modules):
            return tuple(views), None, user
        if user.target == torch.ops.aten.relu.default:
            observers = tuple(
                candidate
                for candidate in user.users
                if _prepared_output_observer(candidate, modules)
            )
            if len(observers) == 1:
                return tuple(views), user, observers[0]
            return None
        source = _pe_boundary_view_source(user, current)
        if source is not current:
            return None
        views.append(user)
        current = user
    return None


def _insert_fixed_point_reference(
    model: GraphModule,
    conv: Node,
    views,
    relu: Node | None,
    observer: Node,
) -> None:
    graph = model.graph
    module_name = _unique_module_name(
        model,
        f"{conv.name}_ppu_fixed_point",
    )
    fixed_point = MagnetAlphaPEFakeEpilogue()
    fixed_point.train(model.training)
    model.add_module(module_name, fixed_point)

    if relu is None and not views:
        with graph.inserting_before(observer):
            fixed_node = graph.call_module(module_name, (conv,))
        observer.replace_input_with(conv, fixed_node)
    elif relu is None:
        # Keep the PPU reference on the resident producer output; validated
        # PE-transparent views remain logical aliases afterward.
        with graph.inserting_after(conv):
            fixed_node = graph.call_module(module_name, (conv,))
        views[0].replace_input_with(conv, fixed_node)
    elif not views:
        with graph.inserting_before(relu):
            fixed_node = graph.call_module(module_name, (conv,))
        relu.replace_input_with(conv, fixed_node)
    else:
        # ReLU is elementwise, so it commutes with these validated
        # PE-transparent views. Keep Alpha's PPU on the resident Conv output.
        with graph.inserting_after(conv):
            fixed_node = graph.call_module(module_name, (conv,))
        with graph.inserting_after(fixed_node):
            relocated_relu = graph.call_function(
                torch.ops.aten.relu.default,
                (fixed_node,),
            )
        relocated_relu.meta = copy.copy(conv.meta)
        views[0].replace_input_with(conv, relocated_relu)
        relu.replace_all_uses_with(views[-1])
        graph.erase_node(relu)

    fixed_node.meta = copy.copy(conv.meta)
    fixed_node.meta["magnet_alpha_ppu_fixed_point"] = True


def insert_magnet_alpha_fixed_point_epilogues(
    model: GraphModule,
) -> GraphModule:
    """Put one trainable normalized-Q8.8 reference at each PE boundary."""
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))
    inserted = 0

    for conv in list(graph.nodes):
        if conv.target != torch.ops.aten.conv2d.default:
            continue
        if get_arg_value(conv, 6, "groups", 1) != 1:
            continue
        boundary = _prepared_pe_boundary(conv, modules)
        if boundary is None:
            continue
        views, relu, observer = boundary
        _insert_fixed_point_reference(
            model,
            conv,
            views,
            relu,
            observer,
        )
        inserted += 1

    graph.lint()
    model.recompile()
    logger.info("Inserted %d Magnet Alpha PE fixed-point references", inserted)
    return model


def _insert_nmp_fixed_point_reference(
    model: GraphModule,
    conv: Node,
    views,
    relu: Node | None,
    observer: Node,
    *,
    lane_width: int,
) -> None:
    """Attach one exact-capable reference to a prepared depthwise boundary."""
    input_value = get_arg_value(conv, 0, "input")
    weight_value = get_arg_value(conv, 1, "weight")
    if not isinstance(input_value, Node) or not isinstance(weight_value, Node):
        raise ValueError(f"{conv.name}: depthwise Conv2d operands are missing")
    input_source = input_value.args[0] if input_value.args else None
    weight_source = weight_value.args[0] if weight_value.args else None
    if not isinstance(input_source, Node) or not isinstance(
        weight_source,
        Node,
    ):
        raise ValueError(
            f"{conv.name}: depthwise W8 observers have no tensor sources"
        )

    module_name = _unique_module_name(
        model,
        f"{conv.name}_nmp_fixed_point",
    )
    fixed_point = MagnetAlphaNMPFakeEpilogue(
        lane_width=lane_width,
        stride=get_arg_value(conv, 3, "stride", (1, 1)),
        padding=get_arg_value(conv, 4, "padding", (0, 0)),
        dilation=get_arg_value(conv, 5, "dilation", (1, 1)),
        groups=int(get_arg_value(conv, 6, "groups", 1)),
    )
    fixed_point.train(model.training)
    model.add_module(module_name, fixed_point)

    with model.graph.inserting_after(conv):
        fixed_node = model.graph.call_module(
            module_name,
            (conv, input_source, weight_source),
        )
    first_user = views[0] if views else relu
    if first_user is None:
        first_user = observer
    first_user.replace_input_with(conv, fixed_node)
    fixed_node.meta = copy.copy(conv.meta)


def insert_magnet_alpha_nmp_fixed_point_epilogues(
    model: GraphModule,
    hardware_config,
) -> GraphModule:
    """Put one exact-capable reference at each prepared Alpha DWC boundary."""
    if hardware_config is None or not hasattr(hardware_config, "nmp_lanes"):
        return model
    lane_width = int(hardware_config.nmp_lanes(8))
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))
    inserted = 0

    for conv in list(graph.nodes):
        if conv.target != torch.ops.aten.conv2d.default:
            continue
        groups = int(get_arg_value(conv, 6, "groups", 1))
        if groups <= 1:
            continue
        input_value = get_arg_value(conv, 0, "input")
        weight_value = get_arg_value(conv, 1, "weight")
        if (
            _prepared_w8_observer(input_value, modules) is None
            or _prepared_w8_observer(weight_value, modules) is None
        ):
            continue
        weight_source = weight_value.args[0] if weight_value.args else None
        if not isinstance(weight_source, Node):
            continue
        weight_shape = _node_shape(weight_source)
        if (
            len(weight_shape) != 4
            or weight_shape[0] != groups
            or weight_shape[1] != 1
        ):
            continue
        boundary = _prepared_pe_boundary(conv, modules)
        if boundary is None:
            continue
        views, relu, observer = boundary
        _insert_nmp_fixed_point_reference(
            model,
            conv,
            views,
            relu,
            observer,
            lane_width=lane_width,
        )
        inserted += 1

    if inserted:
        graph.lint()
        model.recompile()
    logger.info("Inserted %d Magnet Alpha NMP fixed-point references", inserted)
    return model


def _is_calibration_side_user(
    current: Node,
    user: Node,
    modules,
) -> bool:
    """Identify non-semantic operand edges into exact calibration modules."""
    if user.op != "call_module":
        return False
    module = modules.get(str(user.target))
    if not isinstance(
        module, (MagnetAlphaPEFakeEpilogue, MagnetAlphaNMPFakeEpilogue)
    ):
        return False
    return any(argument is current for argument in user.args[1:])


def _prepared_exact_output(
    fixed_node: Node,
    modules,
) -> tuple[Node, FusedAmaxObsFakeQuantize, bool] | None:
    """Find the O8/O4 observer and activation owned by one prepared PE."""
    current = fixed_node
    relu = False
    visited = set()
    while current not in visited:
        visited.add(current)
        users = [
            user
            for user in current.users
            if not _is_calibration_side_user(current, user, modules)
        ]
        if len(users) != 1:
            return None
        user = users[0]
        if user.target == torch.ops.aten.relu.default:
            if relu:
                return None
            relu = True
            current = user
            continue
        if user.op == "call_module" and isinstance(
            modules.get(str(user.target)),
            FusedAmaxObsFakeQuantize,
        ):
            observer = modules[str(user.target)]
            if observer.dtype == "int8" and not observer.is_per_channel:
                return user, observer, relu
            if (
                observer.dtype == "int4"
                and observer.block_size == ALPHA_VSQ_BLOCK_VALUES
                and observer.scale_dtype == "uint8"
                and not observer.coarse_is_per_channel
            ):
                return user, observer, relu
            return None
        source = _pe_boundary_view_source(user, current)
        if source is not current:
            return None
        current = user
    return None


def _prepared_w8_observer(
    value,
    modules,
) -> FusedAmaxObsFakeQuantize | None:
    if not isinstance(value, Node) or value.op != "call_module":
        return None
    observer = modules.get(str(value.target))
    if not isinstance(observer, FusedAmaxObsFakeQuantize):
        return None
    if observer.dtype != "int8":
        return None
    return observer


def _prepared_vsq_observer(
    value,
    modules,
) -> FusedAmaxObsFakeQuantize | None:
    if not isinstance(value, Node) or value.op != "call_module":
        return None
    observer = modules.get(str(value.target))
    if not isinstance(observer, FusedAmaxObsFakeQuantize):
        return None
    if (
        observer.dtype != "int4"
        or observer.block_size != ALPHA_VSQ_BLOCK_VALUES
        or observer.scale_dtype != "uint8"
        or observer.coarse_dtype != "int8"
    ):
        return None
    return observer


@dataclass(frozen=True)
class _PreparedExactPEOperands:
    """Finalized prepared operands for one exact Alpha PE simulation."""

    mode: str
    real_bias: torch.Tensor
    output_scale: torch.Tensor
    output_precision: int
    uses_pvs: bool
    relu: bool
    output_observer_node: Node
    output_observer: FusedAmaxObsFakeQuantize
    accumulator_scale: torch.Tensor | None = None
    accumulator_real_amax: torch.Tensor | None = None
    input_observer: FusedAmaxObsFakeQuantize | None = None
    weight_observer: FusedAmaxObsFakeQuantize | None = None
    input_value: Node | None = None
    weight_value: Node | None = None
    stride: object = (1, 1)
    padding: object = (0, 0)
    dilation: object = (1, 1)
    groups: int = 1


@dataclass(frozen=True)
class _PreparedExactNMPOperands:
    """Finalized prepared operands for one exact Alpha NMP simulation."""

    input_observer: FusedAmaxObsFakeQuantize
    weight_observer: FusedAmaxObsFakeQuantize
    real_bias: torch.Tensor
    output_scale: torch.Tensor
    output_precision: int
    uses_pvs: bool
    relu: bool
    output_observer_node: Node
    output_observer: FusedAmaxObsFakeQuantize


def _prepared_real_bias(
    model: GraphModule,
    conv: Node,
    modules,
    channel_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Resolve the prepared Conv bias in its semantic real-value domain."""
    bias_node = get_arg_value(conv, 2, "bias")
    if bias_node is None:
        return torch.zeros_like(channel_scale)
    if not isinstance(bias_node, Node):
        return None
    if bias_node.op == "get_attr":
        bias = fetch_attr(model, bias_node.target)
        return bias.detach() if isinstance(bias, torch.Tensor) else None
    if bias_node.op != "call_module":
        return None
    bias_observer = modules.get(str(bias_node.target))
    source = bias_node.args[0] if bias_node.args else None
    if (
        not isinstance(bias_observer, _DerivedObserverOrFakeQuantize)
        or not isinstance(source, Node)
        or source.op != "get_attr"
    ):
        return None
    bias = fetch_attr(model, source.target)
    if not isinstance(bias, torch.Tensor):
        return None
    derived_scale = bias_observer.calculate_qparams().detach().reshape(-1)
    expected_scale = (
        channel_scale.detach()
        .reshape(-1)
        .to(
            device=derived_scale.device,
            dtype=derived_scale.dtype,
        )
    )
    if derived_scale.shape != expected_scale.shape or not torch.allclose(
        derived_scale,
        expected_scale,
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            f"{conv.name}: derived bias qparams do not match the active "
            "input and weight observers"
        )
    with torch.no_grad():
        return bias_observer(bias.detach()).detach()


def _prepared_w8_weight_payload(
    model: GraphModule,
    weight_node: Node,
    observer: FusedAmaxObsFakeQuantize,
) -> torch.Tensor | None:
    """Materialize the same frozen W8 weight codes used after conversion."""
    source = weight_node.args[0] if weight_node.args else None
    if not isinstance(source, Node) or source.op != "get_attr":
        return None
    weight = fetch_attr(model, source.target)
    if not isinstance(weight, torch.Tensor):
        return None
    scale = (
        observer.calculate_qparams()
        .detach()
        .to(
            device=weight.device,
            dtype=weight.dtype,
        )
    )
    qmap = observer.qmap.to(weight.device)
    if observer.rounding == "floor":
        qmap = None
    return torch.ops.alpha_quant.quantize.default(
        weight.detach(),
        scale,
        None,
        None,
        None,
        qmap,
        None,
        observer.rounding,
        observer.quant_min,
        observer.quant_max,
    )


def _prepared_exact_pe_operands(
    model: GraphModule,
    fixed_node: Node,
    modules,
) -> _PreparedExactPEOperands | None:
    """Resolve finalized prepared operands for one exact fake epilogue."""
    if len(fixed_node.args) not in (1, 3):
        return None
    conv = fixed_node.args[0]
    if (
        not isinstance(conv, Node)
        or conv.target != torch.ops.aten.conv2d.default
        or get_arg_value(conv, 6, "groups", 1) != 1
    ):
        return None

    input_value = get_arg_value(conv, 0, "input")
    weight_value = get_arg_value(conv, 1, "weight")
    output = _prepared_exact_output(fixed_node, modules)
    if output is None:
        return None
    output_observer_node, output_observer, relu = output

    input_w8 = _prepared_w8_observer(input_value, modules)
    weight_w8 = _prepared_w8_observer(weight_value, modules)
    if input_w8 is not None and weight_w8 is not None:
        uses_pvs = output_observer.dtype == "int4"
        output_precision = 4 if uses_pvs else 8
        input_scale = input_w8.calculate_qparams().detach()
        weight_scale = weight_w8.calculate_qparams().detach()
        accumulator_scale = input_scale * weight_scale.reshape(-1)
        real_bias = _prepared_real_bias(
            model,
            conv,
            modules,
            accumulator_scale,
        )
        if real_bias is None:
            return None
        weight_payload = _prepared_w8_weight_payload(
            model,
            weight_value,
            weight_w8,
        )
        if weight_payload is None:
            return None
        accumulator_real_amax = alpha_w8_accumulator_real_bound(
            weight_payload,
            accumulator_scale,
        )
        return _PreparedExactPEOperands(
            mode="w8_o4" if uses_pvs else "w8_o8",
            accumulator_scale=accumulator_scale,
            accumulator_real_amax=accumulator_real_amax,
            real_bias=real_bias,
            output_scale=output_observer.calculate_qparams().detach(),
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            output_observer_node=output_observer_node,
            output_observer=output_observer,
        )

    input_vsq = _prepared_vsq_observer(input_value, modules)
    weight_vsq = _prepared_vsq_observer(weight_value, modules)
    if input_vsq is None or weight_vsq is None:
        return None
    input_source = input_value.args[0] if input_value.args else None
    weight_source = weight_value.args[0] if weight_value.args else None
    if not isinstance(input_source, Node) or not isinstance(
        weight_source,
        Node,
    ):
        return None
    if input_vsq.block_size != weight_vsq.block_size:
        return None
    uses_pvs = output_observer.dtype == "int4"
    output_precision = 4 if uses_pvs else 8
    weight_scale = weight_vsq.calculate_qparams().detach().reshape(-1)
    real_bias = _prepared_real_bias(
        model,
        conv,
        modules,
        weight_scale,
    )
    if real_bias is None:
        return None
    return _PreparedExactPEOperands(
        mode="vsq_o4" if uses_pvs else "vsq_o8",
        real_bias=real_bias,
        output_scale=output_observer.calculate_qparams().detach(),
        output_precision=output_precision,
        uses_pvs=uses_pvs,
        relu=relu,
        output_observer_node=output_observer_node,
        output_observer=output_observer,
        input_observer=input_vsq,
        weight_observer=weight_vsq,
        input_value=input_source,
        weight_value=weight_source,
        stride=get_arg_value(conv, 3, "stride", (1, 1)),
        padding=get_arg_value(conv, 4, "padding", (0, 0)),
        dilation=get_arg_value(conv, 5, "dilation", (1, 1)),
        groups=int(get_arg_value(conv, 6, "groups", 1)),
    )


def _prepared_exact_nmp_operands(
    model: GraphModule,
    fixed_node: Node,
    modules,
) -> _PreparedExactNMPOperands | None:
    """Resolve finalized W8 depthwise operands for exact NMP simulation."""
    if len(fixed_node.args) != 3:
        return None
    conv = fixed_node.args[0]
    if (
        not isinstance(conv, Node)
        or conv.target != torch.ops.aten.conv2d.default
    ):
        return None
    groups = int(get_arg_value(conv, 6, "groups", 1))
    weight_value = get_arg_value(conv, 1, "weight")
    weight_source = (
        weight_value.args[0]
        if isinstance(weight_value, Node) and weight_value.args
        else None
    )
    weight_shape = (
        _node_shape(weight_source) if isinstance(weight_source, Node) else ()
    )
    if (
        groups <= 1
        or len(weight_shape) != 4
        or weight_shape[0] != groups
        or weight_shape[1] != 1
    ):
        return None

    input_observer = _prepared_w8_observer(
        get_arg_value(conv, 0, "input"),
        modules,
    )
    weight_observer = _prepared_w8_observer(weight_value, modules)
    if input_observer is None or weight_observer is None:
        return None
    output = _prepared_exact_output(fixed_node, modules)
    if output is None:
        return None
    output_observer_node, output_observer, relu = output
    uses_pvs = output_observer.dtype == "int4"
    output_precision = 4 if uses_pvs else 8
    input_scale = input_observer.calculate_qparams().detach().reshape(-1)
    weight_scale = weight_observer.calculate_qparams().detach().reshape(-1)
    if input_scale.numel() != 1 or weight_scale.numel() != 1:
        return None
    real_bias = _prepared_real_bias(
        model,
        conv,
        modules,
        input_scale * weight_scale,
    )
    if real_bias is None:
        return None
    return _PreparedExactNMPOperands(
        input_observer=input_observer,
        weight_observer=weight_observer,
        real_bias=real_bias,
        output_scale=output_observer.calculate_qparams().detach(),
        output_precision=output_precision,
        uses_pvs=uses_pvs,
        relu=relu,
        output_observer_node=output_observer_node,
        output_observer=output_observer,
    )


_MAGNET_ALPHA_EXACT_OUTPUT_OBSERVER = "magnet_alpha_exact_output_observer"


def _rebind_downstream_derived_bias_observers(
    model: GraphModule,
    original: FusedAmaxObsFakeQuantize,
    isolated: FusedAmaxObsFakeQuantize,
) -> None:
    """Keep derived bias qparams attached to the active operand observer.

    PT2E's derived bias observers retain direct Python references to their
    input and weight observers. Retargeting an activation observer call site
    to an isolated module therefore does not update a downstream bias
    observer automatically. Rebind only bias observers whose consuming Conv
    now receives the isolated module, leaving genuinely shared observer call
    sites untouched.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    for bias_node in model.graph.nodes:
        if bias_node.op != "call_module":
            continue
        bias_observer = modules.get(str(bias_node.target))
        if not isinstance(
            bias_observer,
            _DerivedObserverOrFakeQuantize,
        ):
            continue

        consumers = [
            user
            for user in bias_node.users
            if user.target == torch.ops.aten.conv2d.default
            and get_arg_value(user, 2, "bias") is bias_node
        ]
        if not consumers:
            continue
        if len(consumers) != 1:
            raise ValueError(
                f"{bias_node.name}: derived bias observer must feed one Conv2d"
            )
        conv = consumers[0]
        operand_modules = []
        for position, keyword in ((0, "input"), (1, "weight")):
            operand = get_arg_value(conv, position, keyword)
            if not isinstance(operand, Node) or operand.op != "call_module":
                continue
            operand_modules.append(modules.get(str(operand.target)))
        if isolated not in operand_modules:
            continue

        dependencies = list(bias_observer.obs_or_fqs)
        updated = [
            isolated if dependency is original else dependency
            for dependency in dependencies
        ]
        if all(
            dependency is updated_dependency
            for dependency, updated_dependency in zip(
                dependencies,
                updated,
            )
        ):
            continue
        bias_observer.obs_or_fqs = updated


def _isolate_exact_output_observer(
    model: GraphModule,
    observer_node: Node,
    observer: FusedAmaxObsFakeQuantize,
) -> FusedAmaxObsFakeQuantize:
    """Give one exact boundary an independently controlled output observer."""
    if observer_node.meta.get(_MAGNET_ALPHA_EXACT_OUTPUT_OBSERVER):
        isolated = model.get_submodule(str(observer_node.target))
        if not isinstance(isolated, FusedAmaxObsFakeQuantize):
            raise TypeError(
                f"{observer_node.name}: exact output observer is not "
                "a FusedAmaxObsFakeQuantize"
            )
    else:
        isolated = copy.deepcopy(observer)
        module_name = _unique_module_name(
            model,
            f"{observer_node.name}_magnet_alpha_exact",
        )
        model.add_module(module_name, isolated)
        observer_node.target = module_name
        observer_node.meta[_MAGNET_ALPHA_EXACT_OUTPUT_OBSERVER] = True
        _rebind_downstream_derived_bias_observers(
            model,
            observer,
            isolated,
        )

    # Preserve the first-pass qparams copied into the isolated module while
    # collecting fresh second-pass history. The exact producer already emits
    # the target output domain, so fake quantization here would apply it twice.
    isolated.amax_history.zero_()
    isolated.enable_observer()
    isolated.disable_fake_quant()
    return isolated


def _configure_magnet_alpha_pe_fake_epilogues(
    model: GraphModule,
    *,
    configured_only: bool,
    reset_output_observers: bool,
    recompile: bool = True,
) -> int:
    modules = dict(model.named_modules(remove_duplicate=False))
    configured = 0
    for fixed_node in model.graph.nodes:
        if fixed_node.op != "call_module":
            continue
        fixed_module = modules.get(str(fixed_node.target))
        if not isinstance(fixed_module, MagnetAlphaPEFakeEpilogue):
            continue
        if configured_only and not fixed_module.exact_enabled:
            continue
        operands = _prepared_exact_pe_operands(
            model,
            fixed_node,
            modules,
        )
        if operands is None:
            continue
        fine_endpoint, coarse_endpoint = _observer_scale_endpoints(
            operands.output_observer
        )
        if operands.mode.startswith("w8_"):
            assert operands.accumulator_scale is not None
            assert operands.accumulator_real_amax is not None
            fixed_module.configure_exact(
                operands.accumulator_scale,
                operands.real_bias,
                operands.output_scale,
                accumulator_real_amax=operands.accumulator_real_amax,
                output_precision=operands.output_precision,
                uses_pvs=operands.uses_pvs,
                relu=operands.relu,
                fine_quant_max=fine_endpoint,
                coarse_scale_quant_max=coarse_endpoint,
            )
        else:
            assert operands.input_observer is not None
            assert operands.weight_observer is not None
            assert operands.input_value is not None
            assert operands.weight_value is not None
            fixed_module.configure_exact_vsq(
                operands.input_observer,
                operands.weight_observer,
                operands.real_bias,
                operands.output_scale,
                output_precision=operands.output_precision,
                uses_pvs=operands.uses_pvs,
                stride=operands.stride,
                padding=operands.padding,
                dilation=operands.dilation,
                groups=operands.groups,
                relu=operands.relu,
                fine_quant_max=fine_endpoint,
                coarse_scale_quant_max=coarse_endpoint,
            )
            fixed_node.args = (
                fixed_node.args[0],
                operands.input_value,
                operands.weight_value,
            )
        configured += 1
        if reset_output_observers:
            isolated = _isolate_exact_output_observer(
                model,
                operands.output_observer_node,
                operands.output_observer,
            )
            modules[str(operands.output_observer_node.target)] = isolated
    if configured and recompile:
        model.graph.lint()
        model.recompile()
    return configured


def _configure_magnet_alpha_nmp_fake_epilogues(
    model: GraphModule,
    *,
    configured_only: bool,
    reset_output_observers: bool,
    recompile: bool = True,
) -> int:
    modules = dict(model.named_modules(remove_duplicate=False))
    configured = 0
    for fixed_node in model.graph.nodes:
        if fixed_node.op != "call_module":
            continue
        fixed_module = modules.get(str(fixed_node.target))
        if not isinstance(fixed_module, MagnetAlphaNMPFakeEpilogue):
            continue
        if configured_only and not fixed_module.exact_enabled:
            continue
        operands = _prepared_exact_nmp_operands(
            model,
            fixed_node,
            modules,
        )
        if operands is None:
            continue
        fine_endpoint, coarse_endpoint = _observer_scale_endpoints(
            operands.output_observer
        )
        fixed_module.configure_exact(
            operands.input_observer,
            operands.weight_observer,
            operands.real_bias,
            operands.output_scale,
            output_precision=operands.output_precision,
            uses_pvs=operands.uses_pvs,
            relu=operands.relu,
            fine_quant_max=fine_endpoint,
            coarse_scale_quant_max=coarse_endpoint,
        )
        configured += 1
        if reset_output_observers:
            isolated = _isolate_exact_output_observer(
                model,
                operands.output_observer_node,
                operands.output_observer,
            )
            modules[str(operands.output_observer_node.target)] = isolated
    if configured and recompile:
        model.graph.lint()
        model.recompile()
    return configured


_MAGNET_ALPHA_PREPARED_PVS_SIDECAR = "magnet_alpha_prepared_pvs_sidecar"
_MAGNET_ALPHA_PREPARED_PVS_COMPONENT = "magnet_alpha_prepared_pvs_component"


_PREPARED_PVS_FORMAT = PreparedDomainFormatConstraint(
    dtype="int4",
    block_size=ALPHA_VSQ_BLOCK_VALUES,
    scale_dtype="uint8",
    coarse_dtype="int8",
)


@dataclass(frozen=True)
class _PreparedPVSValue:
    """Explicit prepared-time physical components and their static contract."""

    coarse: Node
    fine: Node
    payload: Node
    coarse_value: torch.Tensor
    logical_coarse_value: torch.Tensor
    contract: PreparedQuantizationContract
    relu_applied: bool


def _prepared_pvs_contract(
    observer: FusedAmaxObsFakeQuantize,
) -> PreparedQuantizationContract | None:
    """Return the complete observer contract for an Alpha O4/PVS value."""
    contract = PreparedQuantizationContract.from_observer(observer)
    if not _PREPARED_PVS_FORMAT.accepts(contract):
        return None
    return contract


def _prepared_pvs_coarse_value(
    module: MagnetAlphaPEFakeEpilogue | MagnetAlphaNMPFakeEpilogue,
) -> torch.Tensor:
    if isinstance(module, MagnetAlphaPEFakeEpilogue):
        value = module.exact_output_scale
    else:
        value = module.output_scale
    return value.detach().reshape(-1).clone()


def _matching_prepared_pvs_sidecars(
    model: GraphModule,
) -> dict[str, Node]:
    sidecars: dict[str, Node] = {}
    for node in model.graph.nodes:
        source_name = node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_SIDECAR)
        if source_name is None:
            continue
        if source_name in sidecars:
            raise RuntimeError(
                f"{source_name}: duplicate prepared PVS sidecars"
            )
        sidecars[source_name] = node
    return sidecars


def _prepared_pvs_component_nodes(
    graph,
    sidecar: Node,
) -> tuple[Node, Node, Node]:
    components = {
        int(user.args[1]): user
        for user in sidecar.users
        if (
            user.target == operator.getitem
            and len(user.args) == 2
            and isinstance(user.args[1], int)
        )
    }
    if components:
        if set(components) != {0, 1, 2}:
            raise RuntimeError(
                f"{sidecar.name}: malformed prepared PVS components"
            )
        return components[0], components[1], components[2]

    created = []
    previous = sidecar
    for index in range(3):
        with graph.inserting_after(previous):
            component = graph.call_function(
                operator.getitem,
                (sidecar, index),
            )
        component.meta[_MAGNET_ALPHA_PREPARED_PVS_COMPONENT] = True
        created.append(component)
        previous = component
    coarse, fine, payload = created
    fine.meta["dtype"] = "uint8"
    payload.meta["dtype"] = "int4"
    set_physical_quantization_contract(
        payload,
        PhysicalQuantizationContract(
            "int4",
            ("uint8",),
            ALPHA_VSQ_BLOCK_VALUES,
        ),
    )
    return coarse, fine, payload


def _get_or_create_prepared_pvs_sidecar(
    model: GraphModule,
    fixed_node: Node,
    sidecars: dict[str, Node],
) -> tuple[Node, Node, Node]:
    graph = model.graph
    sidecar = sidecars.get(fixed_node.name)
    kwargs = dict(fixed_node.kwargs)
    kwargs["return_components"] = True
    if sidecar is None:
        with graph.inserting_after(fixed_node):
            sidecar = graph.call_module(
                str(fixed_node.target),
                fixed_node.args,
                kwargs,
            )
        sidecar.meta = {
            _MAGNET_ALPHA_PREPARED_PVS_SIDECAR: fixed_node.name,
            "dtype": (None, "uint8", "int4"),
        }
        set_physical_quantization_contract(
            sidecar,
            PhysicalQuantizationContract(
                "int4",
                ("uint8",),
                ALPHA_VSQ_BLOCK_VALUES,
            ),
        )
        sidecars[fixed_node.name] = sidecar
    else:
        sidecar.args = fixed_node.args
        sidecar.kwargs = kwargs
    return _prepared_pvs_component_nodes(graph, sidecar)


def _mark_prepared_pvs_component(node: Node, dtype: str) -> None:
    node.meta[_MAGNET_ALPHA_PREPARED_PVS_COMPONENT] = True
    node.meta["dtype"] = dtype
    if dtype == "int4":
        set_physical_quantization_contract(
            node,
            PhysicalQuantizationContract(
                "int4",
                ("uint8",),
                ALPHA_VSQ_BLOCK_VALUES,
            ),
        )


def _clone_prepared_pvs_unary(
    graph,
    semantic_node: Node,
    component: Node,
    *,
    dtype: str,
) -> Node:
    args = list(semantic_node.args)
    args[0] = component
    with graph.inserting_before(semantic_node):
        cloned = graph.call_function(
            semantic_node.target,
            tuple(args),
            dict(semantic_node.kwargs),
        )
    _mark_prepared_pvs_component(cloned, dtype)
    return cloned


def _same_prepared_pvs_scale(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    return left.shape == right.shape and torch.allclose(
        left.to(torch.float64),
        right.to(torch.float64),
        rtol=1e-6,
        atol=0.0,
    )


def _join_prepared_pvs_values(
    graph,
    join: Node,
    values: tuple[_PreparedPVSValue, ...],
) -> _PreparedPVSValue:
    reference = values[0]
    if any(value.contract != reference.contract for value in values[1:]):
        raise ValueError(
            f"{join.name}: incompatible prepared PVS observer contracts"
        )
    if any(
        not _same_prepared_pvs_scale(
            value.coarse_value,
            reference.coarse_value,
        )
        or not _same_prepared_pvs_scale(
            value.logical_coarse_value,
            reference.logical_coarse_value,
        )
        for value in values[1:]
    ):
        raise ValueError(
            f"{join.name}: prepared PVS inputs do not share one coarse scale"
        )

    input_shapes = tuple(_node_shape(node) for node in join.args[0])
    rank = len(input_shapes[0])
    axis = reference.contract.axes
    if len(axis) != 1:
        raise ValueError(
            f"{join.name}: prepared PVS join requires one block axis"
        )
    channel_axis = axis[0] % rank
    join_axis = int(get_arg_value(join, 1, "dim", 0)) % rank
    if join_axis != channel_axis:
        raise ValueError(
            f"{join.name}: prepared PVS join must concatenate blocks"
        )
    block_size = reference.contract.block_size
    if any(
        len(shape) != rank or shape[channel_axis] % block_size
        for shape in input_shapes
    ):
        raise ValueError(
            f"{join.name}: prepared PVS inputs must be block aligned"
        )

    args = list(join.args)
    args[0] = tuple(value.payload for value in values)
    with graph.inserting_before(join):
        payload = graph.call_function(
            join.target,
            tuple(args),
            dict(join.kwargs),
        )
    args[0] = tuple(value.fine for value in values)
    with graph.inserting_before(join):
        fine = graph.call_function(
            join.target,
            tuple(args),
            dict(join.kwargs),
        )
    _mark_prepared_pvs_component(payload, "int4")
    _mark_prepared_pvs_component(fine, "uint8")
    return _PreparedPVSValue(
        coarse=reference.coarse,
        fine=fine,
        payload=payload,
        coarse_value=reference.coarse_value,
        logical_coarse_value=reference.logical_coarse_value,
        contract=reference.contract,
        relu_applied=all(value.relu_applied for value in values),
    )


def _slice_prepared_pvs_value(
    graph,
    slice_node: Node,
    value: _PreparedPVSValue,
) -> _PreparedPVSValue:
    payload = _clone_prepared_pvs_unary(
        graph,
        slice_node,
        value.payload,
        dtype="int4",
    )
    fine = value.fine
    source = get_leading_channel_slice_source(slice_node)
    if source is not None:
        result_shape = _node_shape(slice_node)
        rank = len(result_shape)
        axes = value.contract.axes
        if len(axes) != 1 or axes[0] % rank != 1:
            raise ValueError(
                f"{slice_node.name}: unsupported prepared PVS block axis"
            )
        fine_channels = math.ceil(result_shape[1] / value.contract.block_size)
        with graph.inserting_before(slice_node):
            fine = graph.call_function(
                torch.ops.aten.slice.Tensor,
                (value.fine, 1, 0, fine_channels, 1),
            )
        _mark_prepared_pvs_component(fine, "uint8")
    return _PreparedPVSValue(
        coarse=value.coarse,
        fine=fine,
        payload=payload,
        coarse_value=value.coarse_value,
        logical_coarse_value=value.logical_coarse_value,
        contract=value.contract,
        relu_applied=value.relu_applied,
    )


def _validate_prepared_pvs_input(
    fixed_node: Node,
    input_observer: FusedAmaxObsFakeQuantize,
    value: _PreparedPVSValue,
) -> None:
    contract = _prepared_pvs_contract(input_observer)
    if contract != value.contract:
        raise ValueError(
            f"{fixed_node.name}: encoded input observer contract differs "
            "from its prepared PVS producer"
        )
    logical_scale = input_observer.calculate_qparams().detach().reshape(-1)
    if not _same_prepared_pvs_scale(
        logical_scale,
        value.logical_coarse_value,
    ):
        raise ValueError(
            f"{fixed_node.name}: encoded input coarse scale differs from "
            "its prepared PVS producer"
        )


def _clear_prepared_pvs_transfers(model: GraphModule) -> None:
    """Remove derived carrier edges while retaining producer sidecars."""
    graph = model.graph
    changed = False
    for node in graph.nodes:
        if "input_components" not in node.kwargs:
            continue
        kwargs = dict(node.kwargs)
        kwargs.pop("input_components")
        node.kwargs = kwargs
        changed = True

    sidecars = {
        node
        for node in graph.nodes
        if node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_SIDECAR)
    }
    direct_components = {
        node
        for node in graph.nodes
        if (
            node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_COMPONENT)
            and node.target == operator.getitem
            and isinstance(node.args[0], Node)
            and node.args[0] in sidecars
        )
    }
    derived = {
        node
        for node in graph.nodes
        if (
            node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_COMPONENT)
            and node not in direct_components
        )
    }
    for node in reversed(tuple(graph.nodes)):
        if node not in derived or node.users:
            continue
        graph.erase_node(node)
        changed = True
    dangling = [node.name for node in derived if node in graph.nodes]
    if dangling:
        raise RuntimeError(
            "Prepared PVS transfers were not removable: " + ", ".join(dangling)
        )
    if changed:
        graph.lint()
        model.recompile()


def _prepared_pvs_transfer_rules(
    modules: dict[str, torch.nn.Module],
) -> PreparedDomainRuleSet:
    """Declare the graph forms through which Alpha PVS may propagate."""

    def is_prepared_observer(node: Node) -> bool:
        if node.op != "call_module":
            return False
        return isinstance(
            modules.get(str(node.target)),
            FusedAmaxObsFakeQuantize,
        )

    def is_supported_slice(node: Node) -> bool:
        return (
            get_identity_slice_source(node) is not None
            or get_leading_channel_slice_source(node) is not None
        )

    map_targets = (
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.upsample_nearest2d.vec,
    )
    rules = [
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.MAP_COMPONENTS,
        )
        for target in map_targets
    ]
    rules.extend(
        (
            PreparedDomainTransferRule(
                target=torch.ops.aten.relu.default,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
            ),
            PreparedDomainTransferRule(
                target=torch.ops.aten.slice.Tensor,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.MAP_COMPONENTS,
                predicate=is_supported_slice,
            ),
            PreparedDomainTransferRule(
                target=torch.ops.aten.cat.default,
                input_format=PreparedDomainInputFormat.SEQUENCE,
                relation=PreparedDomainRelation.JOIN,
            ),
            PreparedDomainTransferRule(
                target=None,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
                predicate=is_prepared_observer,
            ),
        )
    )
    return PreparedDomainRuleSet(tuple(rules))


def _materialize_prepared_pvs_values(model: GraphModule) -> None:
    """Carry exact O4 components through prepared data movement into VSQ."""
    _clear_prepared_pvs_transfers(model)
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))
    sidecars = _matching_prepared_pvs_sidecars(model)
    original_nodes = [
        node
        for node in graph.nodes
        if not node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_COMPONENT)
        and not node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_SIDECAR)
    ]

    def seed_value(
        node: Node,
        values: dict[Node, _PreparedPVSValue],
    ) -> _PreparedPVSValue | None:
        module = (
            modules.get(str(node.target)) if node.op == "call_module" else None
        )
        producer_module = None
        if isinstance(module, MagnetAlphaPEFakeEpilogue):
            if not module.exact_enabled:
                return None
            if module.exact_mode.startswith("vsq_"):
                input_source = node.args[1] if len(node.args) >= 2 else None
                input_value = values.get(input_source)
                kwargs = dict(node.kwargs)
                if input_value is None:
                    kwargs.pop("input_components", None)
                else:
                    conv = node.args[0]
                    input_observer = _prepared_vsq_observer(
                        get_arg_value(conv, 0, "input"),
                        modules,
                    )
                    if input_observer is None:
                        raise ValueError(
                            f"{node.name}: encoded input has no VSQ observer"
                        )
                    _validate_prepared_pvs_input(
                        node,
                        input_observer,
                        input_value,
                    )
                    kwargs["input_components"] = (
                        input_value.coarse,
                        input_value.fine,
                        input_value.payload,
                    )
                node.kwargs = kwargs
            if not module.uses_pvs:
                return None
            producer_module = module
        elif isinstance(module, MagnetAlphaNMPFakeEpilogue):
            if not module.exact_enabled or not module.uses_pvs:
                return None
            producer_module = module

        if producer_module is None:
            return None
        output = _prepared_exact_output(node, modules)
        if output is None:
            raise ValueError(
                f"{node.name}: exact O4 producer has no output observer"
            )
        _, observer, relu = output
        contract = _prepared_pvs_contract(observer)
        if contract is None:
            raise ValueError(
                f"{node.name}: exact O4 producer has no PVS contract"
            )
        coarse, fine, payload = _get_or_create_prepared_pvs_sidecar(
            model,
            node,
            sidecars,
        )
        return _PreparedPVSValue(
            coarse=coarse,
            fine=fine,
            payload=payload,
            coarse_value=_prepared_pvs_coarse_value(producer_module),
            logical_coarse_value=(
                observer.calculate_qparams().detach().reshape(-1).clone()
            ),
            contract=contract,
            relu_applied=relu,
        )

    def transfer_value(
        node: Node,
        transfer: PreparedDomainTransfer,
        input_values: tuple[_PreparedPVSValue, ...],
    ) -> _PreparedPVSValue | None:
        source_value = input_values[0]
        module = (
            modules.get(str(node.target)) if node.op == "call_module" else None
        )
        if isinstance(module, FusedAmaxObsFakeQuantize):
            contract = _prepared_pvs_contract(module)
            if contract is None:
                return None
            if contract != source_value.contract:
                raise ValueError(
                    f"{node.name}: prepared PVS observer contract changed"
                )
            logical_scale = module.calculate_qparams().detach().reshape(-1)
            if not _same_prepared_pvs_scale(
                logical_scale,
                source_value.logical_coarse_value,
            ):
                raise ValueError(
                    f"{node.name}: prepared PVS alias changed coarse scale"
                )
            return source_value

        if node.target == torch.ops.aten.relu.default:
            if not source_value.relu_applied:
                raise ValueError(
                    f"{node.name}: prepared PVS ReLU was not encoded"
                )
            return source_value

        if (
            get_identity_slice_source(node) is not None
            or get_leading_channel_slice_source(node) is not None
        ):
            return _slice_prepared_pvs_value(
                graph,
                node,
                source_value,
            )

        if transfer.rule.relation is PreparedDomainRelation.JOIN:
            return _join_prepared_pvs_values(
                graph,
                node,
                input_values,
            )

        if transfer.rule.relation is PreparedDomainRelation.MAP_COMPONENTS:
            return _PreparedPVSValue(
                coarse=source_value.coarse,
                fine=_clone_prepared_pvs_unary(
                    graph,
                    node,
                    source_value.fine,
                    dtype="uint8",
                ),
                payload=_clone_prepared_pvs_unary(
                    graph,
                    node,
                    source_value.payload,
                    dtype="int4",
                ),
                coarse_value=source_value.coarse_value,
                logical_coarse_value=source_value.logical_coarse_value,
                contract=source_value.contract,
                relu_applied=source_value.relu_applied,
            )
        return source_value

    def partial_transfer(
        node: Node,
        transfer: PreparedDomainTransfer,
        input_values: tuple[_PreparedPVSValue | None, ...],
    ) -> None:
        if transfer.rule.relation is PreparedDomainRelation.JOIN:
            raise ValueError(
                f"{node.name}: cannot mix encoded and semantic PVS inputs"
            )

    propagate_prepared_domain_values(
        original_nodes,
        _prepared_pvs_transfer_rules(modules),
        seed_value=seed_value,
        transfer_value=transfer_value,
        partial_transfer=partial_transfer,
    )

    graph.lint()
    model.recompile()


def _remove_prepared_pvs_values(model: GraphModule) -> None:
    """Remove calibration-only component edges before target lowering."""
    modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        if not isinstance(
            modules.get(str(node.target)),
            MagnetAlphaPEFakeEpilogue,
        ):
            continue
        if "input_components" not in node.kwargs:
            continue
        kwargs = dict(node.kwargs)
        kwargs.pop("input_components")
        node.kwargs = kwargs

    removable = {
        node
        for node in model.graph.nodes
        if (
            node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_COMPONENT)
            or node.meta.get(_MAGNET_ALPHA_PREPARED_PVS_SIDECAR)
        )
    }
    for node in reversed(tuple(model.graph.nodes)):
        if node not in removable:
            continue
        remaining = [user for user in node.users if user not in removable]
        if remaining:
            names = ", ".join(user.name for user in remaining)
            raise RuntimeError(
                f"{node.name}: prepared PVS sidecar still feeds {names}"
            )
        if not node.users:
            model.graph.erase_node(node)
    dangling = [node.name for node in removable if node in model.graph.nodes]
    if dangling:
        raise RuntimeError(
            "Prepared PVS sidecars were not removable: " + ", ".join(dangling)
        )
    model.graph.lint()
    model.recompile()


def _unwrap_magnet_alpha_fake_epilogues(
    model: GraphModule,
) -> GraphModule:
    """Remove prepared-only PE and NMP reference modules."""
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))
    removed = 0
    for fixed_node in list(graph.nodes):
        if fixed_node.op != "call_module":
            continue
        if not isinstance(
            modules.get(str(fixed_node.target)),
            (MagnetAlphaPEFakeEpilogue, MagnetAlphaNMPFakeEpilogue),
        ):
            continue
        semantic_output = fixed_node.args[0] if fixed_node.args else None
        if not isinstance(semantic_output, Node):
            raise ValueError(
                f"{fixed_node.name}: Magnet Alpha reference has no "
                "semantic output"
            )
        fixed_node.replace_all_uses_with(semantic_output)
        graph.erase_node(fixed_node)
        removed += 1
    if removed:
        graph.lint()
        graph.eliminate_dead_code()
        model.recompile()
        model.delete_all_unused_submodules()
    return model


def cleanup_quantization_preparation(
    model: GraphModule,
) -> GraphModule:
    """Discard Alpha calibration/QAT instrumentation before conversion."""
    _remove_prepared_pvs_values(model)
    _unwrap_magnet_alpha_fake_epilogues(model)
    handles = model.__dict__.pop(
        "_magnet_alpha_qat_bias_hook_handles",
        (),
    )
    for handle in handles:
        handle.remove()
    return model


def prepare_quantization_recalibration(
    model: GraphModule,
    *,
    reset_observers: bool,
) -> int:
    """Configure exact Alpha PE and NMP simulation for a second pass."""
    # Every encoded join must have one physical coarse scale. Reusing the
    # normal region finalizer before each configure is idempotent when the
    # histories are already synchronized and closes drift between samples.
    _finalize_quantization_regions(model)
    pe_count = _configure_magnet_alpha_pe_fake_epilogues(
        model,
        configured_only=False,
        reset_output_observers=reset_observers,
    )
    nmp_count = _configure_magnet_alpha_nmp_fake_epilogues(
        model,
        configured_only=False,
        reset_output_observers=reset_observers,
    )
    _materialize_prepared_pvs_values(model)
    return pe_count + nmp_count


def apply_quantization_rewrites(model: GraphModule) -> GraphModule:
    """Lower supported reference quantization into Magnet Alpha target IR."""
    lower_magnet_alpha_vsq_convolutions(model)
    lower_magnet_alpha_fixed_point_epilogues(model)
    lower_magnet_alpha_depthwise_fixed_point(model)
    lower_magnet_alpha_saturating_adds(model)
    lower_magnet_alpha_internal_scalar_int8_qdq(model)
    lower_magnet_alpha_adaptive_average_pool2d(model)
    lower_magnet_alpha_physical_int8_pvs_bridges(model)
    lower_magnet_alpha_pvs_outputs(model)
    lower_magnet_alpha_integer_data_movement(model)
    finalize_magnet_alpha_vsq_operands(model)
    lower_magnet_alpha_physical_int8_relu_qdq(model)
    return fold_magnet_alpha_physical_int8_relus(model)


def prepare_quantization(
    model: GraphModule,
    region_qspec=None,
    hardware_config=None,
    operator_traits=None,
) -> GraphModule:
    """Apply Magnet Alpha's ordered PT2E observer-domain preparation."""
    _move_output_observers_after_epilogues(
        model,
        operator_traits,
    )
    insert_magnet_alpha_fixed_point_epilogues(model)
    insert_magnet_alpha_nmp_fixed_point_epilogues(model, hardware_config)
    _prepare_quantization_regions(
        model,
        region_qspec,
        operator_traits,
    )
    _bind_magnet_alpha_residual_pe_output_domains(model)
    _prepare_vsq_join_regions(model, operator_traits)
    # Region propagation can expose a producer boundary that was previously
    # separated from its output observer by a dequantize/quantize join. Give
    # those newly explicit PE boundaries the same calibrated epilogue.
    insert_magnet_alpha_fixed_point_epilogues(model)
    return insert_magnet_alpha_nmp_fixed_point_epilogues(model, hardware_config)


def finalize_quantization_calibration(model: GraphModule) -> None:
    """Merge calibrated maxima across Magnet Alpha physical domains."""
    _finalize_quantization_regions(model)
    _configure_magnet_alpha_pe_fake_epilogues(
        model,
        configured_only=True,
        reset_output_observers=False,
    )
    _configure_magnet_alpha_nmp_fake_epilogues(
        model,
        configured_only=True,
        reset_output_observers=False,
    )


__all__ = [
    "apply_quantization_rewrites",
    "cleanup_quantization_preparation",
    "finalize_quantization_calibration",
    "prepare_quantization",
    "prepare_quantization_recalibration",
]
