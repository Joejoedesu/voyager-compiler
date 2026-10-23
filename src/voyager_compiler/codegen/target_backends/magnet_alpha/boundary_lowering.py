# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha quantized boundary lowering."""

import copy
import logging
import operator
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.target_backends.magnet_alpha.fixed_point import (
    AlphaOutputDomain,
    alpha_output_domain,
    encode_alpha_nmp_pipeline,
    ideal_int8_qdq_expected_codes,
    prove_exact_identity_dwc_encoding,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering_utils import (
    direct_residual_output_scale,
    node_shape,
    relu_reaches_alpha_eadd,
    residual_dequantized_path,
    residual_quantizer_scale,
    resolve_static_tensor,
    vsq_quantizer_scale_endpoints,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.residual_lowering import (
    materialize_identity_dwc,
)
from voyager_compiler.pt2e_utils import (
    create_getattr_from_value,
    propagate_shape,
)
from voyager_compiler.quantization_contracts import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    get_physical_dtype,
    get_physical_quantization_contract,
    has_physical_payload,
    set_physical_quantization_contract,
)
from voyager_compiler.quantization_regions import (
    _QUANTIZATION_DOMAIN_ALIAS,
    _QUANTIZATION_DOMAIN_ENTRY,
)

logger = logging.getLogger(__name__)


INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META = (
    "magnet_alpha_internal_scalar_int8_qdq_summary"
)
PHYSICAL_INT8_PVS_BRIDGE_SUMMARY_META = (
    "magnet_alpha_physical_int8_pvs_bridge_summary"
)


@dataclass(frozen=True)
class AlphaPVSQuantizerInfo:
    """Validated static O4/PVS facts from one reference VSQ quantizer."""

    semantic_coarse: Node
    coarse_value: torch.Tensor
    physical_coarse_value: torch.Tensor
    output_domain: AlphaOutputDomain
    output_precision: int
    block_size: int


def _alpha_pvs_quantizer_info(
    model: GraphModule,
    quantizer: Node,
) -> AlphaPVSQuantizerInfo | None:
    if quantizer.target != torch.ops.quantized_ops.quantize_vsq.default:
        return None
    try:
        axes = tuple(int(axis) for axis in get_arg_value(quantizer, 4, "axes"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{quantizer.name}: Alpha PVS requires one static VSQ axis"
        ) from error
    block_size_value = get_arg_value(quantizer, 5, "block_size")
    if type(block_size_value) is not int:
        raise ValueError(
            f"{quantizer.name}: Alpha PVS block size must be an integer"
        )
    block_size = block_size_value
    if axes != (1,) or block_size != ALPHA_VSQ_BLOCK_VALUES:
        return None

    dtypes = quantizer.meta.get("dtype")
    if not isinstance(dtypes, (tuple, list)) or len(dtypes) != 3:
        raise ValueError(
            f"{quantizer.name}: Alpha PVS requires VSQ output dtypes"
        )
    scale_dtype, payload_dtype = dtypes[1], dtypes[2]
    if scale_dtype != "uint8" or payload_dtype != "int4":
        raise ValueError(
            f"{quantizer.name}: Alpha PVS requires a uint8 companion "
            "and an int4 payload"
        )
    output_precision = int(payload_dtype.removeprefix("int"))
    semantic_coarse = get_arg_value(quantizer, 1, "coarse_scale")
    coarse_value = resolve_static_tensor(model, semantic_coarse)
    if (
        not isinstance(semantic_coarse, Node)
        or coarse_value is None
        or coarse_value.numel() != 1
        or not torch.is_floating_point(coarse_value)
        or not torch.isfinite(coarse_value).all()
        or coarse_value.reshape(-1)[0] <= 0
    ):
        raise ValueError(
            f"{quantizer.name}: Alpha PVS requires one static positive "
            "floating coarse scale"
        )
    fine_endpoint, coarse_endpoint = vsq_quantizer_scale_endpoints(quantizer)
    output = alpha_output_domain(
        coarse_value,
        output_precision,
        uses_pvs=True,
        fine_quant_max=fine_endpoint,
        coarse_scale_quant_max=coarse_endpoint,
    )
    return AlphaPVSQuantizerInfo(
        semantic_coarse=semantic_coarse,
        coarse_value=coarse_value,
        physical_coarse_value=output.scale.reshape(coarse_value.shape),
        output_domain=output,
        output_precision=output_precision,
        block_size=block_size,
    )


def lower_magnet_alpha_pvs_outputs(model: GraphModule) -> GraphModule:
    """Replace eligible O4 VSQ boundaries with Alpha PVS encoding."""
    graph = model.graph
    lowered = 0
    for quantizer in list(graph.nodes):
        if quantizer.target != torch.ops.quantized_ops.quantize_vsq.default:
            continue
        source = get_arg_value(quantizer, 0, "self")
        if not isinstance(source, Node):
            continue
        if source.target == PPU_FIXED_RELU_OPERATOR and not bool(
            source.kwargs.get("integer_accumulator", False)
        ):
            raise ValueError(
                f"{quantizer.name}: Alpha PVS requires an exact integer "
                "pre-PVS producer; W8-to-O4 PE lowering is unsupported"
            )
        if source.target not in (
            NMP_FIXED_REQUANTIZE_OPERATOR,
            PPU_FIXED_RELU_OPERATOR,
        ):
            continue
        info = _alpha_pvs_quantizer_info(model, quantizer)
        if info is None:
            logger.debug(
                "%s: leaving non-Alpha VSQ boundary unlowered",
                quantizer.name,
            )
            continue
        with graph.inserting_before(quantizer):
            physical_coarse = create_getattr_from_value(
                model,
                graph,
                f"{quantizer.name}_alpha_coarse_scale",
                info.physical_coarse_value,
            )
            target_node = graph.call_function(
                PVS_QUANTIZE_OPERATOR,
                (source, physical_coarse),
                {
                    "output_precision": info.output_precision,
                    "block_size": info.block_size,
                },
            )
        propagate_shape(physical_coarse, model)
        physical_coarse.meta["magnet_alpha_instruction_operand"] = True
        info.semantic_coarse.meta["magnet_alpha_semantic_operand"] = True
        target_node.meta = copy.copy(quantizer.meta)
        target_node.meta["dtype"] = (None, "uint8", "int4")
        domain = PhysicalQuantizationContract(
            "int4",
            ("uint8",),
            info.block_size,
        )
        set_physical_quantization_contract(target_node, domain)
        quantizer.replace_all_uses_with(target_node)
        for user in target_node.users:
            if user.target != operator.getitem:
                continue
            component = int(user.args[1])
            if component == 1:
                user.meta["dtype"] = "uint8"
            elif component == 2:
                user.meta["dtype"] = "int4"
                set_physical_quantization_contract(user, domain)
        graph.erase_node(quantizer)
        lowered += 1
    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info("Lowered %d Magnet Alpha PVS outputs", lowered)
    return model


@dataclass(frozen=True)
class PhysicalInt8PVSBridgePlan:
    """One branch-local physical O8 to O4/PVS transition."""

    source: Node
    dequantize: Node
    relu: Node | None
    quantizer: Node
    quantizer_info: AlphaPVSQuantizerInfo
    source_scale: float
    shape: tuple[int, ...]
    scale_factor: int
    binary_point: int


def _physical_int8_pvs_bridge_plan(
    model: GraphModule,
    quantizer: Node,
) -> PhysicalInt8PVSBridgePlan | None:
    if quantizer.target != torch.ops.quantized_ops.quantize_vsq.default:
        return None
    boundary = get_arg_value(quantizer, 0, "self")
    if not isinstance(boundary, Node):
        return None
    relu = None
    if boundary.target == torch.ops.aten.relu.default:
        relu = boundary
        dequantize = get_arg_value(relu, 0, "self")
    else:
        dequantize = boundary
    if (
        not isinstance(dequantize, Node)
        or dequantize.target != torch.ops.quantized_ops.dequantize.default
    ):
        return None
    source = get_arg_value(dequantize, 0, "input")
    if not has_physical_payload(source, "int8"):
        return None
    assert isinstance(source, Node)
    expected_user = relu if relu is not None else quantizer
    if (
        tuple(dequantize.users) != (expected_user,)
        or get_arg_value(quantizer, 0, "self") is not boundary
        or get_physical_quantization_contract(dequantize) is not None
        or get_physical_dtype(dequantize) is not None
    ):
        raise ValueError(
            f"{quantizer.name}: Alpha O8-to-PVS dequantizer must be one "
            "uncontracted branch-local scalar boundary"
        )
    if relu is not None and (
        tuple(relu.users) != (quantizer,)
        or get_arg_value(relu, 0, "self") is not dequantize
        or len(relu.all_input_nodes) != 1
        or get_physical_quantization_contract(relu) is not None
        or get_physical_dtype(relu) is not None
    ):
        raise ValueError(
            f"{quantizer.name}: Alpha O8-to-PVS ReLU must be an "
            "uncontracted branch-local unary boundary"
        )
    if source.target == PPU_FIXED_RELU_OPERATOR:
        if (
            source.kwargs.get("integer_accumulator") is not True
            or source.kwargs.get("output_precision") != 8
        ):
            raise ValueError(
                f"{quantizer.name}: Alpha PVS bridge requires an exact O8 "
                "integer PPU source"
            )
    elif source.target == NMP_FIXED_REQUANTIZE_OPERATOR:
        if source.kwargs.get("output_precision") != 8:
            raise ValueError(
                f"{quantizer.name}: Alpha PVS bridge requires an O8 NMP "
                "source"
            )
    else:
        raise ValueError(
            f"{quantizer.name}: physical O8 source {source.name} is not a "
            "proved Alpha PPU or NMP producer"
        )
    try:
        path, source_scale = residual_dequantized_path(model, dequantize)
    except ValueError as error:
        raise ValueError(
            f"{quantizer.name}: invalid Alpha O8-to-PVS dequantizer: {error}"
        ) from error
    producer_scale = direct_residual_output_scale(source)
    if (
        path.source is not source
        or path.rescale_output_scales
        or producer_scale != source_scale
    ):
        raise ValueError(
            f"{quantizer.name}: Alpha O8-to-PVS must consume one direct "
            "physical code source in its declared scalar domain"
        )
    info = _alpha_pvs_quantizer_info(model, quantizer)
    if info is None:
        raise ValueError(
            f"{quantizer.name}: physical O8 bridge requires Alpha block-32 "
            "channel-axis O4/PVS quantization"
        )
    boundary_nodes = (dequantize,) if relu is None else (dequantize, relu)
    shape = node_shape(source)
    if not shape:
        for boundary_node in reversed(boundary_nodes):
            shape = node_shape(boundary_node)
            if shape:
                break
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[1] <= 1
        or shape[1] % info.block_size
        or any(extent <= 0 for extent in shape)
        or any(
            boundary_shape and boundary_shape != shape
            for boundary_shape in map(node_shape, boundary_nodes)
        )
    ):
        raise ValueError(
            f"{quantizer.name}: Alpha O8-to-PVS requires equal static NCHW "
            "shapes with N=1 and block-aligned C>1"
        )
    try:
        pipeline = encode_alpha_nmp_pipeline(
            info.coarse_value.new_tensor([source_scale]),
            None,
            info.output_domain,
            relu=relu is not None,
        )
    except ValueError as error:
        raise ValueError(
            f"{quantizer.name}: Alpha O8-to-PVS NMP is not encodable: "
            f"{error}"
        ) from error
    bias = pipeline.bias_relu.bias
    if (
        pipeline.output.precision != 4
        or not pipeline.output.uses_pvs
        or bias is None
        or bias.numel() != 1
        or int(bias.reshape(-1)[0].item()) != 0
    ):
        raise AssertionError("Alpha O8-to-PVS pipeline contract drifted")
    return PhysicalInt8PVSBridgePlan(
        source=source,
        dequantize=dequantize,
        relu=relu,
        quantizer=quantizer,
        quantizer_info=info,
        source_scale=source_scale,
        shape=shape,
        scale_factor=int(pipeline.primary.multiplier.reshape(-1)[0].item()),
        binary_point=pipeline.primary.shift,
    )


def lower_magnet_alpha_physical_int8_pvs_bridges(
    model: GraphModule,
) -> GraphModule:
    """Materialize proved physical O8 input branches for Alpha PVS."""
    existing = model.meta.get(PHYSICAL_INT8_PVS_BRIDGE_SUMMARY_META)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("schema_version") != 1
        ):
            raise ValueError("invalid existing Alpha O8-to-PVS bridge summary")
        return model
    plans = [
        plan
        for node in list(model.graph.nodes)
        if (plan := _physical_int8_pvs_bridge_plan(model, node)) is not None
    ]
    boundaries = []
    for plan in plans:
        boundary = plan.relu if plan.relu is not None else plan.dequantize
        physical_scale = float(
            plan.quantizer_info.physical_coarse_value.reshape(-1)[0].item()
        )
        output = materialize_identity_dwc(
            model,
            plan.quantizer,
            plan.source,
            plan.source_scale,
            physical_scale,
            relu=plan.relu is not None,
            prefix=f"{plan.quantizer.name}_physical_o8_to_pvs",
            output_meta=boundary.meta,
            proved_encoding=(plan.scale_factor, plan.binary_point),
            proved_shape=plan.shape,
            output_precision=4,
        )
        plan.quantizer.replace_input_with(boundary, output)
        if plan.relu is not None:
            model.graph.erase_node(plan.relu)
        model.graph.erase_node(plan.dequantize)
        boundaries.append(
            {
                "source": plan.source.name,
                "dequantize": plan.dequantize.name,
                "relu": plan.relu.name if plan.relu is not None else None,
                "quantizer": plan.quantizer.name,
                "source_scale": plan.source_scale,
                "physical_coarse_scale": physical_scale,
                "shape": plan.shape,
                "scale_factor": plan.scale_factor,
                "binary_point": plan.binary_point,
                "output_precision": 4,
                "uses_pvs": True,
            }
        )
    if plans:
        model.graph.lint()
        model.graph.eliminate_dead_code()
        model.recompile()
        model.delete_all_unused_submodules()
    model.meta[PHYSICAL_INT8_PVS_BRIDGE_SUMMARY_META] = {
        "schema_version": 1,
        "semantic_contract": "target_native_o8_to_o4_nmp_pvs",
        "lowered_count": len(plans),
        "relu_count": sum(plan.relu is not None for plan in plans),
        "boundaries": tuple(boundaries),
    }
    logger.info(
        "Lowered %d physical Alpha O8-to-PVS input bridges",
        len(plans),
    )
    return model


@dataclass(frozen=True)
class InternalScalarInt8QDQPlan:
    """One bare internal signed-INT8 Q(DQ(code)) transition."""

    source: Node
    dequantize: Node
    quantize: Node
    source_scale: float
    output_scale: float
    shape: tuple[int, ...]
    adopt_quantize_domain: bool

    @property
    def is_identity(self) -> bool:
        return self.source_scale == self.output_scale


def _internal_scalar_int8_qdq_plan(
    model: GraphModule,
    dequantize: Node,
) -> InternalScalarInt8QDQPlan | None:
    if (
        dequantize.op != "call_function"
        or dequantize.target != torch.ops.quantized_ops.dequantize.default
    ):
        return None
    source = get_arg_value(dequantize, 0, "input")
    if (
        not isinstance(source, Node)
        or source.op not in ("call_function", "call_module")
        or not has_physical_payload(source, "int8")
    ):
        return None
    users = tuple(dequantize.users)
    if (
        len(users) != 1
        or users[0].target != torch.ops.quantized_ops.quantize.default
    ):
        return None
    quantize = users[0]
    if get_arg_value(quantize, 0, "input") is not dequantize:
        return None
    if not quantize.users or any(
        user.op == "output" for user in quantize.users
    ):
        return None
    shape = node_shape(source)
    quantize_shape = node_shape(quantize)
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[1] <= 1
        or any(extent <= 0 for extent in shape)
        or (quantize_shape and quantize_shape != shape)
    ):
        raise ValueError(
            f"{quantize.name}: internal Alpha INT8 QDQ requires equal "
            "static NCHW shapes with N=1 and C>1"
        )
    if get_physical_quantization_contract(dequantize) is not None:
        raise ValueError(
            f"{dequantize.name}: scalar dequantization cannot carry a "
            "physical output contract"
        )
    source_contract = get_physical_quantization_contract(source)
    quantize_contract = get_physical_quantization_contract(quantize)
    if quantize_contract not in (None, source_contract):
        raise ValueError(
            f"{quantize.name}: output physical contract differs from its "
            "signed-INT8 code source"
        )
    try:
        path, source_scale = residual_dequantized_path(model, dequantize)
        output_scale = residual_quantizer_scale(model, quantize)
    except ValueError as error:
        raise ValueError(
            f"{quantize.name}: invalid internal Alpha scalar INT8 QDQ: "
            f"{error}"
        ) from error
    if path.source is not source or path.rescale_output_scales:
        raise ValueError(
            f"{quantize.name}: internal Alpha scalar INT8 QDQ must consume "
            "one direct physical code source"
        )
    if path.source_scale != source_scale:
        raise AssertionError("internal Alpha QDQ source-scale proof drifted")
    source_region = source.meta.get(QUANTIZATION_REGION)
    quantize_region = quantize.meta.get(QUANTIZATION_REGION)
    adopt_quantize_domain = False
    if source_scale == output_scale:
        if (
            source_region is not None
            and quantize_region is not None
            and source_region != quantize_region
        ):
            raise ValueError(
                f"{quantize.name}: output quantization region differs from "
                "its signed-INT8 code source"
            )
        if source_region is None and quantize_region is not None:
            if (
                tuple(source.users) != (dequantize,)
                or quantize.meta.get(_QUANTIZATION_DOMAIN_ENTRY) is not True
            ):
                raise ValueError(
                    f"{quantize.name}: output quantization region cannot be "
                    "adopted by a branched or non-entry code source"
                )
            adopt_quantize_domain = True
    return InternalScalarInt8QDQPlan(
        source,
        dequantize,
        quantize,
        source_scale,
        output_scale,
        shape,
        adopt_quantize_domain,
    )


def _adopt_internal_qdq_output_domain(
    source: Node,
    quantize: Node,
) -> None:
    for key in (
        PHYSICAL_QUANTIZATION_CONTRACT,
        QUANTIZATION_REGION,
        _QUANTIZATION_DOMAIN_ENTRY,
        _QUANTIZATION_DOMAIN_ALIAS,
    ):
        if key in quantize.meta:
            source.meta[key] = copy.copy(quantize.meta[key])


def lower_magnet_alpha_internal_scalar_int8_qdq(
    model: GraphModule,
) -> GraphModule:
    """Canonicalize proved internal scalar QDQ in Alpha's integer domain."""
    existing = model.meta.get(INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("schema_version") != 1
            or existing.get("semantic_contract")
            != "target_native_signed_int8_codes"
        ):
            raise ValueError(
                "invalid existing Alpha internal scalar INT8 QDQ summary"
            )
        return model
    graph = model.graph
    plans = [
        plan
        for node in list(graph.nodes)
        if (plan := _internal_scalar_int8_qdq_plan(model, node)) is not None
    ]
    boundaries = []
    lowered = 0
    identity_count = 0
    requantize_count = 0
    unsupported_count = 0
    for plan in plans:
        expected_codes = ideal_int8_qdq_expected_codes(
            plan.source_scale,
            plan.output_scale,
        )
        if plan.is_identity and not torch.equal(
            expected_codes,
            torch.arange(-128, 128, dtype=torch.int64),
        ):
            raise AssertionError(
                f"{plan.quantize.name}: equal Alpha QDQ scales are not an "
                "ideal signed-INT8 identity"
            )
        try:
            proved_encoding = prove_exact_identity_dwc_encoding(
                plan.source_scale,
                plan.output_scale,
                relu=False,
                context=f"{plan.quantize.name} internal scalar INT8 QDQ",
                expected_codes=expected_codes,
            )
        except ValueError as error:
            boundaries.append(
                {
                    "source": plan.source.name,
                    "dequantize": plan.dequantize.name,
                    "quantize": plan.quantize.name,
                    "source_scale": plan.source_scale,
                    "output_scale": plan.output_scale,
                    "shape": plan.shape,
                    "disposition": "unencodable",
                    "reason": str(error),
                    "alpha_compute_owner_added": False,
                }
            )
            unsupported_count += 1
            continue
        if plan.is_identity:
            if plan.adopt_quantize_domain:
                _adopt_internal_qdq_output_domain(plan.source, plan.quantize)
            plan.quantize.replace_all_uses_with(plan.source)
            disposition = "identity_alias"
            owner = plan.source.name
            identity_count += 1
            owner_added = False
        else:
            output = materialize_identity_dwc(
                model,
                plan.quantize,
                plan.source,
                plan.source_scale,
                plan.output_scale,
                relu=False,
                prefix=f"{plan.quantize.name}_internal_qdq",
                output_meta=plan.quantize.meta,
                proved_encoding=proved_encoding,
            )
            plan.quantize.replace_all_uses_with(output)
            disposition = "depthwise_nmp_requantize"
            owner = output.name
            requantize_count += 1
            owner_added = True
        graph.erase_node(plan.quantize)
        graph.erase_node(plan.dequantize)
        lowered += 1
        boundaries.append(
            {
                "source": plan.source.name,
                "dequantize": plan.dequantize.name,
                "quantize": plan.quantize.name,
                "source_scale": plan.source_scale,
                "output_scale": plan.output_scale,
                "shape": plan.shape,
                "disposition": disposition,
                "owner": owner,
                "scale_factor": proved_encoding[0],
                "binary_point": proved_encoding[1],
                "ideal_integer_all_code_proof": True,
                "alpha_compute_owner_added": owner_added,
                "quantization_domain_adopted": plan.adopt_quantize_domain,
            }
        )
    if lowered:
        graph.lint()
        graph.eliminate_dead_code()
        model.recompile()
        model.delete_all_unused_submodules()
    model.meta[INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META] = {
        "schema_version": 1,
        "semantic_contract": "target_native_signed_int8_codes",
        "float32_container_roundtrip_preserved": False,
        "converted_to_target_numerical_gate_required": True,
        "recognized_count": len(plans),
        "lowered_count": lowered,
        "identity_alias_count": identity_count,
        "depthwise_nmp_requantize_count": requantize_count,
        "unencodable_count": unsupported_count,
        "boundaries": tuple(boundaries),
    }
    logger.info(
        "Canonicalized %d internal Alpha scalar INT8 QDQ boundaries "
        "(%d identity aliases, %d DWC/NMP requantizers, %d unsupported)",
        len(plans),
        identity_count,
        requantize_count,
        unsupported_count,
    )
    return model


@dataclass(frozen=True)
class PhysicalInt8ReluQDQPlan:
    """One exact physical PPU/DQ/ReLU/Q transition lowered through NMP."""

    ppu: Node
    dequantize: Node
    relu: Node
    quantize: Node
    source_scale: float
    output_scale: float
    scale_factor: int
    binary_point: int


def _physical_int8_relu_qdq_plan(
    model: GraphModule,
    ppu: Node,
) -> PhysicalInt8ReluQDQPlan | None:
    if (
        ppu.op != "call_function"
        or ppu.target != PPU_FIXED_RELU_OPERATOR
        or ppu.kwargs.get("relu") is not False
        or ppu.kwargs.get("integer_accumulator") is not True
        or type(ppu.kwargs.get("output_precision")) is not int
        or ppu.kwargs["output_precision"] != 8
        or ppu.kwargs.get("saturate") is not True
        or get_physical_quantization_contract(ppu)
        != PhysicalQuantizationContract("int8")
        or get_physical_dtype(ppu) != "int8"
        or ppu.meta.get(QUANTIZATION_REGION) is not None
    ):
        return None
    ppu_users = tuple(ppu.users)
    if (
        len(ppu_users) != 1
        or ppu_users[0].op != "call_function"
        or ppu_users[0].target != torch.ops.quantized_ops.dequantize.default
    ):
        return None
    dequantize = ppu_users[0]
    users = tuple(dequantize.users)
    if (
        len(users) != 1
        or users[0].op != "call_function"
        or users[0].target != torch.ops.aten.relu.default
    ):
        return None
    relu = users[0]
    users = tuple(relu.users)
    if (
        len(users) != 1
        or users[0].op != "call_function"
        or users[0].target != torch.ops.quantized_ops.quantize.default
    ):
        return None
    quantize = users[0]
    if (
        get_arg_value(dequantize, 0, "input") is not ppu
        or get_arg_value(relu, 0, "self") is not dequantize
        or get_arg_value(quantize, 0, "input") is not relu
        or len(relu.all_input_nodes) != 1
        or relu_reaches_alpha_eadd(relu)
        or any(
            get_physical_quantization_contract(node) is not None
            for node in (dequantize, relu, quantize)
        )
        or get_physical_dtype(dequantize) is not None
        or get_physical_dtype(relu) is not None
        or get_physical_dtype(quantize) != "int8"
        or any(
            node.meta.get(QUANTIZATION_REGION) is not None
            for node in (dequantize, relu, quantize)
        )
    ):
        return None
    shape = node_shape(relu)
    if (
        len(shape) != 4
        or shape[0] != 1
        or shape[1] <= 1
        or any(extent <= 0 for extent in shape)
    ):
        return None
    try:
        path, source_scale = residual_dequantized_path(model, dequantize)
        output_scale = residual_quantizer_scale(model, quantize)
    except ValueError:
        return None
    if path.source is not ppu or path.rescale_output_scales:
        return None
    source_value = resolve_static_tensor(
        model,
        get_arg_value(dequantize, 1, "scale"),
    )
    output_value = resolve_static_tensor(
        model,
        get_arg_value(quantize, 1, "scale"),
    )
    if source_value is None or output_value is None:
        return None
    source_value = source_value.detach().cpu().reshape(-1)
    output_value = output_value.detach().cpu().reshape(-1)
    if (
        source_value.numel() != 1
        or output_value.numel() != 1
        or not torch.is_floating_point(source_value)
        or not torch.is_floating_point(output_value)
        or not torch.isfinite(source_value).all()
        or not torch.isfinite(output_value).all()
        or source_value[0] <= 0
        or output_value[0] <= 0
        or float(source_value[0].item()) != source_scale
        or float(output_value[0].item()) != output_scale
    ):
        return None
    codes = torch.arange(-128, 128, dtype=torch.int64)
    expected = torch.clamp(
        torch.floor(torch.relu(codes * source_value[0]) / output_value[0]),
        -128,
        127,
    ).to(torch.int64)
    try:
        scale_factor, binary_point = prove_exact_identity_dwc_encoding(
            source_scale,
            output_scale,
            relu=True,
            context=f"{relu.name} physical PPU/DQ/ReLU/Q",
            expected_codes=expected,
        )
    except ValueError:
        return None
    return PhysicalInt8ReluQDQPlan(
        ppu,
        dequantize,
        relu,
        quantize,
        source_scale,
        output_scale,
        scale_factor,
        binary_point,
    )


def lower_magnet_alpha_physical_int8_relu_qdq(
    model: GraphModule,
) -> GraphModule:
    """Lower proved PPU/DQ/ReLU/Q chains to identity DWC plus NMP ReLU."""
    graph = model.graph
    plans = [
        plan
        for node in list(graph.nodes)
        if (plan := _physical_int8_relu_qdq_plan(model, node)) is not None
    ]
    summaries = []
    for plan in plans:
        output = materialize_identity_dwc(
            model,
            plan.relu,
            plan.ppu,
            plan.source_scale,
            plan.output_scale,
            relu=True,
            prefix=f"{plan.relu.name}_physical_qdq",
            output_meta=plan.quantize.meta,
            proved_encoding=(plan.scale_factor, plan.binary_point),
        )
        plan.quantize.replace_all_uses_with(output)
        graph.erase_node(plan.quantize)
        graph.erase_node(plan.relu)
        graph.erase_node(plan.dequantize)
        summaries.append(
            {
                "source": plan.ppu.name,
                "source_scale": plan.source_scale,
                "output_scale": plan.output_scale,
                "scale_factor": plan.scale_factor,
                "binary_point": plan.binary_point,
            }
        )
    if plans:
        graph.lint()
        graph.eliminate_dead_code()
        model.recompile()
        model.delete_all_unused_submodules()
    model.meta["magnet_alpha_physical_int8_relu_qdq_summary"] = {
        "lowered_count": len(plans),
        "boundaries": tuple(summaries),
    }
    logger.info(
        "Lowered %d exact physical INT8 PPU/DQ/ReLU/Q boundaries through "
        "Alpha identity DWC/NMP ReLU",
        len(plans),
    )
    return model


__all__ = [
    "INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META",
    "PHYSICAL_INT8_PVS_BRIDGE_SUMMARY_META",
    "lower_magnet_alpha_internal_scalar_int8_qdq",
    "lower_magnet_alpha_physical_int8_pvs_bridges",
    "lower_magnet_alpha_physical_int8_relu_qdq",
    "lower_magnet_alpha_pvs_outputs",
]
