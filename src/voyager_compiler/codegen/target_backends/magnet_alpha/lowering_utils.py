# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared facts and graph helpers for Magnet Alpha target lowering."""

import copy
import math
import operator
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.passes.utils import (
    get_arg_value,
    get_identity_slice_source,
    get_leading_channel_slice_source,
)
from voyager_compiler.codegen.storage_predicates import is_address_only_view
from voyager_compiler.codegen.target_backends.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
from voyager_compiler.pt2e_utils import fetch_attr
from voyager_compiler.quantization_contracts import (
    QUANTIZATION_REGION,
    has_physical_payload,
)
from voyager_compiler.quantization_regions import (
    _PREPARED_DOMAIN_JOIN_TARGETS,
    _PREPARED_DOMAIN_PASSTHROUGH_TARGETS,
)

RESIDUAL_ADD_TARGETS = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    }
)
RESIDUAL_RELU_TARGETS = frozenset(
    {
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
    }
)


@dataclass(frozen=True)
class ResidualCodePath:
    """Physical source plus each generic Q/DQ scale transition in order."""

    source: Node
    source_scale: float
    rescale_output_scales: tuple[float, ...] = ()

    @property
    def output_scale(self) -> float:
        if self.rescale_output_scales:
            return self.rescale_output_scales[-1]
        return self.source_scale


def resolve_static_tensor(model: GraphModule, value):
    """Return one statically bound tensor operand, if available."""
    if not isinstance(value, Node) or value.op != "get_attr":
        return None
    resolved = fetch_attr(model, value.target)
    return resolved if isinstance(resolved, torch.Tensor) else None


def clone_bias_free_conv(graph, conv: Node) -> Node:
    """Clone one target accumulator while leaving bias for its epilogue."""
    if conv.target == VSQ_CONV2D_OPERATOR:
        args = list(conv.args)
        kwargs = dict(conv.kwargs)
        if len(args) > 2:
            args[2] = None
        else:
            kwargs["bias"] = None
        target = conv.target
        target_args = tuple(args)
        target_kwargs = kwargs
    else:
        input_value = get_arg_value(conv, 0, "input")
        weight_value = get_arg_value(conv, 1, "weight")
        if input_value is None or weight_value is None:
            raise ValueError(
                f"{conv.name}: integer Conv2d has no input or weight"
            )
        target = INT8_CONV2D_ACCUMULATOR_OPERATOR
        target_args = (
            input_value,
            weight_value,
            None,
            get_arg_value(conv, 3, "stride", (1, 1)),
            get_arg_value(conv, 4, "padding", (0, 0)),
            get_arg_value(conv, 5, "dilation", (1, 1)),
            get_arg_value(conv, 6, "groups", 1),
        )
        target_kwargs = {}
    with graph.inserting_before(conv):
        raw_conv = graph.call_function(
            target,
            target_args,
            target_kwargs,
        )
    raw_conv.meta = copy.copy(conv.meta)
    raw_conv.meta["dtype"] = "int32"
    return raw_conv


def vsq_quantizer_scale_endpoints(quantizer: Node) -> tuple[float, float]:
    """Return the explicit fine/coarse endpoint pair of one VSQ node."""
    fine_endpoint = float(
        get_arg_value(
            quantizer,
            6,
            "fine_quant_max",
            ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        )
    )
    coarse_endpoint = get_arg_value(
        quantizer,
        10,
        "coarse_scale_quant_max",
        ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    )
    if coarse_endpoint is None:
        coarse_endpoint = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
    return fine_endpoint, float(coarse_endpoint)


def vsq_scale_endpoints(
    scale,
    visited: set[Node] | None = None,
) -> tuple[float, float]:
    """Resolve the logical VSQ endpoints that produced one fine scale."""
    if not isinstance(scale, Node):
        return (
            ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
            ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
        )
    if visited is None:
        visited = set()
    if scale in visited:
        raise ValueError(f"{scale.name}: cyclic VSQ scale provenance")
    visited.add(scale)

    if (
        scale.target == operator.getitem
        and len(scale.args) == 2
        and scale.args[1] == 1
        and isinstance(scale.args[0], Node)
        and scale.args[0].target == torch.ops.quantized_ops.quantize_vsq.default
    ):
        quantizer = scale.args[0]
        return vsq_quantizer_scale_endpoints(quantizer)

    if scale.target == torch.ops.aten.cat.default:
        inputs = tuple(scale.args[0])
        if inputs:
            endpoints = {
                vsq_scale_endpoints(value, visited.copy()) for value in inputs
            }
            if len(endpoints) != 1:
                raise ValueError(
                    f"{scale.name}: joined VSQ scales use mixed endpoints"
                )
            return endpoints.pop()

    source = get_identity_slice_source(scale)
    if source is None:
        source = get_leading_channel_slice_source(scale)
    if source is None and scale.target in {
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.upsample_nearest2d.vec,
    }:
        candidate = get_arg_value(scale, 0, "input")
        source = candidate if isinstance(candidate, Node) else None
    if source is not None:
        return vsq_scale_endpoints(source, visited)

    return (
        ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    )


def node_shape(node: Node) -> tuple[int, ...]:
    """Return one finalized static FX tensor shape."""
    shape = getattr(node, "shape", None)
    if shape is None:
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        shape = getattr(node.meta.get("val"), "shape", ())
    return tuple(int(extent) for extent in shape)


def static_quantizer_info(model: GraphModule, quantizer: Node):
    """Resolve one static scalar output-domain description."""
    if quantizer.target == torch.ops.quantized_ops.quantize_vsq.default:
        scale_value = get_arg_value(quantizer, 1, "coarse_scale")
        dtype = quantizer.meta.get("dtype")
        if isinstance(dtype, (tuple, list)) and len(dtype) >= 3:
            dtype = dtype[2]
        uses_vector_scale = True
        fine_endpoint, coarse_endpoint = vsq_quantizer_scale_endpoints(
            quantizer
        )
    elif quantizer.target == torch.ops.quantized_ops.quantize.default:
        scale_value = get_arg_value(quantizer, 1, "scale")
        dtype = quantizer.meta.get("dtype")
        uses_vector_scale = False
        fine_endpoint = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
        coarse_endpoint = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
    else:
        return None

    scale = resolve_static_tensor(model, scale_value)
    if scale is None or scale.numel() != 1:
        return None
    scale_float = float(scale.detach().reshape(-1)[0].item())
    if not torch.isfinite(scale).all() or scale_float <= 0:
        return None
    if dtype not in ("int4", "int8"):
        return None
    return (
        scale_float,
        int(dtype.removeprefix("int")),
        uses_vector_scale,
        fine_endpoint,
        coarse_endpoint,
    )


def output_quantizers(
    model: GraphModule,
    boundary: Node,
) -> tuple[Node, ...]:
    """Find quantizers owned by one output domain without crossing compute."""
    transparent = (
        _PREPARED_DOMAIN_JOIN_TARGETS | _PREPARED_DOMAIN_PASSTHROUGH_TARGETS
    )
    quantizers = []
    queue = list(boundary.users)
    visited: set[Node] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate in visited:
            continue
        visited.add(candidate)
        if static_quantizer_info(model, candidate) is not None:
            quantizers.append(candidate)
        elif candidate.target in transparent:
            queue.extend(candidate.users)
        elif (
            get_identity_slice_source(candidate) is not None
            or get_leading_channel_slice_source(candidate) is not None
        ):
            queue.extend(candidate.users)
    if quantizers:
        return tuple(quantizers)

    region = boundary.meta.get(QUANTIZATION_REGION)
    if region is not None:
        quantizers = [
            candidate
            for candidate in model.graph.nodes
            if candidate.meta.get(QUANTIZATION_REGION) == region
            and static_quantizer_info(model, candidate) is not None
        ]
    return tuple(quantizers)


def output_quantizer_info(model: GraphModule, boundary: Node):
    """Require every branch to expose one equal static output domain."""
    quantizers = output_quantizers(model, boundary)
    infos = [static_quantizer_info(model, node) for node in quantizers]
    if not quantizers:
        return None
    if any(info is None for info in infos):
        raise ValueError(
            f"{boundary.name}: output quantizer has no static Alpha domain"
        )
    infos = [info for info in infos if info is not None]
    reference = infos[0]
    for info in infos[1:]:
        if info[1:] != reference[1:] or not math.isclose(
            info[0],
            reference[0],
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"{boundary.name}: branched output quantizers do not share "
                "one Alpha output domain"
            )
    return reference


def static_positive_residual_scale(
    model: GraphModule,
    value,
    context: str,
) -> float:
    """Require one static positive scalar residual scale."""
    scale = resolve_static_tensor(model, value)
    if (
        scale is None
        or scale.numel() != 1
        or not torch.is_floating_point(scale)
        or not torch.isfinite(scale).all()
        or scale.reshape(-1)[0] <= 0
    ):
        raise ValueError(
            f"{context}: Alpha residual scale must be one static positive "
            "floating value"
        )
    return float(scale.detach().reshape(-1)[0].item())


def residual_quantizer_scale(model: GraphModule, node: Node) -> float:
    """Return the scale of one legal Alpha scalar INT8 quantizer."""
    if node.target != torch.ops.quantized_ops.quantize.default:
        raise ValueError(f"{node.name}: expected an INT8 quantizer")
    if (
        get_arg_value(node, 2, "zero_point") is not None
        or get_arg_value(node, 3, "axes") is not None
        or get_arg_value(node, 4, "block_size") is not None
        or get_arg_value(node, 6, "output_code") is not None
        or get_arg_value(node, 7, "rounding", "nearest") != "floor"
        or get_arg_value(node, 8, "quant_min") != -128.0
        or get_arg_value(node, 9, "quant_max") != 127.0
        or node.meta.get("dtype") != "int8"
    ):
        raise ValueError(
            f"{node.name}: Alpha residual quantizer is not symmetric "
            "floor-clamped signed INT8"
        )
    return static_positive_residual_scale(
        model,
        get_arg_value(node, 1, "scale"),
        node.name,
    )


def direct_residual_output_scale(node: Node) -> float | None:
    """Return the explicit O8 domain of one target-native producer."""
    if node.target not in (
        PPU_FIXED_RELU_OPERATOR,
        NMP_FIXED_REQUANTIZE_OPERATOR,
    ):
        return None
    scale = float(node.kwargs.get("output_scale", math.nan))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"{node.name}: Alpha residual producer has no positive output "
            "scale"
        )
    return scale


def residual_dequantized_code(
    model: GraphModule,
    node: Node,
    visited: set[Node] | None = None,
) -> tuple[Node, float]:
    """Resolve the physical code source of one scalar dequantizer."""
    path, scale = residual_dequantized_path(model, node, visited)
    return path.source, scale


def residual_dequantized_path(
    model: GraphModule,
    node: Node,
    visited: set[Node] | None = None,
) -> tuple[ResidualCodePath, float]:
    """Resolve one nested scalar Q/DQ path to physical INT8 codes."""
    if node.target != torch.ops.quantized_ops.dequantize.default:
        raise ValueError(f"{node.name}: expected an INT8 dequantizer")
    if (
        get_arg_value(node, 2, "zero_point") is not None
        or get_arg_value(node, 3, "axes") is not None
        or get_arg_value(node, 4, "block_size") is not None
        or get_arg_value(node, 5, "input_qmap") is not None
        or get_arg_value(node, 6, "output_qmap") is not None
    ):
        raise ValueError(
            f"{node.name}: Alpha residual dequantizer must be a scalar "
            "symmetric INT8 code mapping"
        )
    scale = static_positive_residual_scale(
        model,
        get_arg_value(node, 1, "scale"),
        node.name,
    )
    source = get_arg_value(node, 0, "input")
    if not isinstance(source, Node):
        raise ValueError(
            f"{node.name}: Alpha residual dequantizer has no signed-INT8 "
            "code source"
        )
    return (
        resolve_residual_code_path(
            model,
            source,
            scale,
            visited=visited,
        ),
        scale,
    )


def resolve_residual_code_path(
    model: GraphModule,
    node: Node,
    declared_scale: float,
    *,
    visited: set[Node] | None = None,
) -> ResidualCodePath:
    """Trace nested scalar Q/DQ stages back to an addressable code."""
    if visited is None:
        visited = set()
    if node in visited:
        raise ValueError(f"{node.name}: cyclic Alpha residual code provenance")
    visited.add(node)

    if node.target == torch.ops.quantized_ops.quantize.default:
        quantized_scale = residual_quantizer_scale(model, node)
        if quantized_scale != declared_scale:
            raise ValueError(
                f"{node.name}: nested residual code scale {quantized_scale} "
                f"does not equal declared scale {declared_scale}"
            )
        dequantize = get_arg_value(node, 0, "input")
        if (
            not isinstance(dequantize, Node)
            or dequantize.target != torch.ops.quantized_ops.dequantize.default
        ):
            if isinstance(dequantize, Node) and has_physical_payload(
                dequantize,
                "int8",
            ):
                source_scale = direct_residual_output_scale(dequantize)
                if source_scale is None:
                    raise ValueError(
                        f"{dequantize.name}: physical residual producer has "
                        "no explicit output scale"
                    )
                output_scales = ()
                if source_scale != quantized_scale:
                    output_scales = (quantized_scale,)
                return ResidualCodePath(
                    dequantize,
                    source_scale,
                    output_scales,
                )
            raise ValueError(
                f"{node.name}: nested residual quantizer is not an explicit "
                "Q(DQ(code)) stage"
            )
        path, source_scale = residual_dequantized_path(
            model,
            dequantize,
            visited,
        )
        output_scales = path.rescale_output_scales
        if source_scale != quantized_scale:
            output_scales = (*output_scales, quantized_scale)
        return ResidualCodePath(
            path.source,
            path.source_scale,
            output_scales,
        )

    if not has_physical_payload(node, "int8"):
        raise ValueError(
            f"{node.name}: Alpha residual code source is not physical "
            "signed INT8"
        )
    direct_scale = direct_residual_output_scale(node)
    if direct_scale is not None and direct_scale != declared_scale:
        raise ValueError(
            f"{node.name}: physical residual producer scale {direct_scale} "
            f"does not equal declared scale {declared_scale}"
        )
    return ResidualCodePath(node, declared_scale)


def relu_reaches_alpha_eadd(relu: Node) -> bool:
    """Return whether one ReLU reaches EADD without intervening compute."""
    queue = [(relu, user) for user in relu.users]
    visited = set()
    qdq_targets = {
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
    }
    residual_add_targets = {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    }
    while queue:
        source, user = queue.pop(0)
        if user in visited:
            continue
        visited.add(user)
        if (
            user.target == SATURATING_ADD_OPERATOR
            or user.target in residual_add_targets
        ):
            return True
        if user.target in qdq_targets or is_address_only_view(user, source):
            queue.extend((user, next_user) for next_user in user.users)
    return False


__all__ = [
    "RESIDUAL_ADD_TARGETS",
    "RESIDUAL_RELU_TARGETS",
    "clone_bias_free_conv",
    "direct_residual_output_scale",
    "node_shape",
    "output_quantizer_info",
    "output_quantizers",
    "relu_reaches_alpha_eadd",
    "residual_dequantized_code",
    "residual_dequantized_path",
    "residual_quantizer_scale",
    "resolve_static_tensor",
    "vsq_scale_endpoints",
]
