# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Physical injection policy for the shared Gemmini Conv2d region."""

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.mapping_utils import (
    LOCAL_SCHEDULE,
    LOCAL_SCHEDULER_WORKLOAD,
    TARGET_CODEGEN_KWARGS,
)
from voyager_compiler.codegen.target_backends.gemmini.hardware import (
    GemminiHardwareConfig,
)

_QUANTIZE = torch.ops.quantized_ops.quantize.default
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_CONV2D_TARGETS = (
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
)
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
_TILING_META = ("l2_tiling", "tiled_shapes", "tile_strides")

_CONV_CORE_ROLES = (
    "conv2d",
    "accumulator_dequantize",
    "output_quantize",
)
_PUBLIC_CONV_ROLES = ("input_quantize", *_CONV_CORE_ROLES)
_RESIDUAL_INPUT_ROLES = (
    "input_a_dequantize",
    "input_b_dequantize",
    "add",
    "output_quantize",
    "code_relu",
)
_RESIDUAL_CONNECTED_ROLES = ("add", "output_quantize", "code_relu")
_AVERAGE_ROLES = (
    "average_input_dequantize",
    "global_average",
    "flatten",
    "output_quantize",
)
_LINEAR_CORE_ROLES = (
    "linear",
    "accumulator_dequantize",
    "output_quantize",
)
_NORMALIZATION_ROLES = (
    "normalization_input_dequantize",
    "normalization",
    "output_quantize",
)


class GemminiInjectionError(RuntimeError):
    """Raised when final FX is not the supported shared injection region."""


@dataclass(frozen=True)
class GemminiInjectionRegion:
    """The semantic nodes consumed by one native ``tiled_conv_auto`` call."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, Node, Node, Node]
    terminal_dequantize: Node
    member_roles: tuple[str, ...]


@dataclass(frozen=True)
class GemminiWorkloadRegion:
    """One helper-representable Conv2d group in a larger workload."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, ...]
    core_members: tuple[Node, ...]
    input_quantize: Node | None
    activation: Node | None
    pooling: Node | None
    output_quantize: Node
    member_roles: tuple[str, ...]


@dataclass(frozen=True)
class GemminiWorkloadAddRegion:
    """One shared residual Add/ReLU group for ``tiled_resadd_auto``."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, ...]
    input_boundaries: tuple[Node, Node]
    addition: Node
    output_quantize: Node
    activation: Node
    member_roles: tuple[str, ...]


@dataclass(frozen=True)
class GemminiWorkloadLinearRegion:
    """One shared Q/Linear/DQ/Q group for ``tiled_matmul_auto``."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, ...]
    input_quantize: Node | None
    linear: Node
    accumulator_dequantize: Node
    output_quantize: Node
    member_roles: tuple[str, ...]


@dataclass(frozen=True)
class GemminiWorkloadAverageRegion:
    """One shared global-average group for a scale-aware native helper."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, ...]
    input_dequantize: Node
    average: Node
    flatten: Node | None
    output_quantize: Node
    member_roles: tuple[str, ...]


@dataclass(frozen=True)
class GemminiWorkloadNormRegion:
    """One shared normalization binding for unavailable tiled_norm_auto."""

    outer: Node
    submodule: GraphModule
    members: tuple[Node, Node, Node]
    input_dequantize: Node
    normalization: Node
    output_quantize: Node
    member_roles: tuple[str, ...]


def _node_input(node: Node, index: int, name: str):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name)


def _single_graph_output(model: GraphModule) -> Node:
    outputs = tuple(node for node in model.graph.nodes if node.op == "output")
    if len(outputs) != 1:
        raise GemminiInjectionError("Gemmini requires one graph output")
    return outputs[0]


def _validate_conv_core(
    outer: Node,
    members: tuple[Node, ...],
) -> tuple[Node, Node, Node, Node]:
    if len(members) != 4:
        raise GemminiInjectionError(
            f"{outer.name}: Gemmini requires four fused semantic operators"
        )
    input_quantize, convolution, accumulator_dequantize, output_quantize = (
        members
    )
    if input_quantize.target != _QUANTIZE:
        raise GemminiInjectionError(
            f"{input_quantize.name}: fused member 0 must be input quantize"
        )
    if convolution.target not in _CONV2D_TARGETS:
        raise GemminiInjectionError(
            f"{convolution.name}: fused member 1 must be shared Conv2d"
        )
    if accumulator_dequantize.target != _DEQUANTIZE:
        raise GemminiInjectionError(
            f"{accumulator_dequantize.name}: fused member 2 must be "
            "accumulator dequantize"
        )
    if output_quantize.target != _QUANTIZE:
        raise GemminiInjectionError(
            f"{output_quantize.name}: fused member 3 must be output quantize"
        )

    dependencies = (
        (_node_input(convolution, 0, "input"), input_quantize),
        (
            _node_input(accumulator_dequantize, 0, "input"),
            convolution,
        ),
        (_node_input(output_quantize, 0, "input"), accumulator_dequantize),
    )
    if any(actual is not expected for actual, expected in dependencies):
        raise GemminiInjectionError(
            f"{outer.name}: fused semantic tensor identities are not linear"
        )

    return members


def _validate_inner_region(
    outer: Node,
    submodule: GraphModule,
) -> tuple[Node, Node, Node, Node]:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    core = _validate_conv_core(outer, members)

    output = _single_graph_output(submodule)
    if tuple(output.all_input_nodes) != (core[-1],):
        raise GemminiInjectionError(
            f"{outer.name}: fused output must be the output-quantize tensor"
        )
    return core


def _validate_workload_region(
    outer: Node,
    submodule: GraphModule,
) -> GemminiWorkloadRegion:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    if len(members) not in (3, 4, 5, 6):
        raise GemminiInjectionError(
            f"{outer.name}: unsupported workload-region member count"
        )
    has_input_quantize = members[0].target == _QUANTIZE
    core_size = 4 if has_input_quantize else 3
    core = members[:core_size]
    if has_input_quantize:
        _validate_conv_core(outer, core)
    else:
        convolution, accumulator_dequantize, output_quantize = core
        if convolution.target not in _CONV2D_TARGETS:
            raise GemminiInjectionError(
                f"{convolution.name}: workload core must begin with Conv2d"
            )
        if accumulator_dequantize.target != _DEQUANTIZE:
            raise GemminiInjectionError(
                f"{accumulator_dequantize.name}: Conv must feed dequantize"
            )
        if output_quantize.target != _QUANTIZE:
            raise GemminiInjectionError(
                f"{output_quantize.name}: Conv core must end with quantize"
            )
        dependencies = (
            (_node_input(accumulator_dequantize, 0, "input"), convolution),
            (_node_input(output_quantize, 0, "input"), accumulator_dequantize),
        )
        if any(actual is not expected for actual, expected in dependencies):
            raise GemminiInjectionError(
                f"{outer.name}: Conv core tensor identities are not linear"
            )

    epilogue = members[core_size:]
    activation = None
    pooling = None
    output_quantize = core[-1]
    if epilogue:
        if len(epilogue) not in (1, 2):
            raise GemminiInjectionError(
                f"{outer.name}: malformed code-value ReLU epilogue"
            )
        activation = epilogue[0]
        pooling = epilogue[1] if len(epilogue) == 2 else None
        if activation.target != _CODE_RELU:
            raise GemminiInjectionError(
                f"{activation.name}: workload epilogue must use code ReLU"
            )
        if _node_input(activation, 0, "input") is not output_quantize:
            raise GemminiInjectionError(
                f"{activation.name}: code ReLU must consume final qcodes"
            )
        if pooling is not None:
            if pooling.target != _MAX_POOL:
                raise GemminiInjectionError(
                    f"{pooling.name}: code epilogue must end in max-pool"
                )
            if _node_input(pooling, 0, "input") is not activation:
                raise GemminiInjectionError(
                    f"{pooling.name}: max-pool must consume code ReLU"
                )
    output = _single_graph_output(submodule)
    expected_output = pooling or activation or output_quantize
    if tuple(output.all_input_nodes) != (expected_output,):
        raise GemminiInjectionError(
            f"{outer.name}: workload output differs from final fused member"
        )
    member_roles = (
        (("input_quantize",) if has_input_quantize else ())
        + _CONV_CORE_ROLES
        + (("code_relu",) if activation is not None else ())
        + (("max_pool2d",) if pooling is not None else ())
    )
    return GemminiWorkloadRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        core_members=core,
        input_quantize=core[0] if has_input_quantize else None,
        activation=activation,
        pooling=pooling,
        output_quantize=output_quantize,
        member_roles=member_roles,
    )


def _validate_workload_add_region(
    outer: Node,
    submodule: GraphModule,
) -> GemminiWorkloadAddRegion:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    if len(members) == 5:
        (
            input_dequantize,
            other_dequantize,
            addition,
            output_quantize,
            activation,
        ) = members
        if (
            input_dequantize.target != _DEQUANTIZE
            or other_dequantize.target != _DEQUANTIZE
        ):
            raise GemminiInjectionError(
                f"{outer.name}: residual inputs must both be dequantize"
            )
        if addition.target not in _ADD_TARGETS:
            raise GemminiInjectionError(
                f"{addition.name}: residual helper requires shared Add"
            )
        if output_quantize.target != _QUANTIZE:
            raise GemminiInjectionError(
                f"{output_quantize.name}: residual Add must feed final Q"
            )
        if activation.target != _CODE_RELU:
            raise GemminiInjectionError(
                f"{activation.name}: residual helper requires code ReLU"
            )
        addition_inputs = {
            _node_input(addition, 0, "input"),
            _node_input(addition, 1, "other"),
        }
        if addition_inputs != {input_dequantize, other_dequantize}:
            raise GemminiInjectionError(
                f"{addition.name}: Add must consume both residual DQs"
            )
        dependencies = (
            (_node_input(output_quantize, 0, "input"), addition),
            (_node_input(activation, 0, "input"), output_quantize),
        )
        if any(actual is not expected for actual, expected in dependencies):
            raise GemminiInjectionError(
                f"{outer.name}: residual code-domain identities are not linear"
            )
        output = _single_graph_output(submodule)
        if tuple(output.all_input_nodes) != (activation,):
            raise GemminiInjectionError(
                f"{outer.name}: residual output must be code ReLU qcodes"
            )
        return GemminiWorkloadAddRegion(
            outer=outer,
            submodule=submodule,
            members=members,
            input_boundaries=(input_dequantize, other_dequantize),
            addition=addition,
            output_quantize=output_quantize,
            activation=activation,
            member_roles=_RESIDUAL_INPUT_ROLES,
        )
    if len(members) != 3:
        raise GemminiInjectionError(
            f"{outer.name}: residual helper requires 3 or 5 operators"
        )
    addition, output_quantize, activation = members
    if addition.target not in _ADD_TARGETS:
        raise GemminiInjectionError(
            f"{addition.name}: residual helper requires shared Add"
        )
    if output_quantize.target != _QUANTIZE:
        raise GemminiInjectionError(
            f"{output_quantize.name}: residual Add must feed Q"
        )
    if activation.target != _CODE_RELU:
        raise GemminiInjectionError(
            f"{activation.name}: residual helper requires code ReLU"
        )
    addition_input = _node_input(addition, 0, "input")
    addition_other = _node_input(addition, 1, "other")
    if (
        not isinstance(addition_input, Node)
        or not isinstance(addition_other, Node)
        or addition_input is addition_other
    ):
        raise GemminiInjectionError(
            f"{addition.name}: residual helper requires two tensor inputs"
        )
    dependencies = (
        (_node_input(output_quantize, 0, "input"), addition),
        (_node_input(activation, 0, "input"), output_quantize),
    )
    if any(actual is not expected for actual, expected in dependencies):
        raise GemminiInjectionError(
            f"{outer.name}: residual code-domain identities are not linear"
        )
    output = _single_graph_output(submodule)
    if tuple(output.all_input_nodes) != (activation,):
        raise GemminiInjectionError(
            f"{outer.name}: residual output must be code ReLU qcodes"
        )
    return GemminiWorkloadAddRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        input_boundaries=(addition_input, addition_other),
        addition=addition,
        output_quantize=output_quantize,
        activation=activation,
        member_roles=_RESIDUAL_CONNECTED_ROLES,
    )


def _validate_workload_linear_region(
    outer: Node,
    submodule: GraphModule,
) -> GemminiWorkloadLinearRegion:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    if len(members) not in (3, 4):
        raise GemminiInjectionError(
            f"{outer.name}: Linear helper requires three or four operators"
        )
    input_quantize = members[0] if len(members) == 4 else None
    linear, accumulator_dequantize, output_quantize = members[-3:]
    targets = (
        linear.target,
        accumulator_dequantize.target,
        output_quantize.target,
    )
    if targets != (_LINEAR, _DEQUANTIZE, _QUANTIZE):
        raise GemminiInjectionError(
            f"{outer.name}: Linear helper has the wrong semantic members"
        )
    if input_quantize is not None and input_quantize.target != _QUANTIZE:
        raise GemminiInjectionError(
            f"{input_quantize.name}: Linear input member must be quantize"
        )
    dependencies = [
        (_node_input(accumulator_dequantize, 0, "input"), linear),
        (_node_input(output_quantize, 0, "input"), accumulator_dequantize),
    ]
    if input_quantize is not None:
        dependencies.insert(
            0, (_node_input(linear, 0, "input"), input_quantize)
        )
    if any(actual is not expected for actual, expected in dependencies):
        raise GemminiInjectionError(
            f"{outer.name}: Linear semantic tensor identities are not linear"
        )
    output = _single_graph_output(submodule)
    if tuple(output.all_input_nodes) != (output_quantize,):
        raise GemminiInjectionError(
            f"{outer.name}: Linear output must be output qcodes"
        )
    return GemminiWorkloadLinearRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        input_quantize=input_quantize,
        linear=linear,
        accumulator_dequantize=accumulator_dequantize,
        output_quantize=output_quantize,
        member_roles=(
            (("input_quantize",) if input_quantize is not None else ())
            + _LINEAR_CORE_ROLES
        ),
    )


def _validate_workload_average_region(
    outer: Node,
    submodule: GraphModule,
) -> GemminiWorkloadAverageRegion:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    if len(members) != 4:
        raise GemminiInjectionError(
            f"{outer.name}: global-average helper requires four operators"
        )
    input_dequantize, average, flatten, output_quantize = members
    expected_targets = (_DEQUANTIZE, _ADAPTIVE_AVG_POOL, _FLATTEN, _QUANTIZE)
    if tuple(node.target for node in members) != expected_targets:
        raise GemminiInjectionError(
            f"{outer.name}: global-average helper has wrong semantic members"
        )
    dependencies = [
        (_node_input(average, 0, "input"), input_dequantize),
    ]
    dependencies.append((_node_input(flatten, 0, "input"), average))
    dependencies.append((_node_input(output_quantize, 0, "input"), flatten))
    if any(actual is not expected for actual, expected in dependencies):
        raise GemminiInjectionError(
            f"{outer.name}: global-average tensor identities are not linear"
        )
    output_size = _node_input(average, 1, "output_size")
    if output_size not in (1, [1, 1], (1, 1)):
        raise GemminiInjectionError(
            f"{average.name}: only global 1x1 adaptive average is supported"
        )
    start_dim = _node_input(flatten, 1, "start_dim")
    end_dim = _node_input(flatten, 2, "end_dim")
    if start_dim not in (None, 1) or end_dim not in (None, -1):
        raise GemminiInjectionError(
            f"{flatten.name}: classifier flatten must span 1..-1"
        )
    output = _single_graph_output(submodule)
    if tuple(output.all_input_nodes) != (output_quantize,):
        raise GemminiInjectionError(
            f"{outer.name}: global-average output must be final qcodes"
        )
    return GemminiWorkloadAverageRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        input_dequantize=input_dequantize,
        average=average,
        flatten=flatten,
        output_quantize=output_quantize,
        member_roles=_AVERAGE_ROLES,
    )


def _validate_workload_norm_region(
    outer: Node,
    submodule: GraphModule,
) -> GemminiWorkloadNormRegion:
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    if len(members) != 3:
        raise GemminiInjectionError(
            f"{outer.name}: normalization binding requires three operators"
        )
    input_dequantize, normalization, output_quantize = members
    if (
        input_dequantize.target != _DEQUANTIZE
        or normalization.target not in _NORM_TARGETS
        or output_quantize.target != _QUANTIZE
    ):
        raise GemminiInjectionError(
            f"{outer.name}: normalization binding has wrong semantic members"
        )
    dependencies = (
        (_node_input(normalization, 0, "input"), input_dequantize),
        (_node_input(output_quantize, 0, "input"), normalization),
    )
    if any(actual is not expected for actual, expected in dependencies):
        raise GemminiInjectionError(
            f"{outer.name}: normalization tensor identities are not linear"
        )
    output = _single_graph_output(submodule)
    if tuple(output.all_input_nodes) != (output_quantize,):
        raise GemminiInjectionError(
            f"{outer.name}: normalization output must be final qcodes"
        )
    return GemminiWorkloadNormRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        input_dequantize=input_dequantize,
        normalization=normalization,
        output_quantize=output_quantize,
        member_roles=_NORMALIZATION_ROLES,
    )


def gemmini_injection_region(model: GraphModule) -> GemminiInjectionRegion:
    """Read and validate the sole shared Gemmini execution region."""
    outer_nodes = tuple(
        node for node in model.graph.nodes if node.op == "call_module"
    )
    if len(outer_nodes) != 1:
        raise GemminiInjectionError(
            "Gemmini requires exactly one fused execution region"
        )
    outer = outer_nodes[0]
    submodule = model.get_submodule(str(outer.target))
    if not isinstance(submodule, GraphModule):
        raise GemminiInjectionError(
            f"{outer.name}: fused target must be an FX GraphModule"
        )
    members = _validate_inner_region(outer, submodule)

    top_level_calls = tuple(
        node
        for node in model.graph.nodes
        if node.op in {"call_function", "call_module"}
    )
    if len(top_level_calls) != 2 or top_level_calls[0] is not outer:
        raise GemminiInjectionError(
            "Gemmini permits only the fused region and terminal dequantize"
        )
    terminal = top_level_calls[1]
    if terminal.op != "call_function" or terminal.target != _DEQUANTIZE:
        raise GemminiInjectionError(
            f"{terminal.name}: Gemmini public boundary must be dequantize"
        )
    if _node_input(terminal, 0, "input") is not outer:
        raise GemminiInjectionError(
            f"{terminal.name}: terminal dequantize must consume fused qcodes"
        )
    if tuple(outer.users) != (terminal,):
        raise GemminiInjectionError(
            f"{outer.name}: fused qcodes must feed only terminal dequantize"
        )
    output = _single_graph_output(model)
    if tuple(output.all_input_nodes) != (terminal,):
        raise GemminiInjectionError(
            f"{terminal.name}: graph output must preserve the float boundary"
        )
    return GemminiInjectionRegion(
        outer=outer,
        submodule=submodule,
        members=members,
        terminal_dequantize=terminal,
        member_roles=_PUBLIC_CONV_ROLES,
    )


def _fused_module_candidates(
    model: GraphModule,
) -> tuple[tuple[Node, GraphModule], ...]:
    candidates = []
    for outer in model.graph.nodes:
        if outer.op != "call_module":
            continue
        submodule = model.get_submodule(str(outer.target))
        if isinstance(submodule, GraphModule):
            candidates.append((outer, submodule))
    return tuple(candidates)


def _has_public_injection_shape(model: GraphModule) -> bool:
    """Distinguish the R public boundary from a one-region W workload.

    Selection is based on the semantic member sequence, not the number of
    fused modules. A valid W workload may contain only one extended Conv, Add,
    Linear, or Average region and must still use workload validation.
    """
    candidates = _fused_module_candidates(model)
    if len(candidates) != 1:
        return False
    outer, submodule = candidates[0]
    members = tuple(
        node for node in submodule.graph.nodes if node.op == "call_function"
    )
    targets = tuple(member.target for member in members)
    if (
        len(targets) != 4
        or targets[0] != _QUANTIZE
        or targets[1] not in _CONV2D_TARGETS
        or targets[2:] != (_DEQUANTIZE, _QUANTIZE)
    ):
        return False
    top_level_calls = tuple(
        node
        for node in model.graph.nodes
        if node.op in {"call_function", "call_module"}
    )
    return (
        len(top_level_calls) == 2
        and top_level_calls[0] is outer
        and top_level_calls[1].op == "call_function"
        and top_level_calls[1].target == _DEQUANTIZE
    )


def gemmini_workload_regions(
    model: GraphModule,
) -> tuple[GemminiWorkloadRegion, ...]:
    """Return every complete shared Conv2d fusion in a multi-region graph.

    This recognizes the target-neutral Conv core and optional ReLU/max-pool
    epilogues that one ``tiled_conv_auto`` call can implement.  Residual and
    linear operations remain separate helper candidates.
    """
    regions = []
    for outer, submodule in _fused_module_candidates(model):
        members = tuple(
            node for node in submodule.graph.nodes if node.op == "call_function"
        )
        if not any(member.target in _CONV2D_TARGETS for member in members):
            continue
        regions.append(_validate_workload_region(outer, submodule))
    return tuple(regions)


def gemmini_workload_add_regions(
    model: GraphModule,
) -> tuple[GemminiWorkloadAddRegion, ...]:
    """Return every complete shared residual Add/ReLU helper group."""
    regions = []
    for outer, submodule in _fused_module_candidates(model):
        members = tuple(
            node for node in submodule.graph.nodes if node.op == "call_function"
        )
        if not any(member.target in _ADD_TARGETS for member in members):
            continue
        regions.append(_validate_workload_add_region(outer, submodule))
    return tuple(regions)


def gemmini_workload_linear_regions(
    model: GraphModule,
) -> tuple[GemminiWorkloadLinearRegion, ...]:
    """Return every complete shared Q/Linear/DQ/Q helper group."""
    regions = []
    for outer, submodule in _fused_module_candidates(model):
        members = tuple(
            node for node in submodule.graph.nodes if node.op == "call_function"
        )
        if not any(member.target == _LINEAR for member in members):
            continue
        regions.append(_validate_workload_linear_region(outer, submodule))
    return tuple(regions)


def gemmini_workload_average_regions(
    model: GraphModule,
) -> tuple[GemminiWorkloadAverageRegion, ...]:
    """Return every complete shared classifier global-average group."""
    regions = []
    for outer, submodule in _fused_module_candidates(model):
        members = tuple(
            node for node in submodule.graph.nodes if node.op == "call_function"
        )
        if not any(member.target == _ADAPTIVE_AVG_POOL for member in members):
            continue
        regions.append(_validate_workload_average_region(outer, submodule))
    return tuple(regions)


def gemmini_workload_norm_regions(
    model: GraphModule,
) -> tuple[GemminiWorkloadNormRegion, ...]:
    """Return shared normalization candidates for fail-closed binding."""
    regions = []
    for outer, submodule in _fused_module_candidates(model):
        members = tuple(
            node for node in submodule.graph.nodes if node.op == "call_function"
        )
        if not any(member.target in _NORM_TARGETS for member in members):
            continue
        regions.append(_validate_workload_norm_region(outer, submodule))
    return tuple(regions)


def _shape(value) -> tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise GemminiInjectionError(
            "tiled shape must be an integer tuple"
        ) from error
    if any(type(dimension) is not int for dimension in result):
        raise GemminiInjectionError("tiled shape must contain integers")
    return result


def _node_shape(node: Node) -> tuple[int, ...]:
    candidates = (
        getattr(node, "shape", None),
        getattr(getattr(node, "value", None), "shape", None),
        getattr(node.meta.get("val"), "shape", None),
        getattr(node.meta.get("tensor_meta"), "shape", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return _shape(candidate)
    raise GemminiInjectionError(f"{node.name}: missing logical tensor shape")


def _validate_full_tensor_annotations(node: Node) -> None:
    tiling = node.meta.get("l2_tiling")
    tiled_shapes = node.meta.get("tiled_shapes")
    tile_strides = node.meta.get("tile_strides")
    if tile_strides not in (None, {}):
        raise GemminiInjectionError(
            f"{node.name}: Gemmini rejects explicit tile strides"
        )
    if tiling is None:
        if tiled_shapes not in (None, {}):
            raise GemminiInjectionError(
                f"{node.name}: tiled shapes require identity L2 tiling"
            )
        return
    if (
        not isinstance(tiling, (tuple, list))
        or not tiling
        or any(type(factor) is not int or factor != 1 for factor in tiling)
    ):
        raise GemminiInjectionError(
            f"{node.name}: Gemmini rejects split or malformed L2 tiling"
        )
    if tiled_shapes is None:
        return
    if not isinstance(tiled_shapes, dict):
        raise GemminiInjectionError(
            f"{node.name}: tiled shapes must be a node mapping"
        )
    if set(tiled_shapes) == {"input", "output"}:
        input_shape = _shape(tiled_shapes["input"])
        output_shape = _shape(tiled_shapes["output"])
        if input_shape != output_shape or len(output_shape) != len(tiling):
            raise GemminiInjectionError(
                f"{node.name}: Gemmini rejects decomposed domain-transfer tiles"
            )
        return
    for tensor, tiled_shape in tiled_shapes.items():
        if not isinstance(tensor, Node):
            raise GemminiInjectionError(
                f"{node.name}: tiled-shape keys must be shaped FX nodes"
            )
        if _shape(tiled_shape) != _node_shape(tensor):
            raise GemminiInjectionError(
                f"{node.name}: Gemmini rejects decomposed tensor tiles"
            )


def _reject_target_control(node: Node) -> None:
    if node.meta.get(LOCAL_SCHEDULE) is not None:
        raise GemminiInjectionError(
            f"{node.name}: Gemmini rejects generic local schedules"
        )
    if node.meta.get(LOCAL_SCHEDULER_WORKLOAD) is not None:
        raise GemminiInjectionError(
            f"{node.name}: Gemmini rejects scheduler workloads"
        )
    target_kwargs = node.meta.get(TARGET_CODEGEN_KWARGS, {})
    if not isinstance(target_kwargs, dict):
        raise GemminiInjectionError(
            f"{node.name}: target codegen kwargs must be a dictionary"
        )
    if target_kwargs:
        raise GemminiInjectionError(
            f"{node.name}: Gemmini target ABI fields are forbidden"
        )
    if node.meta.get("physical_scratchpad_map") is not None:
        raise GemminiInjectionError(
            f"{node.name}: Gemmini does not own shared scratchpad placement"
        )


def _clear_identity_tiling(node: Node) -> None:
    _validate_full_tensor_annotations(node)
    for field in _TILING_META:
        node.meta.pop(field, None)


def _clear_workload_region_tiling(node: Node) -> None:
    """Discard generic partition metadata for one target-owned full region.

    The W-stage adapter consumes each admitted Conv2d as one full logical
    tensor operation; the native helper performs its own traversal.  Generic
    L2 partitions are therefore neither a semantic input nor a target ABI.
    We still reject malformed metadata before removing it, while permitting
    non-identity factors that the R single-region anchor intentionally rejects.
    """
    tiling = node.meta.get("l2_tiling")
    if tiling is not None and (
        not isinstance(tiling, (tuple, list))
        or not tiling
        or any(type(factor) is not int or factor <= 0 for factor in tiling)
    ):
        raise GemminiInjectionError(
            f"{node.name}: malformed generic L2 tiling metadata"
        )
    for field in ("tiled_shapes", "tile_strides"):
        mapping = node.meta.get(field)
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            raise GemminiInjectionError(
                f"{node.name}: {field} must be a metadata mapping"
            )
        for tiled_shape in mapping.values():
            _shape(tiled_shape)
    for field in _TILING_META:
        node.meta.pop(field, None)


def apply_memory_rewrites(
    model: GraphModule,
    hardware: GemminiHardwareConfig,
) -> GraphModule:
    """Skip generic inner scheduling for shared Gemmini Conv2d regions."""
    if not isinstance(hardware, GemminiHardwareConfig):
        raise TypeError("Gemmini memory policy requires GemminiHardwareConfig")
    norm_regions = gemmini_workload_norm_regions(model)
    if norm_regions:
        names = ", ".join(region.outer.name for region in norm_regions)
        raise GemminiInjectionError(
            "tiled_norm_auto requires HAS_NORMALIZATIONS, which is absent "
            f"from the pinned Gemmini configuration: {names}"
        )
    if _has_public_injection_shape(model):
        region = gemmini_injection_region(model)
        regions = ((region.outer, region.members, region.terminal_dequantize),)
    else:
        regions = tuple(
            (region.outer, region.members, None)
            for region in gemmini_workload_regions(model)
        )
        regions += tuple(
            (region.outer, region.members, None)
            for region in gemmini_workload_add_regions(model)
        )
        regions += tuple(
            (region.outer, region.members, None)
            for region in gemmini_workload_linear_regions(model)
        )
        regions += tuple(
            (region.outer, region.members, None)
            for region in gemmini_workload_average_regions(model)
        )
    for outer, members, terminal in regions:
        controlled_nodes = (outer, *members)
        if terminal is not None:
            controlled_nodes += (terminal,)
        for node in controlled_nodes:
            _reject_target_control(node)
            if terminal is None:
                _clear_workload_region_tiling(node)
            else:
                _clear_identity_tiling(node)
        scheduler_skips = (outer,)
        if terminal is not None:
            scheduler_skips += (terminal,)
        for node in scheduler_skips:
            scratchpad = node.meta.get("scratchpad_map")
            if scratchpad not in (None, {}):
                raise GemminiInjectionError(
                    f"{node.name}: Gemmini rejects preexisting scratchpad "
                    "mapping"
                )
            node.meta["scratchpad_map"] = {}
    return model


__all__ = [
    "GemminiInjectionError",
    "GemminiInjectionRegion",
    "GemminiWorkloadRegion",
    "GemminiWorkloadAddRegion",
    "GemminiWorkloadLinearRegion",
    "GemminiWorkloadAverageRegion",
    "GemminiWorkloadNormRegion",
    "apply_memory_rewrites",
    "gemmini_injection_region",
    "gemmini_workload_regions",
    "gemmini_workload_add_regions",
    "gemmini_workload_linear_regions",
    "gemmini_workload_average_regions",
    "gemmini_workload_norm_regions",
]
