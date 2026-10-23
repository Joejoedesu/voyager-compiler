# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import copy
import logging
import math
import operator
import os
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable, Dict, List, Union

import graphviz
import torch
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from transformers.utils.import_utils import is_torch_greater_or_equal

from scheduler.dispatch import (
    resolve_local_scheduler,
    schedule_operation,
)
from scheduler.types import OperationOperand
from scheduler.voyager import extract_operation_workload
from voyager_compiler.codegen.artifacts import prepare_output_directory
from voyager_compiler.codegen.banking import (
    _get_scope,
    get_banking_strategies_for_op,
)
from voyager_compiler.codegen.end_to_end import (
    DPExecutionOrderError,
    solve_execution_order_dp,
)
from voyager_compiler.codegen.execution_order import (
    DEFAULT_EXECUTION_ORDER_MODE,
    DP_EXECUTION_ORDER_MODE,
    EXECUTION_ORDER_RESULT_META,
    FX_EXECUTION_ORDER_MODE,
    execution_order_for_model,
    fx_execution_order,
    validate_execution_order_mode,
)
from voyager_compiler.codegen.hardware import (
    HardwareConfig,
    TensorRole,
    get_dtype_width,
    resolve_unroll_dims,
)
from voyager_compiler.codegen.mapping_utils import (
    SERIALIZED_UNSUPPORTED_OPERATION,
    SOURCE_OP_NAME,
    TARGET_CODEGEN_KWARGS,
    TARGET_INSTRUCTION_NOP,
    bind_local_schedule,
    build_tiling_proto,
    is_bmm,
    is_conv2d,
    is_declared_unsupported_operation,
    is_elementwise_op,
    is_fully_connected,
    is_gemm_op,
    is_nop,
    is_prunable_op,
    is_reshape_op,
    map_node,
    propagate_unsupported_operation,
    set_output_field,
    set_tensor_field,
)
from voyager_compiler.codegen.memory import (
    END_TO_END_SCRATCHPAD_MAPPING,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    SCRATCHPAD_MAPPING_MODES,
    MemoryAllocator,
    ScratchpadAllocator,
    physical_access_requirements,
)
from voyager_compiler.codegen.operands import (
    classify_storage_role,
    operand_descriptors,
    require_allocation,
)
from voyager_compiler.codegen.param_pb2 import Model, Operation, Tensor
from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.codegen.storage import (
    StorageGraph,
    StoragePlane,
    StorageRelationKind,
    ValueRef,
    materialize_aggregate_copies,
)
from voyager_compiler.codegen.target_backends import (
    allocate_target_operation_scope,
    apply_target_memory_rewrites,
    classify_target_scheduler_operation,
    create_scratchpad_allocator,
    get_target_operator_traits,
    get_target_scheduler_providers,
    prepare_target_serialization_metadata,
    prepare_target_serialized_artifacts,
    target_has_operation_scope,
    target_supports_operation_scoped_scratchpad,
    target_uses_generic_aggregate_copies,
    validate_target_end_to_end_scratchpad_operations,
    validate_target_hardware_config,
    validate_target_operation_scopes,
    validate_target_serialization_metadata,
)
from voyager_compiler.codegen.targets import (
    COMPILATION_TARGET_META,
    DEFAULT_COMPILATION_TARGET,
    bind_compilation_target,
    resolve_compilation_target,
)
from voyager_compiler.pt2e_utils import (
    create_getattr_from_value,
    dtype_byte_size,
    fetch_attr,
    propagate_shape,
    update_submod_user_meta,
)
from voyager_compiler.quantization_contracts import (
    QUANTIZATION_DOMAIN_DTYPE,
    get_physical_dtype,
    get_physical_quantization_contract,
    inherit_physical_quantization_contract,
)

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_SIZE = torch.finfo(torch.float32).max


def replace_node_with_graph_module(
    model: GraphModule,
    source: Node,
    replacement: GraphModule,
    value_remap=None,
) -> List[Node]:
    """Copy ``replacement`` (an exported subgraph) into ``model`` just before
    ``source``, mapping its placeholders to ``source.all_input_nodes``; return
    the output value node(s).
    """
    graph = model.graph
    if value_remap is None:
        value_remap = {}

    arg_nodes = iter(source.all_input_nodes)
    output = None

    for n in list(replacement.graph.nodes):
        if n.op == "placeholder":
            value_remap[n] = next(arg_nodes, None)
            continue
        if n.op == "output":
            output = n.args[0]
            if len(output) == 1:
                source.replace_all_uses_with(value_remap[output[0]])
            else:
                for user in list(source.users):
                    assert user.target == operator.getitem
                    idx = user.args[1]
                    user.replace_all_uses_with(value_remap[output[idx]])
            continue
        with graph.inserting_before(source):
            if n.op == "get_attr":
                attr = fetch_attr(replacement, n.target)
                if isinstance(attr, GraphModule):
                    # cond / body subgraph of a while_loop: register as a submodule
                    name = get_new_node_name_with_prefix(n.target)(model)
                    setattr(model, name, attr)
                    value_remap[n] = graph.create_node("get_attr", name)
                else:
                    value_remap[n] = create_getattr_from_value(
                        model, graph, "_tensor_constant_", attr
                    )
            else:
                value_remap[n] = graph.node_copy(n, lambda n: value_remap[n])
            propagate_shape(value_remap[n], model)

    return [value_remap[n] for n in output]


def _create_subgraph(nodes: List[Node]):
    new_args = []
    new_graph = torch.fx.Graph()
    value_remap = {}

    for node in nodes:
        for n in node.all_input_nodes:
            if n not in value_remap:
                value_remap[n] = new_graph.placeholder(n.name)
                new_args.append(n)
                value_remap[n].meta["source_node"] = n
        value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

    new_graph.output(value_remap[nodes[-1]])
    new_graph.lint()
    gm = torch.fx.GraphModule(torch.nn.Module(), new_graph)
    return gm, tuple(new_args)


OP_PARAM_ARG_INDEX = {
    torch.ops.aten.conv2d.default: 1,
    torch.ops.aten.linear.default: 1,
    torch.ops.aten.layer_norm.default: 2,
    torch.ops.quantized_ops.conv2d.default: 1,
    torch.ops.quantized_ops.linear.default: 1,
    torch.ops.quantized_ops.conv2d_mx.default: 1,
    torch.ops.quantized_ops.linear_mx.default: 1,
}


def _parameter_operand_index(operator_traits=None):
    operand_index = dict(OP_PARAM_ARG_INDEX)
    if operator_traits is not None:
        operand_index.update(operator_traits.parameter_operand_index)
    return operand_index


def get_unique_node_name(node: Node, operator_traits=None):
    """
    Generate a unique and meaningful name for the node based on its parameter.
    """
    if source_op_name := node.meta.get(SOURCE_OP_NAME):
        return source_op_name

    parameter_operand_index = _parameter_operand_index(operator_traits)
    if (pos := parameter_operand_index.get(node.target)) is not None:
        weight_node = node.args[pos]
        # There are cases where weights are sliced. Trace up to find the
        # get_attr node and use the parameter name
        while weight_node.target == torch.ops.aten.slice.Tensor:
            weight_node = weight_node.args[0]

        if weight_node.op == "get_attr":
            return weight_node.name.split("_weight")[0]

    return node.name


def get_new_node_name_with_prefix(prefix: str):
    """
    Generate a new attribute name with a given prefix that is not already used
    in the module's graph.
    """
    prefix = prefix.replace(".", "_")

    def get_new_node_name(module: torch.nn.Module):
        existing_names = {n.name for n in module.graph.nodes}
        existing_names.update(dict(module.named_modules()).keys())

        if prefix not in existing_names:
            return prefix

        i = 1
        while f"{prefix}_{i}" in existing_names:
            i += 1

        node_name = f"{prefix}_{i}"
        logger.debug(f"Generated new unique node name: {node_name}")
        return node_name

    return get_new_node_name


def get_submodule_name(module, nodes: List[Node], operator_traits=None):
    prefix = "submodule"
    if is_torch_greater_or_equal("2.5"):
        parameter_operand_index = _parameter_operand_index(operator_traits)
        anchor_node = None
        for n in nodes:
            if n.target in parameter_operand_index or is_gemm_op(
                n, operator_traits
            ):
                anchor_node = n
                break
            if (
                n.op == "call_function"
                and not is_nop(n)
                and not is_reshape_op(n)
                and (
                    anchor_node is None
                    or anchor_node.target
                    == torch.ops.quantized_ops.dequantize.default
                )
            ):
                anchor_node = n
        prefix = get_unique_node_name(anchor_node, operator_traits)
        if len(nodes) > 1:
            prefix += "_fused"

    get_new_node_name = get_new_node_name_with_prefix(prefix)
    return get_new_node_name(module)


def rename_nodes_with_param_names(model: GraphModule, operator_traits=None):
    if not is_torch_greater_or_equal("2.5"):
        return
    graph = model.graph
    parameter_operand_index = _parameter_operand_index(operator_traits)
    named_modules = dict(model.named_modules())
    for node in list(graph.nodes):
        if node.target in parameter_operand_index:
            prefix = get_unique_node_name(
                node,
                operator_traits,
            ).replace(".", "_")
            existing_names = {
                existing.name
                for existing in graph.nodes
                if existing is not node
            }
            node_name = prefix
            suffix = 1
            while node_name in existing_names:
                node_name = f"{prefix}_{suffix}"
                suffix += 1
            node.name = node_name
            update_submod_user_meta(model, node, named_modules)
    graph.lint()
    model.recompile()


def _create_and_insert_subgraph(
    nodes: List[Node],
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    node_order: Dict[Node, int] = None,
    operator_traits=None,
) -> Node:
    if node_order is None:
        node_order = {n: i for i, n in enumerate(model.graph.nodes)}
    nodes.sort(key=lambda n: node_order[n])
    submodule, new_args = _create_subgraph(nodes)
    node_name = get_submodule_name(
        model,
        nodes,
        operator_traits,
    )
    setattr(model, node_name, submodule)
    named_modules[node_name] = submodule
    with model.graph.inserting_after(nodes[-1]):
        new_node = model.graph.create_node(
            "call_module", node_name, new_args, {}
        )
    nodes[-1].replace_all_uses_with(new_node)
    for node in reversed(nodes):
        if not node.users:
            model.graph.erase_node(node)
    new_node.meta["submodule"] = submodule
    if (dtype := nodes[-1].meta.get("dtype", None)) is not None:
        new_node.meta["dtype"] = dtype
    inherit_physical_quantization_contract(new_node, nodes[-1])
    propagate_unsupported_operation(nodes, new_node)
    return new_node


def _nodes_sequential(
    nodes: List[Node],
    order: Dict[Node, int],
    operator_traits=None,
) -> bool:
    prior_nodes = set()
    prev_node = None
    for n in nodes:
        # Check if the current node is a user of the previous node
        if prev_node is not None and n not in prev_node.users:
            return False
        # We only handle dequantize after GEMM here
        if (
            n.target == torch.ops.quantized_ops.dequantize.default
            and not is_gemm_op(n.args[0], operator_traits)
        ):
            return False
        if prev_node is not None:
            predecessor_inputs = getattr(
                operator_traits,
                "execution_fusion_predecessor_inputs",
                {},
            )
            required_input = predecessor_inputs.get(n.target)
            if required_input is not None:
                schema = getattr(n.target, "_schema", None)
                arguments = getattr(schema, "arguments", ())
                kwarg_name = (
                    arguments[required_input].name
                    if required_input < len(arguments)
                    else None
                )
                if (
                    get_arg_value(
                        n,
                        required_input,
                        kwarg_name,
                    )
                    is not prev_node
                ):
                    return False
                for arg in n.all_input_nodes:
                    if arg is prev_node:
                        continue
                    if arg not in order or order[arg] >= order[n]:
                        return False
                    pending = [arg]
                    visited = set()
                    while pending:
                        dependency = pending.pop()
                        if dependency in prior_nodes or dependency is n:
                            return False
                        if dependency in visited:
                            continue
                        visited.add(dependency)
                        pending.extend(dependency.all_input_nodes)
                prior_nodes.add(n)
                prev_node = n
                continue
            for arg in n.all_input_nodes:
                if id(arg) == id(prev_node):
                    continue

                while is_nop(arg):
                    arg = arg.all_input_nodes[0]

                if arg.op == "call_function" and order[arg] > order[prev_node]:
                    return False
        prior_nodes.add(n)
        prev_node = n
    return True


def find_sequential_nodes_(
    pattern: List[List[Callable]],
    order: Dict[Node, int],
    nodes_by_source: Dict[Callable, List[Node]],
    require_all_pattern_stages: bool = False,
    operator_traits=None,
):
    def get_matched_nodes(matcher):
        return [
            n
            for tgt in matcher.targets
            for n in nodes_by_source[tgt]
            if matcher.matches(n)
        ]

    def collect_nop_chain(node):
        nops = []
        cur = next(iter(node.users), None)
        while cur and is_nop(cur) and len(cur.users) == 1:
            nops.append(cur)
            cur = next(iter(cur.users), None)
        return nops

    fused_chain = []
    fused_nodes = set()
    singleton_nodes = set(get_matched_nodes(pattern[0]))

    for stage_sources in pattern[1:]:
        stage_nodes = [
            n for n in get_matched_nodes(stage_sources) if n not in fused_nodes
        ]
        if not stage_nodes:
            if require_all_pattern_stages:
                return []
            continue

        fusion_candidates = fused_chain + [[n] for n in singleton_nodes]
        new_chains = []
        for nodes in fusion_candidates:
            if len(nodes) == 1 and nodes[0] in fused_nodes:
                continue

            last_node = nodes[-1]

            # Skip if the last node has multiple users
            if len(last_node.users) > 1:
                if len(nodes) > 1:
                    new_chains.append(nodes)
                continue

            nops = collect_nop_chain(last_node)
            matched = False

            for node in list(stage_nodes):
                if node in fused_nodes or order[node] < order[last_node]:
                    continue
                candidate = nodes + nops + [node]
                if _nodes_sequential(
                    candidate,
                    order,
                    operator_traits,
                ):
                    new_chains.append(candidate)
                    fused_nodes.update(candidate)
                    matched = True
                    break

            if (
                not matched
                and len(nodes) > 1
                and not require_all_pattern_stages
            ):
                new_chains.append(nodes)

        fused_chain = new_chains
        if require_all_pattern_stages:
            # A required execution pattern can only begin at its first stage.
            # Do not turn a later-stage singleton into a suffix fusion.
            singleton_nodes = set()
            if not fused_chain:
                return []
        else:
            singleton_nodes = (singleton_nodes | set(stage_nodes)) - fused_nodes

    return fused_chain


def find_sequential_nodes(
    model: GraphModule,
    patterns: List[List[List[Any]]],
    require_all_pattern_stages: bool = False,
    operator_traits=None,
):
    graph = model.graph
    nodes_order = {node: i for i, node in enumerate(graph.nodes)}

    all_sources = {
        op for pattern in patterns for stage in pattern for op in stage.targets
    }

    nodes_by_source = defaultdict(list)
    for node in list(graph.nodes):
        if node.target in all_sources:
            nodes_by_source[node.target].append(node)

    all_candidates = []
    for pattern in patterns:
        candidates = find_sequential_nodes_(
            pattern,
            nodes_order,
            nodes_by_source,
            require_all_pattern_stages=require_all_pattern_stages,
            operator_traits=operator_traits,
        )
        all_candidates.extend(candidates)

    all_candidates.sort(key=lambda group: len(group), reverse=True)

    final_groups = []
    seen_nodes = set()

    for group in all_candidates:
        if all(node not in seen_nodes for node in group):
            final_groups.append(group)
            seen_nodes.update(group)

    return final_groups


def is_tranpose(node: Node):
    """
    Transpose operations are characterized by swapping the last two dimensions
    """
    if node.target == torch.ops.aten.transpose.int:
        ndim = node.args[0].value.ndim
        axes = {x if x >= 0 else x + ndim for x in node.args[1:]}
        return axes == {ndim - 2, ndim - 1}

    if node.target == torch.ops.aten.permute.default:
        permute_dims = node.args[1]
        tranpose_dims = list(range(len(permute_dims)))
        tranpose_dims[-2], tranpose_dims[-1] = (
            tranpose_dims[-1],
            tranpose_dims[-2],
        )
        return permute_dims == tranpose_dims

    return False


def is_mha_qkv_permute(node):
    """
    Check if the node is a permutation used in multi-head attention (MHA)
    operations. It has characteristics that last dimension is a power of 2 and
    the permuted dimensions are the middle two dimensions (2 and 3) of a 4D
    tensor.
    """
    shape = getattr(node, "shape", None)
    if shape is None or len(shape) != 4:
        return False

    head_dim = shape[-1]
    if not isinstance(head_dim, int):
        return False

    # Don't support head dimension not being a power of 2
    if head_dim <= 0 or not math.log2(head_dim).is_integer():
        return False

    if node.target == torch.ops.aten.permute.default:
        dims = node.args[1]
        return len(dims) == 4 and dims == [0, 2, 1, 3]

    if node.target == torch.ops.aten.transpose.int:
        dims = {x if x >= 0 else x + 4 for x in node.args[1:]}
        return node.value.ndim == 4 and dims == {1, 2}

    return False


def search_group(node, node_lists):
    for l in node_lists:
        if node in l:
            return l
    return None


def duplicate_shared_nodes(
    graph: torch.fx.Graph, nodes: List[Node]
) -> List[Node]:
    """
    Ensures that nodes in the given list are independent by duplicating any
    node that has multiple users.

    This function processes the given list of nodes in topological order,
    identifying any node with multiple users. If such a node exists, it is
    duplicated so that all nodes in the list can be grouped together without
    affecting other nodes in the DAG.

    Args:
        graph (torch.fx.Graph): The FX graph being processed.
        nodes (List[Node]): A list of nodes to check for shared usage.

    Returns:
        List[Node]: A new list where shared nodes have been duplicated.
    """
    nodes_order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes = sorted(nodes, key=lambda n: nodes_order[n])

    for i in reversed(range(len(nodes) - 1)):
        node, next_node = nodes[i], nodes[i + 1]

        if len(node.users) == 1:
            continue

        with graph.inserting_before(next_node):
            new_node = graph.node_copy(node, lambda n: n)
        propagate_shape(new_node)

        next_node.replace_input_with(node, new_node)
        nodes[i] = new_node

    return nodes


def move_transpose_after_select(graph: torch.fx.Graph, nodes: List[Node]):
    transpose_node = nodes[0]

    select_nodes = [n for n in nodes if n.target == torch.ops.aten.select.int]
    chain = [transpose_node] + select_nodes
    for n, next_n in zip(chain[:-1], chain[1:]):
        if next_n not in n.users or next_n.args[1] != 0:
            return nodes

    if len(select_nodes) == 0:
        return nodes

    user_node = next(iter(select_nodes[-1].users))
    ndim = transpose_node.value.ndim
    dims = [
        (x + ndim if x < 0 else x) - len(select_nodes)
        for x in transpose_node.args[1:]
    ]

    with graph.inserting_before(user_node):
        new_node = graph.call_function(
            torch.ops.aten.transpose.int,
            (select_nodes[-1], *dims),
        )

    user_node.replace_input_with(select_nodes[-1], new_node)
    select_nodes[0].replace_input_with(transpose_node, transpose_node.args[0])
    graph.erase_node(transpose_node)

    for n in select_nodes + [new_node]:
        propagate_shape(n)

    nodes = [n for n in nodes if n not in select_nodes and n != transpose_node]
    nodes.insert(0, new_node)
    return nodes


def _fuse_reshape_with_input_impl(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    current_node: Node,
    fused_nodes: List[Node],
    simulate: bool = False,
    operator_traits=None,
) -> Union[bool, List[Node]]:
    reshape_node = fused_nodes[0]
    fused_nodes.append(current_node)

    # Check if fusion is valid
    if is_gemm_op(current_node, operator_traits) and not is_fully_connected(
        current_node
    ):
        input_node = fused_nodes[-2]
        if is_mha_qkv_permute(reshape_node):
            can_fuse = input_node == current_node.args[0]
        elif is_tranpose(reshape_node):
            can_fuse = input_node in current_node.args[:2]
        else:
            can_fuse = False
    elif is_elementwise_op(current_node):
        can_fuse = not is_tranpose(reshape_node)
    else:
        can_fuse = False

    if "tiled_shapes" not in current_node.meta and can_fuse:
        if simulate:
            return True
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        fused_nodes = move_transpose_after_select(graph, fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]
        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        return [fused_nodes[0]]
    else:
        logger.warning(f"Cannot fuse {reshape_node} with {current_node}")

    if not is_nop(current_node) and not (
        is_tranpose(reshape_node)
        and current_node.target == torch.ops.aten.select.int
        and current_node.args[1] == 0
    ):
        logger.info(f"Cannot fuse {reshape_node} with {current_node}")
        return False if simulate else []

    all_results = []
    for user in list(current_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            list(fused_nodes),
            simulate,
            operator_traits,
        )
        if simulate:
            if not result:
                return False
        else:
            all_results.extend(result)

    return True if simulate else all_results


def fuse_reshape_with_input(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node,
    operator_traits=None,
):
    # First pass: simulate fusion to ensure all users can be fused
    for user in list(reshape_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            [reshape_node],
            simulate=True,
            operator_traits=operator_traits,
        )
        if not result:
            logger.info(
                f"Skipping fusion for {reshape_node} due to unfusable path"
            )
            return

    # Second pass: perform actual fusion
    for user in list(reshape_node.users):
        result = _fuse_reshape_with_input_impl(
            graph,
            candidates,
            nodes_map,
            user,
            [reshape_node],
            simulate=False,
            operator_traits=operator_traits,
        )


def fuse_reshape_with_output(
    graph: torch.fx.Graph,
    candidates: List[List[Node]],
    nodes_map: Dict[Node, Node],
    reshape_node: Node,
    operator_traits=None,
) -> bool:
    if not is_mha_qkv_permute(reshape_node):
        return False

    curr_node = reshape_node.all_input_nodes[0]
    fused_nodes = [reshape_node]

    while not (
        is_gemm_op(curr_node, operator_traits) or is_elementwise_op(curr_node)
    ):
        if (
            len(curr_node.users) > 1
            or len(curr_node.all_input_nodes) != 1
            or not is_nop(curr_node)
        ):
            logger.debug(f"Cannot fuse {reshape_node} with {curr_node}")
            return False

        fused_nodes.insert(0, curr_node)
        curr_node = curr_node.all_input_nodes[0]

    def _is_tiled(n):
        return "tiled_shapes" in getattr(n, "meta", {})

    if len(curr_node.users) > 1 or _is_tiled(curr_node):
        return False

    group = search_group(curr_node, candidates)

    if group is not None:
        if any(_is_tiled(n) for n in group):
            return False
        else:
            group.extend(n for n in fused_nodes if n not in group)
    else:
        candidates.append([curr_node, *fused_nodes])

    nodes_map[reshape_node] = curr_node

    return True


def fuse_dequantize_with_gemm_or_elementwise(
    graph,
    candidates,
    nodes_map,
    node_to_fuse,
    operator_traits=None,
):
    for user in list(node_to_fuse.users):
        _fuse_dequantize_recursive(
            graph,
            candidates,
            nodes_map,
            user,
            [node_to_fuse],
            operator_traits,
        )


def _fuse_dequantize_recursive(
    graph,
    candidates,
    nodes_map,
    current_node,
    fused_nodes,
    operator_traits=None,
):
    fused_nodes.append(current_node)

    if (
        is_gemm_op(current_node, operator_traits)
        or is_elementwise_op(current_node)
        or current_node.target
        in [
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.softmax.int,
        ]
    ):
        fused_nodes = duplicate_shared_nodes(graph, fused_nodes)
        fused_nodes = move_dq_after_select(graph, fused_nodes)
        nodes_map[fused_nodes[0]] = fused_nodes[-2]

        if (group := search_group(current_node, candidates)) is not None:
            group.extend(n for n in fused_nodes if n not in group)
        else:
            candidates.append(fused_nodes)
        return

    if current_node.target != torch.ops.aten.select.int and not is_nop(
        current_node
    ):
        logger.info(f"Cannot fuse {fused_nodes[0]} with {current_node}")
        return

    for user in list(current_node.users):
        _fuse_dequantize_recursive(
            graph,
            candidates,
            nodes_map,
            user,
            list(fused_nodes),
            operator_traits,
        )


def move_dq_after_select(graph: torch.fx.Graph, nodes: List[Node]):
    node_to_move = nodes[0]

    for n in nodes[1:]:
        if is_prunable_op(n):
            n.replace_all_uses_with(n.args[0])
            graph.erase_node(n)
            nodes.remove(n)

    # Pick select nodes in the chain
    select_nodes = [n for n in nodes if n.target == torch.ops.aten.select.int]
    chain = [node_to_move] + select_nodes
    for n, next_n in zip(chain[:-1], chain[1:]):
        if next_n not in n.users or next_n.args[1] != 0:
            return nodes

    if len(select_nodes) == 0:
        return nodes

    user_node = next(iter(select_nodes[-1].users))

    def insert_select_ops(arg):
        if arg == node_to_move.args[0]:
            return select_nodes[-1]

        if "qmap" in arg.name or "code" in arg.name:
            return arg

        # Some dims are broadcasted, thus don't need to apply all selects
        for sel_node in select_nodes:
            arg_ndim = arg.value.ndim
            ndim = sel_node.args[0].value.ndim
            if sel_node.args[1] < ndim - arg_ndim:
                continue
            with graph.inserting_before(user_node):
                arg = graph.call_function(
                    torch.ops.aten.select.int,
                    (arg,) + sel_node.args[1:],
                )
            propagate_shape(arg)
            arg.meta["dtype"] = arg.args[0].meta.get("dtype", None)
        return arg

    with graph.inserting_before(user_node):
        new_node = graph.node_copy(node_to_move, insert_select_ops)

    # Update dequantize axes arguments
    axes = get_arg_value(new_node, 3, "axes")
    if axes is not None:
        axes = tuple(a - len(select_nodes) if a >= 0 else a for a in axes)
        new_node.update_arg(3, axes)

    user_node.replace_input_with(select_nodes[-1], new_node)
    select_nodes[0].replace_input_with(node_to_move, node_to_move.args[0])

    if len(node_to_move.users) == 0:
        graph.erase_node(node_to_move)

    for n in select_nodes + [new_node]:
        propagate_shape(n)
        n.meta["dtype"] = n.args[0].meta.get("dtype", None)

    # Respect the order of nodes appearing in the graph
    nodes = [n for n in nodes if n not in select_nodes and n != node_to_move]
    nodes.insert(0, new_node)
    return nodes


def fuse_operator(
    model: GraphModule,
    operations: List[List[Callable]] = None,
    fuse_reshape: bool = True,
    require_all_pattern_stages: bool = False,
    operator_traits=None,
):
    """
    Fuse reshape, slicing, and dequantize operations with their immediate users.

    The logic for fusing reshape operations is that we first trace up the graph
    until we reach a GEMM or elementwise operation. If we encounter a reshape operation
    or a node that has either multiple inputs or users along the way, we stop and
    try to fuse the node with its immediate user. For fusing a node with its user,
    we trace down the graph until we reach a GEMM or elementwise operation. If we
    encounter a node with multiple users, we duplicate all the elements on the path
    and perform fusion on each branch.
    """
    graph = model.graph
    named_modules = dict(model.named_modules(remove_duplicate=False))

    nodes_map = {}
    fused_nodes_list = []

    if operations is not None:
        fused_nodes_list = find_sequential_nodes(
            model,
            operations,
            require_all_pattern_stages=require_all_pattern_stages,
            operator_traits=operator_traits,
        )

    for node in list(graph.nodes):
        # Try to fuse MHA QKV permute with preceeding GEMM
        if fuse_reshape_with_output(
            graph,
            fused_nodes_list,
            nodes_map,
            node,
            operator_traits,
        ):
            continue

        # Attempt to fuse it with its immediate user
        if fuse_reshape and is_reshape_op(node):
            fuse_reshape_with_input(
                graph,
                fused_nodes_list,
                nodes_map,
                node,
                operator_traits,
            )

    for node in list(graph.nodes):
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        # If the node is already fused, skip it
        if search_group(node, fused_nodes_list) is not None:
            continue

        fuse_dequantize_with_gemm_or_elementwise(
            graph,
            fused_nodes_list,
            nodes_map,
            node,
            operator_traits,
        )

    # Sort nodes based on their order of appearance in the graph
    nodes_order = {node: i for i, node in enumerate(graph.nodes)}
    for nodes in fused_nodes_list:
        nodes.sort(key=lambda n: nodes_order[n])
    fused_nodes_list.sort(key=lambda g: nodes_order[g[-1]])

    nodes_map = {v.name: k.name for k, v in nodes_map.items()}

    for fused_nodes in fused_nodes_list:
        node = _create_and_insert_subgraph(
            fused_nodes,
            model,
            named_modules,
            operator_traits=operator_traits,
        )
        update_submod_user_meta(model, node)
        propagate_shape(node, model)
        gm = named_modules[node.target]

        for n in list(gm.graph.nodes):
            if (name := nodes_map.get(n.name, None)) is None:
                continue

            fused_node = next(iter(n for n in gm.graph.nodes if n.name == name))
            fused_node.meta["fused"] = True

            if is_reshape_op(fused_node):
                if next(iter(fused_node.users)).op == "output":
                    node.meta["reshape"] = fused_node
                else:
                    n.meta["reshape"] = fused_node

            if fused_node.target == torch.ops.quantized_ops.dequantize.default:
                n.meta["dequantize"] = fused_node

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    return model


def get_node_bytes(n: Node):
    if (dtype := n.meta.get("dtype", None)) is None:
        if isinstance(n.value, (list, tuple)):
            return (dtype_byte_size(t.dtype) for t in (n.value))
        else:
            return dtype_byte_size(n.value.dtype)

    if isinstance(dtype, (list, tuple)):
        dtypes = [
            t if t is not None else v.dtype for t, v in zip(dtype, n.value)
        ]
        return (dtype_byte_size(t) for t in dtypes)

    return dtype_byte_size(dtype if dtype is not None else n.value.dtype)


def get_tiled_tensor(arg, tiled_shapes=None):
    if not isinstance(arg, torch.fx.Node):
        return arg

    if tiled_shapes is None or arg not in tiled_shapes:
        return arg.value.clone()

    tensor = arg.value.clone()
    shape = tiled_shapes[arg]
    n = len(shape)
    slices = [slice(None)] * (tensor.ndim - n) + [slice(0, s) for s in shape]
    return tensor[tuple(slices)]


def get_node_to_key_map(node, operator_traits=None):
    """Project occurrence descriptors to legacy node-keyed tiling shapes."""
    result = {}
    for descriptor in operand_descriptors(
        node,
        operator_traits=operator_traits,
    ):
        tensor = descriptor.tensor.meta.get(
            "source_node",
            descriptor.tensor,
        )
        result.setdefault(tensor, descriptor.key)
    result[node] = "output"
    return result


def normalize_shape(node, shape, operator_traits=None):
    node_to_key = get_node_to_key_map(node, operator_traits)
    shape = {n: shape[k] for n, k in node_to_key.items() if k in shape}
    return shape


def get_allocation_role(
    node: Node,
    operator_traits=None,
) -> TensorRole:
    """Classify a graph value for the configured global memory level."""
    return classify_storage_role(node, operator_traits)


def get_anchor_node(nodes, operator_traits=None):
    anchor_node = None
    for n in nodes:
        if is_gemm_op(n, operator_traits):
            return n
        if n.op == "call_function" and (
            anchor_node is None
            or anchor_node.target == torch.ops.quantized_ops.dequantize.default
        ):
            anchor_node = n
    return anchor_node


def _build_conv2d_and_gemm_shape_map(
    node,
    output_shape,
    operator_traits=None,
):
    from voyager_compiler.codegen.passes.tiling import (
        _build_gemm_shape_map,
        _conv2d_layout,
    )

    input_node = node.args[0]
    transposed = node.meta.get("transposed", False)
    bs = node.kwargs.get("block_size", 1)

    if is_conv2d(node, operator_traits):
        # Certain conv2d layers (e.g. conv1) have extra pixels for alignment
        # purpose. We directly take the input shape from the input node.
        N, iy_tiled, ix_tiled, c_tiled = _conv2d_layout(
            node.args[0].shape, False, not transposed
        )
        kH, kW, actual_weight_c, _ = _conv2d_layout(
            node.args[1].shape,
            True,
            not transposed,
        )
        _, _, _, k_tiled = _conv2d_layout(output_shape, False, not transposed)
        groups = get_arg_value(node, 6, "groups", 1)
        if (
            not isinstance(groups, int)
            or isinstance(groups, bool)
            or groups <= 0
        ):
            raise ValueError(
                f"{node.name}: convolution groups must be a positive integer"
            )
        if c_tiled % groups:
            raise ValueError(
                f"{node.name}: input channels {c_tiled} are not divisible "
                f"by groups {groups}"
            )
        weight_c = c_tiled // groups
        if actual_weight_c != weight_c:
            raise ValueError(
                f"{node.name}: weight input channels {actual_weight_c} "
                f"do not match input channels/groups {weight_c}"
            )

        new_shapes = {
            "input": (N, c_tiled, iy_tiled, ix_tiled),
            "weight": (k_tiled, weight_c, kH, kW),
            "bias": (k_tiled,),
            "input_scale": (N, c_tiled // bs, iy_tiled, ix_tiled),
            "weight_scale": (k_tiled, weight_c // bs, kH, kW),
            "folded_coarse_scale": (k_tiled,),
        }

        new_shapes = {
            k: (
                v
                if k in {"bias", "folded_coarse_scale"}
                else _conv2d_layout(v, "weight" in k, transposed)
            )
            for k, v in new_shapes.items()
        }
    else:
        if is_bmm(node):
            x_tiled = output_shape[-2]
        else:
            x_tiled = math.prod(output_shape[:-1])
        k_tiled = output_shape[-1]
        c_tiled = input_node.shape[-1]

        new_shapes = _build_gemm_shape_map(node, (x_tiled, c_tiled, k_tiled))

    return normalize_shape(
        node,
        new_shapes,
        operator_traits,
    )


def run_submod_l2_tiling(
    node,
    module,
    key_to_node,
    tiled_shapes,
    unroll_dims,
    strategy,
    cache_size,
    bank_width,
    bank_size,
    memory_level=None,
    operator_traits=None,
):
    from voyager_compiler.codegen.passes.tiling import (
        _fits_memory_level,
        _merge_tiling,
        compute_output_tiled_shapes,
        compute_tiled_shape,
        get_valid_tiling,
    )

    anchor_node = get_anchor_node(
        module.graph.nodes,
        operator_traits,
    )

    total_size, scratchpad_map = strategy.evaluate(
        key_to_node,
        node,
        tiled_shapes,
        bank_width,
        bank_size,
        unroll_dims[1],
        memory_level=memory_level,
        operator_traits=operator_traits,
    )

    # Skip if GEMM already has fused reshape op
    is_gemm = is_gemm_op(anchor_node, operator_traits)
    has_sparse_outputs = anchor_node.kwargs.get("A_indptr") is not None
    submod_nodes = list(module.graph.nodes) + [node]
    if is_gemm and any("reshape" in n.meta for n in submod_nodes):
        logger.info(f"Skip submodule {node} which is a GEMM fused with reshape")
        return total_size, scratchpad_map, None

    min_sizes = None
    transposed = anchor_node.meta.get("transposed", False)

    output_shape = tiled_shapes[node]
    if isinstance(node.value, (list, tuple)):
        output_shape = output_shape[-1]

    if is_gemm:
        args = map_arg(node.args, lambda n: get_tiled_tensor(n, tiled_shapes))
        ShapeProp(module).propagate(*args)

        # We are not doing tiling on Y, X and C dimensions for conv layers here
        if is_conv2d(anchor_node, operator_traits):
            dim = 3 if transposed else 1
            min_sizes = (
                output_shape[:dim] + (unroll_dims[0],) + output_shape[dim + 1 :]
            )
        else:
            x_min_size = min(sum(unroll_dims), math.prod(output_shape[:-1]))
            min_sizes = (x_min_size, unroll_dims[0])

    for tile_sizes, tiling in get_valid_tiling(
        output_shape, min_sizes=min_sizes, reverse=is_gemm
    ):
        new_shapes = {}

        if is_gemm:
            new_shapes = _build_conv2d_and_gemm_shape_map(
                anchor_node,
                tile_sizes,
                operator_traits=operator_traits,
            )
            propagate_tiled_shapes_upstream(
                anchor_node,
                new_shapes,
                operator_traits,
            )

        for n in node.all_input_nodes:
            if n not in new_shapes and require_allocation(
                n,
                operator_traits,
            ):
                new_shapes[n] = compute_tiled_shape(tiled_shapes[n], tiling)

        new_shapes[node] = compute_output_tiled_shapes(
            node,
            tiling,
            tiled_shapes[node],
            has_sparse_outputs=has_sparse_outputs,
        )

        logger.debug("Proposed new shapes:")
        for n, s in new_shapes.items():
            logger.debug(f"  {n}: {s}")

        total_size, scratchpad_map = strategy.evaluate(
            key_to_node,
            node,
            new_shapes,
            bank_width,
            bank_size,
            unroll_dims[1],
            memory_level=memory_level,
            operator_traits=operator_traits,
        )
        logger.debug(f"  Total size: {total_size}, Available: {cache_size}")

        fits = (
            _fits_memory_level(
                anchor_node,
                new_shapes,
                memory_level,
                bank_width,
                operator_traits=operator_traits,
            )
            and memory_level.fits_segments(scratchpad_map.values())
            if memory_level is not None
            else total_size <= cache_size
        )
        if fits:
            # Tiling adjustment for linear and matmul layers
            if is_gemm and not is_conv2d(
                anchor_node,
                operator_traits,
            ):
                tiling = (math.prod(tiling[:-1]), tiling[-1])

            tiling = _merge_tiling(tiling, anchor_node.meta.get("l2_tiling"))

            if math.prod(tiling) == 1:
                new_shapes = None
                anchor_node.meta.pop("l2_tiling", None)
            else:
                anchor_node.meta["l2_tiling"] = tiling

                if is_gemm and anchor_node.kwargs.get("A_indptr") is not None:
                    input_node = anchor_node.args[0]
                    input_node = input_node.meta.get("source_node", input_node)
                    tile_strides = anchor_node.meta["tile_strides"]
                    tile_strides["A_indptr"] = new_shapes[input_node][:-1]

            anchor_node.meta.pop("tiled_shapes", None)

            return total_size, scratchpad_map, new_shapes

    logger.warning(f"Failed to adjust tiling for {node}")
    return total_size, None, None


def propagate_tiled_shapes_upstream(
    start_node,
    tiled_shapes,
    operator_traits=None,
):
    """
    Propagates tiling constraints backwards from the start_node's inputs
    (e.g., GEMM inputs) to their upstream dependencies (e.g., Dequantize inputs).

    Args:
        start_node: The main operation node (e.g., GEMM) whose inputs
                    already have defined tiled shapes.
        tiled_shapes: Dictionary mapping nodes to their tiled shapes.
    """
    from voyager_compiler.codegen.passes.tiling import compute_tiled_shape

    queue = list(start_node.all_input_nodes)
    visited = set(queue)

    while queue:
        node = queue.pop(0)

        if (tiled_shape := tiled_shapes.get(node)) is None:
            continue

        orig_shape = node.shape

        assert len(tiled_shape) == len(
            orig_shape
        ), f"Rank mismatch: {len(tiled_shape)} vs {len(orig_shape)}"
        factors = tuple(o // s for o, s in zip(orig_shape, tiled_shape))

        for n in node.all_input_nodes:
            if n in tiled_shapes or not require_allocation(
                n,
                operator_traits,
            ):
                continue

            node_key = n.meta.get("source_node", n)
            tiled_shapes[node_key] = compute_tiled_shape(n.shape, factors)

            if n not in visited and n.all_input_nodes:
                visited.add(n)
                queue.append(n)


def _get_scheduler_operand_bindings(
    output_node,
    anchor_node,
    named_modules,
    operator_traits=None,
):
    """Bind typed semantic operand occurrences to current graph nodes."""
    source_to_current = {}
    if output_node.op == "call_module":
        module = named_modules[output_node.target]
        placeholders = [
            node for node in module.graph.nodes if node.op == "placeholder"
        ]
        for index, placeholder in enumerate(placeholders):
            current = (
                output_node.args[index]
                if index < len(output_node.args)
                else output_node.kwargs.get(str(placeholder.target))
            )
            if isinstance(current, Node):
                source_to_current[
                    placeholder.meta.get("source_node", placeholder)
                ] = current

    descriptors = operand_descriptors(
        anchor_node,
        operator_traits=operator_traits,
    )
    bindings = []
    for descriptor in descriptors:
        source_node = descriptor.tensor.meta.get(
            "source_node",
            descriptor.tensor,
        )
        current_node = source_to_current.get(source_node, source_node)
        bindings.append((descriptor, current_node))
    return tuple(bindings)


def _scheduler_node_bits(node):
    return _scheduler_value_bits(ValueRef(node))


def _value_at_ref(ref):
    value = getattr(ref.node, "value", None)
    try:
        for index in ref.path:
            value = value[index]
    except (IndexError, TypeError):
        return None
    return value


def _scheduler_value_dtype(ref):
    """Resolve one scheduler value's compiler-declared physical dtype."""
    if not ref.path:
        return get_physical_dtype(ref.node)

    dtype = ref.node.meta.get(
        QUANTIZATION_DOMAIN_DTYPE,
        ref.node.meta.get("dtype"),
    )
    for index in ref.path:
        if dtype is None:
            break
        if not isinstance(dtype, (tuple, list)):
            return dtype
        try:
            dtype = dtype[index]
        except IndexError as error:
            raise ValueError(
                f"{ref.name}: compiler dtype does not describe the "
                "selected output"
            ) from error
    return dtype


def _declared_physical_payload(ref_node):
    """Resolve a unique tuple payload from its physical format contract."""
    contract = get_physical_quantization_contract(ref_node)
    if contract is None:
        return None
    dtype = ref_node.meta.get(
        QUANTIZATION_DOMAIN_DTYPE,
        ref_node.meta.get("dtype"),
    )
    if not isinstance(dtype, (tuple, list)):
        return None
    matches = tuple(
        index
        for index, component_dtype in enumerate(dtype)
        if component_dtype == contract.payload_dtype
    )
    if len(matches) != 1:
        return None
    return ValueRef(ref_node, matches)


def _scheduler_value_bits(ref):
    dtype = _scheduler_value_dtype(ref)
    if dtype is None:
        value = _value_at_ref(ref)
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
    return None if dtype is None else get_dtype_width(dtype)


def _is_multi_output_shape(shape):
    return (
        isinstance(shape, tuple) and bool(shape) and isinstance(shape[0], tuple)
    )


def _get_scheduler_operands(
    output_node,
    anchor_node,
    tiled_shapes,
    named_modules,
    operator_traits=None,
):
    """Freeze typed operand occurrences for the scheduler boundary."""
    bindings = _get_scheduler_operand_bindings(
        output_node,
        anchor_node,
        named_modules,
        operator_traits,
    )
    return tuple(
        OperationOperand(
            port=descriptor.key or "operand",
            shape=tiled_shapes.get(current_node),
            bits=_scheduler_node_bits(current_node),
            occurrence=descriptor.occurrence,
        )
        for descriptor, current_node in bindings
    )


def _resolve_scheduler_output(
    output_node,
    anchor_node,
    tiled_shapes,
    operator_traits=None,
):
    """Resolve the one physical result scheduled for an operation."""
    output_shape = tiled_shapes.get(output_node)
    if output_shape is None:
        raise ValueError(
            f"{output_node.name}: scheduler output shape is unresolved"
        )
    if not _is_multi_output_shape(output_shape):
        return OperationOperand(
            port="output",
            shape=output_shape,
            bits=_scheduler_node_bits(output_node),
        )

    output_values = output_node.value
    if not isinstance(output_values, (tuple, list)) or len(
        output_values
    ) != len(output_shape):
        raise ValueError(
            f"{output_node.name}: scheduler output values do not match "
            "the resolved output shapes"
        )

    payload = output_node.meta.get(
        TARGET_CODEGEN_KWARGS,
        {},
    ).get("output_payload")
    if payload is None:
        payload = _declared_physical_payload(output_node)
    is_sparse_gemm = (
        is_gemm_op(anchor_node, operator_traits)
        and anchor_node.kwargs.get("A_indptr") is not None
    )
    if payload is None:
        if not is_sparse_gemm:
            contract = get_physical_quantization_contract(output_node)
            declared_dtype = output_node.meta.get(
                QUANTIZATION_DOMAIN_DTYPE,
                output_node.meta.get("dtype"),
            )
            raise ValueError(
                f"{output_node.name}: multi-output scheduling requires an "
                "explicit output_payload ValueRef; "
                f"physical_contract={contract!r}, dtype={declared_dtype!r}"
            )
        payload = ValueRef(output_node, (len(output_shape) - 1,))
    elif not isinstance(payload, ValueRef) or payload.node is not output_node:
        raise ValueError(
            f"{output_node.name}: output_payload must reference this "
            "operation"
        )

    if len(payload.path) != 1:
        raise ValueError(
            f"{output_node.name}: output_payload must select one "
            "top-level result"
        )
    output_index = payload.path[0]
    if (
        isinstance(output_index, bool)
        or not isinstance(output_index, int)
        or not 0 <= output_index < len(output_shape)
    ):
        raise ValueError(
            f"{output_node.name}: output_payload index "
            f"{output_index!r} is out of range"
        )

    return OperationOperand(
        port="output",
        shape=output_shape[output_index],
        bits=_scheduler_value_bits(payload),
        occurrence=payload.path,
    )


def build_scheduler_workload(
    output_node,
    anchor_node,
    tiled_shapes,
    hardware_config,
    named_modules,
    target,
    operator_traits=None,
):
    """Freeze graph facts into one backend-neutral scheduler workload.

    Unclaimed tuple-valued graph utilities return ``None`` because they have
    no single physical output for a scheduler to consume.
    """
    operands = _get_scheduler_operands(
        output_node,
        anchor_node,
        tiled_shapes,
        named_modules,
        operator_traits,
    )
    operation = classify_target_scheduler_operation(
        anchor_node,
        target,
        operands,
    )
    output_shape = tiled_shapes.get(output_node)
    output_payload = output_node.meta.get(
        TARGET_CODEGEN_KWARGS,
        {},
    ).get("output_payload")
    if (
        _is_multi_output_shape(output_shape)
        and output_payload is None
        and operation is None
        and not is_gemm_op(anchor_node, operator_traits)
    ):
        # Tuple-valued graph utilities such as chunk are not scheduler
        # operations. Do not invent one representative output merely to send
        # an operation that no scheduler can claim through dispatch.
        return None
    output = _resolve_scheduler_output(
        output_node,
        anchor_node,
        tiled_shapes,
        operator_traits,
    )
    input_width = next(
        (
            operand.bits
            for name in ("input", "input1")
            for operand in operands
            if operand.port == name and operand.bits is not None
        ),
        None,
    )
    has_hp_vector_input = False
    if output_node.op == "call_module" and input_width is not None:
        gm = named_modules[output_node.target]
        for node in gm.graph.nodes:
            if node.op != "placeholder" or anchor_node in node.users:
                continue
            source = node.meta.get("source_node", node)
            source_width = _scheduler_node_bits(source)
            if source_width is not None and source_width > input_width:
                has_hp_vector_input = True
                break

    has_sparse_op = anchor_node.kwargs.get("A_indptr") is not None
    return extract_operation_workload(
        anchor_node,
        operands,
        output,
        hardware_config,
        operator_traits,
        operation=operation,
        has_high_precision_vector_input=has_hp_vector_input,
        has_sparse_op=has_sparse_op,
    )


def _scheduler_options_for_node(options, output_node):
    if getattr(options, "artifact_dir", None) is None:
        return options
    return replace(
        options,
        artifact_dir=os.path.join(
            str(options.artifact_dir),
            output_node.name,
        ),
    )


def _log_local_schedule(output_node, workload, result):
    if workload.gemm is not None:
        gemm = workload.gemm
        logger.info(
            "[%s] %s: IC=%s OC=%s H=%s W=%s kH=%s kW=%s",
            result.backend,
            output_node.name,
            gemm.nifm,
            gemm.nofm,
            gemm.hofm,
            gemm.wofm,
            gemm.hfil,
            gemm.wfil,
        )
    logger.info(
        "[%s] %s: runtime=%s",
        result.backend,
        output_node.name,
        result.runtime,
    )
    if result.mapping is not None:
        logger.info("%s", result.mapping)


def _schedule_node(
    output_node,
    anchor_node,
    tiled_shapes,
    hardware_config,
    provider_cache,
    result_cache,
    named_modules,
    target,
    scheduler_backend,
    scheduler_providers,
    scheduler_options,
    operator_traits=None,
):
    workload = build_scheduler_workload(
        output_node,
        anchor_node,
        tiled_shapes,
        hardware_config,
        named_modules,
        target,
        operator_traits,
    )
    if workload is None:
        return False
    scheduler_options = _scheduler_options_for_node(
        scheduler_options,
        output_node,
    )
    result = schedule_operation(
        workload,
        hardware_config,
        provider_cache,
        scheduler_options,
        result_cache,
        providers=scheduler_providers,
        backend=scheduler_backend,
    )
    if result is None:
        return False
    _log_local_schedule(output_node, workload, result)
    bind_local_schedule(output_node, result, workload)
    return True


def adjust_tiling(
    node: Node,
    named_modules: dict,
    cache_size: int,
    num_banks: int = None,
    bank_width: int = None,
    unroll_dims=None,
    local_scheduler=None,
    hardware_config: HardwareConfig | None = None,
    operator_traits=None,
):
    """Tile one node for the scratchpad and return its ``scratchpad_map``.

    For a fused ``call_module`` this runs per-submodule L2 tiling
    (``run_submod_l2_tiling``), fitting the reference op's tiling to ``cache_size``
    and node-keying the tiling dict; for a standalone tiled node it evaluates the
    banking strategy directly.  Mutates ``node.meta`` (``tiled_shapes`` /
    ``tile_strides``) and the reference op's ``l2_tiling`` in place. When
    ``local_scheduler`` is given, it schedules GEMMs and target-supported
    operations. Returns the ``scratchpad_map`` for the caller to attach
    (``run_memory_mapping`` does; the bufferized path, which only needs the
    tiling, ignores it).

    ``local_scheduler`` contains the hardware configuration, caches,
    double-buffering mode, compilation target, and scheduler backend, or is
    ``None``.
    Was ``run_memory_mapping.allocate_scratchpad`` (closures lifted to params).
    """
    from voyager_compiler.codegen.passes.tiling import (
        _fits_memory_level,
        compute_output_tiled_shapes,
        compute_tiled_shape,
    )

    if cache_size is None and hardware_config is None:
        return None
    if operator_traits is None and hardware_config is not None:
        operator_traits = get_target_operator_traits(hardware_config.target)

    bank_size = (
        cache_size // num_banks
        if cache_size is not None and num_banks is not None
        else None
    )

    if node.op == "call_module":
        mod = named_modules[node.target]
        anchor_node = get_anchor_node(
            mod.graph.nodes,
            operator_traits,
        )

        for n in list(mod.graph.nodes):
            if n != anchor_node:
                n.meta.pop("l2_tiling", None)
    else:
        anchor_node = node

    has_sparse_outputs = anchor_node.kwargs.get("A_indptr") is not None
    effective_unroll_dims = unroll_dims
    if is_gemm_op(anchor_node, operator_traits):
        effective_unroll_dims = resolve_unroll_dims(
            node.args[0],
            unroll_dims if hardware_config is None else None,
            hardware_config,
        )

    l2_tiling = anchor_node.meta.get("l2_tiling")
    tiled_shapes = normalize_shape(
        anchor_node,
        anchor_node.meta.get("tiled_shapes") or {},
        operator_traits,
    )

    if not tiled_shapes:
        tiled_shapes = {n: n.shape for n in node.all_input_nodes}
        if isinstance(node.value, torch.Tensor):
            tiled_shapes[node] = node.shape
        else:
            tiled_shapes[node] = tuple(t.shape for t in node.value)
    elif node.op == "call_module":
        # TODO Skip if GEMM does not have fused dequantize op
        args = map_arg(node.args, lambda n: get_tiled_tensor(n))
        ShapeProp(mod).propagate(*args)
        propagate_tiled_shapes_upstream(
            anchor_node,
            tiled_shapes,
            operator_traits,
        )

        # Calculate tiled shape for other input/output nodes
        for n in node.all_input_nodes:
            if n not in tiled_shapes:
                tiled_shapes[n] = compute_tiled_shape(n.shape, l2_tiling)

        tiled_shapes[node] = compute_output_tiled_shapes(
            node,
            l2_tiling,
            has_sparse_outputs=has_sparse_outputs,
        )

    tiled_shapes = {
        n: s
        for n, s in tiled_shapes.items()
        if (
            n in node.all_input_nodes and require_allocation(n, operator_traits)
        )
        or n is node
    }

    op_scope = _get_scope(anchor_node.target)
    node_to_key = get_node_to_key_map(
        anchor_node,
        operator_traits,
    )
    key_to_node = {f"{op_scope}::{v}": k for k, v in node_to_key.items()}

    strategies = get_banking_strategies_for_op(
        anchor_node.target,
        operator_traits,
    )

    logger.debug(f"Allocating scratchpad for {node} with strategies:")

    final_tiled_shapes = tiled_shapes
    for strategy in strategies:
        if node.op == "call_module":
            total_size, scratchpad_map, new_shapes = run_submod_l2_tiling(
                node,
                mod,
                key_to_node,
                tiled_shapes,
                effective_unroll_dims,
                strategy,
                cache_size=cache_size,
                bank_width=bank_width,
                bank_size=bank_size,
                memory_level=(
                    hardware_config.global_memory
                    if hardware_config is not None
                    else None
                ),
                operator_traits=operator_traits,
            )
        else:
            total_size, scratchpad_map = strategy.evaluate(
                key_to_node,
                node,
                tiled_shapes,
                bank_width,
                bank_size,
                effective_unroll_dims[1],
                memory_level=(
                    hardware_config.global_memory
                    if hardware_config is not None
                    else None
                ),
                operator_traits=operator_traits,
            )
            new_shapes = tiled_shapes if l2_tiling else None

        fits = (
            _fits_memory_level(
                anchor_node,
                new_shapes or tiled_shapes,
                hardware_config.global_memory,
                bank_width,
                operator_traits=operator_traits,
            )
            and hardware_config.global_memory.fits_segments(
                scratchpad_map.values()
            )
            if hardware_config is not None
            else total_size <= cache_size
        )
        if fits:
            node.meta["tiled_shapes"] = new_shapes
            if new_shapes is not None:
                final_tiled_shapes = new_shapes
            logger.info(
                f"  Successfully tiled {node} with strategy: {strategy}"
            )
            break

    logger.debug("Scratchpad allocation result:")
    for n, s in scratchpad_map.items():
        logger.debug(f"  {n}: {s}")

    strides = anchor_node.meta.get("tile_strides")
    if strides is not None:
        node.meta["tile_strides"] = normalize_shape(
            anchor_node,
            strides,
            operator_traits,
        )

    strategy.print_banking_info(
        key_to_node,
        node,
        operator_traits,
    )

    if not fits:
        logger.warning(
            "[MEM_ALLOC_FAIL] %s: Could not allocate operand tiles in %s",
            node,
            (
                hardware_config.global_memory.name
                if hardware_config is not None
                else f"{cache_size} bytes"
            ),
        )
    # Run the local scheduler when a hardware configuration is provided.
    if local_scheduler is not None:
        (
            _scheduler_config,
            _scheduler_provider_cache,
            _scheduler_cache,
            scheduler_target,
            scheduler_backend,
            scheduler_providers,
            scheduler_options,
        ) = local_scheduler
        if _scheduler_config is not None:
            _schedule_node(
                node,
                anchor_node,
                final_tiled_shapes,
                _scheduler_config,
                _scheduler_provider_cache,
                _scheduler_cache,
                named_modules,
                scheduler_target,
                scheduler_backend,
                scheduler_providers,
                scheduler_options,
                operator_traits,
            )

    return scratchpad_map


def run_memory_mapping(
    model: GraphModule,
    allocator: MemoryAllocator = None,
    cache_size: int = None,
    num_banks: int = None,
    bank_width: int = None,
    unroll_dims=None,
    double_buffered_accum_buffer: bool = False,
    input_buffer_size: int = None,
    weight_buffer_size: int = None,
    accum_buffer_size: int = None,
    hardware_config: HardwareConfig = None,
    scheduler_backend: str | None = None,
    scheduler_options=None,
    scratchpad_allocator: ScratchpadAllocator = None,
    scratchpad_mapping: str = PER_OPERATION_SCRATCHPAD_MAPPING,
    execution_order: str = DEFAULT_EXECUTION_ORDER_MODE,
):
    validate_execution_order_mode(
        execution_order,
        bufferized=False,
        end_to_end=(scratchpad_mapping == END_TO_END_SCRATCHPAD_MAPPING),
    )
    model.meta.pop(EXECUTION_ORDER_RESULT_META, None)
    model.meta.pop("storage_plan", None)

    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    bind_compilation_target(
        model,
        hardware_config,
        stage="run_memory_mapping",
    )
    operator_traits = get_target_operator_traits(target)
    scheduler_providers = get_target_scheduler_providers(target)
    scheduler_backend = resolve_local_scheduler(
        scheduler_backend,
        providers=scheduler_providers,
    )
    if hardware_config is not None:
        profile_double_buffered_accum = (
            hardware_config.memory_is_double_buffered(
                hardware_config.local_memory,
                TensorRole.OUTPUT,
            )
        )
        if double_buffered_accum_buffer and not profile_double_buffered_accum:
            raise ValueError(
                "hardware_config conflicts with " "double_buffered_accum_buffer"
            )
        double_buffered_accum_buffer = profile_double_buffered_accum
        profile_values = (
            ("unroll_dims", unroll_dims, hardware_config.unroll_dims),
            (
                "num_banks",
                num_banks,
                hardware_config.global_num_banks,
            ),
            (
                "bank_width",
                bank_width,
                hardware_config.memory_alignment_bytes,
            ),
        )
        conflicts = [
            name
            for name, value, expected in profile_values
            if value is not None and value != expected
        ]
        if conflicts:
            raise ValueError(
                "hardware_config conflicts with legacy mapping arguments: "
                + ", ".join(conflicts)
            )
        legacy_buffer_values = (
            input_buffer_size,
            weight_buffer_size,
            accum_buffer_size,
        )
        if any(value is not None for value in legacy_buffer_values):
            raise ValueError(
                "hardware_config cannot be combined with legacy local "
                "buffer sizes"
            )
        unroll_dims = hardware_config.unroll_dims
        num_banks = hardware_config.global_num_banks
        bank_width = hardware_config.memory_alignment_bytes
    if scratchpad_mapping not in SCRATCHPAD_MAPPING_MODES:
        raise ValueError(
            f"Unknown scratchpad mapping mode: {scratchpad_mapping}"
        )
    end_to_end_scratchpad = scratchpad_mapping == END_TO_END_SCRATCHPAD_MAPPING
    if end_to_end_scratchpad and hardware_config is None:
        raise ValueError(
            "end_to_end scratchpad mapping requires a hardware profile"
        )

    apply_target_memory_rewrites(
        model,
        target,
        hardware_config,
    )
    graph = model.graph
    graph.eliminate_dead_code()
    model.recompile()
    declared_unsupported = tuple(
        node
        for node in model.graph.nodes
        if is_declared_unsupported_operation(node)
    )
    if end_to_end_scratchpad and declared_unsupported:
        details = []
        for node in declared_unsupported:
            declaration = node.meta["unsupported_operation"]
            details.append(f"{node.name} ({declaration['reason_code']})")
        raise ValueError(
            "end_to_end scratchpad mapping cannot execute explicit "
            "unsupported boundaries: " + ", ".join(details)
        )
    # A target can replace aggregate copies only when its model-wide
    # scratchpad allocator is actually selected.  Per-operation mapping uses
    # the generic allocator, so it must retain the generic copy fallback.
    if not end_to_end_scratchpad or target_uses_generic_aggregate_copies(
        target
    ):
        materialize_aggregate_copies(model)
    selected_execution_order = fx_execution_order(model)
    execution_nodes = selected_execution_order.nodes
    named_modules = dict(model.named_modules(remove_duplicate=False))

    if isinstance(unroll_dims, int):
        unroll_dims = (unroll_dims, unroll_dims)

    # Defer architecture construction until each node's logical dtype is known.
    _scheduler_config = None
    _scheduler_provider_cache = {}
    _scheduler_cache = {}
    if hardware_config is not None:
        _scheduler_config = hardware_config
    elif unroll_dims and all(
        x is not None
        for x in (
            input_buffer_size,
            weight_buffer_size,
            accum_buffer_size,
        )
    ):
        _scheduler_config = HardwareConfig.from_legacy(
            unroll_dims=unroll_dims,
            cache_size=cache_size,
            input_buffer_size=input_buffer_size,
            weight_buffer_size=weight_buffer_size,
            accum_buffer_size=accum_buffer_size,
            num_banks=num_banks,
            bank_width=bank_width,
            double_buffered_accum_buffer=(double_buffered_accum_buffer),
            target=target,
        )

    if allocator is None:
        allocator = MemoryAllocator(DEFAULT_MEMORY_SIZE, bank_width)
    operation_scoped_scratchpad = (
        not end_to_end_scratchpad
        and target_supports_operation_scoped_scratchpad(
            target,
        )
    )
    uses_target_scratchpad = (
        end_to_end_scratchpad or operation_scoped_scratchpad
    )
    if (
        hardware_config is not None
        and scratchpad_allocator is None
        and uses_target_scratchpad
    ):
        scratchpad_allocator = create_scratchpad_allocator(hardware_config)
    allocator.bind_operator_traits(operator_traits)
    bind_scratchpad_traits = getattr(
        scratchpad_allocator,
        "bind_operator_traits",
        None,
    )
    if bind_scratchpad_traits is not None:
        bind_scratchpad_traits(operator_traits)

    storage_key = "scratchpad" if end_to_end_scratchpad else "memory"
    storage_allocator = (
        scratchpad_allocator if end_to_end_scratchpad else allocator
    )
    storage_planes = (
        (StoragePlane.DRAM, StoragePlane.SCRATCHPAD)
        if uses_target_scratchpad
        else (StoragePlane.DRAM,)
    )
    storage_graph = StorageGraph(model, storage_planes)

    def has_logical_users(node):
        return any(
            storage_graph.index.logical_tensors[output].users
            for output in storage_graph.index.node_outputs.get(node, ())
        )

    target_storage_allocator = (
        scratchpad_allocator if uses_target_scratchpad else storage_allocator
    )
    target_storage_key = (
        StoragePlane.SCRATCHPAD.value if uses_target_scratchpad else storage_key
    )
    prepare_graph = getattr(
        target_storage_allocator,
        "prepare_graph",
        None,
    )
    if prepare_graph is not None:
        prepare_graph(
            model,
            storage_graph=storage_graph,
            storage_plane=target_storage_key,
            persistent_storage=end_to_end_scratchpad,
        )
    if end_to_end_scratchpad:
        validate_target_end_to_end_scratchpad_operations(
            target,
            target_storage_allocator,
            model,
        )

    if end_to_end_scratchpad:
        for node in execution_nodes:
            if node.op not in {"placeholder", "get_attr"}:
                continue
            if node.op == "placeholder" and not has_logical_users(node):
                continue
            if node.op == "get_attr" and not require_allocation(
                node,
                operator_traits,
            ):
                continue
            storage_graph.add_transfer(
                node,
                source_plane=StoragePlane.DRAM,
                destination_plane=StoragePlane.SCRATCHPAD,
                kind="load",
            )

    prepare_target_serialized_artifacts(
        model,
        target,
        storage_graph,
        target_storage_key,
        target_storage_allocator,
    )
    storage_plan = storage_graph.finalize(
        execution_order=selected_execution_order,
    )
    dp_result = None
    resident_physical_ids = ()

    def bind_selected_storage_plan() -> None:
        """Bind only the final order-derived plan before allocation."""

        def storage_access_requirements(value):
            """Return every exact role/unit access to one physical tensor."""
            return physical_access_requirements(
                storage_plan,
                hardware_config,
                target_storage_key,
                value,
                operator_traits,
            )

        bind_access_requirements = getattr(
            target_storage_allocator,
            "bind_access_requirements",
            None,
        )
        if bind_access_requirements is not None:
            bind_access_requirements(storage_access_requirements)

        allocator.bind_storage_plan(storage_plan, StoragePlane.DRAM)
        if uses_target_scratchpad:
            scratchpad_allocator.bind_storage_plan(
                storage_plan,
                StoragePlane.SCRATCHPAD,
            )

    def resident_storage_identities(plan):
        """Return the exact physical backings this mapping path allocates."""
        resident = {
            endpoint
            for transfer in plan.transfers
            for endpoint in (transfer.source, transfer.destination)
            if endpoint.plane == storage_key
        }
        resident.update(
            physical
            for node in model.graph.nodes
            if node.op in {"call_function", "call_module"}
            for physical in plan.produced_physical_outputs(
                node,
                storage_key,
            )
        )
        resident.update(
            physical
            for physical, tensor in plan.physical_tensors[storage_key].items()
            if tensor.accesses
        )
        return tuple(sorted(resident, key=lambda physical: physical.name))

    if execution_order == FX_EXECUTION_ORDER_MODE:
        bind_selected_storage_plan()

    nested_outputs = [
        output.name
        for node in execution_nodes
        for output in storage_plan.output_refs(node)
        if len(output.path) > 1
    ]
    if nested_outputs:
        raise NotImplementedError(
            "memory mapping does not yet serialize nested tensor output "
            "containers: " + ", ".join(nested_outputs)
        )

    # A dim-0 cat/stack is emitted as address-only. Every input must therefore
    # have been lowered into the aggregate's exact physical backing by either
    # generic copy materialization or the selected target allocator.
    for aggregate in execution_nodes:
        if aggregate.target not in {
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        }:
            continue
        dim = aggregate.kwargs.get(
            "dim",
            aggregate.args[1] if len(aggregate.args) > 1 else 0,
        )
        if dim != 0 or not aggregate.args:
            continue
        tensors = aggregate.args[0]
        if not isinstance(tensors, (tuple, list)):
            continue
        backing = storage_plan.physical_id(aggregate, storage_key)
        unlowered = [
            (
                tensor,
                storage_plan.relations[storage_key][storage_plan.ref(tensor)],
            )
            for tensor in tensors
            if (
                isinstance(tensor, Node)
                and storage_plan.physical_id(tensor, storage_key) != backing
            )
        ]
        if unlowered:
            aggregate_relation = storage_plan.relations[storage_key][
                storage_plan.ref(aggregate)
            ]
            raise ValueError(
                f"{aggregate.name} ({aggregate_relation.kind.value}): "
                "address-only dim-0 aggregate has "
                "unlowered inputs: "
                + ", ".join(
                    f"{tensor.name} ({relation.kind.value}, "
                    f"shape={tuple(getattr(tensor, 'shape', ()))})"
                    for tensor, relation in unlowered
                )
                + f"; aggregate shape={tuple(getattr(aggregate, 'shape', ()))}"
            )

    local_scheduler = (
        _scheduler_config,
        _scheduler_provider_cache,
        _scheduler_cache,
        target,
        scheduler_backend,
        scheduler_providers,
        scheduler_options,
    )

    def map_operation_scratchpad(node: Node):
        if (
            node.target == operator.getitem
            or is_nop(node)
            or node.meta.get(TARGET_INSTRUCTION_NOP, False)
            or is_declared_unsupported_operation(node)
        ):
            node.meta.pop("scratchpad_map", None)
            node.meta.pop("physical_scratchpad_map", None)
            return

        # Tiling + scratchpad logic lives in the module-level ``adjust_tiling``
        # (shared with the bufferized path). End-to-end mapping retains the
        # tiling and schedule but uses each tensor's model-wide scratchpad base
        # instead of an independently allocated per-operation tile address.
        scratchpad_map = adjust_tiling(
            node,
            named_modules,
            cache_size,
            num_banks,
            bank_width,
            unroll_dims,
            local_scheduler,
            hardware_config,
            operator_traits,
        )
        if end_to_end_scratchpad:
            node.meta.pop("scratchpad_map", None)
        else:
            node.meta["scratchpad_map"] = scratchpad_map
            if target_has_operation_scope(
                target,
                target_storage_allocator,
                node,
            ):
                physical_map = allocate_target_operation_scope(
                    target,
                    target_storage_allocator,
                    node,
                )
                if not isinstance(physical_map, dict) or not physical_map:
                    raise ValueError(
                        f"{node.name}: target operation scope is empty"
                    )

    preallocated = {
        node for node in execution_nodes if storage_key in node.meta
    }
    for node in execution_nodes:
        if node.op not in {"call_function", "call_module"}:
            continue
        if (
            node not in preallocated
            and not end_to_end_scratchpad
            and node.meta.get("scratchpad_map") is not None
        ):
            continue
        map_operation_scratchpad(node)

    if execution_order == DP_EXECUTION_ORDER_MODE:
        operation_constraints = getattr(
            storage_allocator,
            "operation_constraints",
            None,
        )
        if not callable(operation_constraints):
            raise DPExecutionOrderError(
                "DP execution order requires an end-to-end allocator "
                "with authoritative operation constraints"
            )
        constraints = tuple(operation_constraints())
        if not all(
            isinstance(entry, tuple) and len(entry) == 2
            for entry in constraints
        ):
            raise DPExecutionOrderError(
                "End-to-end operation constraints are malformed"
            )
        executable_nodes = tuple(
            operation for operation, _constraint in constraints
        )
        resident_physical_ids = resident_storage_identities(storage_plan)
        dp_result = solve_execution_order_dp(
            model,
            storage_plan,
            executable_nodes,
            storage_plane=StoragePlane.SCRATCHPAD,
            resident_physical_ids=resident_physical_ids,
        )
        selected_execution_order = dp_result.order
        execution_nodes = selected_execution_order.nodes
        storage_plan = storage_graph.finalize(
            execution_order=selected_execution_order,
        )
        final_resident_physical_ids = resident_storage_identities(storage_plan)
        if final_resident_physical_ids != resident_physical_ids:
            raise DPExecutionOrderError(
                "Selected execution order changed resident physical "
                "storage membership"
            )
        allocation_physicals = set(resident_physical_ids)
        measured_peak = max(
            (
                sum(
                    storage_plan.backing_size_bytes(
                        physical,
                        storage_key,
                    )
                    for physical, interval in storage_plan.intervals[
                        storage_key
                    ].items()
                    if (
                        interval.birth <= position < interval.death
                        and physical in allocation_physicals
                    )
                )
                for position in range(len(execution_nodes))
            ),
            default=0,
        )
        if measured_peak != dp_result.peak_live_bytes:
            raise DPExecutionOrderError(
                "Selected storage-plan liveness disagrees with the DP "
                f"objective: plan={measured_peak}, "
                f"solver={dp_result.peak_live_bytes}"
            )
        logger.info(
            "Selected exact DP execution order: candidates=%d, "
            "states=%d/%d, peak_live_bytes=%d, "
            "source_candidate_peak_live_bytes=%d",
            dp_result.candidate_count,
            dp_result.explored_states,
            dp_result.state_limit,
            dp_result.peak_live_bytes,
            dp_result.source_candidate_peak_live_bytes,
        )
        bind_selected_storage_plan()

    logical_segments = {}

    def publish_node_segments(node: Node) -> None:
        """Index exact logical results by their already-lowered segments."""
        outputs = storage_plan.output_refs(node)
        if not outputs:
            return
        segments = node.meta.get(f"{storage_key}_output_segments")
        if segments is not None:
            if len(segments) != len(outputs):
                raise ValueError(
                    f"{node.name}: logical output/segment count mismatch"
                )
            logical_segments.update(zip(outputs, segments))
            return
        segment = node.meta.get(storage_key)
        if len(outputs) == 1 and segment is not None:
            logical_segments[outputs[0]] = segment

    # Inputs and parameters retain DRAM backing addresses. In end-to-end mode,
    # compute intermediates are intentionally scratchpad-only.
    if end_to_end_scratchpad:
        for node in execution_nodes:
            if node.op == "get_attr" and require_allocation(
                node,
                operator_traits,
            ):
                node.meta["memory"] = allocator.allocate_memory(node)
            elif node.op == "placeholder":
                if not has_logical_users(node):
                    continue
                node.meta["memory"] = allocator.allocate_memory(node)
        allocator.snapshot()

    # Parameters remain persistent in the selected execution storage.
    for node in execution_nodes:
        if node.op == "get_attr" and require_allocation(
            node,
            operator_traits,
        ):
            node.meta[storage_key] = storage_allocator.allocate_memory(
                node,
                get_allocation_role(node, operator_traits),
            )
            publish_node_segments(node)

    # Store model inputs in the selected execution storage.
    for node in execution_nodes:
        if node.op == "placeholder":
            if not has_logical_users(node):
                continue
            node.meta[storage_key] = storage_allocator.allocate_memory(
                node,
                TensorRole.ACTIVATION,
            )
            publish_node_segments(node)

    storage_allocator.snapshot()

    def get_unused_values(user: Node):
        """Return physical tensors whose complete alias class is dead."""
        return tuple(
            storage_plan.storage_key(physical, storage_key)
            for physical in storage_plan.release_after(
                user,
                storage_key,
            )
        )

    def release_unused_values(user: Node) -> None:
        for storage in get_unused_values(user):
            storage_allocator.free_memory(storage)

    def get_storage_tensor_size(node: Node):
        target_size = getattr(
            storage_allocator,
            "tensor_size_bytes",
            None,
        )
        if target_size is not None:
            size = target_size(node)
            if size is not None:
                return size
        return node.value.numel() * get_node_bytes(node)

    def storage_relation(node: Node):
        try:
            ref = storage_plan.ref(node)
        except (KeyError, ValueError):
            return None
        return storage_plan.relations[storage_key][ref]

    def allocation_identities(node: Node):
        """Return the exact physical results allocated for one FX node.

        Tensor-valued nodes have a root logical ``ValueRef`` and may be
        passed through the allocator normally. Tuple/list-producing nodes are
        only logical containers: each produced physical root is allocated
        independently unless storage analysis deliberately combines the
        results in one container.
        """
        produced_outputs = storage_plan.produced_physical_outputs(
            node,
            storage_key,
        )
        if produced_outputs:
            return produced_outputs
        raise ValueError(
            f"{node.name}: has no physical tensor output produced by "
            "this operation"
        )

    def physical_output_size(physical):
        """Return packed bytes for one exact multi-output physical root."""
        return storage_plan.backing_size_bytes(
            physical,
            storage_key,
        )

    def lower_analyzed_view(node: Node):
        """Lower one shared storage relation without reparsing FX syntax."""
        relation = storage_relation(node)
        if relation is None or relation.kind not in {
            StorageRelationKind.ALIAS,
            StorageRelationKind.VIEW,
            StorageRelationKind.WRITE_INTO,
            StorageRelationKind.COMPONENT,
        }:
            return None
        if (
            relation.transform is not None
            and relation.transform.kind == "aggregate_slot"
        ):
            return None

        source_segment = (
            logical_segments.get(relation.source)
            if relation.source is not None
            else None
        )
        if source_segment is None:
            physical = storage_plan.physical_tensor(
                node,
                storage_key,
            )
            source_segment = storage_allocator.allocation_segment(
                physical.id,
            )
        if source_segment is None:
            physical = storage_plan.physical_tensor(
                node,
                storage_key,
            )
            raise RuntimeError(
                f"{node.name}: analyzed physical tensor "
                f"{physical.id.name} has no live {storage_key} allocation"
            )

        transform = relation.transform
        if (
            relation.kind is StorageRelationKind.VIEW
            and transform is not None
            and transform.kind == "contiguous_block"
        ):
            block_index = transform.parameter("block_index")
            if (
                isinstance(block_index, bool)
                or not isinstance(block_index, int)
                or block_index < 0
            ):
                raise ValueError(
                    f"{node.name}: invalid contiguous-block index "
                    f"{block_index!r}"
                )
            size = get_storage_tensor_size(node)
            return storage_allocator.create_view(
                source_segment,
                block_index * size,
                size,
            )
        return copy.deepcopy(source_segment)

    def bind_multi_output_segments(node, allocated):
        """Publish one segment per logical result for getitem/serialization."""
        outputs = storage_plan.output_refs(node)
        if not outputs:
            return
        segments = []
        for output in outputs:
            physical = storage_plan.storage_key(output, storage_key)
            segment = allocated.get(physical)
            if segment is None:
                segment = physical.owner.node.meta.get(storage_key)
            relation = storage_plan.relations[storage_key][output]
            if (
                segment is not None
                and relation.transform is not None
                and relation.transform.kind == "tuple_output"
            ):
                result_index = relation.transform.parameter("result_index")
                output_sizes = node.meta["output_sizes"]
                segment = storage_allocator.create_view(
                    segment,
                    sum(output_sizes[:result_index]),
                    output_sizes[result_index],
                )
            segments.append(segment)
            if segment is not None:
                logical_segments[output] = segment
        node.meta[f"{storage_key}_output_segments"] = tuple(segments)
        node.meta[storage_key] = next(
            (segment for segment in segments if segment is not None),
            None,
        )

    def allocate_aggregate_view(node: Node):
        """Lower one analyzed direct-write relation to a byte view."""
        relation = storage_relation(node)
        if (
            relation is None
            or relation.kind != StorageRelationKind.WRITE_INTO
            or relation.source is None
            or relation.transform is None
            or relation.transform.kind != "aggregate_slot"
        ):
            return node.meta.get(storage_key)

        aggregate = relation.source.node
        storage = aggregate.meta.get(storage_key)
        if storage is None:
            storage = storage_allocator.allocate_memory(
                aggregate,
                TensorRole.OUTPUT,
            )
            aggregate.meta[storage_key] = storage
            publish_node_segments(aggregate)

        slot = relation.transform.parameter("slot")
        tensors = aggregate.args[0]
        tensor_sizes = [get_storage_tensor_size(tensor) for tensor in tensors]
        byte_offset = sum(tensor_sizes[:slot])
        size = tensor_sizes[slot]
        try:
            segment = storage_allocator.create_view(
                storage,
                byte_offset,
                size,
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot place {node.name} as input {slot} of "
                f"{aggregate.name}: {error}"
            ) from error

        node.meta[storage_key] = segment
        return segment

    for node in execution_nodes:
        if node.op not in ["call_function", "call_module"]:
            continue

        if storage_key in node.meta:
            output_segments_key = f"{storage_key}_output_segments"
            if (
                not storage_plan.has_logical_value(node)
                and output_segments_key not in node.meta
            ):
                produced = allocation_identities(node)
                bind_multi_output_segments(
                    node,
                    {physical: node.meta[storage_key] for physical in produced},
                )
            publish_node_segments(node)
            release_unused_values(node)
            continue

        skip_allocation = False

        create_split_views = getattr(
            storage_allocator,
            "create_split_views",
            None,
        )
        if (
            create_split_views is not None
            and node.args
            and isinstance(node.args[0], Node)
        ):
            input_storage = node.args[0].meta.get(storage_key)
            output_segments = create_split_views(node, input_storage)
            if output_segments is not None:
                node.meta[storage_key] = output_segments[0]
                node.meta[f"{storage_key}_output_segments"] = output_segments
                publish_node_segments(node)
                skip_allocation = True

        # Instruction emission and storage identity are independent.  Only
        # propagate storage for relations that explicitly share a backing.
        if (
            is_nop(node)
            or node.meta.get(TARGET_INSTRUCTION_NOP, False)
            or node.target == operator.getitem
        ):
            segment = lower_analyzed_view(node)
            if segment is not None:
                node.meta[storage_key] = segment
                publish_node_segments(node)
                skip_allocation = True

        # We use the partition of the first input tensor since it preallocates
        # memory for all the tensors in the stack operation
        if node.target in [
            torch.ops.aten.stack.default,
            torch.ops.aten.cat.default,
        ]:
            concat_dim = (
                node.args[1]
                if len(node.args) > 1
                else node.kwargs.get("dim", 0)
            )
            if concat_dim == 0 and node.meta.get(storage_key) is not None:
                continue

        allocate_aggregate_view(node)

        if skip_allocation:
            publish_node_segments(node)
            release_unused_values(node)
            continue

        if node.meta.get(storage_key) is None:
            physical_tensors = allocation_identities(node)
            if len(physical_tensors) == 1:
                physical_tensor = physical_tensors[0]
                node.meta[storage_key] = storage_allocator.allocate_memory(
                    node,
                    TensorRole.OUTPUT,
                    physical_tensor=physical_tensor,
                )
                if not storage_plan.has_logical_value(node):
                    bind_multi_output_segments(
                        node,
                        {
                            physical_tensor: node.meta[storage_key],
                        },
                    )
            else:
                allocated = {
                    physical: storage_allocator.allocate_memory(
                        node,
                        TensorRole.OUTPUT,
                        physical_tensor=physical,
                        tensor_size=physical_output_size(physical),
                    )
                    for physical in physical_tensors
                }
                bind_multi_output_segments(node, allocated)
            storage_allocator.snapshot()

        publish_node_segments(node)
        release_unused_values(node)

    graph.lint()
    model.recompile()

    validate_plan = getattr(storage_allocator, "validate_plan", None)
    allocation_history = None
    if validate_plan is not None:
        allocation_history = validate_plan(model)
    if not end_to_end_scratchpad:
        validate_target_operation_scopes(
            target,
            target_storage_allocator,
            model,
        )
    if dp_result is not None and allocation_history is not None:
        allocated_physicals = {
            record.physical_tensor
            for record in allocation_history
            if getattr(record, "physical_tensor", None) is not None
        }
        expected_physicals = set(resident_physical_ids)
        if allocated_physicals != expected_physicals:
            missing = expected_physicals - allocated_physicals
            unexpected = allocated_physicals - expected_physicals
            details = []
            if missing:
                details.append(
                    "missing="
                    + ", ".join(sorted(physical.name for physical in missing))
                )
            if unexpected:
                details.append(
                    "unexpected="
                    + ", ".join(
                        sorted(physical.name for physical in unexpected)
                    )
                )
            raise DPExecutionOrderError(
                "Allocator residency disagrees with the DP objective: "
                + "; ".join(details)
            )

    model.meta["storage_plan"] = storage_plan
    try:
        prepare_target_serialization_metadata(
            model,
            target,
            target_storage_allocator,
            hardware_config,
        )
    except Exception:
        model.meta.pop("storage_plan", None)
        model.meta.pop(EXECUTION_ORDER_RESULT_META, None)
        raise
    if dp_result is not None:
        model.meta[EXECUTION_ORDER_RESULT_META] = dp_result


def _get_fused_codegen_owner(module, executable_nodes):
    """Return the serialized node that owns the fused output write."""
    output = next(node for node in module.graph.nodes if node.op == "output")
    output_nodes = output.all_input_nodes
    if len(output_nodes) == 1 and output_nodes[0] in executable_nodes:
        return output_nodes[0]
    if len(executable_nodes) == 1:
        return executable_nodes[0]
    return None


def _map_fused_node(node, target_kwargs, output_dir):
    previous = node.meta.get("target_codegen_kwargs")
    if target_kwargs:
        merged = dict(previous or {})
        overlap = set(merged).intersection(target_kwargs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"{node.name}: duplicate target codegen kwargs: {names}"
            )
        merged.update(target_kwargs)
        node.meta["target_codegen_kwargs"] = merged
    try:
        return map_node(
            node,
            target_output_dir=output_dir,
        )
    finally:
        if previous is None:
            node.meta.pop("target_codegen_kwargs", None)
        else:
            node.meta["target_codegen_kwargs"] = previous


def gen_code(model, args, output_dir=None):
    if output_dir is not None:
        prepare_output_directory(output_dir)

    named_modules = dict(model.named_modules(remove_duplicate=False))

    ShapeProp(model).propagate(*args)
    target = model.meta.get(
        COMPILATION_TARGET_META,
        DEFAULT_COMPILATION_TARGET,
    )
    validate_target_serialization_metadata(model, target)
    model_params = Model()

    for node in execution_order_for_model(model).nodes:
        if not isinstance(node.value, (torch.Tensor, tuple, list)):
            continue

        op = Operation()

        if node.op == "placeholder":
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            model_params.inputs.append(tensor)
            continue
        elif node.op == "get_attr":
            tensor = Tensor()
            set_tensor_field(tensor, node, output_dir)
            if (
                node.meta.get("memory") is not None
                or node.meta.get("scratchpad") is not None
            ):
                model_params.parameters.append(tensor)
            continue
        elif node.op == "call_function":
            op.op.CopyFrom(
                map_node(
                    node,
                    target_output_dir=output_dir,
                )
            )
            if is_declared_unsupported_operation(node):
                op.op.op = SERIALIZED_UNSUPPORTED_OPERATION
        elif node.op == "call_module":
            gm = named_modules[node.target]
            assert isinstance(gm, torch.fx.GraphModule)

            if tiled_shapes := node.meta.get("tiled_shapes"):
                args = map_arg(
                    node.args,
                    lambda n: get_tiled_tensor(n, tiled_shapes),
                )
                ShapeProp(gm).propagate(*args)

                for n in gm.graph.nodes:
                    if hasattr(n, "shape"):
                        tiled_shapes.setdefault(n, n.shape)

            args = map_arg(node.args, lambda n: n.value.clone())
            ShapeProp(gm).propagate(*args)

            executable_nodes = [
                inner
                for inner in gm.graph.nodes
                if inner.op == "call_function"
                and not inner.meta.get("fused", False)
                and not is_nop(inner)
            ]
            target_kwargs = node.meta.get("target_codegen_kwargs", {})
            codegen_owner = _get_fused_codegen_owner(
                gm,
                executable_nodes,
            )
            if target_kwargs and codegen_owner is None:
                raise ValueError(
                    f"{node.name}: target codegen kwargs have no unique "
                    "fused output owner"
                )

            operators = []
            for n in executable_nodes:
                n.meta["tiled_shapes"] = tiled_shapes
                n.meta["tile_strides"] = node.meta.get("tile_strides")
                n.meta["scratchpad_map"] = node.meta.get("scratchpad_map")
                physical_scratchpad_map = node.meta.get(
                    "physical_scratchpad_map"
                )
                if physical_scratchpad_map is None:
                    n.meta.pop("physical_scratchpad_map", None)
                else:
                    n.meta["physical_scratchpad_map"] = physical_scratchpad_map

                outer_kwargs = target_kwargs if n is codegen_owner else {}
                operators.append(_map_fused_node(n, outer_kwargs, output_dir))

            if is_declared_unsupported_operation(node):
                for operator_record in operators:
                    operator_record.op = SERIALIZED_UNSUPPORTED_OPERATION

            op.fused_op.name = node.name
            op.fused_op.op_list.extend(operators)
        else:
            continue

        set_output_field(op, node, output_dir)

        if "local_schedule" in node.meta:
            op.tiling.CopyFrom(build_tiling_proto(node))

        model_params.ops.append(op)

    return model_params


def _wrap_long_name(name: str, max_len: int = 20) -> str:
    if len(name) <= max_len:
        return name
    lines = []
    start = 0
    while start + max_len < len(name):
        end = start + max_len
        break_at = end
        for i in range(end, start, -1):
            if not name[i].isalpha():
                break_at = i + 1
                break
        lines.append(name[start:break_at])
        start = break_at
    lines.append(name[start:])
    return "&#92;n".join(lines)


def gen_compute_graph(model, output_file="compute_graph", max_users=10):
    nodes = {}
    edges = []
    named_modules = dict(model.named_modules(remove_duplicate=False))

    for node in model.graph.nodes:
        if node.op == "get_attr" and "qmap" in node.name:
            continue

        header = _wrap_long_name(node.name)

        if isinstance(node.value, torch.Tensor):
            header += f"&#92;n{str(tuple(node.value.shape))}"
            if (dtype := node.meta.get("dtype", None)) is not None:
                header += f"&#92;n{dtype}"
            elif node.value.dtype not in [torch.float, torch.bfloat16]:
                header += f"&#92;n{node.value.dtype}"
        elif isinstance(node.value, (tuple, list)):
            shape_str = ", ".join([str(tuple(t.shape)) for t in node.value])
            header += f"&#92;n{shape_str}"

            dtypes = [t.dtype for t in node.value]
            if (dtype := node.meta.get("dtype", None)) is not None:
                dtypes = [dt or dtypes[i] for i, dt in enumerate(dtype)]

            if any(
                dtype not in [torch.float, torch.bfloat16] for dtype in dtypes
            ):
                header += f"&#92;n{', '.join([str(d) for d in dtypes])}"
        else:
            continue

        body = None
        if node.op == "call_module":
            gm = named_modules[node.target]
            if isinstance(gm, torch.fx.GraphModule):
                body = "&#92;n".join(
                    [n.name for n in gm.graph.nodes if n.op == "call_function"]
                )
        label = f"{{{header}}}" if body is None else f"{{{header}|{body}}}"
        label = label.replace("<", r"\<").replace(">", r"\>")

        nodes[node.name] = {
            "label": label,
            "shape": "Mrecord",
        }

        users = list(node.users)
        num_users = len(users)
        if num_users > max_users:
            num_splits = (num_users + max_users - 1) // max_users
            for i in range(num_splits):
                sub_node = f"{node.name}_split_{i}"
                sub_label = f"{{{sub_node}}}"
                sub_label = sub_label.replace("<", r"\<").replace(">", r"\>")

                # Create a sub-node for this group of users
                nodes[sub_node] = {
                    "label": sub_label,
                    "shape": "Mrecord",
                }

                edges.append((node.name, sub_node))

                # Add edges from sub-node to its users
                start_idx = i * max_users
                end_idx = min(start_idx + max_users, num_users)
                for u in users[start_idx:end_idx]:
                    edges.append((sub_node, u.name))
        else:
            for u in users:
                edges.append((node.name, u.name))

    g = graphviz.Digraph()
    # g.attr(bgcolor="transparent")

    for node, attrs in nodes.items():
        g.node(node, **attrs)

    g.edges(edges)

    g.render(output_file, format="svg", cleanup=True)
