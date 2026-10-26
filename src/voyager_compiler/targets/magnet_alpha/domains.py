# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Prepared PT2E quantization-region construction and finalization."""

import copy
import logging
import operator
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from functools import lru_cache

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.compiler.analysis import is_address_only_view
from voyager_compiler.ir.graph import (
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_INPUT_REGION,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainRuleSet,
    PreparedDomainTransferRule,
    PreparedQuantizationContract,
    QuantizationContractTransferGraph,
    set_physical_quantization_contract,
)
from voyager_compiler.targets.magnet_alpha.fx import (
    get_arg_value,
    is_gemm_op,
    new_attr_name,
)
from voyager_compiler.targets.magnet_alpha.observers import (
    FusedAmaxObsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
)
from voyager_compiler.targets.magnet_alpha.qspec import (
    DerivedQuantizationSpec,
    QScheme,
    QuantizationSpec,
)

logger = logging.getLogger(__name__)


_QUANTIZATION_DOMAIN_ENTRY = "quantization_domain_entry"
_QUANTIZATION_DOMAIN_ALIAS = "quantization_domain_alias"
_VSQ_JOIN_ENTRY = "vsq_join_entry"
_VSQ_JOIN_ALIAS = "vsq_join_alias"
_UNKNOWN_PREPARED_DOMAIN = object()
_CONFLICTING_PREPARED_DOMAIN = object()

# Max-pool and nearest replication preserve per-tensor quantized values.
_PREPARED_DOMAIN_PASSTHROUGH_TARGETS = {
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.upsample_nearest2d.vec,
    torch.ops.aten.view.default,
}
_PREPARED_DOMAIN_REGION_ROOT_TARGETS = {
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.upsample_nearest2d.vec,
}
_PREPARED_DOMAIN_JOIN_TARGETS = {
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
}
_PREPARED_DOMAIN_TARGET_RULE = "prepared_domain_target_rule"


def _create_obs_or_fq_from_qspec(
    quantization_spec,
    obs_or_fq_map,
    is_qat,
):
    """Create an observer or fake quantizer from one quantization spec."""
    if quantization_spec is None:
        return None
    if isinstance(quantization_spec, DerivedQuantizationSpec):
        kwargs = {
            "dtype": quantization_spec.dtype,
            "derive_qparams_fn": quantization_spec.derive_qparams_fn,
        }
        edge_or_nodes = quantization_spec.derived_from
        obs_or_fqs = [obs_or_fq_map[key] for key in edge_or_nodes]
        kwargs["obs_or_fqs"] = obs_or_fqs
        return _DerivedObserverOrFakeQuantize.with_args(**kwargs)()

    assert isinstance(quantization_spec, QuantizationSpec)
    observer_or_fake_quant_ctr = quantization_spec.observer_or_fake_quant_ctr
    kwargs = copy.deepcopy(asdict(quantization_spec))
    kwargs.pop("observer_or_fake_quant_ctr")
    return observer_or_fake_quant_ctr.with_args(**kwargs)()


def _is_chunk_getitem(node: Node) -> bool:
    return (
        node.target == operator.getitem
        and isinstance(node.args[0], Node)
        and node.args[0].target == torch.ops.aten.chunk.default
        and isinstance(node.args[1], int)
    )


def _is_zero_constant_pad(node: Node) -> bool:
    if node.target != torch.ops.aten.pad.default:
        return False
    mode = get_arg_value(node, 2, "mode", "constant")
    value = get_arg_value(node, 3, "value", None)
    return mode == "constant" and (value is None or value == 0)


@lru_cache(maxsize=1)
def _prepared_domain_rule_set() -> PreparedDomainRuleSet:
    rules = [
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.ALIAS,
            region_root=(target in _PREPARED_DOMAIN_REGION_ROOT_TARGETS),
            forcing_root=(target in _PREPARED_DOMAIN_REGION_ROOT_TARGETS),
        )
        for target in _PREPARED_DOMAIN_PASSTHROUGH_TARGETS
    ]
    rules.extend(
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.SEQUENCE,
            relation=PreparedDomainRelation.JOIN,
            region_root=True,
        )
        for target in _PREPARED_DOMAIN_JOIN_TARGETS
    )
    rules.extend(
        (
            PreparedDomainTransferRule(
                target=torch.ops.aten.chunk.default,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
            ),
            PreparedDomainTransferRule(
                target=operator.getitem,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
                predicate=_is_chunk_getitem,
            ),
            PreparedDomainTransferRule(
                target=torch.ops.aten.pad.default,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.MAP_COMPONENTS,
                region_root=True,
                forcing_root=True,
                predicate=_is_zero_constant_pad,
            ),
        )
    )
    return PreparedDomainRuleSet(tuple(rules))


def _selected_prepared_domain_rules(
    operator_traits=None,
) -> tuple[PreparedDomainTransferRule, ...]:
    if operator_traits is not None:
        return tuple(operator_traits.prepared_domain_transfer_rules)
    # This module belongs to Alpha. Keep its domain semantics with its
    # numerical policy rather than relying on the old backend trait registry.
    from .graph import is_channel_shuffle_cat
    from .operators import CHANNEL_SHUFFLE_OPERATOR

    return (
        *(
            PreparedDomainTransferRule(
                target=t,
                input_format=PreparedDomainInputFormat.PREFIX,
                input_count=2,
                relation=PreparedDomainRelation.INPUT_JOIN,
                region_root=True,
                forcing_root=True,
            )
            for t in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
        ),
        *(
            PreparedDomainTransferRule(
                target=t,
                input_format=PreparedDomainInputFormat.FIRST,
                relation=PreparedDomainRelation.ALIAS,
            )
            for t in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
        ),
        PreparedDomainTransferRule(
            target=torch.ops.aten.cat.default,
            input_format=PreparedDomainInputFormat.SEQUENCE,
            relation=PreparedDomainRelation.JOIN,
            region_root=True,
            forcing_root=True,
            predicate=is_channel_shuffle_cat,
        ),
        PreparedDomainTransferRule(
            target=CHANNEL_SHUFFLE_OPERATOR,
            input_format=PreparedDomainInputFormat.PREFIX,
            input_count=2,
            relation=PreparedDomainRelation.JOIN,
            region_root=True,
            forcing_root=True,
        ),
    )


def _selected_quantization_output_epilogues(
    operator_traits=None,
) -> tuple:
    """Return only output epilogues declared by the selected target."""
    return (
        (
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
            torch.ops.aten.silu.default,
        )
        if operator_traits is None
        else tuple(operator_traits.quantization_output_epilogue_operators)
    )


def _effective_prepared_domain_rule_set(
    operator_traits=None,
) -> PreparedDomainRuleSet:
    """Compose selected-target rules ahead of shared semantic rules."""
    return PreparedDomainRuleSet(
        (
            *_selected_prepared_domain_rules(operator_traits),
            *_prepared_domain_rule_set().rules,
        )
    )


def _prepared_domain_data_inputs(node: Node) -> tuple[Node, ...]:
    target_rule = node.meta.get(_PREPARED_DOMAIN_TARGET_RULE)
    if target_rule is not None:
        return target_rule.inputs(node)
    return _prepared_domain_rule_set().inputs(node)


def _bind_prepared_domain_target_rules(
    model: GraphModule,
    operator_traits=None,
) -> None:
    """Bind only the selected target's complete transfer rules."""
    rule_set = PreparedDomainRuleSet(
        _selected_prepared_domain_rules(operator_traits)
    )
    for node in model.graph.nodes:
        node.meta.pop(_PREPARED_DOMAIN_TARGET_RULE, None)
        resolved = rule_set.resolve(node)
        if resolved is not None:
            node.meta[_PREPARED_DOMAIN_TARGET_RULE] = resolved.rule


def _get_module(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
) -> torch.nn.Module | None:
    """Return the module called by ``node``, when one exists."""
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    return None


def _get_prepared_per_tensor_observer(
    node: Node,
    modules: dict[str, torch.nn.Module],
) -> FusedAmaxObsFakeQuantize | None:
    if node.op != "call_module" or not isinstance(node.target, str):
        return None
    module = modules.get(node.target)
    if not isinstance(module, FusedAmaxObsFakeQuantize):
        return None
    if module.qscheme != QScheme.PER_TENSOR_SYMMETRIC:
        return None
    if module.block_size is not None or module.ch_axis is not None:
        return None
    return module


def _get_prepared_vsq_observer(
    node: Node,
    modules: dict[str, torch.nn.Module],
) -> FusedAmaxObsFakeQuantize | None:
    if node.op != "call_module" or not isinstance(node.target, str):
        return None
    module = modules.get(node.target)
    if not isinstance(module, FusedAmaxObsFakeQuantize):
        return None
    if module.qscheme != QScheme.VSQ:
        return None
    if module.coarse_qscheme != QScheme.PER_TENSOR_SYMMETRIC:
        return None
    if module.coarse_ch_axis is not None:
        return None
    return module


def _insert_prepared_observer(
    model: GraphModule,
    modules: dict[str, torch.nn.Module],
    value: Node,
    consumer: Node,
    observer: FusedAmaxObsFakeQuantize,
    prefix: str,
) -> Node:
    graph = model.graph
    observer_name = new_attr_name(model, prefix)
    model.add_module(observer_name, observer)
    modules[observer_name] = observer
    with graph.inserting_before(consumer):
        observer_node = graph.call_module(observer_name, (value,))
    observer_node.meta = copy.copy(value.meta)
    consumer.replace_input_with(value, observer_node)
    return observer_node


def _prepared_domains_compatible(
    left: FusedAmaxObsFakeQuantize,
    right: FusedAmaxObsFakeQuantize,
) -> bool:
    return PreparedQuantizationContract.from_observer(
        left
    ) == PreparedQuantizationContract.from_observer(right)


def _merge_prepared_quantization_plans(plans):
    """Merge compatible regions that share encoded storage frontiers."""
    parents = list(range(len(plans)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(plans):
        left_domain, left_encoded, left_aliases, left_frontiers = left
        for right_index in range(left_index + 1, len(plans)):
            right = plans[right_index]
            (
                right_domain,
                right_encoded,
                right_aliases,
                right_frontiers,
            ) = right
            if not _prepared_domains_compatible(
                left_domain,
                right_domain,
            ):
                continue
            if not (
                left_encoded & right_encoded
                or left_aliases & right_aliases
                or left_frontiers.keys() & right_frontiers.keys()
            ):
                continue
            union(left_index, right_index)

    grouped = OrderedDict()
    for index, plan in enumerate(plans):
        root = find(index)
        if root not in grouped:
            domain, _, _, _ = plans[root]
            grouped[root] = (
                domain,
                set(),
                set(),
                defaultdict(set),
            )
        _, encoded, aliases, frontiers = plan
        _, merged_encoded, merged_aliases, merged_frontiers = grouped[root]
        merged_encoded.update(encoded)
        merged_aliases.update(aliases)
        for value, consumers in frontiers.items():
            merged_frontiers[value].update(consumers)

    merged = []
    for domain, encoded, aliases, frontiers in grouped.values():
        normalized_frontiers = defaultdict(set)
        for value, consumers in frontiers.items():
            if value in encoded:
                continue
            normalized_frontiers[value].update(
                consumer for consumer in consumers if consumer in encoded
            )
        merged.append(
            (
                domain,
                encoded,
                aliases,
                normalized_frontiers,
            )
        )
    return merged


def _analyze_prepared_quantization_regions(
    model: GraphModule,
    domain_graph: QuantizationContractTransferGraph,
):
    modules = dict(model.named_modules(remove_duplicate=False))
    memo = {}
    active = set()

    def join_domains(domains):
        if not domains:
            return _UNKNOWN_PREPARED_DOMAIN
        if any(domain is _CONFLICTING_PREPARED_DOMAIN for domain in domains):
            return _CONFLICTING_PREPARED_DOMAIN
        if any(domain is _UNKNOWN_PREPARED_DOMAIN for domain in domains):
            return _UNKNOWN_PREPARED_DOMAIN

        reference = domains[0]
        reference_module = _get_prepared_per_tensor_observer(
            reference,
            modules,
        )
        if reference_module is None:
            return _CONFLICTING_PREPARED_DOMAIN
        for domain in domains[1:]:
            module = _get_prepared_per_tensor_observer(domain, modules)
            if module is None or not _prepared_domains_compatible(
                reference_module,
                module,
            ):
                return _CONFLICTING_PREPARED_DOMAIN
        return reference

    def required_domain(node):
        if node in memo:
            return memo[node]
        if node in active:
            return _CONFLICTING_PREPARED_DOMAIN

        active.add(node)
        domains = []
        for user in node.users:
            observer = _get_prepared_per_tensor_observer(user, modules)
            if observer is not None and user.args[0] is node:
                domain = user
            elif domain_graph.is_edge(node, user):
                domain = required_domain(user)
            else:
                domain = _UNKNOWN_PREPARED_DOMAIN
            domains.append(domain)

        active.remove(node)
        memo[node] = join_domains(domains)
        return memo[node]

    def collect_region(node, domain, encoded, observers):
        if node in encoded:
            return
        encoded.add(node)
        domain_module = _get_prepared_per_tensor_observer(
            domain,
            modules,
        )
        assert domain_module is not None

        for user in node.users:
            observer = _get_prepared_per_tensor_observer(user, modules)
            if observer is not None:
                if user.args[0] is node and _prepared_domains_compatible(
                    domain_module,
                    observer,
                ):
                    observers.add(user)
                continue
            if not domain_graph.is_edge(node, user):
                continue

            user_domain = required_domain(user)
            user_module = (
                _get_prepared_per_tensor_observer(
                    user_domain,
                    modules,
                )
                if isinstance(user_domain, Node)
                else None
            )
            if user_module is not None and _prepared_domains_compatible(
                domain_module,
                user_module,
            ):
                collect_region(
                    user,
                    domain,
                    encoded,
                    observers,
                )

    seen_nodes = set()
    plans = []
    for root in model.graph.nodes:
        if not domain_graph.is_region_root(root):
            continue
        if root in seen_nodes:
            continue

        domain = required_domain(root)
        if not isinstance(domain, Node):
            continue

        encoded = set()
        observers = set()
        collect_region(root, domain, encoded, observers)
        seen_nodes.update(encoded)

        frontiers = domain_graph.frontiers(encoded)

        if frontiers and observers:
            plans.append((domain, encoded, observers, frontiers))

    return plans


def _find_prepared_data_movement_regions(
    model: GraphModule,
    domain_graph: QuantizationContractTransferGraph,
):
    seen = set()
    regions = []

    for root in model.graph.nodes:
        if not domain_graph.is_region_root(root):
            continue
        if root in seen:
            continue

        encoded = set(domain_graph.connected_component(root))

        seen.update(encoded)
        frontiers = domain_graph.frontiers(encoded)
        exits = domain_graph.exits(encoded)

        if frontiers and exits:
            has_forcing_root = any(
                domain_graph.is_forcing_root(node) for node in encoded
            )
            regions.append(
                (
                    encoded,
                    frontiers,
                    exits,
                    has_forcing_root,
                )
            )

    return regions


def _prepared_region_domain_template(
    region_qspec: QuantizationSpec,
) -> FusedAmaxObsFakeQuantize:
    domain_template = _create_obs_or_fq_from_qspec(
        region_qspec,
        {},
        False,
    )
    if not isinstance(domain_template, FusedAmaxObsFakeQuantize):
        raise TypeError("Encoded quantization regions require an amax observer")
    if domain_template.qscheme != QScheme.PER_TENSOR_SYMMETRIC:
        raise ValueError(
            "Encoded quantization regions require per-tensor quantization"
        )
    return domain_template


def _fresh_prepared_region_observer(
    domain_template: FusedAmaxObsFakeQuantize,
) -> FusedAmaxObsFakeQuantize:
    observer = copy.deepcopy(domain_template)
    observer.enable_fake_quant()
    observer.enable_observer()
    observer.amax_history.resize_(0)
    observer.scale.resize_(1).fill_(1.0)
    return observer


def _prepare_input_join_regions(
    model: GraphModule,
    region_qspec: QuantizationSpec | None,
    operator_traits,
    *,
    first_region_id: int,
) -> int:
    """Give selected inputs one domain without aliasing the node result."""
    rule_set = PreparedDomainRuleSet(
        _selected_prepared_domain_rules(operator_traits)
    )
    joins = tuple(
        (node, resolved)
        for node in model.graph.nodes
        if (resolved := rule_set.input_join(node)) is not None
    )
    if not joins:
        return 0

    modules = dict(model.named_modules(remove_duplicate=False))
    configured_template = (
        None
        if region_qspec is None
        else _prepared_region_domain_template(region_qspec)
    )

    for offset, (node, resolved) in enumerate(joins):
        if len(resolved.inputs) < 2:
            raise ValueError(
                f"{node.name}: input join requires at least two inputs"
            )

        input_modules = tuple(
            _get_prepared_per_tensor_observer(value, modules)
            for value in resolved.inputs
        )
        if configured_template is None:
            if any(module is None for module in input_modules):
                raise ValueError(
                    f"{node.name}: input join requires prepared "
                    "per-tensor observers"
                )
            domain_template = input_modules[0]
            assert domain_template is not None
            if any(
                not _prepared_domains_compatible(domain_template, module)
                for module in input_modules[1:]
            ):
                raise ValueError(
                    f"{node.name}: input join observers are incompatible"
                )
        else:
            domain_template = configured_template

        region_id = first_region_id + offset
        domain_dtype = domain_template.dtype
        physical_contract = PhysicalQuantizationContract(domain_dtype)
        localized = {}
        members = set()

        for input_index, value in enumerate(resolved.inputs):
            if value in localized:
                members.add(localized[value])
                continue

            observer = _get_prepared_per_tensor_observer(value, modules)
            reusable = (
                observer is not None
                and _prepared_domains_compatible(domain_template, observer)
                and set(value.users) == {node}
                and QUANTIZATION_REGION not in value.meta
            )
            if reusable:
                entry = value
            else:
                entry = _insert_prepared_observer(
                    model,
                    modules,
                    value,
                    node,
                    _fresh_prepared_region_observer(domain_template),
                    f"{node.name}_input_{input_index}_domain",
                )
                for key in (
                    QUANTIZATION_REGION,
                    _QUANTIZATION_DOMAIN_ALIAS,
                    _QUANTIZATION_DOMAIN_ENTRY,
                    _VSQ_JOIN_ALIAS,
                    _VSQ_JOIN_ENTRY,
                ):
                    entry.meta.pop(key, None)

            localized[value] = entry
            members.add(entry)

        if len(members) < 2 and len(set(resolved.inputs)) > 1:
            raise ValueError(
                f"{node.name}: input join did not materialize all inputs"
            )

        node.meta[QUANTIZATION_INPUT_REGION] = region_id
        for entry in members:
            entry.meta[QUANTIZATION_REGION] = region_id
            entry.meta[QUANTIZATION_DOMAIN_DTYPE] = domain_dtype
            entry.meta[_QUANTIZATION_DOMAIN_ENTRY] = True
            set_physical_quantization_contract(entry, physical_contract)

    return len(joins)


def _prepare_quantization_regions(
    model: GraphModule,
    region_qspec: QuantizationSpec | None = None,
    operator_traits=None,
) -> GraphModule:
    _bind_prepared_domain_target_rules(model, operator_traits)
    graph = model.graph
    domain_graph = QuantizationContractTransferGraph(
        graph.nodes,
        _effective_prepared_domain_rule_set(operator_traits),
    )
    modules = dict(model.named_modules(remove_duplicate=False))
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    plans = []

    def compatible_observer_users(value, domain_module):
        observers = set()
        for user in value.users:
            observer = _get_prepared_per_tensor_observer(user, modules)
            if (
                observer is not None
                and user.args[0] is value
                and _prepared_domains_compatible(
                    domain_module,
                    observer,
                )
            ):
                observers.add(user)
        return observers

    if region_qspec is None:
        for (
            domain,
            encoded,
            observers,
            frontiers,
        ) in _analyze_prepared_quantization_regions(
            model,
            domain_graph,
        ):
            domain_module = _get_prepared_per_tensor_observer(
                domain,
                modules,
            )
            assert domain_module is not None
            plans.append(
                (
                    domain_module,
                    encoded,
                    observers,
                    frontiers,
                )
            )
    else:
        domain_template = _prepared_region_domain_template(region_qspec)

        for (
            encoded,
            frontiers,
            exits,
            has_forcing_root,
        ) in _find_prepared_data_movement_regions(
            model,
            domain_graph,
        ):
            compatible_exits = [
                _get_prepared_per_tensor_observer(user, modules)
                for _, user in exits
            ]
            if not has_forcing_root and not all(
                observer is not None
                and _prepared_domains_compatible(
                    domain_template,
                    observer,
                )
                for observer in compatible_exits
            ):
                # Cat/stack may join an existing per-tensor domain, but
                # must not replace a VSQ/O4 boundary with the non-GEMM
                # INT8 domain.
                continue

            observers = set()
            for (value, user), existing in zip(
                exits,
                compatible_exits,
            ):
                if existing is not None and _prepared_domains_compatible(
                    domain_template,
                    existing,
                ):
                    observers.add(user)
                    continue

                observer = _fresh_prepared_region_observer(domain_template)
                observer_node = _insert_prepared_observer(
                    model,
                    modules,
                    value,
                    user,
                    observer,
                    f"{value.name}_quantized_output",
                )
                observers.add(observer_node)
            plans.append(
                (
                    domain_template,
                    encoded,
                    observers,
                    frontiers,
                )
            )

    plans = _merge_prepared_quantization_plans(plans)

    for region_id, plan in enumerate(plans):
        domain_module, encoded, aliases, frontiers = plan
        domain_dtype = domain_module.dtype
        physical_contract = PhysicalQuantizationContract(domain_dtype)

        for node in encoded:
            node.meta[QUANTIZATION_REGION] = region_id
            node.meta[QUANTIZATION_DOMAIN_DTYPE] = domain_dtype
            set_physical_quantization_contract(node, physical_contract)

        for alias in aliases:
            alias_module = _get_prepared_per_tensor_observer(
                alias,
                modules,
            )
            assert alias_module is not None
            alias_module.disable_fake_quant()
            alias_module.enable_observer()
            alias.meta[QUANTIZATION_REGION] = region_id
            alias.meta[_QUANTIZATION_DOMAIN_ALIAS] = True
            set_physical_quantization_contract(alias, physical_contract)

        for value, consumers in frontiers.items():
            candidates = compatible_observer_users(
                value,
                domain_module,
            )
            if candidates:
                entry = min(candidates, key=node_order.__getitem__)
            else:
                entry_module = _fresh_prepared_region_observer(domain_module)
                consumer = min(
                    consumers,
                    key=node_order.__getitem__,
                )
                entry = _insert_prepared_observer(
                    model,
                    modules,
                    value,
                    consumer,
                    entry_module,
                    f"{value.name}_quantization_domain_entry",
                )

            entry.meta[QUANTIZATION_REGION] = region_id
            entry.meta[_QUANTIZATION_DOMAIN_ENTRY] = True
            set_physical_quantization_contract(entry, physical_contract)
            for consumer in consumers:
                consumer.replace_input_with(value, entry)

    input_join_count = _prepare_input_join_regions(
        model,
        region_qspec,
        operator_traits,
        first_region_id=len(plans),
    )

    graph.lint()
    model.recompile()
    model.meta["quantization_region_count"] = len(plans) + input_join_count
    return model


def _node_tensor_shape(node: Node) -> tuple[int, ...] | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        return None
    return tuple(value.shape)


def _prepared_vsq_join_exit(
    join: Node,
    modules: dict[str, torch.nn.Module],
):
    path = [join]
    current = join
    while len(current.users) == 1:
        user = next(iter(current.users))
        observer = _get_prepared_vsq_observer(user, modules)
        if observer is not None and user.args[0] is current:
            return tuple(path), user, observer
        if user.target not in _PREPARED_DOMAIN_PASSTHROUGH_TARGETS:
            return None
        if _prepared_domain_data_inputs(user) != (current,):
            return None
        path.append(user)
        current = user
    return None


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _prepared_vsq_join_has_legal_layout(
    join: Node,
    path: tuple[Node, ...],
    observer: FusedAmaxObsFakeQuantize,
) -> bool:
    if not isinstance(observer.ch_axis, int):
        return False
    if not isinstance(observer.block_size, int):
        return False

    inputs = _prepared_domain_data_inputs(join)
    shapes = tuple(_node_tensor_shape(node) for node in inputs)
    if not shapes or any(shape is None for shape in shapes):
        return False
    reference = shapes[0]
    if any(len(shape) != len(reference) for shape in shapes[1:]):
        return False

    channel_axis = _normalize_axis(observer.ch_axis, len(reference))
    if not 0 <= channel_axis < len(reference):
        return False

    if join.target == torch.ops.aten.cat.default:
        join_axis = get_arg_value(join, 1, "dim", 0)
        join_axis = _normalize_axis(join_axis, len(reference))
        if join_axis != channel_axis:
            return False
    elif join.target == torch.ops.aten.stack.default:
        join_axis = get_arg_value(join, 1, "dim", 0)
        join_axis = _normalize_axis(join_axis, len(reference) + 1)
        if join_axis != channel_axis or len(path) < 2:
            return False
        flatten = path[1]
        if flatten.target != torch.ops.aten.flatten.using_ints:
            return False
        start_dim = get_arg_value(
            flatten,
            1,
            "start_dim",
            0,
        )
        end_dim = get_arg_value(
            flatten,
            2,
            "end_dim",
            -1,
        )
        if (
            start_dim,
            end_dim,
        ) != (
            channel_axis,
            channel_axis + 1,
        ):
            return False
    else:
        return False

    block_size = observer.block_size
    channel_sizes = tuple(shape[channel_axis] for shape in shapes)
    if any(
        not isinstance(size, int) or size % block_size for size in channel_sizes
    ):
        return False

    output_shape = _node_tensor_shape(path[-1])
    return output_shape is not None and output_shape[channel_axis] == sum(
        channel_sizes
    )


def _is_prepared_vsq_output_boundary(
    node: Node,
    operator_traits=None,
) -> bool:
    if is_gemm_op(node, operator_traits):
        return True
    if node.target not in _selected_quantization_output_epilogues(
        operator_traits
    ):
        return False
    source = node.args[0]
    return isinstance(source, Node) and is_gemm_op(source, operator_traits)


def _prepared_vsq_join_input(
    node: Node,
    join: Node,
    modules: dict[str, torch.nn.Module],
    template: FusedAmaxObsFakeQuantize,
    operator_traits=None,
):
    observer = _get_prepared_vsq_observer(node, modules)
    if observer is not None:
        if not _prepared_domains_compatible(template, observer):
            return None
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        if not _is_prepared_vsq_output_boundary(source, operator_traits):
            return None
        if tuple(node.users) != (join,) or tuple(source.users) != (node,):
            return None
        return node, observer

    if not _is_prepared_vsq_output_boundary(node, operator_traits):
        return None
    if tuple(node.users) != (join,):
        return None
    return node, None


def _vsq_physical_quantization_contract(
    observer: FusedAmaxObsFakeQuantize,
) -> PhysicalQuantizationContract:
    scale_dtype = (
        "fp8_e8m0"
        if observer.force_scale_power_of_two
        else observer.scale_dtype
    )
    companions = () if scale_dtype is None else (scale_dtype,)
    return PhysicalQuantizationContract(
        observer.dtype,
        companions,
        observer.block_size,
    )


def _prepare_vsq_join_regions(
    model: GraphModule,
    operator_traits=None,
) -> GraphModule:
    """Make legal VSQ channel joins share one calibrated coarse domain."""
    modules = dict(model.named_modules(remove_duplicate=False))
    next_region = model.meta.get("quantization_region_count", 0)
    used_observers = set()
    region_count = 0

    for join in list(model.graph.nodes):
        if join.target not in _PREPARED_DOMAIN_JOIN_TARGETS:
            continue
        target_rule = join.meta.get(_PREPARED_DOMAIN_TARGET_RULE)
        if target_rule is not None and target_rule.forcing_root:
            # A selected backend already owns this join as a mandatory
            # data-movement region. Do not reinterpret it as a VSQ join.
            continue
        exit_match = _prepared_vsq_join_exit(join, modules)
        if exit_match is None:
            continue
        path, exit_node, exit_observer = exit_match
        if exit_node in used_observers:
            continue
        if not _prepared_vsq_join_has_legal_layout(
            join,
            path,
            exit_observer,
        ):
            continue

        inputs = _prepared_domain_data_inputs(join)
        input_matches = [
            _prepared_vsq_join_input(
                input_node,
                join,
                modules,
                exit_observer,
                operator_traits,
            )
            for input_node in inputs
        ]
        if any(match is None for match in input_matches):
            continue

        entries = []
        for input_node, observer in input_matches:
            if observer is None:
                observer = copy.deepcopy(exit_observer)
                observer.amax_history.resize_(0)
                observer.scale.resize_(1).fill_(1.0)
                input_node = _insert_prepared_observer(
                    model,
                    modules,
                    input_node,
                    join,
                    observer,
                    f"{input_node.name}_vsq_output",
                )
            entries.append(input_node)

        if any(entry in used_observers for entry in entries):
            continue

        region_id = next_region + region_count
        region_count += 1
        domain = _vsq_physical_quantization_contract(exit_observer)

        for node in path:
            node.meta[QUANTIZATION_REGION] = region_id
            node.meta[QUANTIZATION_DOMAIN_DTYPE] = domain.payload_dtype
            set_physical_quantization_contract(node, domain)

        for entry in entries:
            entry.meta[QUANTIZATION_REGION] = region_id
            entry.meta[_VSQ_JOIN_ENTRY] = True
            set_physical_quantization_contract(entry, domain)

        exit_node.meta[QUANTIZATION_REGION] = region_id
        exit_node.meta[_VSQ_JOIN_ALIAS] = True
        set_physical_quantization_contract(exit_node, domain)
        exit_observer.disable_fake_quant()
        exit_observer.enable_observer()
        used_observers.update((*entries, exit_node))

    model.graph.lint()
    model.recompile()
    model.meta["quantization_region_count"] = next_region + region_count
    return model


def _finalize_quantization_regions(model: GraphModule) -> None:
    modules = dict(model.named_modules(remove_duplicate=False))
    regions = defaultdict(list)

    for node in model.graph.nodes:
        region_id = node.meta.get(QUANTIZATION_REGION)
        if region_id is None or node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if not isinstance(module, FusedAmaxObsFakeQuantize):
            continue
        regions[region_id].append((node, module))

    for members in regions.values():
        maxima = []
        for _, module in members:
            if module.amax_history.numel() == 0:
                continue
            amax = torch.amax(module.amax_history)
            if torch.isfinite(amax):
                maxima.append(amax)
        if not maxima:
            continue

        group_amax = torch.amax(torch.stack(maxima))
        domain_modules = {id(module): module for _, module in members}.values()
        for module in domain_modules:
            if module.amax_history.numel() == 0:
                module.amax_history.resize_(module.amax_history_len).fill_(0.0)
            module.amax_history[0].copy_(group_amax)
            module.calculate_qparams()


def _move_output_observers_after_epilogues(
    model: GraphModule,
    operator_traits=None,
) -> GraphModule:
    """Calibrate and quantize a fused activation, not its raw input.

    Besides GEMM epilogues, an input-only join may own a following activation
    without transferring its input domain to the join result.  Recognizing the
    join through the selected target's prepared-domain rules keeps this
    placement declarative and avoids target-specific graph walks.  Preserve an
    existing post-activation observer: matching observer configuration does
    not prove that independently calibrated observers share a quantization
    domain.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    graph = model.graph
    rule_set = PreparedDomainRuleSet(
        _selected_prepared_domain_rules(operator_traits)
    )
    output_epilogues = _selected_quantization_output_epilogues(operator_traits)
    moved_count = 0

    for observer in list(graph.nodes):
        module = _get_module(observer, modules)
        if not isinstance(module, FusedAmaxObsFakeQuantize):
            continue
        if len(observer.args) != 1 or len(observer.users) != 1:
            continue

        source = observer.args[0]
        input_join = (
            None
            if not isinstance(source, Node)
            else rule_set.input_join(source)
        )
        if not isinstance(source, Node) or not (
            is_gemm_op(source, operator_traits) or input_join is not None
        ):
            continue
        current = observer
        views = []
        while len(current.users) == 1:
            user = next(iter(current.users))
            view_source = source if current is observer else current
            if not is_address_only_view(user, view_source):
                break
            views.append(user)
            current = user
        if len(current.users) != 1:
            continue
        if (
            views
            and _get_prepared_per_tensor_observer(
                observer,
                modules,
            )
            is None
        ):
            continue
        epilogue = next(iter(current.users))
        if epilogue.target not in output_epilogues:
            continue
        if epilogue.args[0] is not current:
            continue

        epilogue_users = list(epilogue.users)
        output_observers = tuple(
            output_observer
            for user in epilogue_users
            if isinstance(
                output_observer := _get_module(user, modules),
                FusedAmaxObsFakeQuantize,
            )
        )
        if input_join is None and output_observers:
            continue

        if views:
            views[0].replace_input_with(observer, source)
        else:
            epilogue.replace_input_with(observer, source)

        with graph.inserting_after(epilogue):
            moved = graph.call_module(
                str(observer.target),
                (epilogue,),
            )
        moved.meta = copy.copy(epilogue.meta)
        for user in epilogue_users:
            user.replace_input_with(epilogue, moved)
        graph.erase_node(observer)
        moved_count += 1

    graph.lint()
    model.recompile()
    logger.info(
        "Moved %d output observers after target-declared epilogues",
        moved_count,
    )
    return model


__all__ = [
    "_CONFLICTING_PREPARED_DOMAIN",
    "_PREPARED_DOMAIN_JOIN_TARGETS",
    "_PREPARED_DOMAIN_PASSTHROUGH_TARGETS",
    "_PREPARED_DOMAIN_REGION_ROOT_TARGETS",
    "_PREPARED_DOMAIN_TARGET_RULE",
    "_QUANTIZATION_DOMAIN_ALIAS",
    "_QUANTIZATION_DOMAIN_ENTRY",
    "_UNKNOWN_PREPARED_DOMAIN",
    "_VSQ_JOIN_ALIAS",
    "_VSQ_JOIN_ENTRY",
    "_analyze_prepared_quantization_regions",
    "_bind_prepared_domain_target_rules",
    "_create_obs_or_fq_from_qspec",
    "_effective_prepared_domain_rule_set",
    "_finalize_quantization_regions",
    "_find_prepared_data_movement_regions",
    "_get_module",
    "_get_prepared_per_tensor_observer",
    "_get_prepared_vsq_observer",
    "_insert_prepared_observer",
    "_is_chunk_getitem",
    "_is_prepared_vsq_output_boundary",
    "_is_zero_constant_pad",
    "_merge_prepared_quantization_plans",
    "_move_output_observers_after_epilogues",
    "_node_tensor_shape",
    "_normalize_axis",
    "_prepare_quantization_regions",
    "_prepare_vsq_join_regions",
    "_prepared_domain_data_inputs",
    "_prepared_domain_rule_set",
    "_prepared_domains_compatible",
    "_prepared_vsq_join_exit",
    "_prepared_vsq_join_has_legal_layout",
    "_prepared_vsq_join_input",
    "_selected_prepared_domain_rules",
    "_selected_quantization_output_epilogues",
    "_vsq_physical_quantization_contract",
]
