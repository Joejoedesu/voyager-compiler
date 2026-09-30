"""Target-selected graph rules resolved before PT2E observer injection.

A rule finds edges/outputs whose quantization parameters must be equal. The
scanner merges overlapping groups, checks explicit specs, and emits PT2E shared
specs. It does not move quantize operations or depend on quantization folding.
"""

import copy
from dataclasses import dataclass, fields, replace
from typing import Callable

import torch
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

from voyager_compiler.quantization.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    SharedAmaxObsFakeQuantize,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    QScheme,
    QuantizationSpec,
)


@dataclass(frozen=True)
class QuantizationRule:
    name: str
    operations: tuple
    collect: Callable  # (node, context) -> iterable of edges and outputs
    applies: Callable  # (specs, context) -> bool


_CAT = torch.ops.aten.cat.default
_VIEWS = frozenset(
    (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
    )
)


def _annotation(node):
    return node.meta.get("quantization_annotation", QuantizationAnnotation())


def _spec(member):
    if isinstance(member, tuple):
        source, user = member
        return _annotation(user).input_qspec_map.get(source)
    return _annotation(member).output_qspec


def _resolve(spec):
    seen = set()
    while isinstance(spec, SharedQuantizationSpec):
        member = spec.edge_or_node
        if member in seen:
            raise ValueError("Cyclic shared quantization annotation")
        seen.add(member)
        spec = _spec(member)
    return spec


def _concat_members(node, context):
    # Only per-tensor rules use these views: no channel/block-axis inference.
    component, todo = set(), [node]
    while todo:
        current = todo.pop()
        if current in component or current.target not in (_CAT, *_VIEWS):
            continue
        component.add(current)
        todo.extend(current.users)
        todo.extend(
            current.all_input_nodes[:1]
            if current.target in _VIEWS
            else current.args[0]
        )
    members = set(component)
    for current in component:
        inputs = (
            current.args[0] if current.target is _CAT else (current.args[0],)
        )
        members.update((source, current) for source in inputs)
        # Constrain annotated consumers, not unrelated unquantized branches.
        members.update(
            (current, user)
            for user in current.users
            if _annotation(user).input_qspec_map.get(current) is not None
        )
    return members


def _int8_tensor(specs, context):
    return any(
        isinstance(s, QuantizationSpec)
        and s.dtype == "int8"
        and s.qscheme == QScheme.PER_TENSOR_SYMMETRIC
        for s in specs
    )


CONCAT_INT8 = QuantizationRule(
    "concat_int8_shared_scale",
    (_CAT,),
    _concat_members,
    _int8_tensor,
)


def _observer(spec):
    return spec.observer_or_fake_quant_ctr(
        **{
            f.name: getattr(spec, f.name)
            for f in fields(spec)
            if f.name != "observer_or_fake_quant_ctr"
        }
    )


def _signature(spec):
    if not isinstance(spec, QuantizationSpec):
        return None
    # Constructor wrappers have identity-based equality. Compare their actual
    # observer type and options too, without treating equivalent wrappers as
    # conflicting policies.
    observer = _observer(spec)
    return (
        tuple(
            (f.name, getattr(spec, f.name))
            for f in fields(spec)
            if f.name != "observer_or_fake_quant_ctr"
        ),
        type(observer),
        getattr(observer, "force_scale_power_of_two", False),
    )


def scan_quantization_rules(model, rules, context=None):
    """Resolve target-selected equality constraints, retaining edge isolation."""
    groups = []
    for rule in rules:
        visited = set()
        for node in model.graph.nodes:
            if node.target not in rule.operations or node in visited:
                continue
            members = set(rule.collect(node, context))
            visited.update(m for m in members if not isinstance(m, tuple))
            specs = [
                _resolve(_spec(m)) for m in members if _spec(m) is not None
            ]
            if not rule.applies(specs, context):
                continue
            groups.append((members, {rule.name}))

    # Overlapping rules/chains are one transitive constraint, not successive
    # annotation overwrites. Specs are checked before any graph mutation.
    merged = []
    for members, names in groups:
        changed = True
        while changed:
            changed = False
            for other, labels in list(merged):
                if members & other:
                    members |= other
                    names |= labels
                    merged.remove((other, labels))
                    changed = True
        merged.append((members, names))

    order = {node: i for i, node in enumerate(model.graph.nodes)}
    plans = []
    for members, names in merged:
        specs = [_resolve(_spec(m)) for m in members if _spec(m) is not None]
        spec = specs[0]
        label = ", ".join(sorted(names))
        if any(_signature(s) != _signature(spec) for s in specs):
            nodes = sorted(
                {(m[1] if isinstance(m, tuple) else m).name for m in members}
            )
            raise ValueError(
                f"{label}: incompatible quantization specs at {nodes}"
            )
        observer = _observer(spec)
        if (
            type(observer) is not FusedAmaxObsFakeQuantize
            or spec.qscheme != QScheme.PER_TENSOR_SYMMETRIC
            or spec.is_dynamic
        ):
            raise ValueError(
                f"{label}: shared PTQ calibration requires static per-tensor amax"
            )
        spec = replace(
            spec,
            observer_or_fake_quant_ctr=SharedAmaxObsFakeQuantize.with_args(
                force_scale_power_of_two=observer.force_scale_power_of_two,
            ),
        )
        members = sorted(
            members,
            key=lambda m: (
                order[m[1]] if isinstance(m, tuple) else order[m],
                0 if isinstance(m, tuple) else 1,
                order[m[0]] if isinstance(m, tuple) else 0,
            ),
        )
        plans.append((members, names, spec))

    records = []
    for members, names, spec in plans:
        # PT2E's implicit union is initiated from either consumer and compares
        # dtype/range, not observer policy. Disable it on sibling consumers too,
        # or an unrelated edge can pull itself into this calibration group.
        for member in members:
            if not isinstance(member, tuple):
                continue
            for user in member[0].users:
                if "quantization_annotation" in user.meta:
                    annotation = copy.copy(_annotation(user))
                    annotation.allow_implicit_sharing = False
                    user.meta["quantization_annotation"] = annotation
        root = members[0]
        for member in members:
            qspec = spec if member == root else SharedQuantizationSpec(root)
            node = member[1] if isinstance(member, tuple) else member
            annotation = copy.copy(_annotation(node))
            annotation.input_qspec_map = dict(_annotation(node).input_qspec_map)
            annotation.allow_implicit_sharing = False
            annotation._annotated = True
            if isinstance(member, tuple):
                annotation.input_qspec_map[member[0]] = qspec
            else:
                annotation.output_qspec = qspec
            node.meta["quantization_annotation"] = annotation
        records.append(
            dict(
                rules=sorted(names),
                members=[
                    (
                        f"{m[0].name}->{m[1].name}"
                        if isinstance(m, tuple)
                        else m.name
                    )
                    for m in members
                ],
            )
        )
    model.meta["quantization_rule_groups"] = records
    return model
