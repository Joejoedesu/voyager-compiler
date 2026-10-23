# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared prepared and physical quantization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Collection, Iterable, TypeVar

import torch
from torch.fx import Node

PHYSICAL_QUANTIZATION_CONTRACT = "physical_quantization_contract"
QUANTIZATION_DOMAIN_DTYPE = "quantization_domain_dtype"
QUANTIZATION_INPUT_REGION = "quantization_input_region"
QUANTIZATION_REGION = "quantization_region"


@dataclass(frozen=True)
class PhysicalQuantizationContract:
    """The components carried by one addressable physical value."""

    payload_dtype: object
    companion_dtypes: tuple[object, ...] = ()
    block_size: int | None = None


def get_physical_quantization_contract(
    node: Node,
) -> PhysicalQuantizationContract | None:
    contract = node.meta.get(PHYSICAL_QUANTIZATION_CONTRACT)
    return (
        contract if isinstance(contract, PhysicalQuantizationContract) else None
    )


def has_physical_payload(
    node: object,
    payload_dtype: object,
    *,
    companion_dtypes: tuple[object, ...] = (),
) -> bool:
    """Return whether a node carries exactly one declared physical format."""
    if not isinstance(node, Node):
        return False
    contract = get_physical_quantization_contract(node)
    return (
        contract is not None
        and contract.payload_dtype == payload_dtype
        and contract.companion_dtypes == companion_dtypes
    )


def set_physical_quantization_contract(
    node: Node,
    contract: PhysicalQuantizationContract,
) -> None:
    """Attach the encoded-value format carried by a logical tensor.

    Storage ownership and aliasing belong to ``StorageGraph``. Keeping the
    format contract here avoids maintaining a second, node-local ownership
    chain alongside the logical-to-physical storage plan.
    """
    node.meta[PHYSICAL_QUANTIZATION_CONTRACT] = contract


def inherit_physical_quantization_contract(
    target: Node,
    source: Node,
) -> None:
    """Copy an encoded-value contract through a semantic graph rewrite."""
    contract = get_physical_quantization_contract(source)
    if contract is not None:
        target.meta[PHYSICAL_QUANTIZATION_CONTRACT] = contract


def get_physical_dtype(node: Node):
    """Return the payload dtype used by physical placement."""
    contract = get_physical_quantization_contract(node)
    if contract is not None:
        return contract.payload_dtype

    dtype = node.meta.get(
        QUANTIZATION_DOMAIN_DTYPE,
        node.meta.get("dtype"),
    )
    if isinstance(dtype, (tuple, list)):
        dtype = next((item for item in reversed(dtype) if item), None)
    return dtype


def _axis_tuple(axis: Any) -> tuple[int, ...]:
    if axis is None:
        return ()
    if isinstance(axis, int):
        return (axis,)
    return tuple(int(item) for item in axis)


def _optional_scalar(value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() != 1:
        raise ValueError("Quantization-domain attributes must be scalar")
    return value.detach().reshape(-1)[0].item()


@dataclass(frozen=True)
class PreparedQuantizationContract:
    """Complete semantic identity of one prepared quantization domain."""

    dtype: object
    qscheme: object
    quant_min: int | None
    quant_max: int | None
    scale_quant_max: float | None
    rounding: str
    axes: tuple[int, ...]
    block_size: int | None
    scale_dtype: object
    force_scale_power_of_two: bool
    outlier_threshold: float | None
    outlier_pct: float | None
    coarse_dtype: object
    coarse_quant_min: int | None
    coarse_quant_max: int | None
    coarse_scale_quant_max: float | None
    coarse_qscheme: object
    coarse_axes: tuple[int, ...]

    @classmethod
    def from_observer(
        cls,
        observer: object,
    ) -> "PreparedQuantizationContract":
        """Canonicalize the domain-defining attributes of an observer."""
        return cls(
            dtype=getattr(observer, "dtype"),
            qscheme=getattr(observer, "qscheme"),
            quant_min=getattr(observer, "quant_min"),
            quant_max=getattr(observer, "quant_max"),
            scale_quant_max=_optional_scalar(
                getattr(observer, "scale_quant_max", None)
            ),
            rounding=getattr(observer, "rounding"),
            axes=_axis_tuple(getattr(observer, "ch_axis")),
            block_size=getattr(observer, "block_size"),
            scale_dtype=getattr(observer, "scale_dtype"),
            force_scale_power_of_two=bool(
                getattr(observer, "force_scale_power_of_two")
            ),
            outlier_threshold=_optional_scalar(
                getattr(observer, "outlier_threshold")
            ),
            outlier_pct=_optional_scalar(getattr(observer, "outlier_pct")),
            coarse_dtype=getattr(observer, "coarse_dtype"),
            coarse_quant_min=getattr(observer, "coarse_quant_min"),
            coarse_quant_max=getattr(observer, "coarse_quant_max"),
            coarse_scale_quant_max=_optional_scalar(
                getattr(observer, "coarse_scale_quant_max", None)
            ),
            coarse_qscheme=getattr(observer, "coarse_qscheme"),
            coarse_axes=_axis_tuple(getattr(observer, "coarse_ch_axis")),
        )


@dataclass(frozen=True)
class PreparedDomainFormatConstraint:
    """Required encoded format; unspecified semantics remain unconstrained."""

    dtype: object
    block_size: int | None
    scale_dtype: object
    coarse_dtype: object

    def accepts(self, contract: PreparedQuantizationContract) -> bool:
        return (
            contract.dtype == self.dtype
            and contract.block_size == self.block_size
            and contract.scale_dtype == self.scale_dtype
            and contract.coarse_dtype == self.coarse_dtype
        )


class PreparedDomainInputFormat(Enum):
    """How tensor inputs are represented in an FX node's arguments."""

    FIRST = "first"
    SEQUENCE = "sequence"
    PREFIX = "prefix"


class PreparedDomainRelation(Enum):
    """How an operation relates encoded input and result domains."""

    ALIAS = "alias"
    MAP_COMPONENTS = "map_components"
    JOIN = "join"
    INPUT_JOIN = "input_join"


PreparedDomainPredicate = Callable[[Node], bool]


@dataclass(frozen=True)
class PreparedDomainTransferRule:
    """Declarative local domain-transfer rule for one operator shape."""

    target: object | None
    input_format: PreparedDomainInputFormat
    relation: PreparedDomainRelation
    input_count: int | None = None
    region_root: bool = False
    forcing_root: bool = False
    predicate: PreparedDomainPredicate | None = None

    def matches(self, node: Node) -> bool:
        if self.target is not None and node.target != self.target:
            return False
        return self.predicate is None or self.predicate(node)

    def inputs(self, node: Node) -> tuple[Node, ...]:
        if self.input_format is PreparedDomainInputFormat.FIRST:
            inputs = node.args[:1]
        elif self.input_format is PreparedDomainInputFormat.SEQUENCE:
            inputs = node.args[0] if node.args else ()
        else:
            if self.input_count is None:
                raise ValueError("PREFIX domain rules require input_count")
            schema = getattr(node.target, "_schema", None)
            arguments = getattr(schema, "arguments", ())
            resolved = []
            for index in range(self.input_count):
                if index < len(node.args):
                    resolved.append(node.args[index])
                    continue
                name = arguments[index].name if index < len(arguments) else None
                if name is None or name not in node.kwargs:
                    return ()
                resolved.append(node.kwargs[name])
            inputs = tuple(resolved)
        if not isinstance(inputs, (tuple, list)):
            return ()
        if not all(isinstance(item, Node) for item in inputs):
            return ()
        return tuple(inputs)


@dataclass(frozen=True)
class PreparedDomainTransfer:
    """One matched transfer rule with its resolved input values."""

    rule: PreparedDomainTransferRule
    inputs: tuple[Node, ...]


@dataclass(frozen=True)
class PreparedDomainRuleSet:
    """Ordered target-independent or target-specific transfer rules."""

    rules: tuple[PreparedDomainTransferRule, ...]

    def resolve(self, node: Node) -> PreparedDomainTransfer | None:
        """Resolve the first complete rule, including input-only rules."""
        for rule in self.rules:
            if not rule.matches(node):
                continue
            inputs = rule.inputs(node)
            if inputs:
                return PreparedDomainTransfer(rule, inputs)
        return None

    def transfer(self, node: Node) -> PreparedDomainTransfer | None:
        """Resolve a rule that transfers an input domain to the result."""
        resolved = self.resolve(node)
        if (
            resolved is not None
            and resolved.rule.relation is PreparedDomainRelation.INPUT_JOIN
        ):
            return None
        return resolved

    def input_join(self, node: Node) -> PreparedDomainTransfer | None:
        """Resolve a common-domain constraint over only the node inputs."""
        resolved = self.resolve(node)
        if (
            resolved is None
            or resolved.rule.relation is not PreparedDomainRelation.INPUT_JOIN
        ):
            return None
        return resolved

    def inputs(self, node: Node) -> tuple[Node, ...]:
        transfer = self.transfer(node)
        return () if transfer is None else transfer.inputs


class QuantizationContractTransferGraph:
    """Indexed dataflow induced by one declarative transfer-rule set.

    The index deliberately knows nothing about observers, encoded component
    formats, or compilation targets. Callers provide those constraints in
    ``rules`` and use the same edge relation for forward propagation and
    connected-region analysis.
    """

    def __init__(
        self,
        nodes: Iterable[Node],
        rules: PreparedDomainRuleSet,
    ) -> None:
        self.nodes = tuple(nodes)
        self._transfers: dict[Node, PreparedDomainTransfer] = {}
        consumers: dict[Node, list[Node]] = {}
        for node in self.nodes:
            transfer = rules.transfer(node)
            if transfer is None:
                continue
            self._transfers[node] = transfer
            for input_node in transfer.inputs:
                input_consumers = consumers.setdefault(input_node, [])
                if node not in input_consumers:
                    input_consumers.append(node)
        self._consumers = {
            node: tuple(users) for node, users in consumers.items()
        }

    def transfer(self, node: Node) -> PreparedDomainTransfer | None:
        return self._transfers.get(node)

    def inputs(self, node: Node) -> tuple[Node, ...]:
        transfer = self.transfer(node)
        return () if transfer is None else transfer.inputs

    def consumers(self, node: Node) -> tuple[Node, ...]:
        """Return only users connected by a declared domain edge."""
        return self._consumers.get(node, ())

    def is_node(self, node: Node) -> bool:
        return node in self._transfers

    def is_edge(self, source: Node, consumer: Node) -> bool:
        return source in self.inputs(consumer)

    def is_region_root(self, node: Node) -> bool:
        transfer = self.transfer(node)
        return transfer is not None and transfer.rule.region_root

    def is_forcing_root(self, node: Node) -> bool:
        transfer = self.transfer(node)
        return transfer is not None and transfer.rule.forcing_root

    def connected_component(self, root: Node) -> frozenset[Node]:
        """Return the bidirectionally connected declared-transfer region."""
        if not self.is_node(root):
            return frozenset()

        connected: set[Node] = set()
        pending = [root]
        while pending:
            node = pending.pop()
            if node in connected:
                continue
            connected.add(node)
            pending.extend(
                input_node
                for input_node in self.inputs(node)
                if self.is_node(input_node)
            )
            pending.extend(self.consumers(node))
        return frozenset(connected)

    def frontiers(
        self,
        region: Collection[Node],
    ) -> dict[Node, tuple[Node, ...]]:
        """Map each external input to its consumers inside ``region``."""
        region_nodes = frozenset(region)
        frontiers: dict[Node, list[Node]] = {}
        for node in self.nodes:
            if node not in region_nodes:
                continue
            for input_node in self.inputs(node):
                if input_node in region_nodes:
                    continue
                input_consumers = frontiers.setdefault(input_node, [])
                if node not in input_consumers:
                    input_consumers.append(node)
        return {node: tuple(consumers) for node, consumers in frontiers.items()}

    def exits(
        self,
        region: Collection[Node],
    ) -> tuple[tuple[Node, Node], ...]:
        """Return every ordinary FX use that leaves ``region``."""
        region_nodes = frozenset(region)
        return tuple(
            (node, user)
            for node in self.nodes
            if node in region_nodes
            for user in node.users
            if user not in region_nodes
        )


DomainValue = TypeVar("DomainValue")
SeedValue = Callable[
    [Node, dict[Node, DomainValue]],
    DomainValue | None,
]
TransferValue = Callable[
    [Node, PreparedDomainTransfer, tuple[DomainValue, ...]],
    DomainValue | None,
]
PartialTransfer = Callable[
    [Node, PreparedDomainTransfer, tuple[DomainValue | None, ...]],
    None,
]


def propagate_prepared_domain_values(
    nodes: Iterable[Node],
    rules: PreparedDomainRuleSet,
    *,
    seed_value: SeedValue[DomainValue],
    transfer_value: TransferValue[DomainValue],
    partial_transfer: PartialTransfer[DomainValue] | None = None,
) -> dict[Node, DomainValue]:
    """Propagate encoded values over the shared declared-transfer graph."""
    domain_graph = QuantizationContractTransferGraph(nodes, rules)
    values: dict[Node, DomainValue] = {}
    for node in domain_graph.nodes:
        seed = seed_value(node, values)
        if seed is not None:
            values[node] = seed
            continue

        transfer = domain_graph.transfer(node)
        if transfer is None:
            continue
        input_values = tuple(values.get(item) for item in transfer.inputs)
        present = tuple(value for value in input_values if value is not None)
        if not present:
            continue
        if len(present) != len(input_values):
            if partial_transfer is not None:
                partial_transfer(node, transfer, input_values)
            continue

        result = transfer_value(node, transfer, present)
        if result is not None:
            values[node] = result
    return values
