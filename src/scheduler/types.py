# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Backend-neutral local-scheduler workload and result types."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Iterable


class Loop(IntEnum):
    """Canonical GEMM loop order shared by Voyager scheduler backends."""

    FX = 0
    FY = 1
    OX = 2
    OY = 3
    OC = 4
    IC = 5
    ON = 6


NUM_LOOPS = len(Loop)


@dataclass(frozen=True)
class GemmWorkload:
    """Canonical convolution/GEMM loop extents supplied to schedulers.

    The names follow the CNN loop convention shared by Voyager's scheduling
    engines. A matrix multiplication is represented as a 1x1 filter over a
    flattened output position space. Element widths are deliberately absent:
    dispatch supplies them separately because they are properties of one
    scheduler invocation, not loop extents.
    """

    nifm: int
    nofm: int
    wofm: int
    hofm: int
    wfil: int
    hfil: int
    nimg: int = 1
    wstd: int = 1
    hstd: int = 1

    def __post_init__(self):
        for field_name in (
            "nifm",
            "nofm",
            "wofm",
            "hofm",
            "wfil",
            "hfil",
            "nimg",
            "wstd",
            "hstd",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise ValueError(f"{field_name} must be a positive integer")
            normalized = int(value)
            if normalized <= 0 or normalized != value:
                raise ValueError(f"{field_name} must be a positive integer")
            object.__setattr__(self, field_name, normalized)

    @property
    def wifm(self) -> int:
        return self.wfil + (self.wofm - 1) * self.wstd

    @property
    def hifm(self) -> int:
        return self.hfil + (self.hofm - 1) * self.hstd

    @property
    def sizes(self) -> tuple[int, ...]:
        """Return extents in canonical FX,FY,OX,OY,OC,IC,ON order."""
        return (
            self.wfil,
            self.hfil,
            self.wofm,
            self.hofm,
            self.nofm,
            self.nifm,
            self.nimg,
        )


GemmTilingCandidateCheck = Callable[
    [GemmWorkload, GemmWorkload, object, int, bool],
    bool,
]


def _freeze_shape(
    shape,
    field_name: str,
) -> tuple[int, ...] | None:
    if shape is None:
        return None
    result = []
    for dimension in shape:
        normalized = int(dimension)
        if (
            isinstance(dimension, bool)
            or normalized != dimension
            or normalized < 0
        ):
            raise ValueError(
                f"{field_name} dimensions must be non-negative integers"
            )
        result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class OperationOperand:
    """One resolved semantic operand supplied by the compiler adapter.

    ``port`` is the normalized operator-schema port. ``occurrence`` keeps
    repeated tensor leaves distinct without encoding paths into port names.
    """

    port: str
    shape: tuple[int, ...] | None
    bits: int | None
    occurrence: tuple[object, ...] = ()

    def __post_init__(self):
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("operation operand port must be non-empty")
        object.__setattr__(
            self,
            "shape",
            _freeze_shape(self.shape, f"{self.port} shape"),
        )
        if self.bits is not None:
            bits = int(self.bits)
            if isinstance(self.bits, bool) or bits != self.bits or bits <= 0:
                raise ValueError(f"{self.port} bits must be a positive integer")
            object.__setattr__(self, "bits", bits)
        occurrence = tuple(self.occurrence)
        try:
            hash(occurrence)
        except TypeError as error:
            raise TypeError(
                "operation operand occurrence must be hashable"
            ) from error
        object.__setattr__(self, "occurrence", occurrence)

    @property
    def identity(self) -> tuple[str, tuple[object, ...]]:
        """Return the typed identity of this operand occurrence."""
        return self.port, self.occurrence


def _freeze_attribute(value, field_name: str):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attribute(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only scalar or tuple values")


@dataclass(frozen=True)
class OperationClassification:
    """Target-owned operation identity and frozen scalar facts."""

    kind: str
    attributes: tuple[tuple[str, object], ...] | Mapping = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("operation kind must be non-empty")
        items = (
            self.attributes.items()
            if isinstance(self.attributes, Mapping)
            else self.attributes
        )
        attributes = tuple(
            sorted(
                (
                    (
                        str(name),
                        _freeze_attribute(
                            value,
                            f"operation attribute {name!r}",
                        ),
                    )
                    for name, value in items
                ),
                key=lambda item: item[0],
            )
        )
        names = tuple(name for name, _value in attributes)
        if len(set(names)) != len(names):
            raise ValueError("operation attribute names must be unique")
        object.__setattr__(self, "attributes", attributes)

    def attribute(self, name: str, default=None):
        """Return one frozen scalar fact."""
        return next(
            (
                value
                for attribute_name, value in self.attributes
                if attribute_name == name
            ),
            default,
        )


@dataclass(frozen=True)
class OperationWorkload:
    """Resolved input to one scheduler-dispatch decision.

    The compiler and selected target freeze all graph knowledge into
    ``operation`` and typed operands before dispatch. Scheduler providers
    therefore neither inspect FX nodes nor rediscover operand semantics.
    """

    operands: tuple[OperationOperand, ...]
    output: OperationOperand
    operation: OperationClassification | None = None
    gemm: GemmWorkload | None = None
    has_high_precision_vector_input: bool = False
    has_sparse_op: bool = False

    def __post_init__(self):
        operands = tuple(self.operands)
        if any(
            not isinstance(operand, OperationOperand) for operand in operands
        ):
            raise TypeError(
                "operation operands must be OperationOperand values"
            )
        identities = tuple(operand.identity for operand in operands)
        if len(set(identities)) != len(identities):
            raise ValueError(
                "operation operand port/occurrence identities must be unique"
            )
        if any(operand.port == "output" for operand in operands):
            raise ValueError(
                "output must use the dedicated OperationWorkload field"
            )
        if not isinstance(self.output, OperationOperand):
            raise TypeError(
                "operation output must be an OperationOperand value"
            )
        if self.output.port != "output":
            raise ValueError("operation output must use port 'output'")
        if self.output.shape is None:
            raise ValueError("operation output shape must be resolved")
        if self.operation is not None and not isinstance(
            self.operation,
            OperationClassification,
        ):
            raise TypeError(
                "operation must be an OperationClassification or None"
            )
        if self.gemm is not None and not isinstance(
            self.gemm,
            GemmWorkload,
        ):
            raise TypeError("gemm must be a GemmWorkload or None")
        for field_name in (
            "has_high_precision_vector_input",
            "has_sparse_op",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool")
        object.__setattr__(
            self,
            "operands",
            tuple(
                sorted(
                    operands,
                    key=lambda operand: (
                        operand.port,
                        repr(operand.occurrence),
                    ),
                )
            ),
        )

    def operands_for_port(
        self,
        port: str,
    ) -> tuple[OperationOperand, ...]:
        """Return every occurrence bound to one schema port."""
        return tuple(
            operand for operand in self.operands if operand.port == port
        )

    def operand(
        self,
        port: str,
        occurrence: tuple[object, ...] | None = None,
    ) -> OperationOperand | None:
        """Return one unambiguous semantic operand occurrence."""
        matches = self.operands_for_port(port)
        if occurrence is not None:
            occurrence = tuple(occurrence)
            return next(
                (
                    operand
                    for operand in matches
                    if operand.occurrence == occurrence
                ),
                None,
            )
        if len(matches) > 1:
            raise ValueError(
                f"operation workload has multiple {port!r} occurrences"
            )
        return matches[0] if matches else None

    def require_operand(
        self,
        port: str,
        occurrence: tuple[object, ...] | None = None,
    ) -> OperationOperand:
        operand = self.operand(port, occurrence)
        if operand is None:
            raise ValueError(f"operation workload has no {port!r} operand")
        return operand


@dataclass(frozen=True)
class SchedulerProviderSet:
    """A target-selected set of scheduler implementations.

    Compilation-target identity deliberately does not appear here.  The
    compiler selects one immutable provider set, while scheduler dispatch only
    resolves an optional engine name within that set.

    The optional tiling check receives the candidate and full GEMM workloads,
    hardware, input width, and whether the transformed graph can materialize
    output-channel tiles. Target policy decides whether that candidate is
    legal. A target may also lower the generic Conv2d search's minimum square
    spatial extent when spatial dimensions are not part of its PE unroll.
    """

    default_backend: str
    gemm_providers: Mapping
    operation_provider: object | None = None
    gemm_tiling_candidate_fits: GemmTilingCandidateCheck | None = None
    conv2d_tiling_min_spatial_extent: int | None = None

    def __post_init__(self):
        if (
            not isinstance(self.default_backend, str)
            or not self.default_backend
        ):
            raise ValueError("default scheduler backend must be non-empty")
        providers = dict(self.gemm_providers)
        if not providers:
            raise ValueError("at least one GEMM scheduler provider is required")
        if any(
            not isinstance(name, str) or not name or not callable(provider)
            for name, provider in providers.items()
        ):
            raise TypeError(
                "GEMM scheduler providers require non-empty names and "
                "callable values"
            )
        if self.default_backend not in providers:
            raise ValueError(
                "default scheduler backend must have a registered provider"
            )
        if self.operation_provider is not None and not callable(
            self.operation_provider
        ):
            raise TypeError("operation scheduler provider must be callable")
        if self.gemm_tiling_candidate_fits is not None and not callable(
            self.gemm_tiling_candidate_fits
        ):
            raise TypeError("GEMM tiling candidate check must be callable")
        min_spatial = self.conv2d_tiling_min_spatial_extent
        if min_spatial is not None and (
            isinstance(min_spatial, bool)
            or not isinstance(min_spatial, int)
            or min_spatial < 1
        ):
            raise ValueError(
                "Conv2d tiling minimum spatial extent must be a positive "
                "integer"
            )
        object.__setattr__(
            self,
            "gemm_providers",
            MappingProxyType(providers),
        )


def _freeze_matrix(
    values: Iterable[Iterable[int]],
    field_name: str,
) -> tuple[tuple[int, ...], ...]:
    result = tuple(tuple(int(value) for value in row) for row in values)
    if len(result) != NUM_LOOPS:
        raise ValueError(
            f"{field_name} must contain one row per canonical loop"
        )
    if not result or not result[0]:
        raise ValueError(f"{field_name} must contain at least one level")
    num_levels = len(result[0])
    if any(len(row) != num_levels for row in result):
        raise ValueError(f"{field_name} rows must have equal length")
    return result


@dataclass(frozen=True)
class MappingPoint:
    """A complete local mapping independent of a scheduler implementation.

    Matrices use ``[loop][level]`` indexing. Levels are ordered from the
    innermost PE level to the outermost memory level, matching Interstellar's
    factor convention. Code generation deliberately skips level zero.
    """

    loop_orders: tuple[tuple[int, ...], ...]
    loop_blockings: tuple[tuple[int, ...], ...]
    loop_partitionings: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        orders = _freeze_matrix(self.loop_orders, "loop_orders")
        blockings = _freeze_matrix(self.loop_blockings, "loop_blockings")
        partitionings = _freeze_matrix(
            self.loop_partitionings,
            "loop_partitionings",
        )
        num_levels = len(orders[0])
        if len(blockings[0]) != num_levels:
            raise ValueError("loop_blockings level count does not match")
        if len(partitionings[0]) != num_levels:
            raise ValueError("loop_partitionings level count does not match")
        if any(value <= 0 for row in blockings for value in row):
            raise ValueError("loop blocking factors must be positive")
        if any(value <= 0 for row in partitionings for value in row):
            raise ValueError("loop partitioning factors must be positive")
        if any(value < 0 for row in orders for value in row):
            raise ValueError("loop orders must be non-negative")
        object.__setattr__(self, "loop_orders", orders)
        object.__setattr__(self, "loop_blockings", blockings)
        object.__setattr__(self, "loop_partitionings", partitionings)

    @property
    def num_levels(self) -> int:
        return len(self.loop_orders[0])

    def loop_coverage(self, loop: Loop | int) -> int:
        """Return total blocking and spatial coverage for one loop."""
        loop = int(loop)
        return math.prod(self.loop_blockings[loop]) * math.prod(
            self.loop_partitionings[loop]
        )


@dataclass(frozen=True)
class LevelAccessCounts:
    input: int = 0
    output: int = 0
    weight: int = 0

    def __post_init__(self):
        if min(self.input, self.output, self.weight) < 0:
            raise ValueError("access counts must be non-negative")

    @classmethod
    def from_sequence(cls, values) -> "LevelAccessCounts":
        return cls(*(int(value) for value in values))


@dataclass(frozen=True)
class OperandAccessEvents:
    """Scheduler access events for one operand at one memory level."""

    reads: int = 0
    updates: int = 0
    fills: int = 0

    def __post_init__(self):
        if min(self.reads, self.updates, self.fills) < 0:
            raise ValueError("access events must be non-negative")

    @property
    def total(self) -> int:
        return self.reads + self.updates + self.fills


@dataclass(frozen=True)
class LevelAccessEvents:
    """Uncollapsed operand events for one canonical memory level."""

    input: OperandAccessEvents = OperandAccessEvents()
    output: OperandAccessEvents = OperandAccessEvents()
    weight: OperandAccessEvents = OperandAccessEvents()

    @property
    def totals(self) -> LevelAccessCounts:
        return LevelAccessCounts(
            input=self.input.total,
            output=self.output.total,
            weight=self.weight.total,
        )


@dataclass(frozen=True)
class LocalScheduleResult:
    mapping: MappingPoint
    backend: str
    access_counts: tuple[LevelAccessCounts, ...] = ()
    runtime: float | None = None
    energy_pj: float | None = None
    access_events: tuple[LevelAccessEvents, ...] | None = None
    named_factors: tuple[tuple[str, int], ...] | Mapping = ()

    def __post_init__(self):
        factor_items = (
            self.named_factors.items()
            if isinstance(self.named_factors, Mapping)
            else self.named_factors
        )
        named_factors = []
        for name, value in factor_items:
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name.lower() != name
            ):
                raise ValueError(
                    "schedule factor names must be lowercase identifiers"
                )
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"schedule factor {name!r} must be a positive integer"
                )
            named_factors.append((name, value))
        named_factors.sort(key=lambda item: item[0])
        factor_names = tuple(name for name, _value in named_factors)
        if len(set(factor_names)) != len(factor_names):
            raise ValueError("schedule factor names must be unique")
        object.__setattr__(
            self,
            "named_factors",
            tuple(named_factors),
        )

        counts = tuple(self.access_counts)
        if counts and len(counts) != self.mapping.num_levels:
            raise ValueError(
                "access-count levels must match the mapping hierarchy"
            )
        object.__setattr__(self, "access_counts", counts)
        if self.access_events is not None:
            events = tuple(self.access_events)
            if len(events) != self.mapping.num_levels:
                raise ValueError(
                    "access-event levels must match the mapping hierarchy"
                )
            event_totals = tuple(level.totals for level in events)
            if counts and event_totals != counts:
                raise ValueError("access-event totals must match access_counts")
            if not counts:
                object.__setattr__(self, "access_counts", event_totals)
            object.__setattr__(self, "access_events", events)

    def factor(self, name: str, default=None):
        """Return one target-named schedule factor."""
        return next(
            (
                value
                for factor_name, value in self.named_factors
                if factor_name == name
            ),
            default,
        )


__all__ = [
    "GemmTilingCandidateCheck",
    "GemmWorkload",
    "LevelAccessEvents",
    "LevelAccessCounts",
    "LocalScheduleResult",
    "Loop",
    "MappingPoint",
    "NUM_LOOPS",
    "OperationClassification",
    "OperationOperand",
    "OperationWorkload",
    "OperandAccessEvents",
    "SchedulerProviderSet",
]
