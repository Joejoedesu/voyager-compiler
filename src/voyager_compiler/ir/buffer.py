"""The executable buffer graph is authoritative; dumps are projections of it."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class TargetInvoke:
    """A target instruction/macro body and its observable execution contract.

    Native allocation and hardware scheduling are independent authorities.
    The target's registered builder/encoder must implement this body; declaring
    it does not grant permission to emit an unmodeled instruction.
    """

    implementation: str
    opcode: str
    operands: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    resources: tuple[str, ...] = ()
    completion: str = "synchronous"
    allocation_owner: str = "compiler"
    scheduling_owner: str = "compiler"
    implicit_traffic_bytes: int = 0
    native_requirements: tuple = ()

    def __post_init__(self):
        if self.allocation_owner not in (
            "compiler",
            "native",
        ) or self.scheduling_owner not in ("compiler", "hardware"):
            raise ValueError("unknown target invocation authority")
        if self.completion != "synchronous":
            raise ValueError(
                "asynchronous native invocations require an explicit completion token"
            )
        if self.allocation_owner == "native" and not self.native_requirements:
            raise ValueError(
                "native invocations must declare private resource requirements"
            )


@dataclass
class BufferProgram:
    graph: Any
    plan: Any
    bindings: dict[str, str] = field(default_factory=dict)
    frozen_digest: str | None = None
    plan_digest: str | None = None
    validation: dict = field(default_factory=dict)

    def freeze(self, allocation):
        from voyager_compiler.artifacts import content_digest
        from voyager_compiler.compiler.analysis import graph_digest

        self.plan.freeze(allocation)
        self.bindings = MappingProxyType(dict(self.bindings))
        self.validation = MappingProxyType(dict(self.validation))
        self.frozen_digest = graph_digest(self.graph)
        self.plan_digest = content_digest(
            (self.plan, self.bindings, self.validation)
        )

    def assert_frozen(self):
        from voyager_compiler.artifacts import content_digest
        from voyager_compiler.compiler.analysis import graph_digest

        if (
            not self.plan.frozen
            or graph_digest(self.graph) != self.frozen_digest
            or content_digest((self.plan, self.bindings, self.validation))
            != self.plan_digest
        ):
            raise ValueError(
                "requires an unchanged frozen buffer program, allocation and selected plan"
            )
