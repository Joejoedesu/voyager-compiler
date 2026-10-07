"""Shared panel-level DMA/layout implementations for search and realization."""

from dataclasses import dataclass
from functools import lru_cache
from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    OperationEvent,
    RepeatedGraph,
)


@dataclass(frozen=True)
class TransferPanel:
    row: int
    column: int
    partitions: int
    free: int
    bits: int
    store: bool
    transpose: bool
    direct: bool
    ideal_payload_ns: float
    tensor_ns: float = 0
    vector_ns: float = 0
    scalar_ns: float = 0

    @property
    def bytes(self):
        return self.partitions * self.free * self.bits // 8

    @property
    def key(self):
        return (
            self.row,
            self.column,
            self.partitions,
            self.free,
            self.bits,
            self.store,
            self.transpose,
            self.direct,
        )


def partition_payload_ns(config, partitions, free, bits):
    """Payload service on the busiest of 16 eight-partition DMA engines."""
    return min(8, partitions) * free * (bits / 8) / (config.dram_bandwidth / 16)


@lru_cache(maxsize=2048)
def transfer_graph(config, panels):
    """One request and payload event per emitted transfer, with explicit layout.

    Issue occupies DMAIssue. Dispatch latency does not monopolize that resource.
    Payload occupies DMA for its bandwidth demand; observed payload duration is a pipelined completion latency, including endpoint notification. Layout
    consumes TensorE and a copy engine. These engines can overlap across panels.
    """
    nodes = []
    profile = config.timing_profile

    def add(name, resource, service, latency, deps=(), implementation=""):
        i = len(nodes)
        nodes.append(
            OperationEvent(
                name,
                resource,
                service,
                service,
                latency,
                dependencies=tuple(Dependency(x) for x in deps),
                implementation=implementation,
            )
        )
        return i

    for i, p in enumerate(panels):
        dtype = "float32" if p.bits == 32 else "bfloat16"
        previous = None

        def layout(previous):
            if p.tensor_ns:
                from .dependencies import engine_clock
                previous = add(f"panel_{i}_psum_clear", "VectorE",
                               max(64, p.partitions) / engine_clock(config, "VectorE"), None,
                               () if previous is None else (previous,), "nki.isa.memset.VectorE")
            for engine, service in [
                ("TensorE", p.tensor_ns),
                ("VectorE", p.vector_ns),
                ("ScalarE", p.scalar_ns),
            ]:
                if not service:
                    continue
                impl = (
                    f"nki.transpose_copy.{dtype}.ScalarE"
                    if engine == "TensorE"
                    else f"nki.copy.PSUM.{dtype}.{engine}"
                )
                law = profile.operation(impl)
                occupancy, latency = (
                    law.evaluate(service) if law else (service, None)
                )
                previous = add(
                    f"panel_{i}_{engine}",
                    engine,
                    occupancy,
                    latency,
                    () if previous is None else (previous,),
                    impl,
                )
            return previous

        if p.store:
            previous = layout(previous)
        law = profile.store if p.store else profile.load
        issue = law.issue_ns if law else p.ideal_payload_ns
        dispatch = law.dispatch_ns if law else 0
        request = add(
            f"panel_{i}_request",
            "DMAIssue",
            issue,
            dispatch,
            () if previous is None else (previous,),
        )
        payload = (
            law.payload(p.ideal_payload_ns) if law else p.ideal_payload_ns
        )
        completion = payload + (
            law.notification_ns if law else config.dram_access_latency
        )
        previous = add(
            f"panel_{i}_payload",
            "DMA",
            p.ideal_payload_ns,
            completion,
            (request,),
        )
        if not p.store:
            previous = layout(previous)
    return RepeatedGraph(tuple(nodes))
