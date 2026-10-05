"""DMA service estimates, separating shared bus work from endpoint occupancy."""

from dataclasses import dataclass
from collections import defaultdict

from voyager_compiler.codegen.transform.tiling.cost import _step_classes


@dataclass(frozen=True)
class DMAService:
    payload_cycles: float
    command_cycles: float
    endpoint_cycles: float
    latency_cycles: float
    bandwidth_resource: str
    service_resource: str

    @property
    def cycles(self):
        return (
            self.latency_cycles
            + self.command_cycles
            + max(self.payload_cycles, self.endpoint_cycles)
        )


def dma_service(config, connection, size, commands):
    """Price one contiguous stream, with pipeline latency charged only once.

    Connection.startup_ns is non-overlapped overhead per expanded DMA command.
    Endpoint service may instead scale with bytes (e.g. accumulator writeback).
    It cannot be charged to other traffic sharing the external bus.
    """
    edge = config.connection(connection)
    payload = size / config.connection_bytes_per_cycle(connection)
    endpoint = (
        size / edge.service_bandwidth.bytes_per_cycle(config)
        if edge.service_bandwidth is not None
        else payload
    )
    return DMAService(
        payload,
        commands * edge.startup_ns * config.frequency,
        endpoint,
        (
            (
                (
                    config.bandwidth_connection(connection).latency_ns
                    if edge.latency_ns is None
                    else edge.latency_ns
                )
                or 0
            )
            * config.frequency
            if commands
            else 0
        ),
        config.bandwidth_connection(connection).name,
        edge.service_resource or edge.name,
    )


def concurrent_dma_cycles(loads, store=None):
    """Bound concurrent transfer service using the IR's resource identities.

    Traffic sharing a bus or controller accumulates demand on that resource;
    independent resources overlap. The schedule handles dependencies and the
    command submission window separately.
    """
    # Unrelated links/controllers can overlap; shared resources sum demand.
    # Latency remains a conservative stream-fill charge per shared bus.
    work = list(loads) + ([store] if store is not None else [])
    bus, endpoint = defaultdict(float), defaultdict(float)
    for transfer in work:
        bus[transfer.bandwidth_resource] += (
            transfer.payload_cycles + transfer.latency_cycles
        )
        endpoint[transfer.service_resource] += transfer.cycles
    return max([0, *bus.values(), *endpoint.values()])


def dma_sweep_cycles(loads, store, store_steps, steps, compute):
    """Double-buffered service sweep with distinct bus and endpoint budgets.

    Preserve the shared builder's load recurrence and exposed first/last tile.
    Within a steady step, independent engines can overlap, but transfers must
    collectively fit the shared bandwidth. Submission delays are separate.
    """
    transfers = [(1, store_steps)] + [
        (1 << (i + 1), count) for i, (_, count) in enumerate(loads)
    ]

    def service(mask):
        return concurrent_dma_cycles(
            [
                load
                for i, (load, _) in enumerate(loads)
                if mask & (1 << (i + 1))
            ],
            store if mask & 1 else None,
        )

    all_loads = sum(bit for bit, _ in transfers[1:])
    prologue = service(all_loads)
    if steps == 1:
        return prologue + compute + store.cycles
    total = sum(
        count * max(service(int(mask)), compute)
        for mask, count in _step_classes(transfers, steps)
    )
    recurring = sum(bit for bit, count in transfers if count == steps)
    return (
        total
        - max(service(all_loads | 1), compute)
        - max(service(recurring), compute)
        + prologue
        + max(service(recurring & ~1), compute)
        + max(store.cycles, compute)
        + store.cycles
    )
