"""Target-neutral repeated operation timing, independent of candidate enumeration.

Nodes are operation templates, not encoded instructions. Edges carry readiness
milestones and iteration distances. Resource scheduling is dependency-ready ASAP;
this is an analytical availability model, not a claim about backend dispatch.
"""

from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Dependency:
    source: int
    distance: int = 0
    milestone: str = "result"

    def __post_init__(self):
        if (
            type(self.source) is not int
            or self.source < 0
            or type(self.distance) is not int
            or self.distance < 0
        ):
            raise ValueError(
                "Dependencies need nonnegative source and iteration distance"
            )
        if self.milestone not in ("result", "read", "forward", "issue"):
            raise ValueError("Unknown readiness milestone")


@dataclass(frozen=True)
class OperationEvent:
    name: str
    resource: str
    issue_ns: float
    occupancy_ns: float
    latency_ns: float | None = None
    read_ns: float | None = None
    forward_ns: float | None = None
    dependencies: tuple[Dependency, ...] = ()
    period: int = 1
    phase: int = 0
    except_phase: bool = False
    implementation: str = ""

    def __post_init__(self):
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if not self.name or not self.resource:
            raise ValueError("An event needs a name and resource")
        for value in (
            self.issue_ns,
            self.occupancy_ns,
            self.latency_ns,
            self.read_ns,
            self.forward_ns,
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Timing must be finite and nonnegative")
        if (
            type(self.period) is not int
            or self.period < 1
            or type(self.phase) is not int
            or not 0 <= self.phase < self.period
        ):
            raise ValueError("Invalid event repetition condition")
        if self.except_phase and self.period == 1:
            raise ValueError("An event cannot be permanently inactive")

    def active(self, iteration):
        return (iteration % self.period == self.phase) != self.except_phase

    def offset(self, milestone):
        minimum = max(self.issue_ns, self.occupancy_ns)
        if milestone == "issue":
            return self.issue_ns
        if milestone == "result":
            return minimum if self.latency_ns is None else self.latency_ns
        value = self.read_ns if milestone == "read" else self.forward_ns
        # Conservative relative to the supplied completion model, not a
        # fabricated early-read/accumulator-forwarding guarantee.
        return self.offset("result") if value is None else value


@dataclass(frozen=True)
class RepeatedGraph:
    nodes: tuple[OperationEvent, ...]
    repetitions: int = 1

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("A graph needs a positive repetition count")
        if len({n.name for n in self.nodes}) != len(self.nodes):
            raise ValueError("Event names must be unique")
        for i, node in enumerate(self.nodes):
            for edge in node.dependencies:
                if edge.source >= len(self.nodes) or (
                    edge.distance == 0 and edge.source >= i
                ):
                    raise ValueError(
                        "Same-iteration dependencies must be topologically ordered"
                    )


@dataclass(frozen=True)
class ExecutionPlan:
    """Selected implementation and dependency contract; no physical addresses."""

    graph: RepeatedGraph
    implementations: tuple[str, ...] = ()
    instruction_counts: tuple[tuple[str, int], ...] = ()
    buffer_slots: tuple[tuple[str, int], ...] = ()
    scope: str = ""


@dataclass(frozen=True)
class GraphTiming:
    duration_ns: float
    service_ns: tuple[tuple[str, float], ...]
    iteration_finishes_ns: tuple[float, ...]
    unknown_latency: tuple[str, ...]
    resource_bound_ns: float

    @property
    def last_period_ns(self):
        if len(self.iteration_finishes_ns) < 2:
            return None
        return self.iteration_finishes_ns[-1] - self.iteration_finishes_ns[-2]


def evaluate_graph(graph, *, event_timing=None):
    """Schedule finite repetitions exactly under the declared ASAP abstraction.

    Periodic nodes express retained loads and final reduction stores. A
    dependency binds to the last active producer at or before (iteration -
    distance). Negative iterations are boundary values, hence have no edge.
    No steady-state extrapolation or finite-queue assumption is hidden here.
    """
    nodes = graph.nodes
    instances, indices, previous = [], {}, {}
    for iteration in range(graph.repetitions):
        for j, node in enumerate(nodes):
            if node.active(iteration):
                index = len(instances)
                indices[iteration, j] = index
                previous[j] = index
                instances.append((iteration, j))
            # Cache the latest producer even when this node is inactive.
            indices[iteration, j] = previous.get(j)
    count = len(instances)
    successors = [[] for _ in range(count)]
    pending = [0] * count
    ready_at = [0.0] * count
    for dest, (iteration, j) in enumerate(instances):
        for edge in nodes[j].dependencies:
            source = indices.get((iteration - edge.distance, edge.source))
            if source is None:
                continue
            successors[source].append((dest, edge.milestone))
            pending[dest] += 1
    # One ready queue per exclusive resource avoids repeatedly requeuing every
    # ready command each time that resource advances. Preserve the original
    # global (ready time, instance index) ordering exactly.
    future, eligible = {}, {}
    for i, n in enumerate(pending):
        resource = nodes[instances[i][1]].resource
        future.setdefault(resource, [])
        eligible.setdefault(resource, [])
        if n == 0:
            heapq.heappush(future[resource], (0.0, i))
    available, service = {}, {}
    finishes = [0.0] * graph.repetitions
    completed = 0
    while completed < count:
        candidates = []
        for resource, waiting in future.items():
            now = available.get(resource, 0.0)
            runnable = eligible[resource]
            while waiting and waiting[0][0] <= now:
                _, candidate = heapq.heappop(waiting)
                heapq.heappush(runnable, candidate)
            if runnable:
                candidates.append((now, runnable[0], resource, True))
            elif waiting:
                when, candidate = waiting[0]
                candidates.append((when, candidate, resource, False))
        if not candidates:
            break
        start, index, resource, is_eligible = min(candidates)
        if is_eligible:
            heapq.heappop(eligible[resource])
        else:
            heapq.heappop(future[resource])
        iteration, j = instances[index]
        node = nodes[j]
        if event_timing is not None:
            timed = event_timing(node, start)
            if (timed.name, timed.resource, timed.dependencies) != (
                node.name,
                node.resource,
                node.dependencies,
            ):
                raise ValueError(
                    "Dynamic timing may not change graph identity or dependencies"
                )
            node = timed
        available[node.resource] = start + max(node.issue_ns, node.occupancy_ns)
        service[node.resource] = (
            service.get(node.resource, 0) + node.occupancy_ns
        )
        finishes[iteration] = max(
            finishes[iteration], start + node.offset("result")
        )
        for dest, milestone in successors[index]:
            ready_at[dest] = max(ready_at[dest], start + node.offset(milestone))
            pending[dest] -= 1
            if not pending[dest]:
                heapq.heappush(
                    future[nodes[instances[dest][1]].resource],
                    (ready_at[dest], dest),
                )
        completed += 1
    if completed != count:
        raise ValueError("Cyclic execution dependencies")
    return GraphTiming(
        max(finishes, default=0),
        tuple(sorted(service.items())),
        tuple(finishes),
        tuple(node.name for node in nodes if node.latency_ns is None),
        max(service.values(), default=0),
    )
