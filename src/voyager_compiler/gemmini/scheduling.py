"""Lower existing async dependencies to a bounded-window RoCC submission order.

No tiles, transfers or arithmetic are created here. Tasks are expansions of
existing protobuf operations. Semaphore edges and physical RAW/WAR/WAW edges
constrain their order; only independent DMA can pass an execute task.
"""

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubmissionPolicy:
    """Tunable compiler heuristic; deliberately separate from hardware queues."""

    execute_quantum: int = 32
    load_quantum: int = 4
    store_quantum: int = 4

    def __post_init__(self):
        for value in (
            self.execute_quantum,
            self.load_quantum,
            self.store_quantum,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError("Submission quanta must be positive integers")
        if self.execute_quantum % 2:
            raise ValueError(
                "Execute quantum must preserve preload/compute pairs"
            )


@dataclass(frozen=True)
class QueueGeometry:
    input_fifo: int
    reservation_load: int
    reservation_execute: int
    reservation_store: int
    controller_load: int
    controller_execute: int
    controller_store: int

    @classmethod
    def from_hardware(cls, config):
        names = (
            "reservation_load",
            "reservation_execute",
            "reservation_store",
            "controller_load",
            "controller_execute",
            "controller_store",
        )
        return cls(
            input_fifo=config.memory_instance("instruction_queue").size.value,
            **{n: config.memory_instance(n).size.value for n in names},
        )


from .hardware import lean_config

LEAN_QUEUES = QueueGeometry.from_hardware(lean_config())


def funct(command):
    return int(command["instruction"], 16) >> 25


@dataclass
class Task:
    index: int
    name: str
    start: int
    end: int = 0
    dependencies: set = field(default_factory=set)
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    engine: str = ""


def merge_ranges(ranges):
    result = []
    for space, start, end in sorted(set(ranges)):
        if result and result[-1][0] == space and start <= result[-1][2]:
            result[-1] = space, result[-1][1], max(end, result[-1][2])
        else:
            result.append((space, start, end))
    return result


def overlap(left, right):
    return any(
        a == b and x < v and u < y for a, x, y in left for b, u, v in right
    )


class Footprints:
    """Conservative physical ranges, decoded before instruction reordering."""

    def __init__(self):
        self.loads = {}
        self.store_stride = None
        self.a_stride = 1
        self.weight = []
        self.destination = []

    @staticmethod
    def local(value, stride=1, block_stride=None):
        address = value & 0xFFFFFFFF
        if address == 0xFFFFFFFF:
            return []
        rows, cols = value >> 48, (value >> 32) & 0xFFFF
        space = "acc" if address & 0x80000000 else "sp"
        start = address & 0x1FFFFFFF
        if not rows:  # pooled mvout: conservatively retain the entire ACC
            return [(space, 0, 1024 if space == "acc" else 16384)]
        blocks = (cols + 15) // 16
        end = (
            start
            + (blocks - 1) * (block_stride or rows)
            + (rows - 1) * stride
            + 1
        )
        return [(space, start, end)]

    def analyze(self, task, commands):
        engines = set()
        for cmd in commands:
            if cmd["type"] != "command":
                raise ValueError("Unexpected control event inside async task")
            f = funct(cmd)
            a, b = int(cmd["rs1"], 16), int(cmd["rs2"], 16)
            if f == 0:
                kind = a & 3
                if kind == 0:
                    self.a_stride = (a >> 16) & 0xFFFF
                elif kind == 1:
                    self.loads[(a >> 3) & 3] = (
                        b,
                        (a >> 16) & 0xFFFF,
                        bool(a & 4),
                    )
                elif kind == 2:
                    self.store_stride = b & 0xFFFFFFFF
                else:
                    raise ValueError(
                        "Unsupported configuration in async scheduler"
                    )
            elif f in (1, 2, 14):
                engines.add("load")
                index = {2: 0, 1: 1, 14: 2}[f]
                if index not in self.loads:
                    raise ValueError("DMA load without a configuration")
                stride, block_stride, shrunk = self.loads[index]
                task.writes += self.local(b, block_stride=block_stride)
                rows, cols = b >> 48, (b >> 32) & 0xFFFF
                item = 4 if (b & 0x80000000) and not shrunk else 1
                if a:
                    task.reads.append(
                        ("dram", a, a + (rows - 1) * stride + cols * item)
                    )
            elif f == 6:
                engines.add("execute")
                if a & 0xFFFFFFFF != 0xFFFFFFFF:
                    self.weight = self.local(a)
                self.destination = self.local(b)
                task.reads += self.weight
                task.writes += self.destination
            elif f in (4, 5):
                engines.add("execute")
                task.reads += self.local(a, stride=self.a_stride) + self.weight
                task.writes += self.destination
            elif f == 3:
                engines.add("store")
                if self.store_stride is None:
                    raise ValueError("DMA store without a configuration")
                task.reads += self.local(b)
                rows, cols = b >> 48, (b >> 32) & 0xFFFF
                task.writes.append(
                    ("dram", a, a + (rows - 1) * self.store_stride + cols)
                    if rows
                    else ("dram", 0, 1 << 64)
                )
            else:
                raise ValueError(
                    f"Unsupported instruction {f} in async scheduler"
                )
        task.reads = merge_ranges(task.reads)
        task.writes = merge_ranges(task.writes)
        task.engine = (
            "execute"
            if "execute" in engines
            else "store" if "store" in engines else "load"
        )


class AsyncSchedule:
    def __init__(self, commands, queues=LEAN_QUEUES, policy=None):
        self.commands = commands
        self.queues = queues
        self.policy = policy or SubmissionPolicy()
        self.tasks = []
        self.current = None
        self.barrier = set()
        self.tokens = defaultdict(deque)

    @contextmanager
    def task(self, name):
        if self.current is not None:
            yield
            return
        task = Task(
            len(self.tasks),
            name,
            len(self.commands),
            dependencies=set(self.barrier),
        )
        self.tasks.append(task)
        self.current = task
        try:
            yield
        finally:
            task.end = len(self.commands)
            self.current = None

    def seed(self, key, count):
        if count:
            self.tokens[key].append([None, count])

    def signal(self, key, count):
        self.tokens[key].append(
            [self.current.index if self.current else None, count]
        )

    def wait(self, key):
        tokens = self.tokens[key]
        if not tokens:
            raise ValueError(f"Missing semaphore producer: {key}")
        producer, count = tokens[0]
        if producer is not None:
            deps = self.current.dependencies if self.current else self.barrier
            if self.current is None or producer != self.current.index:
                deps.add(producer)
        if count == 1:
            tokens.popleft()
        else:
            tokens[0][1] -= 1

    def reorder(self):
        if not self.tasks:
            return self.commands, {}
        decoder = Footprints()
        # Initial execution configuration precedes all tasks.
        decoder.analyze(
            Task(-1, "initial", 0), self.commands[1 : self.tasks[0].start]
        )
        last_execute = None
        for task in self.tasks:
            decoder.analyze(task, self.commands[task.start : task.end])
            if task.engine == "execute":
                if last_execute is not None:
                    task.dependencies.add(last_execute)
                last_execute = task.index
            for previous in self.tasks[: task.index]:
                if overlap(
                    previous.writes, task.reads + task.writes
                ) or overlap(previous.reads, task.writes):
                    task.dependencies.add(previous.index)
        result = self.commands[: self.tasks[0].start]
        done, pending = set(), set(range(len(self.tasks)))
        emitted = []

        def ready():
            return [
                self.tasks[i]
                for i in sorted(pending)
                if self.tasks[i].dependencies <= done
            ]

        progress = {t.index: t.start for t in self.tasks}
        chunks = defaultdict(int)

        def emit_chunk(task, bounded=True):
            start, end = progress[task.index], task.end
            count = 0
            quantum = {
                "execute": self.policy.execute_quantum,
                "load": self.policy.load_quantum,
                "store": self.policy.store_quantum,
            }[task.engine]
            if bounded:
                for i in range(start, end):
                    f = funct(self.commands[i])
                    if task.engine == "execute":
                        count += f in (4, 5, 6)
                        boundary = f in (4, 5)
                    else:
                        boundary = f in (
                            (1, 2, 14) if task.engine == "load" else (3,)
                        )
                        count += boundary
                    if boundary and count >= quantum:
                        end = i + 1
                        break
            # Each DMA expansion carries its own configuration. Keep that
            # configuration with its data command; deduplicate only afterwards.
            result.extend(self.commands[start:end])
            progress[task.index] = end
            chunks[task.engine] += 1
            if end == task.end:
                emitted.append(task.index)
                done.add(task.index)
                pending.remove(task.index)

        while pending:
            available = ready()
            if not available:
                raise ValueError("Cycle in collateral async dependency graph")
            executes = [t for t in available if t.engine == "execute"]
            if not executes:
                emit_chunk(
                    min(available, key=lambda t: (t.engine != "store", t.index))
                )
                continue
            task = executes[0]
            stores = [t for t in available if t.engine == "store"]
            if stores:
                emit_chunk(stores[0])
            next_engine = "load"
            while task.index in pending:
                transfers = [t for t in ready() if t.engine != "execute"]
                emit_chunk(task, bounded=bool(transfers))
                # Limit the DMA burst too: a full LD queue can otherwise block
                # the remaining EX submissions behind a large prefetch task.
                if task.index in pending and transfers:
                    transfer = min(
                        transfers,
                        key=lambda t: (t.engine != next_engine, t.index),
                    )
                    emit_chunk(transfer)
                    next_engine = (
                        "store" if transfer.engine == "load" else "load"
                    )
        result.extend(self.commands[self.tasks[-1].end :])
        if len(result) != len(self.commands):
            raise ValueError("Async lowering lost instructions")
        return result, dict(
            tasks=len(self.tasks),
            tasks_completed_out_of_source_order=sum(
                i != t for i, t in enumerate(emitted)
            ),
            execute_quantum=self.policy.execute_quantum,
            load_quantum=self.policy.load_quantum,
            store_quantum=self.policy.store_quantum,
            execute_chunks=chunks["execute"],
            load_chunks=chunks["load"],
            store_chunks=chunks["store"],
        )
