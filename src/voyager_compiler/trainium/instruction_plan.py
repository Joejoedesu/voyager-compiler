"""Selected ISA program, independent of NKI source text.

The expression vocabulary only describes static address arithmetic and views.
Instructions carry their selected opcode, operands, destination, engine and
completion edges. Allocation uses the shared best-fit lifetime allocator.
"""

from dataclasses import asdict, dataclass, field
import ast
import math
import operator

import numpy as np

BITS = dict(float32=32, bfloat16=16, float16=16, int64=64, uint32=32, uint8=8)
OPS = {
    "Add": operator.add,
    "Sub": operator.sub,
    "Mult": operator.mul,
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "Div": operator.truediv,
    "BitAnd": operator.and_,
    "BitOr": operator.or_,
    "Lt": operator.lt,
    "LtE": operator.le,
    "Gt": operator.gt,
    "GtE": operator.ge,
    "Eq": operator.eq,
    "NotEq": operator.ne,
}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: object = None

    def names(self):
        return ({self.value} if self.op == "name" else set()).union(
            *(x.names() for x in self.args)
        )

    @classmethod
    def parse(cls, node):
        if isinstance(node, str):
            node = ast.parse(node, mode="eval").body
        if isinstance(node, ast.Constant):
            return cls(
                "ellipsis" if node.value is Ellipsis else "literal",
                value=None if node.value is Ellipsis else node.value,
            )
        if isinstance(node, ast.Name):
            return cls("name", value=node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            return cls(
                "tuple" if isinstance(node, ast.Tuple) else "list",
                tuple(cls.parse(x) for x in node.elts),
            )
        if isinstance(node, ast.Attribute):
            return cls("attr", (cls.parse(node.value),), node.attr)
        if isinstance(node, ast.Subscript):
            return cls("index", (cls.parse(node.value), cls.parse(node.slice)))
        if isinstance(node, ast.Slice):
            return cls(
                "slice",
                tuple(
                    cls.parse(x) if x else cls("literal")
                    for x in (node.lower, node.upper, node.step)
                ),
            )
        if isinstance(node, ast.BinOp):
            return cls(
                type(node.op).__name__,
                (cls.parse(node.left), cls.parse(node.right)),
            )
        if isinstance(node, ast.UnaryOp):
            return cls(type(node.op).__name__, (cls.parse(node.operand),))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return cls(
                type(node.ops[0]).__name__,
                (cls.parse(node.left), cls.parse(node.comparators[0])),
            )
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError(
                    "Address expressions cannot contain keyword calls"
                )
            return cls(
                "call", tuple(cls.parse(x) for x in (node.func, *node.args))
            )
        raise ValueError(
            f"Unsupported static address expression: {ast.dump(node)}"
        )

    @classmethod
    def load(cls, record):
        return cls(
            record["op"],
            tuple(cls.load(x) for x in record["args"]),
            record["value"],
        )


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    dtype: str
    memory: str
    layout: str
    alias: str = ""
    view: Expr | None = None
    constant: str = ""


@dataclass(frozen=True)
class Instruction:
    opcode: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]
    destination: Expr
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    dependencies: tuple[int, ...]
    implementation: str
    accumulate: bool = False


@dataclass(frozen=True)
class Placement:
    memory: str
    start_partition: int
    byte_address: int
    bank: int | None
    bytes_per_partition: int
    first: int
    last: int


@dataclass
class Program:
    tensors: dict[str, Tensor]
    indices: dict[str, Expr]
    instructions: list[Instruction]
    arguments: tuple[str, ...]
    outputs: tuple[str, ...]
    placements: dict[str, Placement] = field(default_factory=dict)
    contracts: dict = field(default_factory=dict)
    version: int = 1
    repeated_regions: list[dict] = field(default_factory=list)
    encoding_loops: list[dict] = field(default_factory=list)
    encoding_storage: str = "arena"
    storage_regions: list[dict] = field(default_factory=list)

    def root(self, name):
        seen = set()
        while self.tensors[name].alias:
            if name in seen:
                raise ValueError(f"Cyclic tensor alias {name}")
            seen.add(name)
            name = self.tensors[name].alias
        return name

    def lifetimes(self):
        spans = {}
        for i, ins in enumerate(self.instructions):
            for name in ins.reads + ins.writes:
                root = self.root(name)
                if self.tensors[root].memory in ("SBUF", "PSUM"):
                    first, _ = spans.get(root, (i, i))
                    spans[root] = first, i
        return spans

    def allocate(
        self, sbuf_bytes=28 << 20, psum_banks=8, *, strategy="best_fit"
    ):
        if strategy not in ("best_fit", "size_classes"):
            raise ValueError("Unknown physical placement strategy")
        if self.placements:
            raise ValueError(
                "Select allocation from an unplaced instruction plan"
            )
        from voyager_compiler.codegen.transform.bufferize.memory_planning import (
            _greedy_best_fit,
        )

        spans = self.lifetimes()
        items, psums = [], []
        for name, (first, last) in spans.items():
            t = self.tensors[name]
            size = (
                math.ceil(math.prod(t.shape[1:]) * BITS[t.dtype] / 8 / 16) * 16
            )
            if t.memory == "PSUM":
                if size > 2048:
                    raise ValueError(
                        f"{name}: PSUM tile exceeds 2 KiB per partition"
                    )
                psums.append((name, 1, first, last, 1))
            else:
                items.append((name, size, first, last, 16))
        if strategy == "best_fit":
            bases, total = _greedy_best_fit(items)
        else:
            # Keep every reused range a whole slot, avoiding partial overlap
            # across differently sized generations. Each class still uses the
            # shared lifetime allocator; the total must fit physical capacity.
            bases, total = {}, 0
            for size in sorted({item[1] for item in items}, reverse=True):
                offsets, extent = _greedy_best_fit(
                    [x for x in items if x[1] == size]
                )
                bases.update(
                    {name: total + offset for name, offset in offsets.items()}
                )
                total += extent
        # PSUM banks are a concurrency resource. Prefer the least recently
        # occupied legal bank rather than reusing bank zero at every source-
        # order lifetime boundary, which serializes TensorE with its readers.
        available = [-1] * psum_banks
        banks = {}
        for name, _, first, last, _ in sorted(psums, key=lambda x: x[2]):
            legal = [i for i, end in enumerate(available) if end < first]
            if not legal:
                raise ValueError("Selected ISA plan exceeds PSUM bank capacity")
            bank = min(legal, key=lambda i: (available[i], i))
            banks[name] = bank
            available[bank] = last
        count = max(banks.values(), default=-1) + 1
        if total * 128 > sbuf_bytes or count > psum_banks:
            raise ValueError(
                f"Selected ISA plan does not fit: SBUF={total * 128}, PSUM banks={count}"
            )
        for name, size, first, last, _ in items + psums:
            memory = self.tensors[name].memory
            self.placements[name] = Placement(
                memory,
                0,
                bases.get(name, 0),
                banks.get(name),
                size if memory == "SBUF" else 2048,
                first,
                last,
            )
        # Completion edges chain successive owners of every address range.
        # Earlier generations are covered transitively rather than adding a
        # quadratic number of redundant edges across repeated regions.
        extra = {}
        for previous, current in self.reuse_edges():
            extra.setdefault(current, set()).add(previous)
        from dataclasses import replace

        self.instructions = [
            replace(
                ins,
                dependencies=tuple(
                    sorted(set(ins.dependencies) | extra.get(i, set()))
                ),
            )
            for i, ins in enumerate(self.instructions)
        ]
        self.validate(sbuf_bytes, psum_banks)

    def bind_disjoint_regions(self):
        """Select separate names only for physically disjoint address unions.

        Every overlapping placement stays in the same arena, including aliases
        across generations. Addresses, instructions and completion edges remain
        unchanged. This limits backend alias analysis without hiding reuse.
        """
        ranges = sorted(
            (p.byte_address, p.byte_address + p.bytes_per_partition)
            for p in self.placements.values()
            if p.memory == "SBUF"
        )
        regions = []
        for lo, hi in ranges:
            if not regions or lo >= regions[-1]["stop"]:
                regions.append(dict(start=lo, stop=hi))
            else:
                regions[-1]["stop"] = max(regions[-1]["stop"], hi)
        self.encoding_storage = "disjoint_arenas"
        self.storage_regions = regions

    @staticmethod
    def overlap(a, b):
        return a.memory == b.memory and (
            a.bank == b.bank
            if a.memory == "PSUM"
            else a.byte_address < b.byte_address + b.bytes_per_partition
            and b.byte_address < a.byte_address + a.bytes_per_partition
        )

    def reuse_edges(self):
        """Check inclusive lifetimes and yield completion edges at address reuse.

        Each disjoint address interval remembers its most recent owner. Partial
        overwrites split intervals, preserving dependencies on uncovered bytes.
        PSUM banks are indivisible in this pinned direct-allocation ABI.
        """
        # Source-order last use does not imply completion of independent
        # earlier readers on another engine. Retire every outstanding reader,
        # pruning only dependencies that explicitly cover an older access.
        frontier = {}
        for index, instruction in enumerate(self.instructions):
            writes = {self.root(n) for n in instruction.writes}
            for name in {
                self.root(n) for n in instruction.reads + instruction.writes
            }:
                active = frontier.setdefault(name, set())
                if name in writes:
                    active.clear()  # Logical WAR/WAW edges are validated below.
                else:
                    active.difference_update(instruction.dependencies)
                active.add(index)
        owners = {"SBUF": [], "PSUM": []}
        for name, p in sorted(
            self.placements.items(), key=lambda item: item[1].first
        ):
            lo = p.bank if p.memory == "PSUM" else p.byte_address
            hi = lo + (1 if p.memory == "PSUM" else p.bytes_per_partition)
            updated = []
            for a, b, old_name, previous in owners[p.memory]:
                if a >= hi or b <= lo:
                    updated.append((a, b, old_name, previous))
                    continue
                if previous.last >= p.first:
                    raise ValueError(
                        f"{name}/{old_name}: live physical allocations overlap"
                    )
                for predecessor in frontier.get(old_name, {previous.last}):
                    yield predecessor, p.first
                if a < lo:
                    updated.append((a, lo, old_name, previous))
                if b > hi:
                    updated.append((hi, b, old_name, previous))
            updated.append((lo, hi, name, p))
            owners[p.memory] = updated

    def validate(self, sbuf_bytes=28 << 20, psum_banks=8):
        if self.encoding_storage not in ("arena", "disjoint_arenas"):
            raise ValueError("Unknown selected physical storage encoding")
        if self.encoding_storage == "disjoint_arenas":
            end = 0
            for region in self.storage_regions:
                lo, hi = region["start"], region["stop"]
                if (
                    lo < end
                    or lo % 16
                    or hi % 16
                    or hi <= lo
                    or hi * 128 > sbuf_bytes
                ):
                    raise ValueError(
                        "Invalid or overlapping selected storage regions"
                    )
                end = hi
            for name, p in self.placements.items():
                if p.memory == "SBUF" and not any(
                    r["start"] <= p.byte_address
                    and p.byte_address + p.bytes_per_partition <= r["stop"]
                    for r in self.storage_regions
                ):
                    raise ValueError(
                        f"{name}: placement crosses selected storage regions"
                    )
        elif self.storage_regions:
            raise ValueError(
                "Single arena encoding has unexpected storage regions"
            )
        for region in self.repeated_regions:
            previous = None
            for first, stop in region["iterations"]:
                if not 0 <= first <= stop <= len(self.instructions) or (
                    previous is not None and first != previous
                ):
                    raise ValueError("Invalid repeated instruction region")
                previous = stop
        last_write, users = {}, {}
        checker = Builder.__new__(Builder)
        checker.program = self
        checker.constants = {}
        for name, e in self.indices.items():
            checker.constants[name] = checker.value(e)
        for name, t in self.tensors.items():
            if (
                t.memory not in ("HBM", "SBUF", "PSUM")
                or t.layout
                != ("contiguous" if t.memory == "HBM" else "partition_free")
                or t.dtype not in BITS
                or not t.shape
                or any(x <= 0 for x in t.shape)
            ):
                raise ValueError(f"{name}: missing or invalid typed layout")
            if t.memory in ("SBUF", "PSUM") and not 0 < t.shape[0] <= 128:
                raise ValueError(
                    f"{name}: invalid partition count {t.shape[0]}"
                )
            self.root(name)
            if t.alias:
                value = checker.value(t.view)
                if (value.shape, value.dtype, value.memory) != (
                    t.shape,
                    t.dtype,
                    t.memory,
                ):
                    raise ValueError(f"{name}: inconsistent typed view")
        for i, ins in enumerate(self.instructions):
            contract = self.contracts.get(ins.implementation)
            if not contract or ins.opcode != contract["opcode"]:
                raise ValueError(
                    f"Instruction {i}: missing explicit ISA implementation"
                )
            for name in ins.writes:
                if self.tensors[name].memory not in contract["result_memories"]:
                    raise ValueError(
                        f"Instruction {i}: illegal result memory for {ins.implementation}"
                    )
            writes = checker.tensor_names((ins.destination,))
            reads = checker.tensor_names(
                (*ins.args, *(v for _, v in ins.kwargs))
            )
            if ins.accumulate:
                reads = tuple(sorted(set(reads + writes)))
            if reads != ins.reads or writes != ins.writes:
                raise ValueError(
                    f"Instruction {i}: inconsistent typed operand bindings"
                )
            engine = dict(ins.kwargs).get("engine")
            if engine is not None:
                selected = {
                    "nisa.vector_engine": "VectorE",
                    "nisa.scalar_engine": "ScalarE",
                    "nisa.tensor_engine": "TensorE",
                }.get(checker.value(engine))
                if selected != contract["engine"]:
                    raise ValueError(
                        f"Instruction {i}: engine differs from selected implementation"
                    )
            if ins.opcode == "nisa.nc_transpose":
                src = checker.value(ins.args[0])
                dst = checker.value(ins.destination)
                if (
                    src.memory != "SBUF"
                    or dst.memory != "SBUF"
                    or src.dtype != dst.dtype
                    or src.dtype != "float32"
                    or src.shape != (32, 32)
                    or dst.shape != (32, 32)
                ):
                    raise ValueError(
                        "Selected stream transpose requires FP32 32x32 SBUF tiles"
                    )
            if ins.opcode == "nisa.nc_matmul":
                a, b = (checker.value(v) for v in ins.args)
                dst = checker.value(ins.destination)
                if (
                    a.memory != "SBUF"
                    or b.memory != "SBUF"
                    or a.shape[0] != b.shape[0]
                    or a.shape[1] > 128
                    or b.shape[1] > 512
                    or dst.shape != (a.shape[1], b.shape[1])
                ):
                    raise ValueError(
                        f"Instruction {i}: incompatible TensorE layout"
                    )
                transpose = dict(ins.kwargs).get("is_transpose")
                if (
                    transpose
                    and checker.value(transpose)
                    and dst.dtype != a.dtype
                ):
                    raise ValueError(
                        f"Instruction {i}: transpose must preserve operand dtype"
                    )
            if any(d < 0 or d >= i for d in ins.dependencies):
                raise ValueError(
                    f"Instruction {i}: invalid completion dependency"
                )
            required = set()
            for n in ins.reads:
                root = self.root(n)
                if root in last_write:
                    required.add(last_write[root])
                elif self.tensors[root].memory in ("SBUF", "PSUM"):
                    raise ValueError(
                        f"Instruction {i}: read of uninitialized {root}"
                    )
            for n in ins.writes:
                root = self.root(n)
                required.update(users.get(root, ()))
                if root in last_write:
                    required.add(last_write[root])
            if not required.issubset(ins.dependencies):
                raise ValueError(
                    f"Instruction {i}: missing operand/reuse completion dependency"
                )
            for n in ins.writes:
                root = self.root(n)
                last_write[root] = i
                users[root] = set()
            for n in ins.reads:
                users.setdefault(self.root(n), set()).add(i)
        spans = self.lifetimes()
        for name, (first, last) in spans.items():
            if name not in self.placements:
                raise ValueError(f"{name}: missing physical placement")
            p, t = self.placements[name], self.tensors[name]
            needed = math.prod(t.shape[1:]) * BITS[t.dtype] // 8
            if (
                (p.first, p.last) != (first, last)
                or p.memory != t.memory
                or p.bytes_per_partition < needed
            ):
                raise ValueError(f"{name}: inconsistent placement/lifetime")
            if (
                p.start_partition != 0
                or p.byte_address < 0
                or p.byte_address % 16
            ):
                raise ValueError(f"{name}: illegal physical address")
            if t.memory == "SBUF" and (
                p.bank is not None
                or (p.byte_address + p.bytes_per_partition) * 128 > sbuf_bytes
            ):
                raise ValueError(f"{name}: SBUF capacity exceeded")
            if t.memory == "PSUM" and (
                p.bank is None
                or not 0 <= p.bank < psum_banks
                or p.byte_address
                or p.bytes_per_partition > 2048
            ):
                raise ValueError(f"{name}: invalid PSUM bank")
        for previous, current in self.reuse_edges():
            if previous not in self.instructions[current].dependencies:
                raise ValueError(
                    f"Instructions {previous}/{current}: missing physical reuse completion"
                )

    def record(self):
        return asdict(self)

    @classmethod
    def load(cls, r):
        if r["version"] != 1:
            raise ValueError("Unsupported selected instruction plan version")
        tensors = {
            n: Tensor(
                **{
                    **t,
                    "shape": tuple(t["shape"]),
                    "view": Expr.load(t["view"]) if t["view"] else None,
                }
            )
            for n, t in r["tensors"].items()
        }
        instructions = [
            Instruction(
                x["opcode"],
                tuple(Expr.load(a) for a in x["args"]),
                tuple((k, Expr.load(v)) for k, v in x["kwargs"]),
                Expr.load(x["destination"]),
                tuple(x["reads"]),
                tuple(x["writes"]),
                tuple(x["dependencies"]),
                x["implementation"],
                x["accumulate"],
            )
            for x in r["instructions"]
        ]
        return cls(
            tensors,
            {n: Expr.load(x) for n, x in r["indices"].items()},
            instructions,
            tuple(r["arguments"]),
            tuple(r["outputs"]),
            {n: Placement(**p) for n, p in r["placements"].items()},
            r["contracts"],
            repeated_regions=r.get("repeated_regions", []),
            encoding_loops=r.get("encoding_loops", []),
            encoding_storage=r.get("encoding_storage", "arena"),
            storage_regions=r.get("storage_regions", []),
        )


class Builder:
    """Compatibility front end for target lowering's scalar address expressions.

    Each instruction is immediately typed; no generated program is stored or
    reparsed. Unsupported language operations fail before collateral emission.
    """

    def __init__(self, arguments, hardware):
        self.program = Program(
            {}, {}, [], tuple(f"a{i}" for i in range(len(arguments))), ()
        )
        self.program.contracts = {
            c.name: asdict(c) for c in hardware.isa_instructions
        }
        self.constants = {}
        self.last_write, self.users = {}, {}
        for i, b in enumerate(arguments):
            self.program.tensors[f"a{i}"] = Tensor(
                f"a{i}", tuple(b.shape), b.dtype, "HBM", "contiguous"
            )

    def value(self, e):
        if e.op == "literal":
            return e.value
        if e.op == "ellipsis":
            return Ellipsis
        if e.op == "name":
            if e.value == "inf":
                return math.inf
            if e.value in self.constants:
                return self.constants[e.value]
            return self.program.tensors.get(e.value, e.value)
        a = [self.value(x) for x in e.args]
        if e.op in ("tuple", "list"):
            return tuple(a)
        if e.op == "slice":
            return slice(*a)
        if e.op in OPS:
            return OPS[e.op](*a)
        if e.op == "USub":
            return -a[0]
        if e.op == "attr":
            if isinstance(a[0], Tensor):
                return getattr(a[0], e.value, e.value)
            return f"{a[0]}.{e.value}"
        if e.op == "index":
            if not isinstance(a[0], Tensor):
                return a[0][a[1]]
            tensor, index = a
            # Zero-stride shape proxy: no tensor payload allocation.
            proxy = np.lib.stride_tricks.as_strided(
                np.zeros(1, dtype=np.uint8),
                shape=tensor.shape,
                strides=(0,) * len(tensor.shape),
            )
            shape = proxy[index].shape
            return Tensor(
                "",
                tuple(shape),
                tensor.dtype,
                tensor.memory,
                tensor.layout,
                tensor.alias or tensor.name,
                e,
            )
        if e.op == "call":
            func = e.args[0]
            if func.op == "attr" and func.value == "broadcast_to":
                raise ValueError(
                    "Partition broadcast requires an explicit ISA expansion"
                )
            if func.op == "attr" and func.value in (
                "reshape",
                "broadcast_to",
                "view",
            ):
                t = self.value(func.args[0])
                shape = a[1]
                dtype = t.dtype
                if func.value == "view":
                    dtype = shape.removeprefix("nl.")
                    shape = (
                        *t.shape[:-1],
                        t.shape[-1] * BITS[t.dtype] // BITS[dtype],
                    )
                if isinstance(shape, int):
                    shape = (shape,)
                return Tensor(
                    "",
                    tuple(shape),
                    dtype,
                    t.memory,
                    t.layout,
                    t.alias or t.name,
                    e,
                )
            if a[0] == "nl.arange":
                return np.arange(a[1], dtype=np.int64)
        raise ValueError(f"Cannot type selected address expression {e}")

    def tensor_names(self, expressions):
        names = set().union(*(e.names() for e in expressions))
        return tuple(sorted(n for n in names if n in self.program.tensors))

    def add(self, code):
        node = ast.parse(code.strip()).body[0]
        accumulate = isinstance(node, ast.AugAssign)
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            dst_node = node.target if accumulate else node.targets[0]
            rhs = node.value
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            rhs = node.value
            kw = {k.arg: k.value for k in rhs.keywords}
            dst_node = kw.get("dst")
            if dst_node is None:
                raise ValueError("Instruction has no explicit destination")
        else:
            raise ValueError(f"Unsupported lowering statement {code}")
        destination = Expr.parse(dst_node)
        name = dst_node.id if isinstance(dst_node, ast.Name) else None
        if (
            not isinstance(rhs, ast.Call)
            or not isinstance(rhs.func, ast.Attribute)
            or not isinstance(rhs.func.value, ast.Name)
            or rhs.func.value.id not in ("nl", "nisa")
        ):
            value = self.value(Expr.parse(rhs))
            if not name:
                raise ValueError("Non-ISA local assignment")
            if isinstance(value, Tensor):
                self.program.tensors[name] = Tensor(
                    name,
                    value.shape,
                    value.dtype,
                    value.memory,
                    value.layout,
                    value.alias or value.name,
                    Expr.parse(rhs),
                )
            else:
                self.constants[name] = value
                self.program.indices[name] = Expr.parse(rhs)
            return
        opcode = f"{rhs.func.value.id}.{rhs.func.attr}"
        args = tuple(Expr.parse(x) for x in rhs.args)
        kwargs = tuple(
            (x.arg, Expr.parse(x.value)) for x in rhs.keywords if x.arg != "dst"
        )
        kw = {k: self.value(v) for k, v in kwargs}
        values = [self.value(x) for x in args]
        if opcode in ("nl.ndarray", "nl.zeros"):
            shape = tuple(values[0])
            dtype = kw["dtype"].removeprefix("nl.")
            memory = {
                "nl.sbuf": "SBUF",
                "nl.psum": "PSUM",
                "nl.shared_hbm": "HBM",
            }[kw["buffer"]]
            self.program.tensors[name] = Tensor(
                name,
                shape,
                dtype,
                memory,
                "partition_free" if memory != "HBM" else "contiguous",
            )
            if opcode == "nl.ndarray":
                return
            opcode = "nisa.memset"
            args = (
                Expr("tuple", tuple(Expr("literal", value=x) for x in shape)),
            )
            kwargs = (
                ("value", Expr("literal", value=0)),
                ("dtype", Expr.parse(f"nl.{dtype}")),
                ("engine", Expr.parse("nisa.vector_engine")),
            )
        elif not opcode.startswith("nisa."):
            raise ValueError(
                f"Explicit selected plan cannot contain language operation {opcode}"
            )
        elif name and opcode != "nisa.dma_copy":
            data = next(
                (v for v in (*values, *kw.values()) if isinstance(v, Tensor)),
                None,
            )
            dtype = kw.get(
                "dtype", f"nl.{data.dtype}" if data else "nl.float32"
            ).removeprefix("nl.")
            memory = "SBUF"
            if opcode == "nisa.memset":
                shape = values[0]
            elif opcode == "nisa.nc_matmul":
                shape = (values[0].shape[1], values[1].shape[1])
                dtype = (
                    values[0].dtype
                    if kw.get("is_transpose", False)
                    else "float32"
                )
                memory = "PSUM"
            elif opcode == "nisa.nc_transpose":
                shape = (math.prod(data.shape[1:]), data.shape[0])
                memory = "SBUF"
            elif opcode == "nisa.tensor_reduce":
                shape = (data.shape[0], 1)
            elif data is not None:
                shape = data.shape
            else:
                raise ValueError(f"{opcode}: cannot infer result type")
            self.program.tensors[name] = Tensor(
                name, tuple(shape), dtype, memory, "partition_free"
            )
            if opcode == "nisa.nc_matmul" and not accumulate:
                # NKI accumulates into a reused PSUM allocation. Every new
                # result generation therefore has an explicit zero instruction.
                self.add(
                    f"{name} = nl.zeros({tuple(shape)!r}, dtype=nl.{dtype}, buffer=nl.psum)"
                )
        reads = self.tensor_names((*args, *(v for _, v in kwargs)))
        writes = self.tensor_names((destination,))
        if accumulate:
            reads = tuple(sorted(set(reads + writes)))
        required = set()
        for n in reads:
            root = self.program.root(n)
            if root in self.last_write:
                required.add(self.last_write[root])
        for n in writes:
            root = self.program.root(n)
            required.update(self.users.get(root, ()))
            if root in self.last_write:
                required.add(self.last_write[root])
        i = len(self.program.instructions)
        for n in writes:
            root = self.program.root(n)
            self.last_write[root] = i
            self.users[root] = set()
        for n in reads:
            self.users.setdefault(self.program.root(n), set()).add(i)
        engine = kw.get(
            "engine",
            {
                "nisa.dma_copy": "DMA",
                "nisa.nc_matmul": "TensorE",
                "nisa.activation": "ScalarE",
            }.get(opcode, "VectorE"),
        )
        engine = {
            "nisa.vector_engine": "VectorE",
            "nisa.scalar_engine": "ScalarE",
            "nisa.tensor_engine": "TensorE",
        }.get(engine, engine)
        implementation = opcode.replace("nisa.", "nki.isa.") + "." + engine
        if implementation not in self.program.contracts:
            raise ValueError(
                f"No hardware IR instruction implementation for {implementation}"
            )
        self.program.instructions.append(
            Instruction(
                opcode,
                args,
                kwargs,
                destination,
                reads,
                writes,
                tuple(sorted(required)),
                implementation,
                accumulate,
            )
        )
