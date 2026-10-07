"""Formal encoding of validated selected instructions. No target policy here."""

import ast
import math
from .instruction_plan import BITS

SYMBOLS = {
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "Div": "/",
    "BitAnd": "&",
    "BitOr": "|",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "Eq": "==",
    "NotEq": "!=",
}


def expression(e):
    if e.op == "literal":
        return repr(e.value)
    if e.op == "ellipsis":
        return "..."
    if e.op == "name":
        return e.value
    a = [expression(x) for x in e.args]
    if e.op == "tuple":
        return "(" + ", ".join(a) + ",)" if a else "()"
    if e.op == "list":
        return "[" + ", ".join(a) + "]"
    if e.op == "slice":
        return ":".join("" if x == "None" else x for x in a)
    if e.op == "attr":
        return a[0] + "." + e.value
    if e.op == "index":
        index = (
            ", ".join(expression(x) for x in e.args[1].args)
            if e.args[1].op == "tuple"
            else a[1]
        )
        return f"{a[0]}[{index}]"
    if e.op == "call":
        return a[0] + "(" + ", ".join(a[1:]) + ")"
    if e.op in SYMBOLS:
        return f"({a[0]} {SYMBOLS[e.op]} {a[1]})"
    if e.op == "USub":
        return f"(-{a[0]})"
    raise ValueError(f"Unknown plan expression {e.op}")


def emit(program, *, _capture=None):
    program.validate()
    if (
        any(i.opcode == "nisa.nc_transpose" for i in program.instructions)
        and program.encoding_storage != "disjoint_arenas"
    ):
        raise ValueError(
            "Pinned SDK stream transpose requires disjoint arena encoding"
        )
    lines = [
        "import numpy as np",
        "import neuronxcc.nki as nki",
        "import neuronxcc.nki.language as nl",
        "import neuronxcc.nki.isa as nisa",
        "import neuronxcc.nki.compiler as ncc",
        "",
        "@nki.compiler.skip_middle_end_transformations",
        "@nki.jit",
        f"def kernel({', '.join(program.arguments)}):",
    ]

    def line(s):
        lines.append("    " + s)

    from .instruction_plan import Expr

    index_cache = {}

    def inline(e):
        if e.op == "name" and e.value in program.indices:
            if e.value not in index_cache:
                index_cache[e.value] = inline(program.indices[e.value])
            return index_cache[e.value]
        return Expr(e.op, tuple(inline(a) for a in e.args), e.value)

    bindings = {}
    if program.encoding_storage == "disjoint_arenas":
        for name, p in program.placements.items():
            if p.memory == "SBUF":
                bindings[name] = next(
                    (i, r["start"])
                    for i, r in enumerate(program.storage_regions)
                    if r["start"] <= p.byte_address
                    and p.byte_address + p.bytes_per_partition <= r["stop"]
                )

    def physical(name, index=None, dtype=None):
        t = program.tensors[name]
        p = program.placements[name]
        dtype = dtype or t.dtype
        base = (
            f"sbuf_{dtype}" if t.memory == "SBUF" else f"psum{p.bank}_{dtype}"
        )
        offset = (
            p.byte_address * 8 // BITS[t.dtype] if t.memory == "SBUF" else 0
        )
        if t.memory == "SBUF" and name in bindings:
            region, address = bindings[name]
            base = f"sbuf{region}_{dtype}"
            offset = (p.byte_address - address) * 8 // BITS[t.dtype]
        from .instruction_plan import Expr

        if index is None or index.op == "ellipsis":
            indices = [
                Expr(
                    "slice",
                    (
                        Expr("literal", value=0),
                        Expr("literal", value=n),
                        Expr("literal"),
                    ),
                )
                for n in t.shape
            ]
        else:
            indices = list(index.args) if index.op == "tuple" else [index]
        rendered = []
        for axis, item in enumerate(indices):
            start = offset if axis == 1 else 0
            if item.op == "slice":
                lo, hi, step = item.args
                lo = (
                    Expr("literal", value=0)
                    if lo.op == "literal" and lo.value is None
                    else lo
                )
                hi = (
                    Expr("literal", value=t.shape[axis])
                    if hi.op == "literal" and hi.value is None
                    else hi
                )
                rendered.append(
                    f"({expression(lo)}+{start}):({expression(hi)}+{start})"
                    + (":" + expression(step) if step.value is not None else "")
                )
            else:
                rendered.append(f"({expression(item)}+{start})")
        return base + "[" + ", ".join(rendered) + "]"

    def encoded(e):
        e = inline(e)
        if (
            e.op == "call"
            and e.args[0].op == "attr"
            and e.args[0].value == "view"
        ):
            base = e.args[0].args[0]
            index = None
            if base.op == "index":
                base, index = base.args
            dtype = expression(e.args[1]).removeprefix("nl.")
            if (
                base.op == "name"
                and base.value in program.placements
                and BITS[dtype] == BITS[program.tensors[base.value].dtype]
            ):
                # Bitcast the arena before slicing: the pinned SDK cannot
                # bitcast an access-pattern object returned by a slice.
                return physical(base.value, index, dtype)
        if e.op == "name" and e.value in program.tensors:
            t = program.tensors[e.value]
            if t.alias:
                return encoded(t.view)
            if e.value in program.placements:
                return physical(e.value)
        if e.op == "index" and e.args[0].op == "name":
            name = e.args[0].value
            if name in program.placements:
                return physical(name, e.args[1])
        if e.op == "attr":
            return encoded(e.args[0]) + "." + e.value
        if e.op == "call":
            return (
                encoded(e.args[0])
                + "("
                + ", ".join(encoded(x) for x in e.args[1:])
                + ")"
            )
        if e.op == "index":
            index = (
                ", ".join(encoded(x) for x in e.args[1].args)
                if e.args[1].op == "tuple"
                else encoded(e.args[1])
            )
            return encoded(e.args[0]) + "[" + index + "]"
        return expression(e)

    used = set(program.arguments + program.outputs)
    for ins in program.instructions:
        used.update(ins.reads + ins.writes)
    for name in tuple(used):
        t = program.tensors[name]
        while t.alias:
            used.add(t.alias)
            t = program.tensors[t.alias]
    size = max(
        (
            p.byte_address + p.bytes_per_partition
            for p in program.placements.values()
            if p.memory == "SBUF"
        ),
        default=0,
    )
    if size and program.encoding_storage == "arena":
        line(
            f"sbuf = nl.ndarray((128,{size}), dtype=nl.uint8, buffer=ncc.sbuf.alloc(lambda idx, pdim_size, fdim_size: (0,0)))"
        )
        for dtype in sorted(
            {
                program.tensors[n].dtype
                for n, p in program.placements.items()
                if p.memory == "SBUF"
            }
        ):
            line(f"sbuf_{dtype} = sbuf.view(nl.{dtype})")
    elif size:
        for i, region in enumerate(program.storage_regions):
            start, stop = region["start"], region["stop"]
            line(
                f"sbuf{i} = nl.ndarray((128,{stop-start}), dtype=nl.uint8, buffer=ncc.sbuf.alloc(lambda idx, pdim_size, fdim_size: (0,{start})))"
            )
            dtypes = {
                t.dtype
                for name, t in program.tensors.items()
                if t.memory == "SBUF"
                and program.root(name) in bindings
                and bindings[program.root(name)][0] == i
            }
            for dtype in sorted(dtypes):
                line(f"sbuf{i}_{dtype} = sbuf{i}.view(nl.{dtype})")
    for bank in sorted(
        {p.bank for p in program.placements.values() if p.memory == "PSUM"}
    ):
        line(
            f"psum{bank} = nl.ndarray((128,512), dtype=nl.float32, buffer=ncc.psum.alloc(lambda idx, pdim_size, fdim_size: ({bank},0,0)))"
        )
        for dtype in sorted(
            {
                program.tensors[n].dtype
                for n, p in program.placements.items()
                if p.memory == "PSUM" and p.bank == bank
            }
        ):
            line(f"psum{bank}_{dtype} = psum{bank}.view(nl.{dtype})")
    for name, t in program.tensors.items():
        if name not in used or name in program.arguments or t.alias:
            continue
        if t.memory == "HBM":
            if t.constant == "identity":
                dtype = "float32" if t.dtype == "bfloat16" else t.dtype
                line(
                    f"{name} = nl.shared_constant(np.eye({t.shape[0]}, dtype=np.{dtype}), dtype=nl.{t.dtype})"
                )
            else:
                line(
                    f"{name} = nl.ndarray({t.shape!r}, dtype=nl.{t.dtype}, buffer=nl.shared_hbm)"
                )
        else:
            p = program.placements[name]
            if len(t.shape) != 2:
                raise ValueError(
                    f"{name}: direct allocation currently requires a 2-D physical tile"
                )
    first_instruction = len(lines)
    for ins in program.instructions:
        args = [encoded(x) for x in ins.args]
        args.extend(k + "=" + encoded(v) for k, v in ins.kwargs)
        dst = encoded(ins.destination)
        if (
            ins.destination.op == "name"
            and ins.destination.value in program.placements
        ):
            dst = physical(ins.destination.value)
        if ins.opcode in ("nisa.dma_copy",):
            line(f"{ins.opcode}(dst={dst}, {', '.join(args)})")
        else:
            # Explicit destination assignment is the pinned SDK's output
            # binding syntax. No automatically allocated ISA results.
            line(
                f"{dst} {'+=' if ins.accumulate else '='} {ins.opcode}({', '.join(args)})"
            )
    instructions = [line[4:] for line in lines[first_instruction:]]
    if _capture is not None:
        _capture.extend(instructions)
    from .compact_encoding import encode

    lines[first_instruction:] = [
        "    " + line for line in encode(program, instructions)
    ]
    from .instruction_plan import Expr

    line(
        "return ("
        + ", ".join(encoded(Expr("name", value=n)) for n in program.outputs)
        + ",)"
    )
    source = "\n".join(lines) + "\n"
    ast.parse(source)
    return source
