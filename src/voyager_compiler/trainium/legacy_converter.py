"""Translate scheduled Voyager protobuf operations to a single NKI program.

Static scalar control is specialized (as in the Gemmini collateral consumer).
No tensor arithmetic is performed on the host and no whole-op tiler is used.
DMA windows, buffer slots, reduction order and fused primitives come from IR.
NKI owns final physical allocation and engine instruction scheduling; source
addresses and semaphore dependencies are retained in the conversion manifest.
"""

import ast
import hashlib
import json
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from google.protobuf import text_format

from voyager_compiler.codegen import voyager_ir_pb2 as ir


@dataclass
class Ref:
    name: str
    level: int
    shape: tuple
    strides: tuple
    offset: int
    slot: int
    dtype: str
    address: int


@dataclass
class PanelValue:
    """Independent [N,M] SBUF panels; no mutable whole-result assembly."""

    m: int
    n: int
    panels: list  # (mi, ni, mm, nn, NKI variable)


def strides(shape):
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


class Converter:
    def __init__(self, model, target, tuning=None, hardware=None):
        from .execution import TrainiumTuning

        self.tuning = tuning or TrainiumTuning()
        from .hardware import neuron_core, TARGETS

        self.hardware = hardware or neuron_core(TARGETS[target])
        self.realized_compute_graphs = []
        self.transfer_records = []
        self.reduction_records = []
        self.vector_records = []
        self.boundary_records = []
        self.matrix_buffer_bindings = []
        self.used_implementations = set()
        if self.tuning.isa_lowering and target != "trainium-v3":
            raise ValueError(
                "The pinned ISA expansion contract is validated for Trainium2 only"
            )
        self.model, self.target = model, target
        self.env, self.semaphores, self.boxes, self.vars = {}, {}, {}, {}
        self.lines, self.stats, self.events = [], Counter(), []
        self.expanded_isa = Counter()
        self.panel_slots = {}
        self.serial = 0
        self.indices = {}
        self.arguments = list(model.inputs) + list(model.parameters)
        self.collect(model)
        self.header = [
            "from math import inf",
            "import neuronxcc.nki as nki",
            "import neuronxcc.nki.language as nl",
            "import neuronxcc.nki.isa as nisa",
            "",
            *(
                ["@nki.compiler.skip_middle_end_transformations"]
                if self.tuning.isa_lowering
                else []
            ),
            "@nki.jit",
            "def kernel("
            + ", ".join("a" + str(i) for i in range(len(self.arguments)))
            + "):",
        ]
        for i, b in enumerate(self.arguments):
            self.vars[b.node, 0] = f"a{i}"
        self.declarations = []
        for b in self.boxes.values():
            if b.memory.level == ir.MEMORY_LEVEL_REGISTER:
                continue
            for slot in range(b.bank_count or 1):
                key = b.node, slot
                if key in self.vars:
                    continue
                var = "buf" + str(len(self.vars))
                self.vars[key] = var
                shape = (
                    self.storage_shape(b)
                    if b.memory.level == ir.MEMORY_LEVEL_SCRATCHPAD
                    else tuple(b.shape)
                )
                memory = (
                    "nl.sbuf"
                    if b.memory.level == ir.MEMORY_LEVEL_SCRATCHPAD
                    else "nl.shared_hbm"
                )
                self.declarations.append(
                    f"    {var} = nl.ndarray({shape!r}, dtype=nl.{b.dtype}, buffer={memory})"
                )

    def collect(self, message):
        if isinstance(message, ir.TensorBox) and message.HasField("memory"):
            b = message
            if b.dtype not in ("float32", "float16", "bfloat16", "int64"):
                raise ValueError(f"Unsupported storage dtype {b.dtype}")
            if b.node not in self.boxes:
                self.boxes[b.node] = b
        for f, v in message.ListFields():
            if f.type != f.TYPE_MESSAGE:
                continue
            if f.is_repeated:
                vals = (
                    [v[k] for k in sorted(v)]
                    if f.message_type.GetOptions().map_entry
                    else v
                )
                for x in vals:
                    self.collect(x)
            else:
                self.collect(v)

    def storage_shape(self, b):
        if not b.shape:
            return 1, 1
        p = min(128, b.shape[-1] if b.shape else 1)
        return p, math.prod(b.shape[:-1]) * math.ceil(b.shape[-1] / p)

    def emit(self, code):
        self.lines.append("    " + code)

    def tmp(self, code):
        self.serial += 1
        v = f"t{self.serial}"
        self.emit(f"{v} = {code}")
        match = re.fullmatch(r"nl.arange\((\d+)\)\[(.*)\]", code)
        if match:
            values = np.arange(int(match[1]), dtype=np.int64)
            self.indices[v] = (
                values[:, None] if match[2] == ":, None" else values[None, :]
            )
        return v

    def scalar(self, v):
        k = v.WhichOneof("value")
        if k == "node":
            return self.env[v.node]
        if k is None:
            raise ValueError("Missing scalar")
        return getattr(v, k)

    def ref(self, r):
        b = r.box
        if not b.HasField("memory"):
            return self.env[b.node]
        shape = tuple(b.shape)
        ss = strides(shape)
        off = 0
        slot = 0
        offsets = [self.scalar(x) for x in r.offsets]
        sizes = list(r.sizes)
        steps = list(r.strides)
        if b.HasField("bank_count") and offsets:
            slot = offsets.pop(0)
            sizes = sizes[1:]
            steps = steps[1:]
        if offsets:
            off = sum(x * s for x, s in zip(offsets, ss))
            ss = tuple(a * b for a, b in zip(ss, steps))
            shape = tuple(sizes)
        out = tuple(r.output_shape)
        if out != shape:
            # Squeezes and contiguous reshapes are lossless. Reject a reshape
            # of a strided subwindow rather than pretending it is contiguous.
            pairs = [(n, s) for n, s in zip(shape, ss) if n != 1]
            if tuple(n for n, s in pairs) == out:
                ss = tuple(s for n, s in pairs)
            elif ss == strides(shape):
                ss = strides(out)
            elif math.prod(out) != math.prod(shape):
                raise ValueError("Invalid view")
            else:
                raise NotImplementedError("Reshape of strided window")
            shape = out
        return Ref(
            b.node,
            b.memory.level,
            shape,
            ss,
            off,
            slot,
            b.dtype,
            b.memory.address + slot * (b.bank_stride_bytes or 0),
        )

    def arg(self, a):
        k = a.WhichOneof("arg_type")
        if k == "tensor_box":
            return self.ref(a.tensor_box)
        if k == "tensor_box_list":
            return [self.ref(x) for x in a.tensor_box_list.values]
        if k == "scalar":
            return self.scalar(a.scalar)
        if k == "scalar_list":
            return [self.scalar(x) for x in a.scalar_list.values]
        if k == "str_value":
            return None if a.str_value == "None" else a.str_value
        raise NotImplementedError(k)

    def outputs(self, o, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(values) != len(o.outputs):
            raise ValueError("Scalar output arity")
        for out, v in zip(o.outputs, values):
            self.env[out.name] = v

    def signal(self, r, n=1):
        key = (r.name, r.slot)
        self.semaphores[key] = self.semaphores.get(key, 0) + n

    def wait(self, r):
        key = (r.name, r.slot)
        if self.semaphores.get(key, 0) < 1:
            raise ValueError(f"Unbalanced wait {key}")
        self.semaphores[key] -= 1
        self.stats["waits"] += 1

    def region(self, ops):
        for o in ops:
            self.stats["expanded_operations"] += 1
            if self.stats["expanded_operations"] > 2000000:
                raise ValueError("Static expansion limit exceeded")
            k = o.WhichOneof("op_type")
            self.stats[k] += 1
            if k == "loop":
                if o.loop.WhichOneof("loop_type") != "for_loop":
                    raise NotImplementedError("Dynamic while loop")
                f = o.loop.for_loop
                state = [self.scalar(a.initial) for a in f.iter_args]
                for i in range(
                    self.scalar(f.start),
                    self.scalar(f.end),
                    self.scalar(f.step),
                ):
                    self.env[f.iv] = i
                    self.env.update(
                        (a.name, v) for a, v in zip(f.iter_args, state)
                    )
                    self.region(f.body.ops)
                    state = [self.scalar(v) for v in f.body.yields]
                self.outputs(o, state)
            elif k == "cond":
                reg = (
                    o.cond.true_region
                    if self.scalar(o.cond.predicate)
                    else o.cond.false_region
                )
                self.region(reg.ops)
                self.outputs(o, [self.scalar(v) for v in reg.yields])
            elif k == "async":
                a = getattr(o, "async")
                for r in a.dependencies:
                    self.wait(self.ref(r))
                self.events.append(
                    dict(
                        kind="async",
                        name=o.name,
                        dependencies=[
                            [self.ref(r).name, self.ref(r).slot]
                            for r in a.dependencies
                        ],
                    )
                )
                self.region(a.body.ops)
                if a.HasField("post"):
                    self.signal(self.ref(a.post))
            elif k == "prim" and o.prim.op == "cpu":
                self.cpu(o)
            elif k in ("prim", "fused"):
                self.compute(o)
            else:
                raise NotImplementedError(k)

    def cpu(self, o):
        p = o.prim
        name = p.target.removeprefix("voyager::")
        kw = {k: self.arg(v) for k, v in p.kwargs.items()}
        if p.target in ("aten::pad", "aten::slice", "aten::permute"):
            self.boundary(o, kw)
            return
        if name in ("alloc", "zeros", "fill"):
            for out in o.outputs:
                if out.WhichOneof("result_type") != "tensor_box":
                    continue
                b = out.tensor_box
                for slot in range(b.bank_count or 1):
                    if b.memory.level == ir.MEMORY_LEVEL_REGISTER:
                        self.semaphores[b.node, slot] = int(
                            kw.get("value", 0)
                        )
                    elif name in ("zeros", "fill"):
                        var = self.vars[b.node, slot]
                        self.emit(f"{var}[...] = {kw.get('value', 0)!r}")
            return
        if name == "async_copy":
            self.copy(**kw)
            return
        if name == "async_wait":
            self.wait(kw["semaphore"])
            return
        if name == "sym_ite":
            self.outputs(o, kw["t"] if kw["b"] else kw["f"])
            return
        if name == "delinearize_index":
            linear = kw["linear"]
            coords = []
            for b in reversed(kw["basis"]):
                coords.append(linear % b)
                linear //= b
            self.outputs(o, list(reversed(coords)))
            return
        functions = {
            n: getattr(operator, n)
            for n in (
                "add",
                "sub",
                "mul",
                "floordiv",
                "mod",
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
                "and_",
                "or_",
                "not_",
                "neg",
            )
        }
        if name in functions:
            a = [kw["input"]] + ([kw["other"]] if "other" in kw else [])
            self.outputs(o, functions[name](*a))
            return
        raise NotImplementedError(f"CPU primitive {p.target}")

    def boundary(self, o, kw):
        b = o.outputs[0].tensor_box
        d = Ref(
            b.node,
            b.memory.level,
            tuple(b.shape),
            strides(b.shape),
            0,
            0,
            b.dtype,
            b.memory.address,
        )
        a = kw["input"]
        name = o.prim.target
        # Boundary materialization is outside matrix search but inside this
        # executable. Preserve it explicitly for whole-program predictions.
        element_bytes = 2 if d.dtype in ("bfloat16", "float16") else 4
        read_elements = math.prod(d.shape)
        if name == "aten::pad":
            pads = kw["pad"]
            valid = list(a.shape)
            for axis in range(len(pads) // 2):
                valid[-1 - axis] = max(
                    0,
                    valid[-1 - axis]
                    + min(0, pads[2 * axis])
                    + min(0, pads[2 * axis + 1]),
                )
            read_elements = math.prod(valid)
        self.boundary_records.append(
            dict(
                operation=name,
                source=a.name,
                destination=d.name,
                input_shape=list(a.shape),
                output_shape=list(d.shape),
                dtype=d.dtype,
                read_bytes=read_elements * element_bytes,
                write_bytes=math.prod(d.shape) * element_bytes,
            )
        )
        if name == "aten::permute":
            dims = tuple(kw["dims"])
            if dims not in ((0, 2, 3, 1), (0, 3, 1, 2)):
                raise NotImplementedError(f"Boundary permutation {dims}")
            nchw = dims == (0, 2, 3, 1)
            batch, channels, height, width = (
                a.shape
                if nchw
                else (a.shape[0], a.shape[3], a.shape[1], a.shape[2])
            )
            for bidx in range(batch):
                for cidx in range(0, channels, 128):
                    for spatial in range(0, height * width, 128):
                        nc = min(128, channels - cidx)
                        ns = min(128, height * width - spatial)
                        ip = self.tmp(
                            f"nl.arange({nc if nchw else ns})[:, None]"
                        )
                        jf = self.tmp(
                            f"nl.arange({ns if nchw else nc})[None, :]"
                        )
                        src_linear = (
                            f"{bidx * channels * height * width} + ({ip}+{cidx})*{height * width}+{jf}+{spatial}"
                            if nchw
                            else f"{bidx * channels * height * width} + ({ip}+{spatial})*{channels}+{jf}+{cidx}"
                        )
                        av = self.vars[a.name, a.slot]
                        dv = self.vars[d.name, d.slot]
                        val = self.tmp(
                            f"nl.load({av}.reshape(({math.prod(a.shape)},))[{src_linear}])"
                        )
                        val = self.tmp(f"nl.transpose({val})")
                        op = self.tmp(
                            f"nl.arange({ns if nchw else nc})[:, None]"
                        )
                        of = self.tmp(
                            f"nl.arange({nc if nchw else ns})[None, :]"
                        )
                        dst_linear = (
                            f"{bidx * channels * height * width}+({op}+{spatial})*{channels}+{of}+{cidx}"
                            if nchw
                            else f"{bidx * channels * height * width}+({op}+{cidx})*{height * width}+{of}+{spatial}"
                        )
                        self.emit(
                            f"nl.store({dv}.reshape(({math.prod(d.shape)},))[{dst_linear}], {val})"
                        )
            return
        if self.tuning.isa_lowering and len(a.shape) == len(d.shape) == 2:
            if self.boundary_2d(a, d, name, kw):
                return
        shape = d.shape
        padding = [0] * len(shape)
        if name == "aten::pad":
            if kw.get("mode", "constant") != "constant":
                raise NotImplementedError("Nonconstant pad")
            for i in range(len(kw["pad"]) // 2):
                padding[-1 - i] = kw["pad"][2 * i]
        for row in range(0, math.prod(shape[:-1]), 128):
            nr = min(128, math.prod(shape[:-1]) - row)
            for col in range(0, shape[-1], 128):
                nc = min(128, shape[-1] - col)
                ip = self.tmp(f"nl.arange({nr})[:, None]")
                jf = self.tmp(f"nl.arange({nc})[None, :]")
                lin = f"(({ip}+{row})*{shape[-1]}+{jf}+{col})"
                dest = [
                    f"(({lin}//{st})%{n})"
                    for st, n in zip(strides(shape), shape)
                ]
                source = [f"({c}-{pa})" for c, pa in zip(dest, padding)]
                if name == "aten::slice":
                    dim = kw["dim"] % len(shape)
                    source[dim] = (
                        f"({dest[dim]}*{kw.get('step', 1)}+{kw.get('start', 0)})"
                    )
                mask = " & ".join(
                    f"(({c})>=0) & (({c})<{n})"
                    for c, n in zip(source, a.shape)
                )
                v = self.tmp(f"nl.load({self.index(a, source)}, mask={mask})")
                v = self.tmp(f"nl.where({mask}, {v}, {kw.get('value', 0)!r})")
                self.emit(f"nl.store({self.index(d, dest)}, {v})")

    def boundary_2d(self, a, d, name, kw):
        """Realize rectangular pad/slice with ISA DMA and optional zero fill.

        No intermediate mask tensors or language-level where expansion. Keep
        unsupported leading/negative padding on the explicit fallback path.
        """
        if name not in ("aten::pad", "aten::slice"):
            return False
        start = [0, 0]
        if name == "aten::pad":
            pads = kw["pad"]
            if (
                kw.get("mode", "constant") != "constant"
                or any(x < 0 for x in pads)
                or any(pads[i] for i in range(0, len(pads), 2))
            ):
                return False
        else:
            if kw.get("step", 1) != 1:
                return False
            start[kw["dim"] % 2] = kw.get("start", 0)
            if any(x < 0 for x in start):
                return False
        panels = []
        for row in range(0, d.shape[0], 128):
            for col in range(0, d.shape[1], 128):
                nr, nc = min(128, d.shape[0] - row), min(
                    128, d.shape[1] - col
                )
                sr, sc = row + start[0], col + start[1]
                vr, vc = max(0, min(nr, a.shape[0] - sr)), max(
                    0, min(nc, a.shape[1] - sc)
                )
                fill = vr != nr or vc != nc
                if fill:
                    value = self.tmp(
                        f'nisa.memset(({nr}, {nc}), value={kw.get("value",0)!r}, dtype=nl.{d.dtype}, engine=nisa.vector_engine)'
                    )
                else:
                    value = self.tmp(
                        f"nl.ndarray(({nr}, {nc}), dtype=nl.{d.dtype}, buffer=nl.sbuf)"
                    )
                if vr and vc:
                    ip = self.tmp(f"nl.arange({vr})[:, None]")
                    jf = self.tmp(f"nl.arange({vc})[None, :]")
                    self.emit(
                        f'nisa.dma_copy(dst={value}[{ip}, {jf}], src={self.index(a,[f"({ip}+{sr})",f"({jf}+{sc})"])})'
                    )
                    self.stats["isa_dma_panels"] += 1
                ip = self.tmp(f"nl.arange({nr})[:, None]")
                jf = self.tmp(f"nl.arange({nc})[None, :]")
                self.emit(
                    f'nisa.dma_copy(dst={self.index(d,[f"({ip}+{row})",f"({jf}+{col})"])}, src={value})'
                )
                self.stats["isa_dma_panels"] += 1
                panels.append(
                    dict(
                        rows=nr,
                        columns=nc,
                        valid_rows=vr,
                        valid_columns=vc,
                        fill=fill,
                    )
                )
        self.boundary_records[-1][
            "implementation"
        ] = "isa.rectangular_pad_slice"
        self.boundary_records[-1]["panels"] = panels
        return True

    def coordinates(self, shape):
        p = min(128, shape[-1] if shape else 1)
        f = math.ceil(math.prod(shape) / p)
        ip = self.tmp(f"nl.arange({p})[:, None]")
        jf = self.tmp(f"nl.arange({f})[None, :]")
        linear = f"({jf} * {p} + {ip})"
        coords = [
            f"(({linear} // {s}) % {n})"
            for n, s in zip(shape, strides(shape))
        ]
        return coords, linear, p, f

    def mask(self, expression):
        names = set(re.findall(r"\bt\d+\b", expression))
        if names.issubset(self.indices):
            value = np.asarray(
                eval(
                    expression,
                    {"__builtins__": {}},
                    {n: self.indices[n] for n in names},
                )
            )
            if np.all(value):
                return "True"
            if not np.any(value):
                return "False"
        return expression

    def affine(self, expression):
        names = set(re.findall(r"\bt\d+\b", expression))
        if not names or not names.issubset(self.indices):
            return expression
        arrays = {n: self.indices[n] for n in names}
        # Only converter-generated integer address expressions, never tensor
        # data or text supplied by collaterals, are evaluated here.
        value = np.asarray(eval(expression, {"__builtins__": {}}, arrays))
        if value.ndim != 2:
            return expression
        row = next((n for n in sorted(names) if arrays[n].shape[0] > 1), None)
        col = next((n for n in sorted(names) if arrays[n].shape[1] > 1), None)
        base = int(value[0, 0])
        dr = int(value[1, 0]) - base if value.shape[0] > 1 else 0
        dc = int(value[0, 1]) - base if value.shape[1] > 1 else 0
        predicted = (
            base
            + dr * np.arange(value.shape[0])[:, None]
            + dc * np.arange(value.shape[1])[None, :]
        )
        if not np.array_equal(predicted, value):
            return expression
        terms = [str(base)]
        if row:
            terms.append(f"{row} * {dr}")
        if col:
            terms.append(f"{col} * {dc}")
        return "(" + " + ".join(terms) + ")"

    def index(self, r, coords):
        linear = " + ".join(
            [str(r.offset)]
            + [f"({c}) * {s}" for c, s in zip(coords, r.strides)]
        )
        linear = self.affine(linear)
        key = r.name, r.slot
        if key in self.panel_slots:
            value = self.panel_slots[key]
            names = set(re.findall(r"\bt\d+\b", linear))
            positions = np.asarray(
                eval(
                    linear,
                    {"__builtins__": {}},
                    {n: self.indices[n] for n in names},
                )
            )
            rows, cols = positions // value.n, positions % value.n
            for mi, ni, mm, nn, var in value.panels:
                if np.all(
                    (rows >= mi)
                    & (rows < mi + mm)
                    & (cols >= ni)
                    & (cols < ni + nn)
                ):
                    return f"{var}[{self.affine(f'({linear}) % {value.n} - {ni}')}, {self.affine(f'({linear}) // {value.n} - {mi}')}]"
            raise NotImplementedError(
                "Read spans independent result panels; use panel-wise compute"
            )
        var = self.vars[key]
        if r.level == ir.MEMORY_LEVEL_SCRATCHPAD:
            box = self.boxes[r.name]
            p, _ = self.storage_shape(box)
            width = box.shape[-1]
            if width % p:
                column = self.affine(f'({linear}) % {width}')
                row = self.affine(f'({linear}) // {width}')
                return f"{var}[{self.affine(f'({column}) % {p}')}, {self.affine(f'({row}) * {math.ceil(width/p)} + ({column}) // {p}')}]"
            return f"{var}[{self.affine(f'({linear}) % {p}')}, {self.affine(f'({linear}) // {p}')}]"
        size = math.prod(self.boxes[r.name].shape)
        return f"{var}.reshape(({size},))[{linear}]"

    def read(self, r, shape=None):
        if isinstance(r, PanelValue):
            return r
        if not isinstance(r, Ref):
            return str(r)
        shape = shape or r.shape
        panel = self.panel_slots.get((r.name, r.slot))
        if (
            panel is not None
            and r.offset == 0
            and math.prod(shape) == panel.m * panel.n
        ):
            return panel
        box = self.boxes[r.name]
        if (r.level == ir.MEMORY_LEVEL_SCRATCHPAD and r.offset == 0
            and r.shape == tuple(box.shape) == tuple(shape)
            and shape[-1] > 128 and shape[-1] % 128):
            return self.vars[r.name, r.slot]
        coords, linear, p, f = self.coordinates(shape)
        # NumPy broadcasting, expressed explicitly for NKI's partition layout.
        if r.shape != shape:
            offset = len(shape) - len(r.shape)
            coords = [
                "0" if n == 1 else coords[i + offset]
                for i, n in enumerate(r.shape)
            ]
        expr = self.index(r, coords)
        if r.level != ir.MEMORY_LEVEL_SCRATCHPAD:
            expr = f"nl.load({expr}, mask={linear} < {math.prod(shape)})"
        return self.tmp(expr)

    def renew_full_slot(self, r):
        """Give a fully overwritten logical slot a new NKI SSA tensor.

        The logical slot and its schedule/lifetime do not change. NKI owns
        physical allocation; distinct definitions prevent stale ISA operands
        when the same source-level ndarray is assigned multiple times.
        """
        if r.level != ir.MEMORY_LEVEL_SCRATCHPAD:
            return
        self.panel_slots.pop((r.name, r.slot), None)
        box = self.boxes[r.name]
        if r.offset != 0 or math.prod(r.shape) != math.prod(box.shape):
            return
        var = self.tmp(
            f"nisa.memset({self.storage_shape(box)!r}, value=0, dtype=nl.{box.dtype}, engine=nisa.vector_engine)"
            if box.shape[-1] > 128 and box.shape[-1] % 128 and self.tuning.isa_lowering
            else f"nl.ndarray({self.storage_shape(box)!r}, dtype=nl.{box.dtype}, buffer=nl.sbuf)"
        )
        self.vars[r.name, r.slot] = var
        self.stats["sbuf_definitions"] += 1

    def write(self, r, value):
        if isinstance(value, PanelValue):
            if (
                r.level != ir.MEMORY_LEVEL_SCRATCHPAD
                or r.offset
                or math.prod(r.shape) != value.m * value.n
            ):
                raise NotImplementedError(
                    "Panel result requires a full local destination"
                )
            self.panel_slots[r.name, r.slot] = value
            self.stats["panel_result_bindings"] += 1
            return
        self.renew_full_slot(r)
        box = self.boxes[r.name]
        if (r.level == ir.MEMORY_LEVEL_SCRATCHPAD and r.offset == 0
            and r.shape == tuple(box.shape) and r.shape[-1] > 128 and r.shape[-1] % 128):
            self.emit(f"{self.vars[r.name,r.slot]}[...] = nisa.tensor_copy({value}, engine=nisa.vector_engine)")
            return
        coords, linear, p, f = self.coordinates(r.shape)
        expr = self.index(r, coords)
        if r.level == ir.MEMORY_LEVEL_SCRATCHPAD:
            self.emit(
                f"{expr} = nisa.tensor_copy({value}, engine=nisa.vector_engine)"
                if self.tuning.isa_lowering
                else f"{expr} = {value}"
            )
        else:
            self.emit(
                f"nl.store({expr}, {value}, mask={linear} < {math.prod(r.shape)})"
            )

    def copy(
        self,
        src,
        dst,
        indices,
        sizes,
        semaphore,
        dims=None,
        strides=None,
        transposed=False,
        pad=None,
        pad_value=0,
        post_count=1,
        count=None,
    ):
        self.stats["dma_copies"] += 1
        self.events.append(
            dict(
                kind="copy",
                src=src.name,
                dst=dst.name,
                src_slot=src.slot,
                dst_slot=dst.slot,
                indices=indices,
                sizes=sizes,
                transposed=transposed,
            )
        )
        step = strides or sizes
        offsets = [0] * len(sizes)
        for idx, d in zip(
            indices, range(len(sizes)) if dims is None else dims
        ):
            offsets[d] = idx * step[d]
        load = src.level != ir.MEMORY_LEVEL_SCRATCHPAD
        if load:
            self.renew_full_slot(dst)
        shape = tuple(sizes)
        if (
            src.level != ir.MEMORY_LEVEL_SCRATCHPAD
            or dst.level != ir.MEMORY_LEVEL_SCRATCHPAD
        ):
            self.transfer_records.append(
                dict(
                    kind="scheduled_copy",
                    source=src.name,
                    destination=dst.name,
                    source_slot=src.slot,
                    destination_slot=dst.slot,
                    shape=list(shape),
                    dtype=dst.dtype,
                    read_bytes=(
                        math.prod(shape)
                        * (2 if src.dtype in ("bfloat16", "float16") else 4)
                        if load
                        else 0
                    ),
                    write_bytes=(
                        0
                        if load
                        else math.prod(shape)
                        * (2 if dst.dtype in ("bfloat16", "float16") else 4)
                    ),
                )
            )
        if not shape:
            shape = (1,)
        cols = shape[-1]
        rows = math.prod(shape[:-1])
        padding = [0] * len(shape)
        if pad:
            if len(pad) != len(shape):
                raise ValueError(
                    "DMA padding must have one entry per dimension"
                )
            padding = list(pad)
        if transposed and len(shape) != 2:
            raise NotImplementedError("Transposed DMA rank > 2")
        # Legal instruction subdivision only: the scheduled DMA window and
        # software-pipeline slot remain unchanged. HBM free dimension must be
        # contiguous, so stage row-major, then physically transpose to SBUF.
        from .execution import dma_panels

        row_boundary = shape[-2] if len(shape) > 1 else 1
        dma_tuning = (
            replace(self.tuning, dma_transpose=False)
            if len(shape) > 2 or any(padding)
            else self.tuning
        )
        panels = dma_panels(
            rows,
            cols,
            row_boundary,
            transpose=not transposed,
            store=not load,
            tuning=dma_tuning,
        )
        for row, col, nr, nc, direct in panels:
            self.stats["isa_dma_panels"] += 1
            ip = self.tmp(f"nl.arange({nr})[:, None]")
            jf = self.tmp(f"nl.arange({nc})[None, :]")
            linear = f"(({ip} + {row}) * {cols} + {jf} + {col})"
            local = [
                f"(({linear} // {st}) % {n})"
                for n, st in zip(shape, globals()["strides"](shape))
            ]
            external = [
                f"({c} + {off} - {pa})"
                for c, off, pa in zip(local, offsets, padding)
            ]
            if load:
                mask = " & ".join(
                    f"(({c}) >= 0) & (({c}) < {n})"
                    for c, n in zip(external, src.shape)
                )
                mask = self.mask(mask)
                if direct and mask != "True":
                    # Masked halo panels use the existing staged path; large
                    # source partition axes cannot be loaded by nl.load.
                    if nr > 128:
                        raise NotImplementedError(
                            "Large masked DMA transpose panels require smaller dma_rows"
                        )
                    direct = False
                if mask == "False":
                    value = None
                elif self.tuning.isa_lowering and not direct:
                    value = self.tmp(
                        f"nl.ndarray(({nr}, {nc}), dtype=nl.{dst.dtype}, buffer=nl.sbuf)"
                    )
                    self.emit(
                        f"nisa.dma_copy(dst={value}, src={self.index(src, external)}, mask={mask})"
                    )
                else:
                    value = self.tmp(
                        f"{'nisa.dma_transpose' if direct else 'nl.load'}({self.index(src, external)}, mask={mask})"
                    )
                if mask != "True":
                    fill = repr(pad_value)
                    if pad_value == float("-inf"):
                        # SDK 2.20 rejects JSON -Infinity immediates. A
                        # bitcast constructs exactly the same FP32 value.
                        bits = self.tmp(
                            f"nl.full(({nr}, {nc}), 4286578688, dtype=nl.uint32)"
                        )
                        fill = self.tmp(f"{bits}.view(nl.float32)")
                    if mask == "False":
                        value = (
                            fill
                            if pad_value == float("-inf")
                            else self.tmp(
                                f"nl.full(({nr}, {nc}), {fill}, dtype=nl.{dst.dtype})"
                            )
                        )
                    else:
                        value = self.tmp(f"nl.where({mask}, {value}, {fill})")
                if transposed:
                    # source [K,N] -> logical [N,K], stored [K,N].
                    self.emit(
                        f"{self.index(dst, list(reversed(local)))} = "
                        + (
                            f"nisa.tensor_copy({value}, engine=nisa.vector_engine)"
                            if self.tuning.isa_lowering
                            else value
                        )
                    )
                else:
                    if not direct:
                        value = self.transpose(
                            value, copy_engine="scalar", dtype=dst.dtype
                        )
                    cp = self.tmp(f"nl.arange({nc})[:, None]")
                    rf = self.tmp(f"nl.arange({nr})[None, :]")
                    lin = f"(({rf} + {row}) * {cols} + {cp} + {col})"
                    target = [
                        f"(({lin} // {st}) % {n})"
                        for n, st in zip(shape, globals()["strides"](shape))
                    ]
                    # A copy may expose a squeezed/reshaped local view.
                    if dst.shape != shape:
                        target = [
                            f"(({lin} // {st}) % {n})"
                            for n, st in zip(
                                dst.shape,
                                (
                                    dst.strides
                                    if dst.strides
                                    == globals()["strides"](dst.shape)
                                    else globals()["strides"](dst.shape)
                                ),
                            )
                        ]
                    self.emit(
                        f"{self.index(dst, target)} = "
                        + (
                            f"nisa.tensor_copy({value}, engine=nisa.scalar_engine)"
                            if self.tuning.isa_lowering
                            else value
                        )
                    )
            else:
                cp = self.tmp(f"nl.arange({nc})[:, None]")
                rf = self.tmp(f"nl.arange({nr})[None, :]")
                lin = f"(({rf} + {row}) * {cols} + {cp} + {col})"
                source = [
                    f"(({lin} // {st}) % {n})"
                    for n, st in zip(
                        src.shape, globals()["strides"](src.shape)
                    )
                ]
                value = self.tmp(self.index(src, source))
                value = self.transpose(value, dtype=src.dtype)
                mask = " & ".join(
                    f"(({c}) >= 0) & (({c}) < {n})"
                    for c, n in zip(external, dst.shape)
                )
                mask = self.mask(mask)
                self.emit(
                    f"nisa.dma_copy(dst={self.index(dst, external)}, src={value}, mask={mask})"
                    if self.tuning.isa_lowering
                    else f"nl.store({self.index(dst, external)}, {value}, mask={mask})"
                )
        self.signal(semaphore, post_count)

    def transpose(self, value, copy_engine="vector", dtype="float32"):
        self.stats["isa_transposes"] += 1
        if self.tuning.isa_lowering:
            from .isa import transpose

            # All transpose panels are <=128x128; counts/constants do not
            # depend on exact shape. Timing is evaluated with actual geometry.
            engine = (
                "ScalarE"
                if self.tuning.copy_policy == "scalar"
                or copy_engine == "scalar"
                else "VectorE"
            )
            expansion = transpose(128, 128, self.hardware, engine, dtype)
            contract = self.hardware.operation_implementation(
                expansion.implementation
            )
            self.used_implementations.add(contract.name)
            if tuple(
                (step.unit, step.operation) for step in contract.steps
            ) != (("TensorE", "transpose"), (engine, "copy")):
                raise ValueError(
                    "Unsupported transpose implementation realization"
                )
            self.expanded_isa.update(
                {
                    name: count
                    for name, count in expansion.instructions
                    if name in ("LDWEIGHTS", "MATMUL_TRANSPOSE")
                }
            )
            psum = self.tmp(
                f"nisa.nc_transpose({value}, engine=nisa.tensor_engine)"
            )
            return self.local_copy(
                psum,
                engine=(
                    "scalar"
                    if self.tuning.copy_policy == "scalar"
                    else copy_engine
                ),
            )
        # SDK neuronx-cc 2.22 produces stale values for nc_transpose on
        # overwritten SBUF slots. Isolated device probes pass with nl.transpose
        # while changing nc_matmul/tensor_copy alone does not fix the error.
        # Keep this realization until an SDK-specific native route is verified.
        return self.tmp(f"nl.transpose({value})")

    def local_copy(self, value, dtype=None, engine="vector"):
        cast = f", dtype=nl.{dtype}" if dtype else ""
        if self.tuning.isa_lowering:
            self.stats[f"isa_{engine}_copies"] += 1
            result = self.tmp(
                f"nl.ndarray({value}.shape, dtype={'nl.' + dtype if dtype else value + '.dtype'}, buffer=nl.sbuf)"
            )
            self.emit(
                f"{result}[...] = nisa.tensor_copy({value}{cast}, engine=nisa.{engine}_engine)"
            )
            return result
        return self.tmp(f"nl.copy({value}{cast})")

    def map_panels(self, expression, *values):
        panel = next((v for v in values if isinstance(v, PanelValue)), None)
        if panel is None:
            return self.tmp(expression(*values))
        results = []
        for idx, (mi, ni, mm, nn, _) in enumerate(panel.panels):
            args = []
            for value in values:
                if isinstance(value, PanelValue):
                    assert (value.m, value.n) == (panel.m, panel.n)
                    assert value.panels[idx][:4] == (mi, ni, mm, nn)
                    args.append(value.panels[idx][4])
                else:
                    try:
                        literal = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        literal = None
                    if isinstance(literal, (int, float)):
                        args.append(value)
                        continue
                    ip = self.tmp(f"nl.arange({nn})[:, None]")
                    jf = self.tmp(f"nl.arange({mm})[None, :]")
                    args.append(
                        f"{value}[{ip}, ({jf}+{mi})*{math.ceil(panel.n / 128)}+{ni // 128}]"
                    )
            results.append((mi, ni, mm, nn, self.tmp(expression(*args))))
        return PanelValue(panel.m, panel.n, results)

    def gemm(self, a, b, m, n, k, transposed, dtype):
        """Lower one scheduled tile into ISA panels without retiling HBM."""
        if self.tuning.isa_lowering:
            from dataclasses import asdict
            from .dependencies import compute_graph

            graph, _ = compute_graph(
                self.hardware,
                m,
                n,
                k,
                32 if a.dtype == "float32" else 16,
                transposed,
                self.tuning,
                a.dtype,
                dtype,
            )
            self.realized_compute_graphs.append(asdict(graph))
            self.matrix_buffer_bindings.append(
                dict(
                    input_name=a.name,
                    weight_name=b.name,
                    input_slots=self.boxes[a.name].bank_count or 1,
                    weight_slots=self.boxes[b.name].bank_count or 1,
                )
            )
            self.used_implementations.update(
                node.implementation
                for node in graph.nodes
                if node.implementation
            )
        p = min(n, 128)
        result = (
            None
            if self.tuning.isa_lowering
            else self.tmp(
                f"nl.ndarray(({p}, {m * n // p}), dtype=nl.{dtype}, buffer=nl.sbuf)"
            )
        )
        result_panels = []
        from .execution import matmul_panels

        for mi, ni, mm, nn, kpanels in matmul_panels(m, n, k):
            acc = self.tmp(
                f"nl.zeros(({nn},{mm}), dtype=nl.float32, buffer=nl.psum)"
            )
            for ki, kk in kpanels:
                ar = replace(
                    a,
                    shape=(mm, kk),
                    offset=a.offset + mi * a.strides[0] + ki * a.strides[1],
                )
                br = replace(
                    b,
                    shape=(nn, kk) if transposed else (kk, nn),
                    offset=b.offset
                    + (
                        ni * b.strides[0] + ki * b.strides[1]
                        if transposed
                        else ki * b.strides[0] + ni * b.strides[1]
                    ),
                )
                av, bv = self.read(ar), self.read(br)
                if m > 512 or n > 128 or k > 128:
                    av = self.local_copy(av)
                    bv = self.local_copy(bv)
                if not transposed:
                    bv = self.transpose(bv, dtype=b.dtype)
                self.emit(
                    (
                        f"{acc} += nisa.nc_matmul({bv}, {av})"
                        if self.tuning.explicit_isa
                        else f"{acc} += nl.matmul({bv}, {av}, transpose_x=True)"
                    )
                )
                self.stats["tensor_instructions"] += 1
                if self.tuning.isa_lowering:
                    from .isa import matmul

                    self.expanded_isa.update(
                        dict(
                            matmul(
                                mm,
                                nn,
                                kk,
                                32 if a.dtype == "float32" else 16,
                                self.hardware,
                                a.dtype,
                            ).instructions
                        )
                    )
            value = self.local_copy(
                acc,
                dtype,
                engine=(
                    "scalar"
                    if self.tuning.copy_policy == "scalar"
                    else "vector"
                ),
            )
            if self.tuning.isa_lowering:
                result_panels.append((mi, ni, mm, nn, value))
                continue
            ip = self.tmp(f"nl.arange({nn})[:, None]")
            jf = self.tmp(f"nl.arange({mm})[None, :]")
            self.emit(
                f"{result}[{ip}, ({jf}+{mi})*{n // p}+{ni // p}] = {value}"
            )
        return (
            PanelValue(m, n, result_panels)
            if self.tuning.isa_lowering
            else result
        )

    def compute(self, o):
        prims = (
            [o.prim] if o.WhichOneof("op_type") == "prim" else o.fused.op_list
        )
        dest = [
            self.ref(x.destination)
            for x in o.outputs
            if x.WhichOneof("result_type") == "destination"
        ]
        if len(dest) != 1:
            raise NotImplementedError("Compute requires one destination")
        d = dest[0]
        if not any(p.target.split("::")[-1] in ("matmul", "linear", "conv2d") for p in prims):
            names = tuple(p.target.split("::")[-1] for p in prims)
            operands = {}
            window = 1
            for p in prims:
                kw = {key: self.arg(value) for key, value in p.kwargs.items()
                      if value.WhichOneof("arg_type") != "tensor_box" or value.tensor_box.box.HasField("memory")}
                for key, value in kw.items():
                    if isinstance(value, Ref) and key not in ("mean", "variance", "normalized", "max", "sum"):
                        operands[value.name, value.slot] = value.shape
                if "kernel_size" in kw:
                    window = math.prod(kw["kernel_size"])
            self.vector_records.append(dict(name=o.name, operations=names, shape=d.shape,
                                            inputs=tuple(operands.values()), pool_window=window))
        for p in prims:
            kw = {k: self.arg(v) for k, v in p.kwargs.items()}
            name = p.target.split("::")[-1]
            a = kw.get("input", kw.get("self"))
            if name in ("matmul", "linear"):
                b = kw.get("other", kw.get("weight"))
                if math.prod(a.shape[:-2]) == 1:
                    a = replace(a, shape=a.shape[-2:], strides=a.strides[-2:])
                if math.prod(b.shape[:-2]) == 1:
                    b = replace(b, shape=b.shape[-2:], strides=b.strides[-2:])
                if len(a.shape) != 2 or len(b.shape) != 2:
                    raise NotImplementedError(
                        "Scheduled GEMM tile must be rank 2"
                    )
                m, k = a.shape
                transposed = (
                    name == "matmul"
                    and p.target.startswith("quantized_ops::")
                ) or (name == "linear" and p.target.startswith("aten::"))
                n = b.shape[0] if transposed else b.shape[1]
                value = self.gemm(a, b, m, n, k, transposed, d.dtype)
                if kw.get("bias") is not None:
                    value = self.map_panels(
                        lambda x, y: (
                            f"nisa.tensor_tensor({x}, {y}, op=nl.add, engine=nisa.vector_engine)"
                            if self.tuning.isa_lowering
                            else f"{x}+{y}"
                        ),
                        value,
                        self.read(kw["bias"], d.shape),
                    )
            elif name in ("layer_norm", "rms_norm", "softmax"):
                from .lowering import realize_reduction
                value = realize_reduction(self, name, kw, d)
            elif name == "conv2d":
                if not p.target.startswith("quantized_ops::"):
                    raise NotImplementedError(
                        "Convolution requires shared NHWC layout"
                    )
                w = kw["weight"]
                batch, ih, iw, channels = a.shape
                rh, rw, wc, outchannels = w.shape
                _, oh, ow, _ = d.shape
                step = kw.get("stride", [1, 1])
                pad = kw.get("padding", [0, 0])
                if (
                    batch != 1
                    or channels > 128
                    or outchannels > 128
                    or oh * ow > 512
                    or wc != channels
                ):
                    raise ValueError("Illegal scheduled convolution tile")
                if (
                    kw.get("groups", 1) != 1
                    or kw.get("dilation", [1, 1]) != [1, 1]
                    or any(pad)
                ):
                    raise NotImplementedError(
                        "Convolution expects bufferized zero-padded halo, groups=dilation=1"
                    )
                if any(
                    ref.offset or ref.strides != strides(ref.shape)
                    for ref in (a, w)
                ):
                    raise NotImplementedError(
                        "Convolution requires contiguous local tiles"
                    )
                av = self.vars[a.name, a.slot]
                wv = self.vars[w.name, w.slot]
                cp = self.tmp(f"nl.arange({channels})[:, None, None]")
                hi = self.tmp(f"nl.arange({oh})[None, :, None]")
                wi = self.tmp(f"nl.arange({ow})[None, None, :]")
                kp = self.tmp(f"nl.arange({outchannels})[:, None]")
                cf = self.tmp(f"nl.arange({channels})[None, :]")
                acc = self.tmp(
                    f"nl.zeros(({outchannels}, {oh * ow}), dtype=nl.float32, buffer=nl.psum)"
                )
                for r in range(rh):
                    for c in range(rw):
                        act = self.tmp(
                            f"{av}[{cp}, ({hi}*{step[0]}+{r})*{iw}+{wi}*{step[1]}+{c}]"
                        )
                        act = self.local_copy(act)
                        act = self.tmp(
                            f"{act}.reshape(({channels},{oh * ow}))"
                        )
                        weight = self.tmp(
                            f"{wv}[{kp}, {r * rw * channels + c * channels}+{cf}]"
                        )
                        weight = self.transpose(weight, dtype=w.dtype)
                        self.emit(
                            f"{acc} += nisa.nc_matmul({weight}, {act})"
                            if self.tuning.isa_lowering
                            else f"{acc} += nl.matmul({weight}, {act}, transpose_x=True)"
                        )
                        self.stats["tensor_instructions"] += 1
                        if self.tuning.isa_lowering:
                            from .isa import matmul

                            self.expanded_isa.update(
                                dict(
                                    matmul(
                                        oh * ow,
                                        outchannels,
                                        channels,
                                        32 if a.dtype == "float32" else 16,
                                    ).instructions
                                )
                            )
                value = self.local_copy(
                    acc,
                    d.dtype,
                    engine=(
                        "scalar"
                        if self.tuning.copy_policy == "scalar"
                        else "vector"
                    ),
                )
                if kw.get("bias") is not None:
                    value = self.tmp(
                        f"{value}+{self.read(kw['bias'], d.shape)}"
                    )
            elif name in (
                "max_pool2d",
                "avg_pool2d",
                "adaptive_avg_pool2d",
                "_adaptive_avg_pool2d",
            ):
                if not p.target.startswith("quantized_ops::"):
                    raise NotImplementedError(
                        "Pooling requires shared NHWC layout normalization"
                    )
                if name in ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
                    if a.shape[0] != 1 or tuple(d.shape[1:3]) != (1, 1):
                        raise NotImplementedError("Global pooling only")
                    av = self.read(a)
                    value = self.tmp(
                        f"nl.sum({av}, axis=[1], keepdims=True) / {math.prod(a.shape[1:3])}"
                    )
                else:
                    if name == "avg_pool2d" and (
                        not kw.get("count_include_pad", True)
                        or kw.get("divisor_override") is not None
                    ):
                        raise NotImplementedError(
                            "Average pooling requires count_include_pad without divisor_override"
                        )
                    if kw.get("ceil_mode", False) or kw.get(
                        "dilation", [1, 1]
                    ) != [1, 1]:
                        raise NotImplementedError("Pool dilation/ceil mode")
                    pair = lambda v: [v, v] if isinstance(v, int) else v
                    kernel = pair(kw["kernel_size"])
                    step = pair(kw.get("stride") or kernel)
                    pad = pair(kw.get("padding", [0, 0]))
                    if a.shape[0] != 1:
                        raise NotImplementedError("Batch pooling tile")
                    if any(pad) or a.offset or a.strides != strides(a.shape):
                        raise NotImplementedError(
                            "Pooling requires contiguous local tiles with DMA halo padding"
                        )
                    _, oh, ow, channels = d.shape
                    cp = self.tmp(f"nl.arange({channels})[:, None, None]")
                    hi = self.tmp(f"nl.arange({oh})[None, :, None]")
                    wi = self.tmp(f"nl.arange({ow})[None, None, :]")
                    av = self.vars[a.name, a.slot]
                    value = None
                    for r in range(kernel[0]):
                        for c in range(kernel[1]):
                            v = self.tmp(
                                f"{av}[{cp}, ({hi}*{step[0]}+{r}-{pad[0]})*{a.shape[2]}+{wi}*{step[1]}+{c}-{pad[1]}]"
                            )
                            value = (
                                v
                                if value is None
                                else self.tmp(
                                    f"nisa.tensor_tensor({value}, {v}, op=nl.maximum, engine=nisa.vector_engine)"
                                    if name == "max_pool2d"
                                    else f"{value}+{v}"
                                )
                            )
                    value = self.tmp(
                        f"{value}.reshape(({channels},{oh * ow}))"
                    )
                    if name == "avg_pool2d":
                        value = self.tmp(f"{value}/{math.prod(kernel)}")
            elif name in ("add", "add_", "sub", "mul", "div", "maximum"):
                a = self.read(a, d.shape)
                b = self.read(kw["other"], d.shape)
                op = {
                    "maximum": "maximum",
                    "add": "add",
                    "add_": "add",
                    "sub": "subtract",
                    "mul": "multiply",
                    "div": "divide",
                }[name]
                alpha = kw.get("alpha", 1)
                if self.tuning.isa_lowering and name != "div":
                    if alpha != 1:
                        b = self.map_panels(
                            lambda x: f"nisa.tensor_scalar({x}, op0=nl.multiply, operand0={alpha}, engine=nisa.vector_engine)",
                            b,
                        )
                    if isinstance(kw["other"], (int, float)):
                        value = self.map_panels(
                            lambda x: f"nisa.tensor_scalar({x}, op0=nl.{op}, operand0={kw['other'] * alpha}, engine=nisa.vector_engine)",
                            a,
                        )
                    else:
                        value = self.map_panels(
                            lambda x, y: f"nisa.tensor_tensor({x}, {y}, op=nl.{op}, engine=nisa.vector_engine)",
                            a,
                            b,
                        )
                else:
                    symbol = {
                        "add": "+",
                        "add_": "+",
                        "sub": "-",
                        "mul": "*",
                        "div": "/",
                    }[name]
                    value = self.map_panels(
                        lambda x, y: f"{x} {symbol} ({y} * {alpha})", a, b
                    )
            elif name in ("relu", "exp", "sigmoid", "tanh", "silu"):
                av = self.read(a, d.shape)

                def unary(x):
                    if self.tuning.isa_lowering and name != "silu":
                        return f"nisa.activation(op=nl.{name}, data={x})"
                    return (
                        f"nl.maximum({x}, 0)"
                        if name == "relu"
                        else (
                            f"{x} * nl.sigmoid({x})"
                            if name == "silu"
                            else f"nl.{name}({x})"
                        )
                    )

                value = self.map_panels(unary, av)
            elif name in ("clone", "alias", "view", "reshape", "as_strided"):
                if name == "as_strided" and (
                    tuple(kw["stride"]) != strides(kw["size"])
                    or kw.get("storage_offset", 0) not in (0, None)
                ):
                    raise NotImplementedError(
                        "Noncontiguous as_strided compute"
                    )
                value = self.read(a, d.shape)
            else:
                raise NotImplementedError(f"Scheduled compute {p.target}")
            self.env[p.name] = value
        self.write(d, value)
        self.events.append(
            dict(
                kind="compute",
                name=o.name,
                dst=d.name,
                slot=d.slot,
                primitives=[p.target for p in prims],
            )
        )

    def source(self):
        if not any(
            o.WhichOneof("op_type") == "prim"
            and o.prim.target.startswith("voyager::")
            for o in self.model.ops
        ):
            raise ValueError(
                "Expected bufferized Voyager collaterals, not semantic ATen export"
            )
        self.region(self.model.ops)
        result = ", ".join(self.vars[b.node, 0] for b in self.model.outputs)
        code = (
            "\n".join(
                self.header
                + self.declarations
                + self.lines
                + ["    return (" + result + ",)"]
            )
            + "\n"
        )
        ast.parse(code)
        return code


def convert(root, output=None, target=None, *, context=None):
    from .hardware import TARGETS

    from .hardware import neuron_core
    from voyager_compiler.compilation import CompilerContext

    root = Path(root)
    if context is None:
        if target is None:
            target = json.loads((root / "hardware.json").read_text())[
                "hardware"
            ]["name"]
        if target not in TARGETS:
            raise ValueError(target)
        context = CompilerContext.from_artifacts(
            root, neuron_core(TARGETS[target])
        )
    if target is not None and target != context.hardware.name:
        raise ValueError("Target differs from recorded compiler context")
    context.check_artifacts(root)
    target = context.hardware.name
    output = Path(output) if output else root / "nki"
    output.mkdir(parents=True, exist_ok=True)
    model = text_format.Parse((root / "model.txt").read_text(), ir.Model())
    converter = Converter(
        model, target, context.policy.tuning, context.hardware
    )
    source = converter.source()
    from .dependencies import audit_realization

    record = json.loads((root / "hardware.json").read_text())
    audit = audit_realization(record.get("execution_plans", ()), converter)
    from .program_analysis import analyze_program

    program_analysis = analyze_program(source, converter, record)

    (output / "program.py").write_text(source)
    manifest = dict(
        format="voyager-bufferized-nki-v1",
        target=target,
        compiler=context.record(),
        source_sha256=hashlib.sha256(
            (root / "model.txt").read_bytes()
        ).hexdigest(),
        stats=converter.stats,
        expanded_isa=dict(converter.expanded_isa),
        execution_contract=(
            "nki-isa-v1" if converter.tuning.isa_lowering else "legacy-nki"
        ),
        high_level_operations=dict(
            Counter(
                re.findall(
                    r"nl\.(load|store|transpose|matmul|copy|sum|maximum|where|sigmoid|exp|tanh)\(",
                    source,
                )
            )
        ),
        events=converter.events,
        dependency_audit=audit,
        program_analysis=program_analysis,
        operation_implementations=sorted(converter.used_implementations),
        arguments=[
            dict(name=b.node, shape=list(b.shape), dtype=b.dtype)
            for b in converter.arguments
        ],
        inputs=len(model.inputs),
        outputs=[
            dict(name=b.node, shape=list(b.shape), dtype=b.dtype)
            for b in model.outputs
        ],
        buffers=[
            dict(
                name=b.node,
                shape=list(b.shape),
                level=b.memory.level,
                address=b.memory.address,
                slots=b.bank_count or 1,
                stride=b.bank_stride_bytes,
            )
            for b in converter.boxes.values()
        ],
        scheduling="Voyager control specialized in source order; NKI schedules engines from tensor dependencies",
        allocation="Voyager buffer identities and slots retained; NKI assigns final physical addresses",
    )
    (output / "plan.json").write_text(json.dumps(manifest, indent=2))
    return manifest
