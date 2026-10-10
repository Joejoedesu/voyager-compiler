"""Select typed target instructions from scheduled Voyager operations.

Static scalar control is specialized (as in the Gemmini collateral consumer).
No tensor arithmetic is performed on the host and no whole-op tiler is used.
DMA windows, buffer slots, reduction order and fused primitives come from IR.
This compile-stage pass fixes all target choices, types every local value and
uses the shared lifetime allocator to select physical SBUF and PSUM placement.
NKI retains final engine instruction scheduling after formal conversion.
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
    """Independent SBUF panels; M-partition panels bind directly to row consumers."""

    m: int
    n: int
    panels: list  # (mi, ni, mm, nn, NKI variable)
    row_major: bool = False


def strides(shape):
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


class InstructionPlanner:
    def __init__(
        self,
        model,
        target,
        tuning=None,
        hardware=None,
        movement_bindings=None,
        matrix_bindings=(),
    ):
        from .execution import TrainiumTuning

        self.tuning = tuning or TrainiumTuning()
        from .hardware import neuron_core, TARGETS

        self.hardware = hardware or neuron_core(TARGETS[target])
        from .movement_search import MovementSelector

        self.movement_selector = (
            MovementSelector(self.hardware, movement_bindings)
            if movement_bindings is not None
            else None
        )
        self._realizing_movement = False
        self.matrix_bindings = matrix_bindings
        self.matrix_choices = []
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
        self.row_buffers = set()
        self.k_weight_buffers = set()
        self.pool_buffers = {}
        self.replicated_parameters = {}
        self.select_layouts(model)
        self.propagate_row_layouts(model)
        self.select_matrix_layouts(model)
        from .instruction_plan import Builder

        self.builder = Builder(self.arguments, self.hardware)
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
                self.builder.add(
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

    def propagate_row_layouts(self, message):
        """Keep same-shape local pointwise edges in their consumer's row layout.

        This closure runs before any instruction/storage definition. Matrix
        boundaries keep their explicit panel-to-row conversion in ``write``.
        HBM edges do not propagate it. Last-axis scalar broadcasts preserve
        the row partition when their leading row extent agrees.
        """
        groups = []

        def visit(msg):
            if isinstance(msg, ir.Operation):
                prims = (
                    [msg.prim]
                    if msg.WhichOneof("op_type") == "prim"
                    else (
                        msg.fused.op_list
                        if msg.WhichOneof("op_type") == "fused"
                        else ()
                    )
                )
                names = {p.target.split("::")[-1] for p in prims}
                if names and names <= {
                    "add",
                    "sub",
                    "mul",
                    "maximum",
                    "relu",
                    "sigmoid",
                    "tanh",
                    "exp",
                    "clone",
                    "reciprocal",
                }:
                    boxes = [
                        v.tensor_box.box
                        for p in prims
                        for v in p.kwargs.values()
                        if v.WhichOneof("arg_type") == "tensor_box"
                    ]
                    boxes += [
                        o.destination.box
                        for o in msg.outputs
                        if o.WhichOneof("result_type") == "destination"
                    ]
                    boxes = [
                        b
                        for b in boxes
                        if b.HasField("memory")
                        and b.memory.level == ir.MEMORY_LEVEL_SCRATCHPAD
                    ]
                    shapes = [tuple(self.boxes[b.node].shape) for b in boxes]
                    row_broadcast = (
                        bool(shapes)
                        and all(
                            len(s) == 2 and s[0] == shapes[0][0] for s in shapes
                        )
                        and all(
                            s[1] in (1, max(t[1] for t in shapes))
                            for s in shapes
                        )
                    )
                    if boxes and (len(set(shapes)) == 1 or row_broadcast):
                        groups.append({b.node for b in boxes})
            for field, value in msg.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    values = (
                        (
                            [value[k] for k in sorted(value)]
                            if field.message_type.GetOptions().map_entry
                            else value
                        )
                        if field.is_repeated
                        else [value]
                    )
                    for child in values:
                        visit(child)

        visit(message)
        changed = True
        while changed:
            before = len(self.row_buffers)
            for group in groups:
                if group & self.row_buffers:
                    self.row_buffers.update(group)
            changed = len(self.row_buffers) != before

    def storage_shape(self, b):
        if b.node in self.k_weight_buffers:
            k, n = b.shape
            p = min(k, 128)
            return p, math.ceil(k / p) * n
        if b.node in self.pool_buffers:
            return self.pool_buffers[b.node]["physical"]
        if b.node in self.row_buffers:
            return math.prod(b.shape[:-1]), b.shape[-1]
        if not b.shape:
            return 1, 1
        p = min(128, b.shape[-1] if b.shape else 1)
        return p, math.prod(b.shape[:-1]) * math.ceil(b.shape[-1] / p)

    def select_matrix_layouts(self, message):
        """Bind saved invariant-weight layouts before DMA or local allocation.

        Only full, untransposed 2-D matrix operands have this contract. A local
        buffer cannot acquire conflicting layouts from different consumers.
        """
        requests = {}

        def visit(message):
            if isinstance(message, ir.Operation):
                prims = (
                    [message.prim]
                    if message.WhichOneof("op_type") == "prim"
                    else (
                        message.fused.op_list
                        if message.WhichOneof("op_type") == "fused"
                        else ()
                    )
                )
                for op in prims:
                    if op.target != "aten::matmul":
                        continue
                    a, b = (
                        op.kwargs["input"].tensor_box,
                        op.kwargs["other"].tensor_box,
                    )
                    dest = [
                        o.destination.box
                        for o in message.outputs
                        if o.WhichOneof("result_type") == "destination"
                    ]
                    ashape = tuple(a.output_shape) or tuple(a.box.shape)
                    bshape = tuple(b.output_shape) or tuple(b.box.shape)
                    if len(ashape) != 2 or len(bshape) != 2 or len(dest) != 1:
                        continue
                    matches = [
                        c
                        for c in self.matrix_bindings
                        if (
                            c["m"],
                            c["n"],
                            c["k"],
                            c["transposed"],
                            c["input_row"],
                            c["output_row"],
                        )
                        == (
                            ashape[0],
                            bshape[1],
                            ashape[1],
                            False,
                            a.box.node in self.row_buffers,
                            dest[0].node in self.row_buffers,
                        )
                    ]
                    layouts = {
                        c.get("weight_layout", "generic") for c in matches
                    } or {
                        (
                            self.tuning.matmul_weight_layout
                            if self.tuning.matmul_weight_layout != "auto"
                            else "generic"
                        )
                    }
                    if len(layouts) != 1:
                        raise ValueError(
                            "Ambiguous selected weight layouts for matrix consumers"
                        )
                    layout = layouts.pop()
                    root = self.boxes[b.box.node]
                    if layout == "k_partitioned" and (
                        tuple(root.shape) != bshape
                        or root.bank_count not in (0, 1, 2)
                        or b.box.node in self.row_buffers
                        or root.memory.level != ir.MEMORY_LEVEL_SCRATCHPAD
                    ):
                        raise ValueError(
                            "K-partitioned weights require a private whole local buffer"
                        )
                    requests.setdefault(b.box.node, set()).add(layout)
            for field, value in message.ListFields():
                if field.type != field.TYPE_MESSAGE:
                    continue
                if field.is_repeated:
                    values = (
                        value.values()
                        if field.message_type.GetOptions().map_entry
                        else value
                    )
                    for child in values:
                        if hasattr(child, "ListFields"):
                            visit(child)
                else:
                    visit(value)

        visit(message)
        for name, layouts in requests.items():
            if len(layouts) != 1:
                raise ValueError(
                    "Conflicting matrix layouts for one local weight buffer"
                )
            if "k_partitioned" in layouts:
                self.k_weight_buffers.add(name)

    def select_layouts(self, message):
        """Bind reduction-compatible layouts before instruction expansion."""
        from .lowering import RECIPES

        if isinstance(message, ir.Operation):
            prims = (
                [message.prim]
                if message.WhichOneof("op_type") == "prim"
                else (
                    message.fused.op_list
                    if message.WhichOneof("op_type") == "fused"
                    else ()
                )
            )
            if any(
                p.target.split("::")[-1] in {*RECIPES, "amax", "sum"}
                for p in prims
            ):
                boxes = [
                    v.tensor_box.box
                    for p in prims
                    for v in p.kwargs.values()
                    if v.WhichOneof("arg_type") == "tensor_box"
                ]
                boxes += [
                    o.destination.box
                    for o in message.outputs
                    if o.WhichOneof("result_type") == "destination"
                ]
                for b in boxes:
                    if (
                        b.HasField("memory")
                        and b.memory.level == ir.MEMORY_LEVEL_SCRATCHPAD
                    ):
                        root = self.boxes[b.node]
                        if math.prod(root.shape[:-1]) > 128:
                            raise ValueError(
                                f"{b.node}: selected row layout exceeds 128 partitions"
                            )
                        self.row_buffers.add(b.node)
            for p in prims:
                if p.target.split("::")[
                    -1
                ] == "max_pool2d" and p.target.startswith("quantized_ops::"):
                    b = p.kwargs["input"].tensor_box.box
                    d = next(
                        o.destination.box
                        for o in message.outputs
                        if o.WhichOneof("result_type") == "destination"
                    )
                    if len(b.shape) == 4 and b.shape[0] == b.shape[-1] == 1:
                        kernel = tuple(
                            self.scalar(v)
                            for v in p.kwargs["kernel_size"].scalar_list.values
                        )
                        step = (
                            tuple(
                                self.scalar(v)
                                for v in p.kwargs["stride"].scalar_list.values
                            )
                            or kernel
                        )
                        if len(kernel) != 2 or step != (1, 1):
                            raise ValueError(
                                "Selected spatial pool layout currently requires unit stride"
                            )
                        oh, ow = d.shape[1:3]
                        if oh > 128:
                            raise ValueError(
                                "Selected pool tile exceeds 128 output rows"
                            )
                        self.pool_buffers[b.node] = dict(
                            physical=(oh, kernel[0] * b.shape[2]),
                            input=True,
                            kernel=kernel,
                            output=(oh, ow),
                        )
                        self.pool_buffers[d.node] = dict(
                            physical=(oh, ow),
                            input=False,
                            kernel=kernel,
                            output=(oh, ow),
                        )
        for f, v in message.ListFields():
            if f.type == f.TYPE_MESSAGE:
                values = (
                    (
                        [v[k] for k in sorted(v)]
                        if f.message_type.GetOptions().map_entry
                        else v
                    )
                    if f.is_repeated
                    else (v,)
                )
                for x in values:
                    self.select_layouts(x)

    def emit(self, code):
        # Assembly copies are movement requests too, including explicit output
        # destinations. Selected chains bypass this hook to prevent recursion.
        if (
            self.movement_selector is not None
            and not self._realizing_movement
            and "nisa.tensor_copy(" in code
        ):
            import ast

            node = ast.parse(code).body[0]
            if isinstance(node, ast.Assign) and isinstance(
                node.value, ast.Call
            ):
                call = node.value
                if all(k.arg in ("engine", "dtype") for k in call.keywords):
                    source = ast.unparse(call.args[0])
                    destination = ast.unparse(node.targets[0])
                    self.movement(
                        source, destination=destination, role="assembly"
                    )
                    return
        self.builder.add(code)

    def movement(
        self,
        source,
        *,
        destination=None,
        transpose=False,
        dtype=None,
        role="local",
        preferred_engine="ScalarE",
    ):
        from .instruction_plan import Expr
        from .movement_search import Endpoint, Request, emit_chain

        src = self.builder.value(Expr.parse(source))
        dst = (
            self.builder.value(Expr.parse(destination)) if destination else None
        )
        request = Request(
            Endpoint(src.memory, src.dtype),
            Endpoint(
                dst.memory if dst else "SBUF",
                dst.dtype if dst else (dtype or src.dtype),
                (1, 0) if transpose else (0, 1),
            ),
            tuple(src.shape),
            role,
        )
        chain = self.movement_selector.choose(request, preferred_engine)
        self._realizing_movement = True
        try:
            return emit_chain(self, request, chain, source, destination)
        finally:
            self._realizing_movement = False

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
                iterations = []
                for i in range(
                    self.scalar(f.start),
                    self.scalar(f.end),
                    self.scalar(f.step),
                ):
                    self.env[f.iv] = i
                    self.env.update(
                        (a.name, v) for a, v in zip(f.iter_args, state)
                    )
                    first = len(self.builder.program.instructions)
                    self.region(f.body.ops)
                    iterations.append(
                        (first, len(self.builder.program.instructions))
                    )
                    state = [self.scalar(v) for v in f.body.yields]
                self.outputs(o, state)
                if len(iterations) > 1:
                    self.builder.program.repeated_regions.append(
                        dict(name=o.name, iterations=iterations)
                    )
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
                        self.semaphores[b.node, slot] = int(kw.get("value", 0))
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
            if channels == 1:
                self.vars[d.name, d.slot] = self.tmp(
                    f"{self.vars[a.name,a.slot]}.reshape({d.shape!r})"
                )
                self.boundary_records.pop()
                return
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
        if self.tuning.isa_lowering and len(a.shape) == len(d.shape) == 1:
            from dataclasses import replace

            two_d = dict(kw)
            if name == "aten::slice":
                two_d["dim"] = 1
            if self.boundary_2d(
                replace(
                    a, shape=(1, *a.shape), strides=(a.shape[0], *a.strides)
                ),
                replace(
                    d, shape=(1, *d.shape), strides=(d.shape[0], *d.strides)
                ),
                name,
                two_d,
            ):
                return
        if (
            self.tuning.isa_lowering
            and len(a.shape) == len(d.shape) > 2
            and a.shape[:-2] == d.shape[:-2]
        ):
            from dataclasses import replace

            two_d = dict(kw)
            compatible = True
            if name == "aten::pad":
                compatible = not any(kw["pad"][4:])
                two_d["pad"] = kw["pad"][:4]
            elif name == "aten::slice":
                dim = kw["dim"] % len(a.shape)
                compatible = dim >= len(a.shape) - 2
                two_d["dim"] = dim - (len(a.shape) - 2)
            else:
                compatible = False
            if compatible:
                panels = []
                for batch in range(math.prod(a.shape[:-2])):
                    aa = replace(
                        a,
                        shape=a.shape[-2:],
                        strides=a.strides[-2:],
                        offset=a.offset + batch * math.prod(a.shape[-2:]),
                    )
                    dd = replace(
                        d,
                        shape=d.shape[-2:],
                        strides=d.strides[-2:],
                        offset=d.offset + batch * math.prod(d.shape[-2:]),
                    )
                    if not self.boundary_2d(aa, dd, name, two_d):
                        raise NotImplementedError(
                            "Unsupported batched rectangular boundary"
                        )
                    panels.extend(self.boundary_records[-1]["panels"])
                self.boundary_records[-1]["panels"] = panels
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
                    f"(({c})>=0) & (({c})<{n})" for c, n in zip(source, a.shape)
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
                nr, nc = min(128, d.shape[0] - row), min(128, d.shape[1] - col)
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
            f"(({linear} // {s}) % {n})" for n, s in zip(shape, strides(shape))
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
            if r.name in self.k_weight_buffers:
                width = box.shape[-1]
                p = min(128, box.shape[0])
                row = self.affine(f"({linear}) // {width}")
                col = self.affine(f"({linear}) % {width}")
                return f"{var}[{self.affine(f'({row}) % {p}')}, {self.affine(f'(({row}) // {p}) * {width} + ({col})')}]"
            if r.name in self.row_buffers:
                width = box.shape[-1]
                return f"{var}[{self.affine(f'({linear}) // {width}')}, {self.affine(f'({linear}) % {width}')}]"
            p, _ = self.storage_shape(box)
            width = box.shape[-1]
            if width % p:
                column = self.affine(f"({linear}) % {width}")
                row = self.affine(f"({linear}) // {width}")
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
        if r.name in self.row_buffers and not r.offset:
            return self.vars[r.name, r.slot]
        panel = self.panel_slots.get((r.name, r.slot))
        if (
            panel is not None
            and r.offset == 0
            and math.prod(shape) == panel.m * panel.n
        ):
            return panel
        box = self.boxes[r.name]
        if (
            r.level == ir.MEMORY_LEVEL_SCRATCHPAD
            and r.offset == 0
            and r.shape == tuple(box.shape) == tuple(shape)
            and shape[-1] > 128
            and shape[-1] % 128
        ):
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
            if box.shape[-1] > 128
            and box.shape[-1] % 128
            and self.tuning.isa_lowering
            else f"nl.ndarray({self.storage_shape(box)!r}, dtype=nl.{box.dtype}, buffer=nl.sbuf)"
        )
        self.vars[r.name, r.slot] = var
        self.stats["sbuf_definitions"] += 1

    def write(self, r, value):
        if r.name in self.row_buffers and isinstance(value, PanelValue):
            # A matrix produces partition=N panels; row reductions consume
            # partition=M. Materialize that required local layout edge.
            local = self.tmp(
                f"nl.ndarray(({value.m}, {value.n}), dtype=nl.{r.dtype}, buffer=nl.sbuf)"
            )
            for mi, ni, mm, nn, panel in value.panels:
                if value.row_major:
                    self.emit(
                        f"{local}[{mi}:{mi + mm}, {ni}:{ni + nn}] = nisa.tensor_copy({panel}, engine=nisa.vector_engine)"
                    )
                    continue
                for row in range(0, mm, 128):
                    count = min(128, mm - row)
                    tile = self.transpose(
                        self.local_copy(f"{panel}[:, {row}:{row + count}]")
                    )
                    self.emit(
                        f"{local}[{mi + row}:{mi + row + count}, {ni}:{ni + nn}] = nisa.tensor_copy({tile}, engine=nisa.vector_engine)"
                    )
            self.vars[r.name, r.slot] = local
            self.panel_slots.pop((r.name, r.slot), None)
            return
        if (
            r.name in self.pool_buffers
            and not self.pool_buffers[r.name]["input"]
        ):
            self.vars[r.name, r.slot] = value
            return
        if r.name in self.row_buffers and not isinstance(value, PanelValue):
            self.vars[r.name, r.slot] = value
            return
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
        if (
            r.level == ir.MEMORY_LEVEL_SCRATCHPAD
            and r.offset == 0
            and r.shape == tuple(box.shape)
            and r.shape[-1] > 128
            and r.shape[-1] % 128
        ):
            self.emit(
                f"{self.vars[r.name,r.slot]}[...] = nisa.tensor_copy({value}, engine=nisa.vector_engine)"
            )
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
        for idx, d in zip(indices, range(len(sizes)) if dims is None else dims):
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
        local_ref = dst if load else src
        if local_ref.name in self.pool_buffers:
            self.pool_copy(
                src,
                dst,
                shape,
                offsets,
                pad or (0,) * len(shape),
                pad_value,
                load,
            )
            self.signal(semaphore, post_count)
            return
        if local_ref.name in self.k_weight_buffers:
            root = self.boxes[local_ref.name]
            if (
                not load
                or transposed
                or any(pad or ())
                or local_ref.offset
                or shape != tuple(root.shape)
                or local_ref.shape != shape
                or local_ref.strides != globals()["strides"](shape)
                or len(src.shape) != 2
                or src.strides != globals()["strides"](src.shape)
                or any(
                    off < 0 or off + size > dim
                    for off, size, dim in zip(offsets, shape, src.shape)
                )
            ):
                raise ValueError(
                    "K-partitioned weight DMA requires a full unpadded contiguous load"
                )
            rows, width = shape
            for row in range(0, rows, 128):
                nr = min(128, rows - row)
                for col in range(0, width, self.tuning.dma_columns):
                    nc = min(self.tuning.dma_columns, width - col)
                    ip = self.tmp(f"nl.arange({nr})[:, None]")
                    jf = self.tmp(f"nl.arange({nc})[None, :]")
                    local = [f"{ip}+{row}", f"{jf}+{col}"]
                    external = [
                        f"({c})+{off}" for c, off in zip(local, offsets)
                    ]
                    self.emit(
                        f"nisa.dma_copy(dst={self.index(dst, local)}, src={self.index(src, external)})"
                    )
                    self.stats["isa_dma_panels"] += 1
                    self.stats["k_weight_dma_panels"] += 1
            self.signal(semaphore, post_count)
            return
        if local_ref.name in self.row_buffers:
            if transposed or any(pad or ()):
                raise ValueError(
                    "Row-layout DMA requires an unpadded, untransposed selected rectangle"
                )
            width = shape[-1]
            rows = math.prod(shape[:-1])
            boundary = shape[-2] if len(shape) > 2 else rows
            for row in range(0, rows, boundary):
                nr = min(boundary, rows - row)
                for col in range(0, width, self.tuning.dma_columns):
                    nc = min(self.tuning.dma_columns, width - col)
                    ip = self.tmp(f"nl.arange({nr})[:, None]")
                    jf = self.tmp(f"nl.arange({nc})[None, :]")
                    lin = f"(({ip}+{row})*{width}+{jf}+{col})"
                    local = [
                        f"(({lin}//{st})%{n})"
                        for st, n in zip(globals()["strides"](shape), shape)
                    ]
                    external = [
                        f"({c}+{off})" for c, off in zip(local, offsets)
                    ]
                    src_view = self.index(src, external if load else local)
                    dst_view = self.index(dst, local if load else external)
                    self.emit(f"nisa.dma_copy(dst={dst_view}, src={src_view})")
                    self.stats["isa_dma_panels"] += 1
            self.signal(semaphore, post_count)
            return
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
            already_oriented = False
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
                elif (
                    self.movement_selector is not None
                    and not direct
                    and not transposed
                    and mask == "True"
                ):
                    self.stats["isa_dma_panels"] -= 1
                    value = self.movement(
                        self.index(src, external),
                        transpose=True,
                        dtype=dst.dtype,
                        role="load",
                    )
                    already_oriented = True
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
                    if not direct and not already_oriented:
                        value = self.transpose(
                            value,
                            copy_engine="scalar",
                            dtype=dst.dtype,
                            role="load",
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
                    for n, st in zip(src.shape, globals()["strides"](src.shape))
                ]
                value = self.tmp(self.index(src, source))
                mask = " & ".join(
                    f"(({c}) >= 0) & (({c}) < {n})"
                    for c, n in zip(external, dst.shape)
                )
                mask = self.mask(mask)
                if self.movement_selector is not None and mask == "True":
                    self.stats["isa_dma_panels"] -= 1
                    self.movement(
                        value,
                        destination=self.index(dst, external),
                        transpose=True,
                        role="store",
                    )
                    continue
                value = self.transpose(value, dtype=src.dtype, role="store")
                self.emit(
                    f"nisa.dma_copy(dst={self.index(dst, external)}, src={value}, mask={mask})"
                    if self.tuning.isa_lowering
                    else f"nl.store({self.index(dst, external)}, {value}, mask={mask})"
                )
        self.signal(semaphore, post_count)

    def pool_copy(self, src, dst, shape, offsets, padding, pad_value, load):
        """Replicated vertical halo: output rows occupy partitions.

        Each kernel row is a distinct DMA rectangle into the free dimension;
        horizontal neighbors are views. No partition-axis gather is implicit.
        """
        local = dst if load else src
        spec = self.pool_buffers[local.name]
        rows, width = spec["physical"]
        kh = spec["kernel"][0] if load else 1
        width //= kh
        local_var = self.vars[local.name, local.slot]
        ext = src if load else dst
        actual_bytes = 0
        for r in range(kh):
            for col in range(0, width, self.tuning.dma_columns):
                nc = min(self.tuning.dma_columns, width - col)
                sr = offsets[1] + r - padding[1]
                sc = offsets[2] + col - padding[2]
                r0, r1 = max(0, -sr), min(rows, ext.shape[1] - sr)
                c0, c1 = max(0, -sc), min(nc, ext.shape[2] - sc)
                if load and (r0 or c0 or r1 != rows or c1 != nc):
                    if pad_value != float("-inf"):
                        raise ValueError(
                            "Maxpool halo requires negative infinity padding"
                        )
                    bits = self.tmp(
                        f"nisa.memset(({rows},{nc}), value=4286578688, dtype=nl.uint32, engine=nisa.vector_engine)"
                    )
                    fill = self.tmp(f"{bits}.view(nl.float32)")
                    self.emit(
                        f"{local_var}[:, {r*width+col}:{r*width+col+nc}] = nisa.tensor_copy({fill}, engine=nisa.vector_engine)"
                    )
                if r1 <= r0 or c1 <= c0:
                    continue
                ip = self.tmp(f"nl.arange({r1-r0})[:, None]")
                jf = self.tmp(f"nl.arange({c1-c0})[None, :]")
                coords = [
                    str(offsets[0]),
                    f"({ip}+{sr+r0})",
                    f"({jf}+{sc+c0})",
                    "0",
                ]
                external = self.index(ext, coords)
                loc = f"{local_var}[{ip}+{r0}, {jf}+{r*width+col+c0}]"
                self.emit(
                    f"nisa.dma_copy(dst={loc if load else external}, src={external if load else loc})"
                )
                self.stats["isa_dma_panels"] += 1
                actual_bytes += (
                    (r1 - r0)
                    * (c1 - c0)
                    * (2 if local.dtype in ("float16", "bfloat16") else 4)
                )
        self.transfer_records[-1][
            "read_bytes" if load else "write_bytes"
        ] = actual_bytes
        self.transfer_records[-1]["implementation"] = (
            "replicated_vertical_halo" if load else "row_store"
        )

    def transpose(
        self, value, copy_engine="vector", dtype="float32", role="operand"
    ):
        if self.movement_selector is not None:
            return self.movement(value, transpose=True, dtype=dtype, role=role)
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
            if not self.tuning.strict_realization:
                # The pinned SDK rejects explicit identity-matmul transposes
                # with native allocation (duplicate memlocSet). Its native
                # transpose owns the shared identity and PSUM allocation.
                psum = self.tmp(
                    f"nisa.nc_transpose({value}, engine=nisa.tensor_engine)"
                )
                return self.local_copy(
                    psum,
                    dtype=dtype,
                    engine=(
                        "scalar"
                        if self.tuning.copy_policy == "scalar"
                        else copy_engine
                    ),
                )
            from .instruction_plan import Tensor

            identity = "identity_" + dtype
            if identity not in self.builder.program.tensors:
                hbm = identity + "_hbm"
                self.builder.program.tensors[hbm] = Tensor(
                    hbm,
                    (128, 128),
                    dtype,
                    "HBM",
                    "contiguous",
                    constant="identity",
                )
                self.builder.add(
                    f"{identity} = nl.ndarray((128,128), dtype=nl.{dtype}, buffer=nl.sbuf)"
                )
                self.builder.add(f"nisa.dma_copy(dst={identity}, src={hbm})")
                self.stats["isa_dma_panels"] += 1
            from .instruction_plan import Expr

            shape = self.builder.value(Expr.parse(value)).shape
            psum = self.tmp(
                f"nisa.nc_matmul({value}, {identity}[:{shape[0]}, :{shape[0]}], is_transpose=True)"
            )
            return self.local_copy(
                psum,
                dtype=dtype,
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
        if self.movement_selector is not None:
            from .instruction_plan import Expr

            data = self.builder.value(Expr.parse(value))
            return self.movement(
                value,
                dtype=dtype,
                role="eviction" if data.memory == "PSUM" else "staging",
                preferred_engine="ScalarE" if engine == "scalar" else "VectorE",
            )
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

    def replicate_parameter(self, parameter, rows, width):
        """Explicit partition broadcast, retained with the shared operand.

        NKI broadcast_to inserts an automatically allocated partition-copy
        temporary. A selected ones-by-row matmul has fully enumerated SBUF,
        PSUM and copy operands and a legal direct-allocation implementation.
        """
        source = self.read(parameter)
        key = (source, rows, width)
        if key in self.replicated_parameters:
            return self.replicated_parameters[key]
        ones = self.tmp(
            f"nisa.memset((1,{rows}), value=1, dtype=nl.float32, engine=nisa.vector_engine)"
        )
        result = self.tmp(
            f"nl.ndarray(({rows},{width}), dtype=nl.float32, buffer=nl.sbuf)"
        )
        for col in range(0, width, 512):
            extent = min(512, width - col)
            panel = self.tmp(f"{source}[:,{col}:{col+extent}]")
            psum = self.tmp(f"nisa.nc_matmul({ones}, {panel})")
            self.emit(
                f"{result}[:,{col}:{col+extent}] = nisa.tensor_copy({psum}, engine=nisa.scalar_engine)"
            )
            self.stats["tensor_instructions"] += 1
            self.stats["parameter_replication_matmuls"] += 1
            self.expanded_isa.update({"LDWEIGHTS": 2, "MATMUL_REGULAR": 2})
        self.replicated_parameters[key] = result
        return result

    def map_panels(self, expression, *values):
        panel = next((v for v in values if isinstance(v, PanelValue)), None)
        if panel is None:
            return self.tmp(expression(*values))
        # NKI cannot index an advanced-index view a second time. Materialize
        # mixed panel/non-panel operands once before taking panel subviews.
        # This copy belongs to the selected ISA, not to formal conversion.
        staged = []
        for value in values:
            tensor = (
                self.builder.program.tensors.get(value)
                if isinstance(value, str)
                else None
            )
            staged.append(
                self.local_copy(value)
                if tensor is not None and tensor.alias
                else value
            )
        values = tuple(staged)
        results = []
        for idx, (mi, ni, mm, nn, _) in enumerate(panel.panels):
            args = []
            for value in values:
                if isinstance(value, PanelValue):
                    assert (value.m, value.n) == (panel.m, panel.n)
                    assert value.row_major == panel.row_major
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
        return PanelValue(panel.m, panel.n, results, row_major=panel.row_major)

    def matrix_activation(self, ref):
        """Convert a row-partition producer tile to TensorE's K partitions."""
        if ref.name not in self.row_buffers:
            # A preceding local matrix/pointwise region can bind a logical
            # destination directly to N-partitioned panels. Read the requested
            # K-by-M activation view from that payload; do not treat the panel
            # container as an address expression or return the entire tensor.
            panels = self.panel_slots.get((ref.name, ref.slot))
            if panels is not None:
                width = self.boxes[ref.name].shape[-1]
                row, col = divmod(ref.offset, width)
                rows, columns = ref.shape
                if ref.strides != (width, 1):
                    raise ValueError(
                        "Panel consumer requires a contiguous logical row tile"
                    )
                for mi, ni, mm, nn, value in panels.panels:
                    if (
                        mi <= row
                        and row + rows <= mi + mm
                        and ni <= col
                        and col + columns <= ni + nn
                    ):
                        if panels.row_major:
                            view = f"{value}[{row-mi}:{row-mi+rows}, {col-ni}:{col-ni+columns}]"
                            return self.transpose(
                                self.local_copy(view), dtype=ref.dtype
                            )
                        view = f"{value}[{col-ni}:{col-ni+columns}, {row-mi}:{row-mi+rows}]"
                        return self.tmp(view)
                raise ValueError(
                    "Matrix input view crosses producer panel boundaries"
                )
            return self.read(ref)
        width = self.boxes[ref.name].shape[-1]
        row, col = divmod(ref.offset, width)
        rows, columns = ref.shape
        if rows > 128 or columns > 128 or ref.strides != (width, 1):
            raise ValueError(
                "Row-to-matrix edge requires a contiguous tile up to 128x128"
            )
        value = self.vars[ref.name, ref.slot]
        panel = self.local_copy(
            f"{value}[{row}:{row + rows}, {col}:{col + columns}]"
        )
        return self.transpose(panel, dtype=ref.dtype)

    def resident_weight_panel(self, ref):
        width = self.boxes[ref.name].shape[-1]
        k, n = ref.shape
        row, col = divmod(ref.offset, width)
        p = min(128, self.boxes[ref.name].shape[0])
        if ref.strides != (width, 1) or row % p + k > p or col + n > width:
            raise ValueError(
                "K-partitioned operand view crosses a physical K block"
            )
        value = self.vars[ref.name, ref.slot]
        offset = (row // p) * width + col
        self.stats["resident_weight_panel_views"] += 1
        return self.tmp(f"{value}[{row % p}:{row % p+k}, {offset}:{offset+n}]")

    def gemm(self, a, b, m, n, k, transposed, dtype, output_row=False):
        """Lower one scheduled tile into ISA panels without retiling HBM."""
        from .orientation import matrix_choice

        weight_layout = (
            "k_partitioned" if b.name in self.k_weight_buffers else "generic"
        )
        choice = matrix_choice(
            self.hardware,
            m,
            n,
            k,
            32 if a.dtype == "float32" else 16,
            transposed,
            self.tuning,
            a.dtype,
            dtype,
            a.name in self.row_buffers,
            output_row,
            weight_layout,
        )
        signature = ("m", "n", "k", "transposed", "input_row", "output_row")
        matches = [
            b
            for b in self.matrix_bindings
            if all(b[s] == choice[s] for s in signature)
        ]
        if matches:
            if any(
                b.get("weight_layout", "generic") != weight_layout
                for b in matches
            ):
                raise ValueError("Search/realization weight layout mismatch")
            if any(b["orientation"] != choice["orientation"] for b in matches):
                raise ValueError(
                    "Search/realization matrix orientation mismatch"
                )
            choice = matches[0]
        elif self.matrix_bindings:
            raise ValueError(f"No selected matrix orientation for {choice}")
        self.matrix_choices.append(choice)
        orientation = choice["orientation"]
        tuning = replace(self.tuning, matmul_orientation=orientation)
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
                tuning,
                a.dtype,
                dtype,
                a.name in self.row_buffers,
                output_row,
                weight_layout,
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
        # Only immutable operand views inside this GEMM invocation are reused.
        # A later shared-buffer write cannot see cached values from this call.
        weight_panels = {}
        activation_panels = {}
        from .execution import matmul_panels

        for mi, ni, mm, nn, kpanels in matmul_panels(m, n, k, orientation):
            stationary, moving = (
                (nn, mm) if orientation == "weights" else (mm, nn)
            )
            acc = self.tmp(
                f"nl.zeros(({stationary},{moving}), dtype=nl.float32, buffer=nl.psum)"
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
                if orientation == "activations":
                    reuse = self.tuning.matmul_operands == "reuse"
                    activation_key = (mi, ki)
                    if reuse and activation_key in activation_panels:
                        av = activation_panels[activation_key]
                        self.stats["reused_activation_panels"] += 1
                    else:
                        av = self.matrix_activation(ar)
                        if self.tuning.matmul_operands == "staged" and (
                            m > 512 or n > 128 or k > 128
                        ):
                            av = self.local_copy(av)
                        if reuse:
                            activation_panels[activation_key] = av
                    weight_key = (ni, ki)
                    if weight_layout == "k_partitioned":
                        bv = self.resident_weight_panel(br)
                        if self.tuning.matmul_operands == "staged" and (
                            m > 512 or n > 128 or k > 128
                        ):
                            bv = self.local_copy(bv)
                    elif reuse and weight_key in weight_panels:
                        bv = weight_panels[weight_key]
                        self.stats["reused_weight_panels"] += 1
                    else:
                        bv = None
                        if nn > 128:
                            bv = self.tmp(
                                f"nl.ndarray(({kk},{nn}), dtype=nl.{b.dtype}, buffer=nl.sbuf)"
                            )
                        for col in range(0, nn, 128):
                            width = min(128, nn - col)
                            wr = replace(
                                br,
                                shape=(
                                    (width, kk) if transposed else (kk, width)
                                ),
                                offset=br.offset
                                + col * b.strides[0 if transposed else 1],
                            )
                            wp = self.read(wr)
                            if self.tuning.matmul_operands == "staged" and (
                                m > 512 or n > 128 or k > 128
                            ):
                                wp = self.local_copy(wp)
                            if not transposed:
                                wp = self.transpose(wp, dtype=b.dtype)
                            if nn > 128:
                                self.emit(
                                    f"{bv}[:,{col}:{col+width}] = nisa.tensor_copy({wp}, engine=nisa.vector_engine)"
                                )
                            else:
                                bv = wp
                        if reuse:
                            weight_panels[weight_key] = bv
                else:
                    if self.tuning.matmul_operands == "staged":
                        av = self.matrix_activation(ar)
                        bv = (
                            self.resident_weight_panel(br)
                            if weight_layout == "k_partitioned"
                            else self.read(br)
                        )
                        if m > 512 or n > 128 or k > 128:
                            av = self.local_copy(av)
                            bv = self.local_copy(bv)
                        if not transposed and weight_layout == "generic":
                            bv = self.transpose(bv, dtype=b.dtype)
                    else:
                        weight_key = (ni, ki)
                        reuse = self.tuning.matmul_operands == "reuse"
                        activation_key = (mi, ki)
                        if (
                            reuse
                            and a.name in self.row_buffers
                            and activation_key in activation_panels
                        ):
                            av = activation_panels[activation_key]
                            self.stats["reused_activation_panels"] += 1
                        else:
                            av = self.matrix_activation(ar)
                            if reuse and a.name in self.row_buffers:
                                activation_panels[activation_key] = av
                        if reuse and weight_key in weight_panels:
                            bv = weight_panels[weight_key]
                            self.stats["reused_weight_panels"] += 1
                        else:
                            bv = (
                                self.resident_weight_panel(br)
                                if weight_layout == "k_partitioned"
                                else self.read(br)
                            )
                            if not transposed and weight_layout == "generic":
                                bv = self.transpose(bv, dtype=b.dtype)
                            if reuse:
                                weight_panels[weight_key] = bv
                left, right = (bv, av) if orientation == "weights" else (av, bv)
                self.emit(
                    (
                        f"{acc} += nisa.nc_matmul({left}, {right})"
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
                                moving,
                                stationary,
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
                if orientation == "activations" and not output_row:
                    # The shared default layout partitions N. Convert explicitly
                    # so downstream copies/pointwise operations keep their ABI.
                    for col in range(0, nn, 128):
                        width = min(128, nn - col)
                        tile = self.transpose(
                            self.local_copy(f"{value}[:,{col}:{col+width}]"),
                            dtype=dtype,
                        )
                        result_panels.append((mi, ni + col, mm, width, tile))
                else:
                    result_panels.append((mi, ni, mm, nn, value))
                continue
            ip = self.tmp(f"nl.arange({nn})[:, None]")
            jf = self.tmp(f"nl.arange({mm})[None, :]")
            self.emit(
                f"{result}[{ip}, ({jf}+{mi})*{n // p}+{ni // p}] = {value}"
            )
        return (
            PanelValue(
                m, n, result_panels, orientation == "activations" and output_row
            )
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
        if not any(
            p.target.split("::")[-1] in ("matmul", "linear", "conv2d")
            for p in prims
        ):
            names = tuple(p.target.split("::")[-1] for p in prims)
            operands = {}
            window = 1
            for p in prims:
                kw = {
                    key: self.arg(value)
                    for key, value in p.kwargs.items()
                    if value.WhichOneof("arg_type") != "tensor_box"
                    or value.tensor_box.box.HasField("memory")
                }
                for key, value in kw.items():
                    if isinstance(value, Ref) and key not in (
                        "mean",
                        "variance",
                        "normalized",
                        "max",
                        "sum",
                    ):
                        operands[value.name, value.slot] = value.shape
                if "kernel_size" in kw:
                    window = math.prod(kw["kernel_size"])
            self.vector_records.append(
                dict(
                    name=o.name,
                    operations=names,
                    shape=d.shape,
                    inputs=tuple(operands.values()),
                    pool_window=window,
                )
            )
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
                    name == "matmul" and p.target.startswith("quantized_ops::")
                ) or (name == "linear" and p.target.startswith("aten::"))
                n = b.shape[0] if transposed else b.shape[1]
                value = self.gemm(
                    a,
                    b,
                    m,
                    n,
                    k,
                    transposed,
                    d.dtype,
                    d.name in self.row_buffers,
                )
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
            elif name in ("amax", "sum"):
                dim = kw.get("dim")
                if isinstance(dim, int):
                    dim = [dim]
                if (
                    dim is None
                    or tuple(i % len(a.shape) for i in dim)
                    != (len(a.shape) - 1,)
                    or not kw.get("keepdim", False)
                ):
                    raise NotImplementedError(
                        "Block reduction requires last axis with keepdim"
                    )
                if a.name not in self.row_buffers:
                    raise ValueError("Block reduction lost its row layout")
                op = "max" if name == "amax" else "add"
                value = self.tmp(
                    f"nisa.tensor_reduce(nl.{op}, {self.read(a)}, axis=[1], keepdims=True, dtype=nl.float32)"
                )
            elif name in ("full_like", "zeros_like", "ones_like"):
                fill = kw.get("fill_value", 1 if name == "ones_like" else 0)
                value = self.tmp(
                    f"nisa.memset({self.storage_shape(self.boxes[d.name])!r}, value={fill}, dtype=nl.{d.dtype}, engine=nisa.vector_engine)"
                )
            elif name == "reciprocal":
                value = self.tmp(f"nisa.reciprocal({self.read(a)})")
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
                        act = self.tmp(f"{act}.reshape(({channels},{oh * ow}))")
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
                    if a.name in self.pool_buffers:
                        spec = self.pool_buffers[a.name]
                        oh, ow = spec["output"]
                        width = a.shape[2]
                        # A rectangular max is separable. Combine the full
                        # vertical strips, then their shifted horizontal views.
                        # This same expansion drives the candidate cost graph.
                        from .operations import pool_reduction_steps

                        local = self.vars[a.name, a.slot]
                        values = {
                            f"row{r}": (local, r * width)
                            for r in range(spec["kernel"][0])
                        }
                        for name, left, right, free in pool_reduction_steps(
                            *spec["kernel"], width
                        ):

                            def operand(view):
                                base, start = view
                                tensor, offset = values[base]
                                return f"{tensor}[:{oh}, {offset+start}:{offset+start+free}]"

                            values[name] = (
                                self.tmp(
                                    f"nisa.tensor_tensor({operand(left)}, {operand(right)}, op=nl.maximum, engine=nisa.vector_engine)"
                                ),
                                0,
                            )
                        value = (
                            values["result"][0]
                            if "result" in values
                            else self.tmp(f"{local}[:{oh}, :{ow}]")
                        )
                        self.env[p.name] = value
                        continue
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
                    value = self.tmp(f"{value}.reshape(({channels},{oh * ow}))")
                    if name == "avg_pool2d":
                        value = self.tmp(f"{value}/{math.prod(kernel)}")
            elif name in ("add", "add_", "sub", "mul", "div", "maximum"):
                row_scalar = (
                    isinstance(kw.get("other"), Ref)
                    and kw["other"].name in self.row_buffers
                    and kw["other"].shape[-1] == 1
                    and d.name in self.row_buffers
                )
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
                    if row_scalar:
                        value = self.map_panels(
                            lambda x, y: f"nisa.tensor_scalar({x}, op0=nl.{op}, operand0={y}, engine=nisa.vector_engine)",
                            a,
                            b,
                        )
                    elif isinstance(kw["other"], (int, float)):
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

    def select(self, allocation_policy="best_fit"):
        self.region(self.model.ops)
        program = self.builder.program
        program.outputs = tuple(
            self.vars[b.node, 0] for b in self.model.outputs
        )
        # The pinned direct-allocation SDK requires HBM temporaries to be I/O
        # tensors. Make these workspace results an explicit ABI component.
        semantic_roots = {program.root(n) for n in program.outputs}
        touched = {
            program.root(n)
            for ins in program.instructions
            for n in ins.reads + ins.writes
        }
        workspace = tuple(
            n
            for n, t in program.tensors.items()
            if n in touched
            and t.memory == "HBM"
            and not t.alias
            and not t.constant
            and n not in program.arguments
            and n not in semantic_roots
        )
        program.outputs += workspace
        if not self.tuning.strict_realization:
            if self.movement_selector is not None:
                raise ValueError(
                    "Movement bindings currently require strict realization"
                )
            program.encoding_storage = "compiler"
            self.selected_allocation_policy = "compiler"
            program.validate(self.hardware.scratchpad_size)
            return program
        # SDK realization constraint, not a smaller physical memory: stream
        # instructions sharing a monolithic arena fail the pinned in-place
        # checker. Size classes yield disjoint declarations, verified on device.
        if getattr(self, "required_movement_storage", "") == "disjoint_arenas":
            allocation_policy = "size_classes"
        self.selected_allocation_policy = (
            "bounded_size_classes"
            if self.tuning.temporary_buffer_depth > 1
            else allocation_policy
        )
        program.allocate(
            self.hardware.scratchpad_size,
            strategy=allocation_policy,
            temporary_buffer_depth=self.tuning.temporary_buffer_depth,
        )
        if allocation_policy == "size_classes":
            program.bind_disjoint_regions()
        return program


def select_plan(
    root, *, context, allocation_policy="best_fit", movement_bindings=None
):
    from .hardware import TARGETS

    from .hardware import neuron_core
    from voyager_compiler.compilation import CompilerContext

    root = Path(root)
    target = context.hardware.name
    output = root
    model = text_format.Parse((root / "model.txt").read_text(), ir.Model())
    record = json.loads((root / "hardware.json").read_text())
    tuning = context.policy.tuning
    trial = record.get("stream_search_trial")
    if trial is not None:
        orientation = trial["orientation"]
        if orientation not in ("weights", "activations"):
            raise ValueError(
                "Invalid expanded-search matrix orientation binding"
            )
        if tuning.matmul_orientation not in ("auto", orientation):
            raise ValueError(
                "Expanded search conflicts with requested orientation"
            )
        tuning = replace(tuning, matmul_orientation=orientation)
    matrix_bindings = [
        e["matrix_choice"]
        for e in record.get("estimates", [])
        if "matrix_choice" in e
    ]
    matrix_bindings += [
        c
        for r in record.get("row_regions", [])
        for c in r.get("selected", {}).get("matrix_choices", [])
    ]
    movement_search = None
    if movement_bindings is not None:
        converter = InstructionPlanner(
            model,
            target,
            tuning,
            context.hardware,
            movement_bindings=movement_bindings,
            matrix_bindings=matrix_bindings,
        )
        program = converter.select(allocation_policy)
        from .program_analysis import analyze_selected

        movement_search = dict(
            strategy="explicit diagnostic bindings",
            selected=dict(
                selected_bindings={
                    k: v["selected"]
                    for k, v in converter.movement_selector.requests.items()
                },
                predicted_ns=analyze_selected(
                    program,
                    context.hardware,
                    execution_model=context.policy.tuning.physical_model,
                )["prediction_ns"],
            ),
            requests=converter.movement_selector.requests,
        )
    elif context.policy.tuning.movement_search_budget:
        from .movement_search import search

        def build(bindings):
            planner = InstructionPlanner(
                model,
                target,
                tuning,
                context.hardware,
                movement_bindings=bindings,
                matrix_bindings=matrix_bindings,
            )
            return planner, planner.select(allocation_policy)

        converter, program, movement_search = search(
            build,
            context.hardware,
            budget=context.policy.tuning.movement_search_budget,
            beam_width=context.policy.tuning.movement_search_beam,
        )
    else:
        converter = InstructionPlanner(
            model,
            target,
            tuning,
            context.hardware,
            matrix_bindings=matrix_bindings,
        )
        program = converter.select(allocation_policy)
    from .plan_emitter import emit

    instruction_lines = []
    emit(program, _capture=instruction_lines)
    from .compact_encoding import select as select_encoding

    program.encoding_loops = select_encoding(program, instruction_lines)
    source = emit(program)
    from .dependencies import audit_realization

    record = json.loads((root / "hardware.json").read_text())
    audit = audit_realization(record.get("execution_plans", ()), converter)
    from .program_analysis import analyze_program

    program_analysis = analyze_program(source, converter, record)
    from .program_analysis import analyze_selected

    selected_analysis = analyze_selected(
        program,
        context.hardware,
        execution_model=context.policy.tuning.physical_model,
    )
    program_analysis["selected_instruction_analysis"] = selected_analysis
    program_analysis["template_prediction_ns"] = program_analysis[
        "whole_program_prediction_ns"
    ]
    program_analysis["whole_program_prediction_ns"] = selected_analysis[
        "prediction_ns"
    ]
    program_analysis["whole_program_timing_complete"] = not selected_analysis[
        "unknown_completion"
    ]
    if selected_analysis["hbm_bytes"] != program_analysis["hbm"]["total_bytes"]:
        raise ValueError(
            "Selected ISA traffic differs from whole-program transfer accounting"
        )

    manifest = dict(
        format="voyager-bufferized-nki-v1",
        target=target,
        compiler=context.record(),
        source_sha256=hashlib.sha256(
            (root / "model.txt").read_bytes()
        ).hexdigest(),
        stats=converter.stats,
        matrix_choices=converter.matrix_choices,
        stream_search_trial=trial,
        k_partitioned_weight_buffers=sorted(converter.k_weight_buffers),
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
    if movement_search is not None:
        manifest["movement_search"] = movement_search
        (root / "movement-search.json").write_text(
            json.dumps(movement_search, indent=2) + "\n"
        )
    strict = context.policy.tuning.strict_realization
    manifest["strict_realization"] = strict
    manifest["native_implicit_dma_commands"] = int(
        not strict and bool(converter.stats["isa_transposes"])
    )
    manifest["allocation"] = (
        "All selected local values bind to direct SBUF/PSUM arena views; NKI retains final engine scheduling"
        if strict
        else "Selected logical SBUF/PSUM buffers; NKI owns physical allocation, reuse and engine scheduling"
    )
    manifest["selected_instruction_plan"] = "instructions.json"
    manifest["requested_physical_placement_strategy"] = allocation_policy
    manifest["physical_placement_strategy"] = (
        converter.selected_allocation_policy
    )
    manifest["temporary_buffering"] = dict(
        depth=context.policy.tuning.temporary_buffer_depth,
        policy=(
            "existing placement"
            if context.policy.tuning.temporary_buffer_depth == 1
            else "per-size-class bounded slots; oldest legal owner recycled"
        ),
        sbuf_reserved_bytes=(
            max(
                (
                    p.byte_address + p.bytes_per_partition
                    for p in program.placements.values()
                    if p.memory == "SBUF"
                ),
                default=0,
            )
            * 128
            if strict
            else None
        ),
        scope="Physical temporary allocation; shared software tile depth is unchanged",
        search_scope="Selected logical program only; compact mapping search does not score this pool depth",
    )
    manifest["abi"] = dict(
        semantic_output_count=len(model.outputs),
        workspace_outputs=list(program.outputs[len(model.outputs) :]),
    )
    manifest["program_analysis"]["buffer_allocation"] = (
        "direct" if strict else "compiler"
    )
    manifest["program_analysis"]["physical_addresses_enforced"] = strict
    if not strict:
        manifest["program_analysis"]["whole_program_timing_complete"] = False
        manifest["program_analysis"][
            "allocation_limitation"
        ] = "Logical dependency estimate only; NKI allocation, physical reuse and spills are not predicted"
    manifest["program_analysis"]["physical_slot_depth_enforced"] = False
    manifest["program_analysis"]["logical_depth_speed_credit"] = False
    (root / "instructions.json").write_text(
        json.dumps(program.record(), separators=(",", ":"))
    )
    manifest["instructions_sha256"] = hashlib.sha256(
        (root / "instructions.json").read_bytes()
    ).hexdigest()
    manifest["hardware_record_sha256"] = hashlib.sha256(
        (root / "hardware.json").read_bytes()
    ).hexdigest()
    (root / "selection.json").write_text(json.dumps(manifest, indent=2))
    return manifest
