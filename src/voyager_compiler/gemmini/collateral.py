"""Lower the standard bufferized Voyager protobuf to pinned Lean RoCC commands.

Loops, scalar control, DMA windows and scheduled compute tiles come from model.txt.
No whole-operation retiling or hardware LoopMatmul/LoopConv instructions are used.
Scalar loop control is specialized for the CPU-free replay interface; tensor
arithmetic is never evaluated to supply accelerator outputs.
"""

import argparse
import itertools
import json
import math
import operator
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from google.protobuf import text_format

from voyager_compiler.codegen import voyager_ir_pb2 as ir

from .isa import ACC, STATUS, command, ex, ld, st, tile
from .scheduling import AsyncSchedule, QueueGeometry
from .execution import retains_weight
from .hardware import lean_config, MEMORY_BASE
from .constraints import accumulator_slot_bytes

DRAM = ir.MEMORY_LEVEL_DRAM
SRAM = ir.MEMORY_LEVEL_SCRATCHPAD
REGISTER = ir.MEMORY_LEVEL_REGISTER
BASE = MEMORY_BASE


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def width(dtype):
    return 1 if dtype == "int8" else 4


@dataclass
class Ref:
    name: str
    level: int
    address: int
    shape: tuple
    strides: tuple
    dtype: str
    slot: int = 0

    @property
    def key(self):
        return self.name, self.slot

    def window(self, offsets, sizes):
        return replace(
            self,
            address=self.address
            + sum(a * b for a, b in zip(offsets, self.strides)),
            shape=tuple(sizes),
        )


@dataclass
class Value:
    """A value in accumulator SRAM, with a pending legal mvout scale/activation."""

    base: int
    shape: tuple
    scale: float = 1.0
    relu: bool = False
    pool: object = None


@dataclass
class InputValue:
    ref: Ref
    scale: float = 1.0


class Converter:
    def __init__(
        self, root, ops, output, produced=(), schedule_async=True, policy=None
    ):
        self.root, self.ops, self.output = root, ops, output
        self.output.mkdir(parents=True, exist_ok=True)
        self.env, self.semaphores, self.values = {}, {}, {}
        self.commands = [dict(type="defaults", status=STATUS), ex()]
        from .mapping import GemminiMappingPolicy

        self.hardware = lean_config()
        self.policy = policy or GemminiMappingPolicy(self.hardware)
        if self.policy.config != self.hardware:
            raise ValueError(
                "Converter policy must use the pinned Lean hardware"
            )
        self.schedule = AsyncSchedule(
            self.commands,
            QueueGeometry.from_hardware(self.hardware),
            self.policy.tuning.submission,
        )
        self.schedule_async = schedule_async
        self.stats = Counter()
        self.dram = {}
        self.regions = []
        self.next_dram = BASE + 4096
        self.acc_slots = {}
        self.matrix_accumulators = set()
        self.acc_generations = {}
        self.next_acc = 0
        self.source_inputs, self.source_outputs = [], []
        self.current_tiling = None
        self.bias_sources = {}
        self.layout_ops = {}
        self.identity_loaded = False
        self.input_layouts = {}
        self.produced = set(produced)
        self.chained_inputs = []

    def scalar(self, value):
        field = value.WhichOneof("value")
        if field == "node":
            return self.env[value.node]
        if field is None:
            raise ValueError("Missing scalar value")
        return getattr(value, field)

    def ref(self, ref):
        b = ref.box
        item = width(b.dtype)
        shape = tuple(b.shape)
        strides = tuple(
            math.prod(shape[i + 1 :]) * item for i in range(len(shape))
        )
        offsets = [self.scalar(v) for v in ref.offsets]
        sizes = tuple(ref.sizes)
        slot = 0
        address = b.memory.address
        if b.memory.level == DRAM:
            address = self.dram[b.node]
        if offsets and b.HasField("bank_count"):
            slot = offsets.pop(0)
            address += slot * b.bank_stride_bytes
            sizes = sizes[1:]
        if offsets:
            address += sum(x * s for x, s in zip(offsets, strides))
            strides = tuple(
                a * b for a, b in zip(strides, ref.strides[-len(strides) :])
            )
            shape = sizes
        out_shape = tuple(ref.output_shape)
        if out_shape and out_shape != shape:
            if math.prod(out_shape) != math.prod(shape):
                raise ValueError("Noncontiguous/invalid view")
            shape = out_shape
            strides = tuple(
                math.prod(shape[i + 1 :]) * item for i in range(len(shape))
            )
        return Ref(
            b.node, b.memory.level, address, shape, strides, b.dtype, slot
        )

    def arg(self, a):
        typ = a.WhichOneof("arg_type")
        if typ == "tensor_box":
            r = a.tensor_box
            if not r.box.HasField("memory"):
                return self.env[r.box.node]
            return self.ref(r)
        if typ == "tensor_box_list":
            return [self.ref(r) for r in a.tensor_box_list.values]
        if typ == "scalar":
            return self.scalar(a.scalar)
        if typ == "scalar_list":
            return [self.scalar(v) for v in a.scalar_list.values]
        if typ == "str_value":
            return a.str_value
        raise NotImplementedError(typ)

    def args(self, p):
        return {k: self.arg(v) for k, v in p.kwargs.items()}

    def constant(self, ref):
        if isinstance(ref, (float, int)):
            return float(ref)
        if ref.level != ir.MEMORY_LEVEL_IMMEDIATE:
            raise NotImplementedError("Scale must be a static tensor")
        data = np.fromfile(
            self.root / "tensor_files" / f"{ref.name}.bin", dtype="<f4"
        )
        if data.size != 1:
            raise NotImplementedError(
                "Per-channel scale is not supported by Lean"
            )
        return float(data[0])

    def signal(self, ref, n=1):
        self.semaphores[ref.key] = self.semaphores.get(ref.key, 0) + n
        self.schedule.signal(ref.key, n)

    def wait(self, ref):
        if self.semaphores.get(ref.key, 0) < 1:
            raise ValueError(f"Unbalanced bufferized semaphore: {ref.key}")
        self.semaphores[ref.key] -= 1
        self.schedule.wait(ref.key)
        # Gemmini's reservation station orders local-memory RAW/WAR/WAW edges.
        # Submission order implements these waits without globally draining DMA.
        self.stats["waits"] += 1

    def outputs(self, o, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        for output, value in zip(o.outputs, values):
            self.env[output.name] = value

    def region(self, ops):
        for o in ops:
            kind = o.WhichOneof("op_type")
            self.stats[kind] += 1
            if kind == "loop":
                loop = o.loop
                if loop.WhichOneof("loop_type") != "for_loop":
                    raise NotImplementedError("Dynamic while loop")
                f = loop.for_loop
                state = [self.scalar(a.initial) for a in f.iter_args]
                for i in range(
                    self.scalar(f.start),
                    self.scalar(f.end),
                    self.scalar(f.step),
                ):
                    self.env[f.iv] = i
                    self.env.update(
                        {a.name: v for a, v in zip(f.iter_args, state)}
                    )
                    self.region(f.body.ops)
                    state = [self.scalar(v) for v in f.body.yields]
                self.outputs(o, state)
            elif kind == "cond":
                region = (
                    o.cond.true_region
                    if self.scalar(o.cond.predicate)
                    else o.cond.false_region
                )
                self.region(region.ops)
                self.outputs(o, [self.scalar(v) for v in region.yields])
            elif kind == "async":
                a = getattr(o, "async")
                with self.schedule.task(o.name):
                    for r in a.dependencies:
                        self.wait(self.ref(r))
                    start = len(self.commands)
                    self.region(a.body.ops)
                    self.stats["max_async_commands_before_dedup"] = max(
                        self.stats["max_async_commands_before_dedup"],
                        len(self.commands) - start,
                    )
                    if a.HasField("post"):
                        self.signal(self.ref(a.post))
            elif kind in ("prim", "fused"):
                if kind == "prim" and o.prim.op == "cpu":
                    self.cpu(o)
                else:
                    self.compute(o)
            else:
                raise NotImplementedError(kind)

    def cpu(self, o):
        p = o.prim
        target = p.target.removeprefix("voyager::")
        kw = self.args(p)
        if p.target == "aten::permute":
            # Boundary layouts are represented explicitly in collaterals. Input
            # packing is performed in prepare(); output layout is host metadata.
            return
        if target in ("alloc", "zeros", "fill"):
            for out in o.outputs:
                if out.WhichOneof("result_type") != "tensor_box":
                    continue
                b = out.tensor_box
                if b.memory.level == REGISTER:
                    for slot in range(b.bank_count or 1):
                        self.semaphores[b.node, slot] = int(kw.get("value", 0))
                        self.schedule.seed(
                            (b.node, slot), int(kw.get("value", 0))
                        )
            return
        if target == "async_copy":
            self.copy(**kw)
            return
        if target == "async_wait":
            self.wait(kw["semaphore"])
            return
        if target == "sym_ite":
            self.outputs(o, kw["t"] if kw["b"] else kw["f"])
            return
        if target == "delinearize_index":
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
        if target not in functions:
            raise NotImplementedError(f"CPU control {target}")
        a = [kw["input"]]
        if "other" in kw:
            a.append(kw["other"])
        self.outputs(o, functions[target](*a))

    def copy(self, **kwargs):
        with self.schedule.task("async_copy"):
            self._copy(**kwargs)

    def _copy(
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
        strides = strides or sizes
        offsets = [0] * len(sizes)
        for i, d in zip(indices, range(len(sizes)) if dims is None else dims):
            offsets[d] = i * strides[d]
        if dst.level == SRAM and src.level == DRAM:
            self.load(src, dst, offsets, sizes, pad, transposed, pad_value)
        elif src.level == SRAM and dst.level == DRAM:
            self.store(src, dst.window(offsets, sizes))
        else:
            raise NotImplementedError(f"Copy {src.level}->{dst.level}")
        self.signal(semaphore, post_count)

    def load(self, src, dst, offsets, sizes, pad, transposed, pad_value=0):
        if transposed:
            raise NotImplementedError(
                "Transposed DMA requires explicit layout lowering"
            )
        if width(dst.dtype) != 1:
            if len(sizes) != 1:
                raise NotImplementedError("Wide DMA input must be static bias")
            self.bias_sources[dst.key] = src.window(offsets, sizes)
            return
        geometry = self.hardware.connection("dram_dma").transfer_geometry
        shape = tuple(sizes)
        rows = math.prod(shape[:-1])
        cols = shape[-1]
        if cols % 16:
            raise ValueError("Scratchpad channels must be padded to 16")
        if dst.address % 16:
            raise ValueError("Unaligned scratchpad address")
        # Block-column layout keeps Gemmini row addresses affine. Slot addresses
        # and tile boundaries come directly from the standard memory planner.
        base = dst.address // 16
        pad_address = 0
        if pad_value != 0:
            if pad_value != -math.inf:
                raise NotImplementedError(
                    "Only zero and max-pool minimum padding are supported"
                )
            pad_address = self.literal("minimum_int8", bytes([128]) * 64)
        for cb in range(0, cols, geometry.max_row_bytes):
            entries = []
            for coord in itertools.product(*(range(s) for s in shape[:-1])):
                ij = list(coord) + ([cb])
                origin = [
                    a + b - (pad[d] if pad else 0)
                    for d, (a, b) in enumerate(zip(ij, offsets))
                ]
                valid = all(0 <= x < n for x, n in zip(origin, src.shape))
                addr = (
                    src.address
                    + sum(x * s for x, s in zip(origin, src.strides))
                    if valid
                    else pad_address
                )
                entries.append(addr)
            r = 0
            while r < rows:
                stride = (
                    entries[r + 1] - entries[r]
                    if r + 1 < rows and entries[r] and entries[r + 1]
                    else 0
                )
                if stride < 0:
                    stride = 0
                n = 1
                while (
                    n < geometry.max_rows
                    and r + n < rows
                    and entries[r + n]
                    == (entries[r] + n * stride if entries[r] else 0)
                ):
                    n += 1
                self.commands += [
                    ld(stride, block_stride=rows),
                    command(
                        2,
                        entries[r],
                        tile(
                            base + (cb // 16) * rows + r,
                            n,
                            min(geometry.max_row_bytes, cols - cb),
                        ),
                    ),
                ]
                r += n
        self.values.pop(dst.key, None)

    def accumulator(self, dst):
        key = (*dst.key, self.acc_generations.get(dst.key, 0))
        if key not in self.acc_slots:
            rows = math.prod(dst.shape[:-1]) * ((dst.shape[-1] + 15) // 16)
            span = rows
            if dst.name in self.matrix_accumulators:
                row_bytes = self.hardware.memory_instance(
                    "accumulator"
                ).row_bytes
                span = (
                    accumulator_slot_bytes(
                        self.hardware,
                        rows * row_bytes,
                        separate_banks=self.policy.tuning.separate_accumulator_banks,
                    )
                    // row_bytes
                )
                self.next_acc = (
                    accumulator_slot_bytes(
                        self.hardware,
                        self.next_acc * row_bytes,
                        separate_banks=self.policy.tuning.separate_accumulator_banks,
                    )
                    // row_bytes
                )
            if self.next_acc + span > self.hardware.accum_buffer_size:
                raise ValueError("Accumulator output slots exceed capacity")
            self.acc_slots[key] = self.next_acc
            self.next_acc += span
        return self.acc_slots[key]

    def compute(self, o):
        synchronous = self.schedule.current is None
        with self.schedule.task(o.name):
            self._compute(o)
        if synchronous:
            self.schedule.barrier.add(self.schedule.tasks[-1].index)

    def _compute(self, o):
        prims = (
            [o.prim] if o.WhichOneof("op_type") == "prim" else o.fused.op_list
        )
        dsts = [
            self.ref(v.destination)
            for v in o.outputs
            if v.WhichOneof("result_type") == "destination"
        ]
        if len(dsts) != 1:
            raise NotImplementedError("Compute must have one destination")
        dst = dsts[0]
        # A standard split-K kernel ends with add(partial, destination).
        # Fold that add into Gemmini's accumulator write bit, retaining the
        # shared compiler's reduction loop and first-round bias behavior.
        accumulate_existing = False
        for p in prims:
            if p.target in ("aten::add", "aten::add_"):
                other = p.kwargs.get("other")
                if (
                    other is not None
                    and other.WhichOneof("arg_type") == "tensor_box"
                ):
                    b = other.tensor_box.box
                    accumulate_existing |= (
                        b.node == dst.name and b.dtype != "int8"
                    )
        prior = self.values.get(dst.key) if accumulate_existing else None
        has_matrix = any(
            p.target.split("::")[-1] in ("matmul", "linear", "conv2d")
            for p in prims
        )
        if has_matrix:
            self.matrix_accumulators.add(dst.name)
        if has_matrix and dst.dtype != "int8" and not accumulate_existing:
            # Keep the previous completed tile alive for the scheduler's
            # lagged store. Both generations fit the reserved output arena.
            self.acc_generations[dst.key] = 1 - self.acc_generations.get(
                dst.key, 0
            )
        last = None
        for p in prims:
            kw = self.args(p)
            name = p.target.split("::")[-1]
            if name in ("matmul", "linear"):
                a = kw.get("input", kw.get("self"))
                b = kw.get("weight", kw.get("other"))
                last = self.matmul(
                    a,
                    b,
                    dst,
                    o.tiling,
                    kw.get("bias"),
                    name == "linear",
                    accumulate_existing,
                )
            elif name == "conv2d":
                last = self.conv(kw, dst, o.tiling, accumulate_existing)
            elif name == "quantize":
                v = kw["input"]
                s = self.constant(kw["scale"])
                if not math.isfinite(s) or s <= 0:
                    raise ValueError(
                        "Quantization scale must be finite and positive"
                    )
                if any(
                    k in kw
                    for k in ("zero_point", "axes", "block_size", "output_code")
                ):
                    raise NotImplementedError(
                        "Lean supports scalar symmetric quantization only"
                    )
                if isinstance(v, Ref):
                    v = self.values.get(v.key, v)
                if kw.get("rounding") != "nearest_even_int8":
                    raise ValueError(
                        "Lean requires native nearest-even INT8 quantization collaterals"
                    )
                if isinstance(v, (Ref, InputValue)):
                    v = self.project(v, dst)
                if not isinstance(v, Value):
                    raise NotImplementedError("Standalone quantization")
                last = replace(v, scale=v.scale / s)
            elif name == "dequantize":
                v = kw["input"]
                s = self.constant(kw["scale"])
                if not math.isfinite(s) or s <= 0:
                    raise ValueError(
                        "Dequantization scale must be finite and positive"
                    )
                if any(
                    k in kw
                    for k in (
                        "zero_point",
                        "axes",
                        "block_size",
                        "input_qmap",
                        "output_qmap",
                    )
                ):
                    raise NotImplementedError(
                        "Lean requires exact scalar accumulator decoding"
                    )
                if isinstance(v, Ref):
                    v = InputValue(v)
                last = replace(v, scale=v.scale * s)
            elif name in ("relu", "relu_"):
                v = kw.get("self", kw.get("input"))
                if isinstance(v, (Ref, InputValue)):
                    v = self.project(v, dst)
                last = replace(v, relu=True)
            elif name in ("add", "add_"):
                a = kw.get("input", kw.get("self"))
                b = kw["other"]
                if kw.get("alpha", 1) != 1:
                    raise NotImplementedError("Scaled add alpha")
                if accumulate_existing:
                    if (
                        not isinstance(prior, Value)
                        or not isinstance(a, Value)
                        or a.scale != prior.scale
                    ):
                        raise NotImplementedError(
                            "Split reduction requires an unchanged accumulator scale"
                        )
                    last = a
                    self.env[p.name] = last
                    continue
                if isinstance(a, Ref):
                    a = InputValue(a)
                if isinstance(b, Ref):
                    b = InputValue(b)
                if not isinstance(a, InputValue) or not isinstance(
                    b, InputValue
                ):
                    raise NotImplementedError(
                        "Cross-tile accumulator reduction requires in-place lowering"
                    )
                unit = min(a.scale, b.scale)
                last = self.project(a, dst, coefficient=a.scale / unit)
                self.project(
                    b, dst, coefficient=b.scale / unit, accumulate=True
                )
                last = replace(last, scale=unit)
            elif name in ("avg_pool2d", "max_pool2d"):
                source = kw.get("input", kw.get("self"))
                last = self.pool(source, dst, kw, name == "max_pool2d")
            else:
                raise NotImplementedError(f"Compute {p.target}")
            self.env[p.name] = last
        if isinstance(last, InputValue):
            last = self.project(last, dst)
        self.values[dst.key] = last

    def literal(self, key, data):
        if key not in self.dram:
            addr = self.next_dram
            self.next_dram += (len(data) + 4095) // 4096 * 4096
            if (
                self.next_dram
                > BASE + self.hardware.memory_instance("dram").size.value
            ):
                raise ValueError("Replay aperture exceeded")
            self.dram[key] = addr
            (self.output / (key + ".bin")).write_bytes(data)
            self.regions.append(
                dict(
                    name=key,
                    address=hex(addr),
                    size=len(data),
                    input=key + ".bin",
                )
            )
        return self.dram[key]

    def identity(self, coefficient=1):
        if not float(coefficient).is_integer() or not 0 <= coefficient <= 127:
            raise NotImplementedError("Non-integral/large residual coefficient")
        key = f"identity_{int(coefficient)}"
        self.literal(
            key, (np.eye(16, dtype=np.int8) * int(coefficient)).tobytes()
        )
        if self.identity_loaded != key:
            self.commands += [
                ld(16),
                command(2, self.dram[key], tile(0, 16, 16)),
            ]
            self.identity_loaded = key

    def project(self, source, dst, coefficient=1, accumulate=False):
        source = InputValue(source) if isinstance(source, Ref) else source
        x = source.ref
        if x.dtype != "int8":
            raise NotImplementedError("Pointwise input must be an INT8 buffer")
        self.identity(coefficient)
        c = self.accumulator(dst)
        m = math.prod(x.shape[:-1])
        n = x.shape[-1]
        self.commands.append(ex())
        first = True
        for j in range(0, n, 16):
            for i in range(0, m, 16):
                rows = min(16, m - i)
                out = ACC + c + (j // 16) * m + i
                self.commands += [
                    command(
                        6,
                        tile(0 if first else 0xFFFFFFFF, 16, 16),
                        tile(out | (0x40000000 if accumulate else 0), rows, 16),
                    ),
                    command(
                        4 if first else 5,
                        tile(x.address // 16 + (j // 16) * m + i, rows, 16),
                        tile(0xFFFFFFFF, rows, 16),
                    ),
                ]
                first = False
        return Value(c, x.shape, source.scale)

    def pool(self, source, dst, kw, is_max):
        source = InputValue(source) if isinstance(source, Ref) else source
        x = source.ref
        batch, h, w, n = x.shape
        if batch != 1 or any(kw.get("padding", [0, 0])):
            raise NotImplementedError(
                "Pool converter requires one already padded image tile"
            )
        if is_max:
            v = self.project(source, replace(dst, shape=x.shape))
            return replace(
                v,
                pool=dict(
                    input_shape=x.shape,
                    output_shape=dst.shape,
                    kernel=kw["kernel_size"],
                    stride=kw.get("stride") or kw["kernel_size"],
                ),
            )
        if tuple(dst.shape[1:3]) != (1, 1) or tuple(kw["kernel_size"]) != (
            h,
            w,
        ):
            raise NotImplementedError(
                "Only global average pooling is supported"
            )
        self.identity()
        c = self.accumulator(dst)
        self.commands.append(ex(a_stride=1))
        first = True
        for j in range(0, n, 16):
            for pixel in range(h * w):
                out = ACC + c + j // 16
                self.commands += [
                    command(
                        6,
                        tile(0 if first else 0xFFFFFFFF, 16, 16),
                        tile(out | (0x40000000 if pixel else 0), 1, 16),
                    ),
                    command(
                        4 if first else 5,
                        tile(
                            x.address // 16 + (j // 16) * h * w + pixel, 1, 16
                        ),
                        tile(0xFFFFFFFF, 1, 16),
                    ),
                ]
                first = False
        return Value(c, dst.shape, source.scale / (h * w))

    def init_bias(self, bias, base, m, n):
        if bias is None:
            return
        source = self.bias_sources[bias.key]
        self.commands.append(ld(0, index=2, block_stride=m))
        for j in range(0, n, 16):
            for i in range(0, m, 16):
                self.commands.append(
                    command(
                        14,
                        source.address + j * 4,
                        tile(
                            ACC + base + (j // 16) * m + i,
                            min(16, m - i),
                            min(16, n - j),
                        ),
                    )
                )

    def coordinates(self, tiling, initial):
        extents = dict(initial)
        loops = []
        for level in tiling.level_tilings:
            for b in level.loop_bounds:
                d = b.loop
                if b.bound > 1:
                    loops.append((d, b.bound, extents.get(d, 1)))
                extents[d] = extents.get(d, 1) * b.bound

        def walk(depth, coord):
            if depth < 0:
                yield coord
                return
            dim, bound, stride = loops[depth]
            for t in range(bound):
                nxt = dict(coord)
                nxt[dim] = nxt.get(dim, 0) + t * stride
                yield from walk(depth - 1, nxt)

        return extents, walk(len(loops) - 1, {})

    def conv(self, kw, dst, tiling, accumulate_existing=False):
        x, w = kw["input"], kw["weight"]
        batch, ih, iw, ci = x.shape
        kh, kwid, wci, co = w.shape
        _, oh, ow, _ = dst.shape
        stride = kw.get("stride", [1, 1])
        dilation = kw.get("dilation", [1, 1])
        if (
            kw.get("groups", 1) != 1
            or any(kw.get("padding", [0, 0]))
            or dilation != [1, 1]
        ):
            raise NotImplementedError(
                "Convolution requires a dense, already halo-padded buffer tile"
            )
        if ci != wci or ci % 16 or co % 16:
            raise ValueError("Convolution channel alignment")
        m = batch * oh * ow
        im = batch * ih * iw
        wm = kh * kwid * ci
        c = self.accumulator(dst)
        bias = kw.get("bias")
        self.init_bias(bias, c, m, co)
        self.commands.append(ex(stride[1]))
        extents, coords = self.coordinates(
            tiling, {ir.LOOP_IC: 16, ir.LOOP_OC: 16}
        )
        expected = {
            ir.LOOP_OX: ow,
            ir.LOOP_OY: oh,
            ir.LOOP_IC: ci,
            ir.LOOP_OC: co,
            ir.LOOP_FX: kwid,
            ir.LOOP_FY: kh,
        }
        if any(extents.get(d, 1) != v for d, v in expected.items()):
            raise ValueError(f"Conv tiling coverage {extents} != {expected}")
        initialized = set()
        previous_weight = None
        for pos in coords:
            ox, oy = pos.get(ir.LOOP_OX, 0), pos.get(ir.LOOP_OY, 0)
            if ox % 16:
                continue
            fx, fy = pos.get(ir.LOOP_FX, 0), pos.get(ir.LOOP_FY, 0)
            z, j = pos.get(ir.LOOP_IC, 0), pos.get(ir.LOOP_OC, 0)
            for b in range(batch):
                row = (b * oh + oy) * ow + ox
                ar = (
                    x.address // 16
                    + (z // 16) * im
                    + (b * ih + oy * stride[0] + fy) * iw
                    + ox * stride[1]
                    + fx
                )
                br = (
                    w.address // 16 + (j // 16) * wm + (fy * kwid + fx) * ci + z
                )
                key = (b, oy, ox, j)
                accumulate = (
                    accumulate_existing
                    or bias is not None
                    or key in initialized
                )
                initialized.add(key)
                out = ACC + c + (j // 16) * m + row
                nrows = min(16, ow - ox)
                reuse = retains_weight(previous_weight, br)
                self.commands += [
                    command(
                        6,
                        tile(0xFFFFFFFF if reuse else br, 16, 16),
                        tile(
                            out | (0x40000000 if accumulate else 0), nrows, 16
                        ),
                    ),
                    command(
                        5 if reuse else 4,
                        tile(ar, nrows, 16),
                        tile(0xFFFFFFFF, nrows, 16),
                    ),
                ]
                previous_weight = br
                self.stats["weight_reuses"] += int(reuse)
                self.stats["array_microtiles"] += 1
        return Value(c, tuple(dst.shape))

    def matmul(
        self,
        a,
        b,
        dst,
        tiling,
        bias=None,
        linear=False,
        accumulate_existing=False,
    ):
        m, k = a.shape
        kb, n = b.shape
        if kb != k or min(m, n, k) <= 0 or n % 16 or k % 16:
            raise ValueError("Matrix geometry")
        c = self.accumulator(dst)
        self.init_bias(bias, c, m, n)
        self.commands.append(ex())
        # Honor the mapping's loop order; group the innermost 16 output rows
        # into the physical array instruction, without selecting new tiles.
        levels = list(tiling.level_tilings)
        extents = {ir.LOOP_OX: 1, ir.LOOP_IC: 16, ir.LOOP_OC: 16}
        loops = []
        for level in levels:
            for bound in level.loop_bounds:
                d = bound.loop
                if bound.bound > 1:
                    loops.append((d, bound.bound, extents.get(d, 1)))
                extents[d] = extents.get(d, 1) * bound.bound
        if (
            extents.get(ir.LOOP_OX, 1),
            extents.get(ir.LOOP_IC, 16),
            extents.get(ir.LOOP_OC, 16),
        ) != (m, k, n):
            raise ValueError(
                f"Tiling does not cover compute: {extents} vs {(m,k,n)}"
            )
        initialized = set()
        previous_weight = None

        def emit(depth, coord):
            nonlocal previous_weight
            if depth < 0:
                i, j, z = (
                    coord.get(ir.LOOP_OX, 0),
                    coord.get(ir.LOOP_OC, 0),
                    coord.get(ir.LOOP_IC, 0),
                )
                if i % 16:
                    return
                rows = min(16, m - i)
                out = ACC + c + (j // 16) * m + i
                accumulate = (
                    accumulate_existing
                    or bias is not None
                    or (i, j) in initialized
                )
                initialized.add((i, j))
                ar = a.address // 16 + (z // 16) * m + i
                br = b.address // 16 + (j // 16) * k + z
                reuse = retains_weight(previous_weight, br)
                self.commands.extend(
                    [
                        command(
                            6,
                            tile(0xFFFFFFFF if reuse else br, 16, 16),
                            tile(
                                out | (0x40000000 if accumulate else 0),
                                rows,
                                16,
                            ),
                        ),
                        command(
                            5 if reuse else 4,
                            tile(ar, rows, 16),
                            tile(0xFFFFFFFF, rows, 16),
                        ),
                    ]
                )
                previous_weight = br
                self.stats["weight_reuses"] += int(reuse)
                self.stats["array_microtiles"] += 1
                return
            dim, bound, stride = loops[depth]
            for t in range(bound):
                nxt = dict(coord)
                nxt[dim] = nxt.get(dim, 0) + t * stride
                emit(depth - 1, nxt)

        emit(len(loops) - 1, {})
        return Value(c, tuple(dst.shape))

    def store(self, src, dst):
        v = self.values.get(src.key)
        if not isinstance(v, Value):
            raise NotImplementedError(
                "Store source has no computed accumulator"
            )
        if dst.dtype != "int8":
            raise NotImplementedError("Lean full-width mvout is disabled")
        if v.pool is not None:
            _, h, w, n = v.pool["input_shape"]
            _, oh, ow, _ = v.pool["output_shape"]
            k = v.pool["kernel"]
            stride = v.pool["stride"]
            if k[0] != k[1] or stride[0] != stride[1]:
                raise NotImplementedError("Asymmetric maxpool")
            for j in range(0, n, 16):
                for y in range(oh):
                    self.commands += [
                        st(
                            dst.strides[-2],
                            v.scale,
                            v.relu,
                            pool=(stride[0], k[0], ow, 1, ow, k[0], w, 0, 0),
                        ),
                        command(
                            3,
                            dst.address + y * dst.strides[1] + j,
                            tile(
                                ACC
                                + v.base
                                + (j // 16) * h * w
                                + y * stride[0] * w,
                                0,
                                16,
                            ),
                        ),
                    ]
            return
        geometry = self.hardware.connection("acc_to_dram").transfer_geometry
        m = math.prod(dst.shape[:-1])
        n = dst.shape[-1]
        addresses = [
            dst.address + sum(c * s for c, s in zip(coord, dst.strides))
            for coord in itertools.product(*(range(s) for s in dst.shape[:-1]))
        ]
        for j in range(0, n, 16):
            i = 0
            while i < m:
                stride = addresses[i + 1] - addresses[i] if i + 1 < m else n
                rows = 1
                while (
                    rows < geometry.max_rows
                    and i + rows < m
                    and addresses[i + rows] == addresses[i] + rows * stride
                ):
                    rows += 1
                self.commands += [
                    st(stride, v.scale, v.relu),
                    command(
                        3,
                        addresses[i] + j,
                        tile(
                            ACC + v.base + (j // 16) * m + i,
                            rows,
                            min(16, n - j),
                        ),
                    ),
                ]
                i += rows

    def prepare(self):
        boxes = {}
        written = set()

        def refs(msg):
            if isinstance(msg, ir.TensorBox):
                if msg.memory.level == DRAM:
                    boxes.setdefault(msg.node, msg)
                return
            for field, value in msg.ListFields():
                if field.type != field.TYPE_MESSAGE:
                    continue
                if field.is_repeated:
                    if field.message_type.GetOptions().map_entry:
                        for x in value.values():
                            refs(x)
                    else:
                        for x in value:
                            refs(x)
                else:
                    refs(value)

        for o in self.ops:
            refs(o)
            if (
                o.WhichOneof("op_type") == "prim"
                and o.prim.target == "aten::permute"
            ):
                self.layout_ops[o.outputs[0].name] = o.prim
            for out in o.outputs:
                if (
                    out.WhichOneof("result_type") == "tensor_box"
                    and out.tensor_box.memory.level == DRAM
                ):
                    if o.prim.target in ("voyager::alloc", "voyager::zeros"):
                        written.add(out.tensor_box.node)
        for name, b in boxes.items():
            if b.dtype not in ("int8", "int32"):
                raise NotImplementedError(
                    f"{name}: Lean cannot spill {b.dtype}; quantize/fuse this boundary"
                )
            if name in self.layout_ops:
                p = self.layout_ops[name]
                source = p.kwargs["input"].tensor_box.box
                if source.node in written:
                    # Postprocessing views do not allocate hardware output.
                    self.dram[name] = self.dram[source.node]
                    continue
                # Verify host packing is only a layout transform of the supplied
                # input, never a reference-computed arithmetic intermediate.
                dims = [
                    self.scalar(v) for v in p.kwargs["dims"].scalar_list.values
                ]
                original = np.fromfile(
                    self.root / "tensor_files" / f"{source.node}.bin",
                    dtype="<f4",
                ).reshape(source.shape)
                packed = original.transpose(dims).copy()
                emitted = np.fromfile(
                    self.root / "tensor_files" / f"{name}.bin", dtype="<f4"
                ).reshape(b.shape)
                if not np.array_equal(packed, emitted):
                    raise ValueError("Input layout collateral mismatch")
                self.input_layouts[name] = dict(
                    source=source.node, dims=dims, shape=list(source.shape)
                )
            arr = np.fromfile(
                self.root / "tensor_files" / f"{name}.bin", dtype="<f4"
            )
            dtype = np.dtype("i1" if b.dtype == "int8" else "<i4")
            if (
                np.any(arr != np.round(arr))
                or np.any(arr < np.iinfo(dtype).min)
                or np.any(arr > np.iinfo(dtype).max)
            ):
                raise ValueError(
                    f"{name}: tensor is not representable as {dtype}"
                )
            data = arr.astype(dtype).tobytes()
            size = math.prod(b.shape) * width(b.dtype)
            if len(data) != size:
                raise ValueError(f"{name}: tensor size mismatch")
            data += bytes((-size) % 16)
            size = len(data)
            addr = self.next_dram
            self.next_dram += (size + 4095) // 4096 * 4096
            if (
                self.next_dram
                > BASE + self.hardware.memory_instance("dram").size.value
            ):
                raise ValueError("Replay aperture exceeded")
            self.dram[name] = addr
            filename = name + (".expected.bin" if name in written else ".bin")
            if name not in self.produced or name in written:
                (self.output / filename).write_bytes(data)
            region = dict(name=name, address=hex(addr), size=size)
            if name in written:
                region.update(fill=165, output=name + ".bin", expected=filename)
                self.source_outputs.append(name)
            else:
                if name in self.produced:
                    filename = "../../state/" + name + ".bin"
                    self.chained_inputs.append(name)
                region.update(input=filename)
                self.source_inputs.append(name)
            self.regions.append(region)
        self.region(self.ops)
        self.commands.append(dict(type="drain"))
        if self.schedule_async:
            self.commands, scheduling = self.schedule.reorder()
            self.stats.update(
                {"schedule_" + k: v for k, v in scheduling.items()}
            )
        # Gemmini configuration registers are sticky. Eliminate redundant writes
        # to the same register without changing the scheduled DMA/compute order.
        config_state = {}
        commands = []
        for cmd in self.commands:
            if (
                cmd["type"] == "command"
                and int(cmd["instruction"], 16) >> 25 == 0
            ):
                rs1 = int(cmd["rs1"], 16)
                kind = rs1 & 3
                key = (kind, (rs1 >> 3) & 3 if kind == 1 else 0)
                value = (cmd["rs1"], cmd["rs2"])
                if config_state.get(key) == value:
                    self.stats["redundant_config_writes"] += 1
                    continue
                config_state[key] = value
            commands.append(cmd)
        self.commands = commands
        (self.output / "commands.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in self.commands)
        )
        write_json(
            self.output / "memory.json",
            dict(
                version=1,
                base=hex(BASE),
                size=self.hardware.memory_instance("dram").size.value,
                alignment=16,
                regions=self.regions,
            ),
        )
        write_json(
            self.output / "lowering.json",
            dict(
                source="standard bufferized model.txt",
                statistics=dict(self.stats),
                inputs=self.source_inputs,
                outputs=self.source_outputs,
                chained_inputs=self.chained_inputs,
                input_layouts=self.input_layouts,
                hardware_loop_instructions=False,
                async_schedule=dict(
                    enabled=self.schedule_async,
                    queues=asdict(self.schedule.queues),
                    policy=asdict(self.policy.tuning),
                    dependencies="source semaphores and physical RAW/WAR/WAW",
                ),
            ),
        )


def convert(root, output=None, segmented=False, policy=None, context=None):
    root = Path(root).resolve()
    from voyager_compiler.compilation import CompilerContext

    hardware = lean_config()
    if context is None:
        context = (
            CompilerContext.resolve(hardware, policy)
            if policy is not None
            else CompilerContext.from_artifacts(root, hardware)
        )
    if context.hardware != hardware or (
        policy is not None and policy is not context.policy
    ):
        raise ValueError(
            "Converter context differs from explicit hardware/policy"
        )
    context.check_artifacts(root)
    policy = context.policy
    output = Path(output).resolve() if output else root / "replay"
    model = text_format.Parse((root / "model.txt").read_text(), ir.Model())
    if segmented:
        boundaries = [
            i
            for i, o in enumerate(model.ops)
            if o.WhichOneof("op_type") == "prim"
            and o.prim.target == "voyager::alloc"
            and any(
                v.WhichOneof("result_type") == "tensor_box"
                and v.tensor_box.memory.level == DRAM
                for v in o.outputs
            )
        ]
        if not boundaries:
            raise ValueError("No bufferized DRAM output allocations")
        boundaries[0] = 0
        boundaries.append(len(model.ops))
        produced, segments, host_outputs = set(), [], []
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            group = model.ops[start:end]
            wide = [
                v.tensor_box
                for o in group
                for v in o.outputs
                if v.WhichOneof("result_type") == "tensor_box"
                and v.tensor_box.memory.level == DRAM
                and v.tensor_box.dtype == "float32"
            ]
            if wide and end == len(model.ops):
                host_outputs.append(
                    output_dequantization(root, group, wide, produced)
                )
                continue
            part = output / "segments" / f"{i:03d}"
            converter = Converter(root, group, part, produced, policy=policy)
            converter.prepare()
            segments.append(
                dict(
                    path=str(part.relative_to(output)),
                    inputs=converter.source_inputs,
                    chained_inputs=converter.chained_inputs,
                    outputs=converter.source_outputs,
                )
            )
            produced.update(converter.source_outputs)
        write_json(
            output / "network.json",
            dict(
                source=str(root / "model.txt"),
                segments=segments,
                outputs=[b.node for b in model.outputs],
                host_outputs=host_outputs,
                logical_output=(
                    json.loads((root / "output.json").read_text())
                    if (root / "output.json").exists()
                    else None
                ),
            ),
        )
    else:
        converter = Converter(root, model.ops, output, policy=policy)
        converter.prepare()
    return output


def output_dequantization(root, ops, outputs, produced):
    """Recognize only a final pointwise decode, executed by the host runtime.

    Lean cannot store full-width accumulators. The CPU consumes actual INT8
    logits and restores their units; no host arithmetic feeds an RTL kernel.
    """

    def walk(ops):
        for op in ops:
            kind = op.WhichOneof("op_type")
            if kind == "prim":
                yield op.prim
            elif kind == "fused":
                yield from op.fused.op_list
            elif kind == "loop":
                yield from walk(op.loop.for_loop.body.ops)
            elif kind == "cond":
                yield from walk(op.cond.true_region.ops)
                yield from walk(op.cond.false_region.ops)
            elif kind == "async":
                yield from walk(getattr(op, "async").body.ops)

    prims = list(walk(ops))
    compute = [p for p in prims if p.op != "cpu"]
    if (
        len(outputs) != 1
        or len(compute) != 1
        or compute[0].target != "quantized_ops::dequantize"
    ):
        raise NotImplementedError(
            "Only final scalar dequantization may execute on the host"
        )
    p = compute[0]
    for key in (
        "zero_point",
        "input_qmap",
        "output_qmap",
        "axes",
        "block_size",
    ):
        if key in p.kwargs:
            raise NotImplementedError(
                "Unsupported output decode parameter: " + key
            )
    scale_name = p.kwargs["scale"].tensor_box.box.node
    scale = np.fromfile(
        root / "tensor_files" / f"{scale_name}.bin", dtype="<f4"
    )
    if scale.size != 1:
        raise NotImplementedError("Output scale must be scalar")
    sources = {
        p.kwargs["src"].tensor_box.box.node
        for p in prims
        if p.target == "voyager::async_copy"
        and p.kwargs["src"].tensor_box.box.memory.level == DRAM
    }
    if len(sources) != 1 or not sources.issubset(produced):
        raise ValueError(
            "Final decode must consume the preceding hardware output"
        )
    return dict(
        input=sources.pop(),
        output=outputs[0].node,
        scale=float(scale[0]),
        shape=list(outputs[0].shape),
        expected=str(root / "tensor_files" / f"{outputs[0].node}.bin"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--segmented", action="store_true")
    args = parser.parse_args()
    print(convert(args.root, args.output, args.segmented))


if __name__ == "__main__":
    main()
