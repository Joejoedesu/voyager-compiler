"""Gemmini deployment arithmetic and shape contracts, independent of encoding."""

import math
from types import SimpleNamespace

import torch
from torch.library import Library

from voyager_compiler.extensions import CandidateRejected

LIBRARY = Library("gemmini", "DEF")
LIBRARY.define(
    "matmul(Tensor a, Tensor b, Tensor bias, float scale, bool relu=False) -> Tensor"
)
LIBRARY.define(
    "conv2d(Tensor x, Tensor weight, Tensor bias, int stride, int pad, float scale, bool relu=False) -> Tensor"
)
LIBRARY.define(
    "depthwise_conv2d(Tensor x, Tensor weight, Tensor bias, int stride, int pad, float scale, bool relu=False) -> Tensor"
)
LIBRARY.define(
    "requantize(Tensor x, float scale=1., bool relu=False) -> Tensor"
)
LIBRARY.define(
    "add(Tensor a, Tensor b, float a_scale, float b_scale, bool relu=False) -> Tensor"
)
LIBRARY.define("average(Tensor x, float scale=1.) -> Tensor")
LIBRARY.define(
    "maxpool(Tensor x, int kernel, int stride, int pad, bool input_nonnegative=False) -> Tensor"
)


def quant(x, scale=1.0, relu=False):
    """The donor's float32 scaler, nearest-even rounding, signed INT8 saturation."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Gemmini scale must be finite and nonnegative")
    return (
        (x.to(torch.float32) * scale)
        .round()
        .clamp(0 if relu else -128, 127)
        .to(torch.int8)
    )


def accumulator(x):
    if not torch.isfinite(x).all() or x.abs().max() >= 2**31:
        raise ValueError("Gemmini INT32 accumulator overflow")
    return x


def matmul(a, b, bias, scale, relu=False):
    return quant(
        accumulator(a.double() @ b.double() + bias.double()), scale, relu
    )


def conv2d(x, weight, bias, stride, pad, scale, relu=False):
    out = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).double(),
        weight.permute(3, 2, 0, 1).double(),
        bias.double(),
        stride,
        pad,
    )
    return quant(accumulator(out), scale, relu).permute(0, 2, 3, 1).contiguous()


def depthwise_conv2d(x, weight, bias, stride, pad, scale, relu=False):
    out = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2).double(),
        weight.double(),
        bias.double(),
        stride,
        pad,
        groups=x.shape[-1],
    )
    return quant(accumulator(out), scale, relu).permute(0, 2, 3, 1).contiguous()


def add(a, b, a_scale, b_scale, relu=False):
    # MVIN scaling saturates each operand BEFORE accumulation.
    return quant(quant(a, a_scale).int() + quant(b, b_scale).int(), 1, relu)


def average(x, scale=1.0):
    return quant(
        accumulator(x.double().sum((1, 2), keepdim=True)),
        scale / (x.shape[1] * x.shape[2]),
    )


def maxpool(x, kernel, stride, pad, input_nonnegative=False):
    if pad and (not input_nonnegative or torch.any(x < 0)):
        raise ValueError(
            "Gemmini zero-padded maxpool requires proven nonnegative input"
        )
    return (
        torch.nn.functional.max_pool2d(
            x.permute(0, 3, 1, 2).float(), kernel, stride, pad
        )
        .to(torch.int8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


for name, fn in dict(
    matmul=matmul,
    conv2d=conv2d,
    depthwise_conv2d=depthwise_conv2d,
    requantize=quant,
    add=add,
    average=average,
    maxpool=maxpool,
).items():
    LIBRARY.impl(name, fn, "CompositeExplicitAutograd")


@torch.library.register_fake("gemmini::matmul")
def fake_matmul(a, b, bias, scale, relu=False):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.int8)


@torch.library.register_fake("gemmini::conv2d")
def fake_conv(x, weight, bias, stride, pad, scale, relu=False):
    return x.new_empty(
        (
            x.shape[0],
            (x.shape[1] + 2 * pad - weight.shape[0]) // stride + 1,
            (x.shape[2] + 2 * pad - weight.shape[1]) // stride + 1,
            weight.shape[3],
        )
    )


@torch.library.register_fake("gemmini::depthwise_conv2d")
def fake_depthwise(x, weight, bias, stride, pad, scale, relu=False):
    return x.new_empty(
        (
            x.shape[0],
            (x.shape[1] + 2 * pad - weight.shape[2]) // stride + 1,
            (x.shape[2] + 2 * pad - weight.shape[3]) // stride + 1,
            x.shape[3],
        )
    )


@torch.library.register_fake("gemmini::requantize")
def fake_quant(x, scale=1.0, relu=False):
    return x.new_empty(x.shape, dtype=torch.int8)


@torch.library.register_fake("gemmini::add")
def fake_add(a, b, a_scale, b_scale, relu=False):
    return a.new_empty(a.shape, dtype=torch.int8)


@torch.library.register_fake("gemmini::average")
def fake_average(x, scale=1.0):
    return x.new_empty((x.shape[0], 1, 1, x.shape[3]))


@torch.library.register_fake("gemmini::maxpool")
def fake_pool(x, kernel, stride, pad, input_nonnegative=False):
    return x.new_empty(
        (
            x.shape[0],
            (x.shape[1] + 2 * pad - kernel) // stride + 1,
            (x.shape[2] + 2 * pad - kernel) // stride + 1,
            x.shape[3],
        )
    )


OPS = tuple(
    getattr(torch.ops.gemmini, name).default
    for name in (
        "matmul",
        "conv2d",
        "depthwise_conv2d",
        "requantize",
        "add",
        "average",
        "maxpool",
    )
)


def _validate(kind, p, operands):
    def positive(*keys):
        for key in keys:
            if type(p[key]) is not int or not 0 < p[key] < 65536:
                raise ValueError(f"{kind}: invalid dimension {key}={p[key]}")

    def shape(key, dims, dtype="int8", flatten=False):
        box = operands[key]
        if box.dtype != dtype or (
            math.prod(box.shape) != math.prod(dims)
            if flatten
            else tuple(box.shape) != tuple(dims)
        ):
            raise ValueError(
                f"{kind}: {key} requires {dims} {dtype}, got {list(box.shape)} {box.dtype}"
            )

    common = {"scale", "relu", "tile"}
    if kind == "matmul":
        allowed = common | {"m", "n", "k"}
        required = {"a", "b", "y"}
        positive("m", "n", "k")
        shape("a", (p["m"], p["k"]), flatten=True)
        shape("b", (p["k"], p["n"]))
        shape("y", (p["m"], p["n"]))
        if "bias" in operands:
            shape("bias", (p["n"],), "int32")
            required.add("bias")
    elif kind in ("conv2d", "depthwise_conv2d"):
        allowed = common | {
            "batch",
            "h",
            "w",
            "ci",
            "co",
            "oh",
            "ow",
            "kernel",
            "stride",
            "pad",
        }
        required = {"x", "weight", "bias", "y"}
        positive("batch", "h", "w", "ci", "co", "oh", "ow", "kernel")
        s, pad, k = p.get("stride", 1), p.get("pad", 0), p["kernel"]
        if (
            type(s) is not int
            or not 1 <= s < 256
            or type(pad) is not int
            or not 0 <= pad < 256
        ):
            raise ValueError("Convolution stride/padding field overflow")
        if (
            p["oh"] != (p["h"] + 2 * pad - k) // s + 1
            or p["ow"] != (p["w"] + 2 * pad - k) // s + 1
        ):
            raise ValueError("Convolution output geometry does not match input")
        shape("x", (p["batch"], p["h"], p["w"], p["ci"]))
        if kind == "depthwise_conv2d":
            if p["ci"] != p["co"]:
                raise ValueError(
                    "Depthwise hardware loop requires channel multiplier one"
                )
            shape("weight", (p["co"], 1, k, k))
        else:
            shape("weight", (k, k, p["ci"], p["co"]))
        shape("bias", (p["co"],), "int32")
        shape("y", (p["batch"], p["oh"], p["ow"], p["co"]))
    elif kind == "maxpool":
        allowed = {
            "batch",
            "h",
            "w",
            "channels",
            "oh",
            "ow",
            "kernel",
            "stride",
            "pad",
            "input_nonnegative",
        }
        required = {"x", "y"}
        positive("h", "w", "channels", "oh", "ow", "kernel", "stride")
        k, s, pad = p["kernel"], p["stride"], p["pad"]
        if (
            not 1 <= k <= 3
            or not 1 <= s <= 3
            or not 0 <= pad <= k // 2
            or max(p["h"], p["w"], p["oh"], p["ow"]) > 255
        ):
            raise ValueError("Unsupported Gemmini pooling geometry")
        if p["h"] * k > 1024 or p["w"] * k > 1024:
            raise ValueError("Pooling strip exceeds accumulator capacity")
        if (
            p["oh"] != (p["h"] + 2 * pad - k) // s + 1
            or p["ow"] != (p["w"] + 2 * pad - k) // s + 1
        ):
            raise ValueError("Pooling output geometry mismatch")
        shape("x", (p.get("batch", 1), p["h"], p["w"], p["channels"]))
        shape("y", (p.get("batch", 1), p["oh"], p["ow"], p["channels"]))
    elif kind == "average":
        allowed = {"batch", "pixels", "channels", "scale"}
        required = {"x", "y"}
        positive("pixels", "channels")
        shape(
            "x", (p.get("batch", 1), p["pixels"], p["channels"]), flatten=True
        )
        shape("y", (p.get("batch", 1), p["channels"]), flatten=True)
    elif kind in ("relu", "requantize", "add"):
        allowed = {"rows", "channels", "scale", "relu", "a_scale", "b_scale"}
        required = {"x", "y"} | ({"b"} if kind == "add" else set())
        positive("rows", "channels")
        for key in required:
            shape(key, (p["rows"], p["channels"]), flatten=True)
    else:
        raise NotImplementedError(f"Unsupported Lean operation {kind}")
    if set(p) - allowed or set(operands) != required:
        raise ValueError(f"{kind}: unsupported parameters/operands")
    if "tile" in p and any(type(v) is not int or v <= 0 for v in p["tile"]):
        raise ValueError("Tile dimensions must be positive integers")


def facts(node):
    name = node.target._schema.name.split("::")[-1]
    args = {a.name: a.default_value for a in node.target._schema.arguments}
    args.update(zip(args, node.args))
    args.update(node.kwargs)
    tensors = {k: v for k, v in args.items() if isinstance(v, torch.fx.Node)}
    p = {k: v for k, v in args.items() if k not in tensors}
    if name == "matmul":
        a, b = tensors["a"].value, tensors["b"].value
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise CandidateRejected(
                "Gemmini matmul requires compatible 2D operands"
            )
        p.update(m=a.shape[0], k=a.shape[1], n=b.shape[1])
    elif name == "conv2d":
        x, w = tensors["x"].value, tensors["weight"].value
        if (
            x.ndim != 4
            or w.ndim != 4
            or w.shape[0] != w.shape[1]
            or x.shape[3] != w.shape[2]
        ):
            raise CandidateRejected(
                "Gemmini convolution requires NHWC and square HWIO weights"
            )
        p.update(
            batch=x.shape[0],
            h=x.shape[1],
            w=x.shape[2],
            ci=x.shape[3],
            co=w.shape[3],
            oh=node.value.shape[1],
            ow=node.value.shape[2],
            kernel=w.shape[0],
        )
    elif name == "depthwise_conv2d":
        x, w = tensors["x"].value, tensors["weight"].value
        if (
            x.ndim != 4
            or w.ndim != 4
            or w.shape[:2] != (x.shape[3], 1)
            or w.shape[2] != w.shape[3]
        ):
            raise CandidateRejected(
                "Depthwise convolution requires NHWC and Cx1xRxR weights"
            )
        p.update(
            batch=x.shape[0],
            h=x.shape[1],
            w=x.shape[2],
            ci=x.shape[3],
            co=x.shape[3],
            oh=node.value.shape[1],
            ow=node.value.shape[2],
            kernel=w.shape[2],
        )
    elif name == "average":
        x = tensors["x"].value
        if x.ndim != 4:
            raise CandidateRejected("Global average requires NHWC")
        p.update(
            batch=x.shape[0],
            pixels=x.shape[1] * x.shape[2],
            channels=x.shape[3],
        )
    elif name == "maxpool":
        x = tensors["x"].value
        if x.ndim != 4:
            raise CandidateRejected("Maxpool requires NHWC")
        p.update(
            batch=x.shape[0],
            h=x.shape[1],
            w=x.shape[2],
            channels=x.shape[3],
            oh=node.value.shape[1],
            ow=node.value.shape[2],
        )
    else:
        if name == "add":
            tensors["x"] = tensors.pop("a")
        x = tensors["x"].value
        p.update(rows=math.prod(x.shape[:-1]), channels=x.shape[-1])
    boxes = {
        k: SimpleNamespace(
            shape=v.value.shape,
            dtype=str(v.value.dtype).removeprefix("torch."),
        )
        for k, v in tensors.items()
    }
    boxes["y"] = SimpleNamespace(
        shape=node.value.shape,
        dtype=str(node.value.dtype).removeprefix("torch."),
    )
    try:
        _validate(name, p, boxes)
    except (ValueError, KeyError) as error:
        raise CandidateRejected(str(error)) from error
    return name, p, tensors


def raw_dimensions(node):
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten._int_mm.default
    ):
        raise CandidateRejected(
            "this Gemmini family supports aten._int_mm only"
        )
    a, b = node.args
    m, k = a.value.shape
    kb, n = b.value.shape
    if (
        k != kb
        or any(t.value.dtype != torch.int8 for t in (a, b))
        or n % 16
        or k % 16
    ):
        raise CandidateRejected(
            "Gemmini requires INT8 matrices with 16-aligned N and K"
        )
    if m % 16 or max(m, n, k) >= 65536:
        raise CandidateRejected(
            "this native family requires 16-aligned M and 16-bit bounds"
        )
    return m, n, k


def tile_storage(kind, params, tile, mode):
    """Physical scratchpad/accumulator bytes, shared by search and emission."""

    def blocks(value):
        return (value + 15) // 16

    if kind == "matmul":
        m, n, k = tile
        if mode == "hardware_loop":
            m, n, k = (blocks(v) * 16 for v in tile)
        return 16 * (m * blocks(k) + k * blocks(n)), 64 * m * blocks(n)
    if kind == "depthwise_conv2d":
        r, c = tile
        return tile_storage("conv2d", params, (r, c, 1, 1), mode)
    if kind == "conv2d":
        r, c, o, i = tile
        kernel, stride = params["kernel"], params["stride"]
        return (
            16
            * (
                (r * stride + kernel - 1)
                * (c * stride + kernel - 1)
                * blocks(i)
                + kernel * kernel * i * blocks(o)
            ),
            64 * r * c * blocks(o),
        )
    raise ValueError(f"No matrix tile storage for {kind}")


def selected_tile(kind, p, mapping):
    """Project the selected on-chip extents; tails remain logical extents."""
    import interstellar

    le = interstellar.le
    loops = (
        (le.OX, le.OC, le.IC)
        if kind == "matmul"
        else (le.OY, le.OX, le.OC, le.IC)
    )
    full = (
        (p["m"], p["n"], p["k"])
        if kind == "matmul"
        else (p["oh"], p["ow"], p["co"], p["ci"])
    )
    return tuple(
        min(
            extent,
            math.prod(
                b * q
                for b, q in zip(
                    mapping.loop_blockings[loop][:2],
                    mapping.loop_partitionings[loop][:2],
                )
            ),
        )
        for loop, extent in zip(loops, full)
    )
