"""Private instruction-word packing for the emission hook; not an extension API."""

import math
import struct

STATUS = dict(prv=3, dprv=3, mprv=0, v=0, dv=0, satp=0)
ACC = 0x80000000


def command(funct, rs1=0, rs2=0):
    if not 0 <= rs1 < 1 << 64 or not 0 <= rs2 < 1 << 64:
        raise ValueError("RoCC operand overflow")
    return dict(
        type="command",
        instruction=hex(
            (funct << 25) | (2 << 20) | (1 << 15) | (3 << 12) | 0x7B
        ),
        rs1=hex(rs1),
        rs2=hex(rs2),
    )


def bits(x):
    if not math.isfinite(x) or x < 0:
        raise ValueError("Scale must be finite and nonnegative")
    return struct.unpack("<I", struct.pack("<f", x))[0]


def tile(addr, rows, cols):
    if not 0 <= rows < 65536 or not 0 < cols < 65536:
        raise ValueError("Invalid DMA tile dimensions")
    return (rows << 48) | (cols << 32) | addr


def ld(stride, index=0, shrunk=False, scale=1.0, block_stride=16):
    return command(
        0,
        (bits(scale) << 32)
        | (block_stride << 16)
        | (1 << 8)
        | (index << 3)
        | (int(shrunk) << 2)
        | 1,
        stride,
    )


def st(stride, scale=1.0, relu=False, pool=None):
    a = 2 | (int(relu) << 2)
    if pool:
        for v, shift in zip(pool, (4, 6, 24, 32, 40, 48, 56, 8, 10)):
            a |= v << shift
    return command(0, a, (bits(scale) << 32) | stride)


def ex(a_stride=1):
    return command(0, (bits(1.0) << 32) | (a_stride << 16) | 4, 1 << 48)


def loop_mm(
    m, n, k, a, b, d, c, sa, sb, sc, *, accumulate=False, full_output=False
):
    """LOOP_WS configuration and launch, matching the pinned gemmini.h."""
    i, j, z = ((v + 15) // 16 for v in (m, n, k))
    if min(i, j, z) < 1 or max(i, j, z) >= 65536:
        raise ValueError("Hardware matrix loop bounds out of range")
    return [
        command(
            9,
            ((z * 16 - k) << 32) | ((j * 16 - n) << 16) | (i * 16 - m),
            (z << 32) | (j << 16) | i,
        ),
        command(10, a, b),
        command(11, d, c),
        command(12, sa, sb),
        command(13, 0, sc),
        command(8, (int(full_output) << 1) | int(accumulate), 0),
    ]


def loop_conv(
    *,
    batch,
    h,
    w,
    ci,
    co,
    oh,
    ow,
    stride,
    kernel,
    rows,
    cols,
    channels,
    kchannels,
    lp,
    rp,
    up,
    down,
    x,
    weight,
    bias,
    y,
    tile_batch=1,
    pad=0,
    no_bias=False,
    relu=False,
):
    """Dense NHWC LOOP_CONV_WS; no pooling, dilation or transpose."""
    fields = (
        batch,
        h,
        w,
        ci,
        co,
        oh,
        ow,
        rows,
        cols,
        channels,
        kchannels,
        tile_batch,
    )
    if any(not 0 < v < 65536 for v in fields) or not 0 < kernel < 256:
        raise ValueError("Hardware convolution loop bounds out of range")
    if (
        not 0 <= pad < 256
        or not 0 < stride < 256
        or any(not 0 <= v < 256 for v in (lp, rp, up, down))
    ):
        raise ValueError("Hardware convolution stride/padding out of range")
    return [
        command(
            16,
            (co << 48) | (ci << 32) | (h << 16) | batch,
            (pad << 56) | (stride << 48) | (ow << 32) | (oh << 16) | oh,
        ),
        command(
            17,
            (kernel << 48) | (ow << 32) | (1 << 16) | (1 << 8),
            (tile_batch << 48) | (rows << 32) | (cols << 16) | channels,
        ),
        command(
            18,
            (kernel << 48) | (kernel << 32) | (kchannels << 16) | lp,
            (rp << 48) | (up << 32) | (down << 24) | w,
        ),
        command(
            19, (rows << 48) | 1, (ci << 48) | (co << 32) | (co << 16) | cols
        ),
        command(20, weight, y),
        command(21, bias, x),
        command(15, (1 << 8) | int(no_bias), 1 | (int(relu) << 3)),
    ]
