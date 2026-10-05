"""Resolved RoCC encodings from the pinned gemmini.h; no CPU instructions."""

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
