"""Project selected Gemmini tiles into commands; no tile selection in emission.

Operation expansions are adapted from the voyager-gemmini donor scheduler.
Residual phase drains preserve its validated conservative load/add/store order.
"""

SPAD_ROWS = 16384
ACC_ROWS = 1024
from ._encoding import ACC, command, ex, ld, loop_conv, loop_mm, st, tile


def ceildiv(a, b):
    return (a + b - 1) // b


def matmul(p, addr):
    m, k, n = p["m"], p["k"], p["n"]
    tm, tn, tk = p["tile"]
    i, j, z = (ceildiv(v, 16) for v in (tm, tn, tk))
    if 16 * i * j > ACC_ROWS // 2 or 16 * (i + j) * z > SPAD_ROWS // 2:
        raise ValueError("Matmul tile exceeds double-buffered Lean capacity")
    a, b, c = addr["a"], addr["b"], addr["y"]
    d = addr.get("bias", 0)
    out = [
        ex(),
        st(n, p.get("scale", 1), p.get("relu", False)),
        ld(k),
        ld(n, 1),
        ld(0, 2),
    ]
    for r in range(0, m, tm):
        for s in range(0, n, tn):
            for t in range(0, k, tk):
                out += loop_mm(
                    min(tm, m - r),
                    min(tn, n - s),
                    min(tk, k - t),
                    a + r * k + t,
                    b + t * n + s,
                    d + s * 4 if d and t == 0 else 0,
                    c + r * n + s if t + tk >= k else 0,
                    k,
                    n,
                    n,
                    accumulate=bool(d or t),
                )
    return out


def conv(p, addr):
    batch, h, w, ci, co, oh, ow, k = (
        p[x] for x in ("batch", "h", "w", "ci", "co", "oh", "ow", "kernel")
    )
    stride, pad = p.get("stride", 1), p.get("pad", 0)
    tr, tc, to, ti = p["tile"]
    ar = (tr * stride + k - 1) * (tc * stride + k - 1) * ceildiv(ti, 16)
    br = k * k * ti * ceildiv(to, 16)
    if ar + br > SPAD_ROWS // 2 or tr * tc * ceildiv(to, 16) > ACC_ROWS // 2:
        raise ValueError(
            "Convolution tile exceeds double-buffered Lean capacity"
        )
    out = [st(co, p.get("scale", 1), p.get("relu", False)), ex(stride)]
    bias = addr.get("bias", 0)
    # Zero bias has real DMA storage too; it starts each reduction context.
    if not bias:
        raise ValueError("Convolution needs an explicit bias/zero-bias region")
    for b in range(batch):
        for r in range(0, oh, tr):
            for c in range(0, ow, tc):
                for oc in range(0, co, to):
                    for ic in range(0, ci, ti):
                        rows, cols = min(tr, oh - r), min(tc, ow - c)
                        ir, ix = r * stride - pad, c * stride - pad
                        up, lp = max(0, -ir), max(0, -ix)
                        down = max(0, ir + rows * stride + k - 1 - h)
                        rp = max(0, ix + cols * stride + k - 1 - w)
                        out += loop_conv(
                            batch=batch,
                            h=h,
                            w=w,
                            ci=ci,
                            co=co,
                            oh=oh,
                            ow=ow,
                            stride=stride,
                            pad=pad,
                            kernel=k,
                            rows=rows,
                            cols=cols,
                            channels=min(to, co - oc),
                            kchannels=min(ti, ci - ic),
                            lp=lp,
                            rp=rp,
                            up=up,
                            down=down,
                            x=addr["x"]
                            + ((b * h + ir + up) * w + ix + lp) * ci
                            + ic,
                            weight=addr["weight"] + ic * co + oc,
                            bias=bias + oc * 4 if ic == 0 else 0,
                            y=(
                                addr["y"] + ((b * oh + r) * ow + c) * co + oc
                                if ic + ti >= ci
                                else 0
                            ),
                            no_bias=False,
                            relu=p.get("relu", False),
                        )
    return out


def pointwise(kind, p, addr):
    channels = p["channels"]
    out = [
        ex(),
        st(channels, p.get("scale", 1), kind == "relu" or p.get("relu", False)),
    ]
    if kind == "average":
        out += [ld(channels, shrunk=True), st(channels, 1 / p["pixels"])]
        for b in range(p.get("batch", 1)):
            for c in range(0, channels, 16):
                nc = min(16, channels - c)
                for r in range(p["pixels"]):
                    out.append(
                        command(
                            2,
                            addr["x"] + (b * p["pixels"] + r) * channels + c,
                            tile(ACC | ((r != 0) << 30), 1, nc),
                        )
                    )
                out.append(
                    command(3, addr["y"] + b * channels + c, tile(ACC, 1, nc))
                )
        return out
    out += [ld(channels, shrunk=True, scale=p.get("a_scale", 1))]
    if kind == "add":
        out += [ld(channels, index=1, shrunk=True, scale=p.get("b_scale", 1))]
        # Retire each DMA phase before a read/modify/write or row reuse. Both
        # overlapped explicit loads and LoopMatmul residual mode dropped some
        # scaled B contributions on this pin; keep the tested conservative path.
        tm, tn = p["tile"]
        out += [
            ld(
                channels,
                shrunk=True,
                scale=p.get("a_scale", 1),
                block_stride=tm,
            ),
            ld(
                channels,
                index=1,
                shrunk=True,
                scale=p.get("b_scale", 1),
                block_stride=tm,
            ),
        ]
        for r in range(0, p["rows"], tm):
            for c in range(0, channels, tn):
                nr, nc = min(tm, p["rows"] - r), min(tn, channels - c)
                for key, funct, flags in (
                    ("x", 2, ACC),
                    ("b", 1, ACC | (1 << 30)),
                    ("y", 3, ACC),
                ):
                    block = 16 if key == "y" else 64
                    for cc in range(0, nc, block):
                        for rr in range(0, nr, 16):
                            local = flags + (cc // 16) * tm + rr
                            offset = (r + rr) * channels + c + cc
                            out.append(
                                command(
                                    funct,
                                    addr[key] + offset,
                                    tile(
                                        local,
                                        min(16, nr - rr),
                                        min(block, nc - cc),
                                    ),
                                )
                            )
                    out.append(dict(type="drain"))
        return out
    for r in range(0, p["rows"], 16):
        for c in range(0, channels, 16):
            nr, nc = min(16, p["rows"] - r), min(16, channels - c)
            offset = r * channels + c
            out.append(command(2, addr["x"] + offset, tile(ACC, nr, nc)))
            out.append(command(3, addr["y"] + offset, tile(ACC, nr, nc)))
    return out


def maxpool(p, addr):
    # Tile one output row, preserving the full input-row stride and true halo.
    h, w, ch, oh, ow, k, stride, pad = (
        p[x]
        for x in ("h", "w", "channels", "oh", "ow", "kernel", "stride", "pad")
    )
    if pad and not p.get("input_nonnegative", False):
        raise NotImplementedError(
            "Gemmini maxpool pads with zero: padded pooling requires a proven nonnegative input (e.g. ReLU)"
        )
    out = [ex(), ld(ch, shrunk=True)]
    for b in range(p.get("batch", 1)):
        for r in range(oh):
            start = r * stride - pad
            up, down = max(0, -start), max(0, start + k - h)
            rows = k - up - down
            for c in range(0, ch, 16):
                nc = min(16, ch - c)
                for y in range(rows):
                    for x in range(0, w, 16):
                        out.append(
                            command(
                                2,
                                addr["x"]
                                + ((b * h + start + up + y) * w + x) * ch
                                + c,
                                tile(ACC + y * w + x, min(16, w - x), nc),
                            )
                        )
                out.append(
                    st(ch, pool=(stride, k, ow, 1, ow, rows, w, up, pad))
                )
                out.append(
                    command(
                        3,
                        addr["y"] + ((b * oh + r) * ow) * ch + c,
                        tile(ACC, 0, nc),
                    )
                )
    return out


def expand(kind, params, addresses):
    if kind == "matmul":
        return matmul(params, addresses)
    if kind == "conv2d":
        return conv(params, addresses)
    if kind in ("add", "relu", "requantize", "average"):
        return pointwise(kind, params, addresses)
    if kind == "maxpool":
        return maxpool(params, addresses)
    raise NotImplementedError(f"Unsupported Lean operation: {kind}")


def explicit_matmul(p, addr):
    """Selected SRAM tile, INT32 accumulation, one scaled store after reduction."""
    m, n, k = (p[key] for key in ("m", "n", "k"))
    tm, tn, tk = p["tile"]
    a_rows = tm * ceildiv(tk, 16)
    if (
        a_rows + tk * ceildiv(tn, 16) > SPAD_ROWS
        or tm * ceildiv(tn, 16) > ACC_ROWS
    ):
        raise ValueError("Explicit matmul tile exceeds physical memory")
    commands = [
        ex(),
        st(n, p.get("scale", 1), p.get("relu", False)),
        ld(k, block_stride=tm),
        ld(n, 1, block_stride=tk),
        ld(0, 2, block_stride=tm),
    ]
    for row in range(0, m, tm):
        for col in range(0, n, tn):
            mr, nr = min(tm, m - row), min(tn, n - col)
            for c in range(0, nr, 16):
                for r in range(0, mr, 16):
                    commands.append(
                        command(
                            14,
                            addr["bias"] + (col + c) * 4,
                            tile(
                                ACC + (c // 16) * tm + r,
                                min(16, mr - r),
                                min(16, nr - c),
                            ),
                        )
                    )
            for depth in range(0, k, tk):
                kr = min(tk, k - depth)
                for d in range(0, kr, 16):
                    for r in range(0, mr, 16):
                        commands.append(
                            command(
                                2,
                                addr["a"] + (row + r) * k + depth + d,
                                tile(
                                    (d // 16) * tm + r,
                                    min(16, mr - r),
                                    min(16, kr - d),
                                ),
                            )
                        )
                for c in range(0, nr, 16):
                    for d in range(0, kr, 16):
                        commands.append(
                            command(
                                1,
                                addr["b"] + (depth + d) * n + col + c,
                                tile(
                                    a_rows + (c // 16) * tk + d,
                                    min(16, kr - d),
                                    min(16, nr - c),
                                ),
                            )
                        )
                for c in range(0, nr, 16):
                    for r in range(0, mr, 16):
                        dest = ACC | (1 << 30) | ((c // 16) * tm + r)
                        for d in range(0, kr, 16):
                            commands.extend(
                                [
                                    command(
                                        6,
                                        tile(
                                            a_rows + (c // 16) * tk + d,
                                            min(16, kr - d),
                                            min(16, nr - c),
                                        ),
                                        tile(
                                            dest,
                                            min(16, mr - r),
                                            min(16, nr - c),
                                        ),
                                    ),
                                    command(
                                        4,
                                        tile(
                                            (d // 16) * tm + r,
                                            min(16, mr - r),
                                            min(16, kr - d),
                                        ),
                                        tile(0xFFFFFFFF, 16, 16),
                                    ),
                                ]
                            )
            for c in range(0, nr, 16):
                for r in range(0, mr, 16):
                    commands.append(
                        command(
                            3,
                            addr["y"] + (row + r) * n + col + c,
                            tile(
                                ACC + (c // 16) * tm + r,
                                min(16, mr - r),
                                min(16, nr - c),
                            ),
                        )
                    )
            # Private rows cannot be reused until the output DMA completes.
            commands.append(dict(type="drain"))
    return commands
