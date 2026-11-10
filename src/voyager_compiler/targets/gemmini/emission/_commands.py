"""Project selected Gemmini tiles into commands; no tile selection in emission.

Operation expansions are adapted from the voyager-gemmini donor scheduler.
Residual phase drains preserve its validated conservative load/add/store order.
"""

from ..hardware import hardware
from ..semantics import tile_storage
from ._encoding import ACC, command, ex, ld, loop_conv, loop_mm, st, tile


def ceildiv(a, b):
    return (a + b - 1) // b


def matmul(p, addr, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    m, k, n = p["m"], p["k"], p["n"]
    tm, tn, tk = p["tile"]
    sp, acc = tile_storage("matmul", p, p["tile"], "hardware_loop", dim)
    if (
        sp > config.memory_instance("scratchpad").size.value // 2
        or acc > config.memory_instance("accumulator").size.value // 2
    ):
        raise ValueError("Matmul tile exceeds double-buffered Lean capacity")
    a, b, c = addr["a"], addr["b"], addr["y"]
    d = addr.get("bias", 0)
    out = [
        ex(),
        st(n, p.get("scale", 1), p.get("relu", False)),
        ld(k, block_stride=dim),
        ld(n, 1, block_stride=dim),
        ld(0, 2, block_stride=dim),
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
                    dim=dim,
                )
    return out


def conv(p, addr, *, depthwise=False, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    batch, h, w, ci, co, oh, ow, k = (
        p[x] for x in ("batch", "h", "w", "ci", "co", "oh", "ow", "kernel")
    )
    stride, pad = p.get("stride", 1), p.get("pad", 0)
    tr, tc, to, ti = (*p["tile"], 1, 1) if depthwise else p["tile"]
    sp, acc = tile_storage(
        "depthwise_conv2d" if depthwise else "conv2d",
        p,
        p["tile"],
        "hardware_loop",
        dim,
    )
    if (
        sp > config.memory_instance("scratchpad").size.value // 2
        or acc > config.memory_instance("accumulator").size.value // 2
    ):
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
                    for ic in ((oc,) if depthwise else range(0, ci, ti)):
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
                            weight=addr["weight"]
                            + (oc * k * k if depthwise else ic * co + oc),
                            bias=bias + oc * 4 if depthwise or ic == 0 else 0,
                            y=(
                                addr["y"] + ((b * oh + r) * ow + c) * co + oc
                                if depthwise or ic + ti >= ci
                                else 0
                            ),
                            no_bias=False,
                            depthwise=depthwise,
                            relu=p.get("relu", False),
                        )
    return out


def pointwise(kind, p, addr, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    channels = p["channels"]
    out = [
        ex(),
        st(channels, p.get("scale", 1), kind == "relu" or p.get("relu", False)),
    ]
    if kind == "average":
        out += [
            ld(channels, shrunk=True, block_stride=dim),
            st(channels, p.get("scale", 1) / p["pixels"]),
        ]
        for b in range(p.get("batch", 1)):
            for c in range(0, channels, dim):
                nc = min(dim, channels - c)
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
    out += [
        ld(channels, shrunk=True, scale=p.get("a_scale", 1), block_stride=dim)
    ]
    if kind == "add":
        out += [
            ld(
                channels,
                index=1,
                shrunk=True,
                scale=p.get("b_scale", 1),
                block_stride=dim,
            )
        ]
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
                    block = dim if key == "y" else 64
                    for cc in range(0, nc, block):
                        for rr in range(0, nr, dim):
                            local = flags + (cc // dim) * tm + rr
                            offset = (r + rr) * channels + c + cc
                            out.append(
                                command(
                                    funct,
                                    addr[key] + offset,
                                    tile(
                                        local,
                                        min(dim, nr - rr),
                                        min(block, nc - cc),
                                    ),
                                )
                            )
                    out.append(dict(type="drain"))
        return out
    for r in range(0, p["rows"], dim):
        for c in range(0, channels, dim):
            nr, nc = min(dim, p["rows"] - r), min(dim, channels - c)
            offset = r * channels + c
            out.append(command(2, addr["x"] + offset, tile(ACC, nr, nc)))
            out.append(command(3, addr["y"] + offset, tile(ACC, nr, nc)))
    return out


def maxpool(p, addr, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    # Tile one output row, preserving the full input-row stride and true halo.
    h, w, ch, oh, ow, k, stride, pad = (
        p[x]
        for x in ("h", "w", "channels", "oh", "ow", "kernel", "stride", "pad")
    )
    if pad and not p.get("input_nonnegative", False):
        raise NotImplementedError(
            "Gemmini maxpool pads with zero: padded pooling requires a proven nonnegative input (e.g. ReLU)"
        )
    out = [ex(), ld(ch, shrunk=True, block_stride=dim)]
    for b in range(p.get("batch", 1)):
        for r in range(oh):
            start = r * stride - pad
            up, down = max(0, -start), max(0, start + k - h)
            rows = k - up - down
            for c in range(0, ch, dim):
                nc = min(dim, ch - c)
                for y in range(rows):
                    for x in range(0, w, dim):
                        out.append(
                            command(
                                2,
                                addr["x"]
                                + ((b * h + start + up + y) * w + x) * ch
                                + c,
                                tile(ACC + y * w + x, min(dim, w - x), nc),
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


def expand(kind, params, addresses, config=None):
    if kind == "matmul":
        return matmul(params, addresses, config)
    if kind in ("conv2d", "depthwise_conv2d"):
        return conv(
            params,
            addresses,
            depthwise=kind == "depthwise_conv2d",
            config=config,
        )
    if kind in ("add", "relu", "requantize", "average"):
        return pointwise(kind, params, addresses, config)
    if kind == "maxpool":
        return maxpool(params, addresses, config)
    raise NotImplementedError(f"Unsupported Lean operation: {kind}")


def explicit_matmul(p, addr, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    """Selected SRAM tile, INT32 accumulation, one scaled store after reduction."""
    m, n, k = (p[key] for key in ("m", "n", "k"))
    tm, tn, tk = p["tile"]
    a_rows = tm * ceildiv(tk, dim)
    sp, acc = tile_storage("matmul", p, p["tile"], "explicit", dim)
    if (
        sp > config.memory_instance("scratchpad").size.value
        or acc > config.memory_instance("accumulator").size.value
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
            for c in range(0, nr, dim):
                for r in range(0, mr, dim):
                    commands.append(
                        command(
                            14,
                            addr["bias"] + (col + c) * 4,
                            tile(
                                ACC + (c // dim) * tm + r,
                                min(dim, mr - r),
                                min(dim, nr - c),
                            ),
                        )
                    )
            for depth in range(0, k, tk):
                kr = min(tk, k - depth)
                for d in range(0, kr, dim):
                    for r in range(0, mr, dim):
                        commands.append(
                            command(
                                2,
                                addr["a"] + (row + r) * k + depth + d,
                                tile(
                                    (d // dim) * tm + r,
                                    min(dim, mr - r),
                                    min(dim, kr - d),
                                ),
                            )
                        )
                for c in range(0, nr, dim):
                    for d in range(0, kr, dim):
                        commands.append(
                            command(
                                1,
                                addr["b"] + (depth + d) * n + col + c,
                                tile(
                                    a_rows + (c // dim) * tk + d,
                                    min(dim, kr - d),
                                    min(dim, nr - c),
                                ),
                            )
                        )
                for c in range(0, nr, dim):
                    for r in range(0, mr, dim):
                        dest = ACC | (1 << 30) | ((c // dim) * tm + r)
                        for d in range(0, kr, dim):
                            commands.extend(
                                [
                                    command(
                                        6,
                                        tile(
                                            a_rows + (c // dim) * tk + d,
                                            min(dim, kr - d),
                                            min(dim, nr - c),
                                        ),
                                        tile(
                                            dest,
                                            min(dim, mr - r),
                                            min(dim, nr - c),
                                        ),
                                    ),
                                    command(
                                        4,
                                        tile(
                                            (d // dim) * tm + r,
                                            min(dim, mr - r),
                                            min(dim, kr - d),
                                        ),
                                        tile(0xFFFFFFFF, dim, dim),
                                    ),
                                ]
                            )
            for c in range(0, nr, dim):
                for r in range(0, mr, dim):
                    commands.append(
                        command(
                            3,
                            addr["y"] + (row + r) * n + col + c,
                            tile(
                                ACC + (c // dim) * tm + r,
                                min(dim, mr - r),
                                min(dim, nr - c),
                            ),
                        )
                    )
            # Private rows cannot be reused until the output DMA completes.
            commands.append(dict(type="drain"))
    return commands
