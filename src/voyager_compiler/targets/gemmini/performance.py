"""Target-owned native service model ported from voyager-gemmini-dse.

Hardware work scales with DIM/capacity/bandwidth. Startup and service constants
are inherited Lean16 calibration, not measurements of this checkout. No layer
identity or timing lookup is used. Explicit and depthwise paths retain their
separate analytical estimates in scheduling.py.
"""

from math import ceil

from voyager_compiler.ir.plan import Cost

from .native_service import NativeService
from .semantics import tile_storage
from .traffic import matrix_records

NATIVE_SERVICE = NativeService()
MODEL_REVISION = "dse-native-service-v1"


def native_matmul(p, tile, config):
    dim = config.pe_array_size[0]
    bw = config.connection_bytes_per_cycle("dram_sram")
    m, n, k = (p[key] for key in ("m", "n", "k"))
    latency = traffic = compute = 0
    for mr, nr, kr, a, b, count in matrix_records(m, n, k, tuple(tile), dim):
        traffic += count * (a + b)
        if kr:
            cycles = ceil(mr / dim) * ceil(nr / dim) * ceil(kr / dim) * dim
            compute += count * cycles
            latency += count * (100 + max(cycles + 2 * dim, (a + b) / bw))
        else:
            latency += count * ((a + b) / bw + 32)
    tm, tn, tk = tile
    # The donor's broadcast-bias service calibration applies only to DIM16,
    # full-K, aligned wide tiles. Never extrapolate this empirical fit to DIM32.
    if (
        dim == 16
        and n % dim == 0
        and m % tm == 0
        and m > tm
        and tk == k
        and tm >= 32
        and tm % dim == 0
        and tn >= 64
    ):
        launches = ceil(m / tm) * ceil(n / tn)
        service = (2.5 * tm + tn + 64) * ceil(tk / dim) * launches
        sp, _ = tile_storage("matmul", p, tile, "hardware_loop", dim)
        bias = ceil(tm / dim) * tn * 4
        latency = (
            max(compute, traffic / bw, service)
            + 2 * (sp + bias + tm * tn) / bw
            - 5
        )
    return Cost(float(latency), int(traffic))


def native_conv(p, tile, config):
    dim = config.pe_array_size[0]
    bw = config.connection_bytes_per_cycle("dram_sram")
    r, c, o, i = tile
    kernel = p["kernel"]
    launch = (
        p["batch"]
        * ceil(p["oh"] / r)
        * ceil(p["ow"] / c)
        * ceil(p["co"] / o)
        * ceil(p["ci"] / i)
    )
    outputs = launch / ceil(p["ci"] / i)
    patch = (
        (r * p["stride"] + kernel - 1)
        * (c * p["stride"] + kernel - 1)
        * ceil(i / dim)
        * dim
    )
    weights = kernel * kernel * ceil(i / dim) * dim * ceil(o / dim) * dim
    bias = r * ceil(c / dim) * o * 4
    write = r * c * o
    compute = (
        ceil(r * c / dim)
        * ceil(o / dim)
        * ceil(i / dim)
        * kernel
        * kernel
        * dim
    )
    narrow = c < 8 and r * c < 128
    if not narrow:
        compute = r * c * ceil(o / dim) * ceil(i / dim) * kernel * kernel
    traffic = launch * (patch + weights) + outputs * (bias + write)
    latency = launch * (
        100 + max(compute + 2 * dim, (patch + weights) / bw)
    ) + outputs * ((bias + write) / bw + 32)
    if narrow:
        latency += (
            patch if kernel == 1 else launch * patch
        ) / bw + launch * weights / (32 * bw)
    else:
        latency += launch * weights / (4 * bw)
    latency = max(
        latency, NATIVE_SERVICE.convolution(dim, kernel, tile, launch)
    )
    return Cost(float(latency), int(traffic))


def maxpool(p, config):
    """Price the actual one-output-row store-unit pooling expansion.

    Serialized window service (14 cycles/output vector) and setup (40/tile)
    come from the donor. Count this emitter's rows/halos, not its old tiler.
    """
    dim = config.pe_array_size[0]
    bw = config.connection_bytes_per_cycle("dram_sram")
    blocks = ceil(p["channels"] / dim) * p["batch"]
    input_rows = sum(
        min(p["h"], row * p["stride"] - p["pad"] + p["kernel"])
        - max(0, row * p["stride"] - p["pad"])
        for row in range(p["oh"])
    )
    inputs = blocks * input_rows * p["w"]
    outputs = blocks * p["oh"] * p["ow"]
    tiles = blocks * p["oh"]
    return Cost(
        inputs * dim / bw + 14 * outputs + 40 * tiles + 32,
        (inputs + outputs) * dim,
    )
