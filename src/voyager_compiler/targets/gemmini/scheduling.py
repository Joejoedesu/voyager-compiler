"""ImplementationSpec candidate, capacity and cost callbacks.

This module proposes and prices alternatives; it never constructs FX or emits
instructions. The shared planner chooses. Estimates retain the current heuristic.
"""

from voyager_compiler.extensions import CandidateRejected, RuntimeModel
from voyager_compiler.ir.plan import BufferRequirement, Cost

from .semantics import facts, raw_dimensions


class GemminiModel(RuntimeModel):
    """Recipes provide whole-operation estimates; no shared tile-model fallback."""

    def prepare(self, node):
        return None


def ceildiv(a, b):
    return (a + b - 1) // b


def demands(kind, p, tile, mode):
    if kind == "matmul":
        m, n, k = tile
        return (
            16 * (m * ceildiv(k, 16) + k * ceildiv(n, 16)),
            64 * m * ceildiv(n, 16),
        )
    if kind == "conv2d":
        r, c, o, i = tile
        kernel, stride = p["kernel"], p["stride"]
        return (
            16
            * (
                (r * stride + kernel - 1)
                * (c * stride + kernel - 1)
                * ceildiv(i, 16)
                + kernel * kernel * i * ceildiv(o, 16)
            ),
            64 * r * c * ceildiv(o, 16),
        )
    if kind == "add":
        return (0, 64 * tile[0] * ceildiv(tile[1], 16))
    if kind == "maxpool":
        return (0, 64 * p["w"] * p["kernel"])
    return (0, 64 * (1 if kind == "average" else 16))


def footprint(kind, p, tile, mode, config):
    sp, acc = demands(kind, p, tile, mode)
    # Hardware macros have two private contexts. Explicit code uses one arena.
    slots = 2 if mode == "hardware_loop" and kind in ("matmul", "conv2d") else 1
    if (
        sp * slots > config.memory_instance("scratchpad").size.value
        or acc * slots > config.memory_instance("accumulator").size.value
    ):
        raise CandidateRejected(
            "Selected Gemmini tile exceeds physical private capacity"
        )
    return tuple(
        BufferRequirement(key, key, size, slots=slots)
        for key, size in [("scratchpad", sp), ("accumulator", acc)]
        if size
    )


def cost(kind, p, tile, mode):
    if kind == "matmul":
        m, n, k = p["m"], p["n"], p["k"]
        tm, tn, tk = tile
        traffic = (
            ceildiv(n, tn) * m * k
            + ceildiv(m, tm) * k * n
            + m * n
            + 4 * ceildiv(m, tm) * n
        )
        compute = ceildiv(m, 16) * ceildiv(n, 16) * ceildiv(k, 16) * 16
        launches = ceildiv(m, tm) * ceildiv(n, tn) * ceildiv(k, tk)
        instructions = (
            6 * launches + 5
            if mode == "hardware_loop"
            else 2 * compute / 16
            + traffic / 256
            + ceildiv(m, tm) * ceildiv(n, tn)
        )
        latency = compute + traffic / 16 + instructions + launches * 32
    elif kind == "conv2d":
        r, c, o, i = tile
        sp, _ = demands(kind, p, tile, mode)
        launches = (
            p["batch"]
            * ceildiv(p["oh"], r)
            * ceildiv(p["ow"], c)
            * ceildiv(p["co"], o)
            * ceildiv(p["ci"], i)
        )
        traffic = (
            launches * sp
            + p["batch"] * p["oh"] * p["ow"] * p["co"]
            + p["co"] * 4
        )
        latency = traffic / 16 + launches * (
            r
            * ceildiv(c, 16)
            * ceildiv(o, 16)
            * p["kernel"] ** 2
            * ceildiv(i, 16)
            * 16
            + 60
        )
    elif kind == "average":
        traffic = p["batch"] * (p["pixels"] + 1) * p["channels"]
        latency = traffic / 16 + p["batch"] * p["pixels"] * ceildiv(
            p["channels"], 16
        )
    elif kind == "maxpool":
        traffic = (
            p["batch"]
            * p["oh"]
            * (p["kernel"] * p["w"] + p["ow"])
            * p["channels"]
        )
        latency = traffic / 16 + p["batch"] * p["oh"] * ceildiv(
            p["channels"], 16
        ) * (p["kernel"] * ceildiv(p["w"], 16) + 2)
    else:
        traffic = p["rows"] * p["channels"] * (3 if kind == "add" else 2)
        latency = (
            traffic / 16
            + ceildiv(p["rows"], 16) * ceildiv(p["channels"], 16) * 3
        )
        if kind == "add":
            latency += (
                3
                * ceildiv(p["rows"], tile[0])
                * ceildiv(p["channels"], tile[1])
                * 32
            )
    return Cost(float(latency), int(traffic))


def candidates(node, context, mode):
    kind, p, _ = facts(node)
    config = context.tiler.config
    tiles = []
    if kind == "matmul":
        m, n, k = p["m"], p["n"], p["k"]
        # All output block counts fitting the pinned accumulator; compare both
        # full and shorter reduction tiles. The ranked shortlist is bounded.
        for j in range(1, min(ceildiv(n, 16), 64) + 1):
            for i in range(1, min(ceildiv(m, 16), 64 // j) + 1):
                cap = config.memory_instance("scratchpad").size.value // (
                    2 if mode == "hardware_loop" else 1
                )
                z = min(ceildiv(k, 16), cap // (256 * (i + j)))
                for depth in sorted({1, z, max(1, z // 2)}):
                    if z:
                        tiles.append((i * 16, j * 16, depth * 16))
    elif kind == "conv2d":
        for i in sorted({min(p["ci"], v) for v in (16, 32, 64, 128, p["ci"])}):
            for o in sorted({min(p["co"], v) for v in (16, 32, 64, 128)}):
                for c in sorted({min(p["ow"], v) for v in (4, 8, 16, p["ow"])}):
                    for r in range(
                        1, min(p["oh"], 512 // (c * ceildiv(o, 16))) + 1
                    ):
                        tiles.append((r, c, o, i))
    elif kind == "add":
        for n in sorted(
            {min(ceildiv(p["channels"], 16) * 16, v) for v in (16, 64, 128)}
        ):
            rows = min(p["rows"], 512 // ceildiv(n, 16))
            tiles.append((rows, n))
    else:
        tiles = [()]
    legal = []
    for tile in set(tiles):
        try:
            footprint(kind, p, tile, mode, config)
        except CandidateRejected:
            continue
        legal.append((cost(kind, p, tile, mode).latency, tile))
    if not legal:
        raise CandidateRejected("No legal Gemmini tile")
    # At most 24 per recipe; the shared planner's budget remains 64.
    for _, tile in sorted(legal)[:24]:
        yield dict(tile=tile, mode=mode)


def raw_requirements(node, context, native):
    m, n, k = raw_dimensions(node)
    demands = (
        BufferRequirement("operands", "scratchpad", m * k + k * n),
        BufferRequirement("accumulation", "accumulator", m * n * 4),
    )
    divisor = 2 if native else 1  # hardware-loop ping/pong contexts
    for demand in demands:
        if (
            demand.bytes
            > context.tiler.config.memory_instance(demand.space).size.value
            // divisor
        ):
            raise CandidateRejected(
                f"{demand.space} does not fit {'private half' if native else 'compiler tile'}"
            )
    return demands


def estimate_raw(node, graph, context, parameters):
    """Analytical command/array/DMA work, not device latency evidence."""
    m, n, k = raw_dimensions(node)
    native = any(
        n.meta.get("target_invoke")
        and n.meta["target_invoke"].allocation_owner == "native"
        for n in graph.graph.nodes
    )
    traffic = m * k + k * n + 4 * m * n
    microtiles = (m // 16) * (n // 16) * (k // 16)
    commands = (
        10
        if native
        else 2 * (m * k + k * n) // 256 + 2 * microtiles + (m * n) // 256 + 2
    )
    return Cost(microtiles * 16 + traffic / 16 + commands, traffic)


def estimate_deployment(node, graph, context, parameters):
    kind, p, _ = facts(node)
    return cost(kind, p, parameters["tile"], parameters["mode"])
