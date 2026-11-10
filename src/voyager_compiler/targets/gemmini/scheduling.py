"""ImplementationSpec candidate, capacity and cost callbacks.

This module proposes and prices alternatives; it never constructs FX or emits
instructions. The shared planner chooses. Native service calibration is target-owned and scoped; hardware geometry propagates from the resolved configuration.
"""

from dataclasses import dataclass

import interstellar
from voyager_compiler.compiler.search.interstellar import (
    memory_level,
    search_matrix,
)
from voyager_compiler.extensions import CandidateRejected, RuntimeModel
from voyager_compiler.ir.plan import (
    BufferRequirement,
    Cost,
    MatrixSpace,
    Placement,
)

from .hardware import hardware
from .performance import MODEL_REVISION, maxpool, native_conv, native_matmul
from .semantics import facts, raw_dimensions, selected_tile, tile_storage


class GemminiModel(RuntimeModel):
    """Recipes provide whole-operation estimates; no shared tile-model fallback."""

    def prepare(self, node):
        return None


def ceildiv(a, b):
    return (a + b - 1) // b


def demands(kind, p, tile, mode, dim=16):
    if kind in ("matmul", "conv2d", "depthwise_conv2d"):
        return tile_storage(kind, p, tile, mode, dim)
    if kind == "add":
        return (0, 4 * dim * tile[0] * ceildiv(tile[1], dim))
    if kind == "maxpool":
        return (0, 4 * dim * p["w"] * p["kernel"])
    return (0, 4 * dim * (1 if kind == "average" else dim))


def footprint(kind, p, tile, mode, config):
    sp, acc = demands(kind, p, tile, mode, config.pe_array_size[0])
    # Hardware macros have two private contexts. Explicit code uses one arena.
    slots = (
        2
        if mode == "hardware_loop"
        and kind in ("matmul", "conv2d", "depthwise_conv2d")
        else 1
    )
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


def cost(kind, p, tile, mode, config=None):
    config = config or hardware()
    dim = config.pe_array_size[0]
    bw = config.connection_bytes_per_cycle("dram_sram")
    if kind == "matmul" and mode == "hardware_loop":
        return native_matmul(p, tile, config)
    if kind == "conv2d":
        return native_conv(p, tile, config)
    if kind == "maxpool":
        return maxpool(p, config)
    if kind == "matmul":
        m, n, k = p["m"], p["n"], p["k"]
        tm, tn, tk = tile
        traffic = (
            ceildiv(n, tn) * m * k
            + ceildiv(m, tm) * k * n
            + m * n
            + 4 * ceildiv(m, tm) * n
        )
        compute = ceildiv(m, dim) * ceildiv(n, dim) * ceildiv(k, dim) * dim
        launches = ceildiv(m, tm) * ceildiv(n, tn) * ceildiv(k, tk)
        instructions = (
            6 * launches + 5
            if mode == "hardware_loop"
            else 2 * compute / dim
            + traffic / (dim * dim)
            + ceildiv(m, tm) * ceildiv(n, tn)
        )
        latency = compute + traffic / bw + instructions + launches * 32
    elif kind == "depthwise_conv2d":
        r, c = tile
        launches = (
            p["batch"] * p["ci"] * ceildiv(p["oh"], r) * ceildiv(p["ow"], c)
        )
        traffic = launches * (
            (r * p["stride"] + p["kernel"] - 1)
            * (c * p["stride"] + p["kernel"] - 1)
            + p["kernel"] ** 2
            + 4
        )
        traffic += p["batch"] * p["oh"] * p["ow"] * p["ci"]
        latency = traffic / bw + launches * (
            r * ceildiv(c, dim) * p["kernel"] ** 2 * dim + 60
        )
    elif kind == "average":
        traffic = p["batch"] * (p["pixels"] + 1) * p["channels"]
        latency = traffic / bw + p["batch"] * p["pixels"] * ceildiv(
            p["channels"], dim
        )
    else:
        traffic = p["rows"] * p["channels"] * (3 if kind == "add" else 2)
        latency = (
            traffic / bw
            + ceildiv(p["rows"], dim) * ceildiv(p["channels"], dim) * 3
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
    dim = config.pe_array_size[0]
    tiles = []
    if kind in ("matmul", "conv2d"):
        # One recipe trial runs an independent, exhaustive factor search. The
        # planner's recipe budget does not truncate the matrix search space.
        yield dict(mode=mode)
        return
    if kind == "depthwise_conv2d":
        from interstellar.mapping_point_generator import factors

        def extents(n):
            return sorted({1, n, *factors(n), *range(dim, n, dim)})

        tiles = [(r, c) for r in extents(p["oh"]) for c in extents(p["ow"])]
    elif kind == "add":
        for n in sorted(
            {
                min(ceildiv(p["channels"], dim) * dim, v)
                for v in (dim, 4 * dim, 8 * dim)
            }
        ):
            rows = min(
                p["rows"],
                (config.memory_instance("accumulator").size.value // (8 * dim))
                // ceildiv(n, dim),
            )
            if rows:
                tiles.append((rows, n))
    else:
        tiles = [()]
    legal = []
    for tile in set(tiles):
        try:
            footprint(kind, p, tile, mode, config)
        except CandidateRejected:
            continue
        legal.append((cost(kind, p, tile, mode, config).latency, tile))
    if not legal:
        raise CandidateRejected("No legal Gemmini tile")
    # At most 24 per recipe; the shared planner's budget remains 64.
    for _, tile in sorted(legal)[:24]:
        yield dict(tile=tile, mode=mode)


def raw_requirements(node, context, native):
    m, n, k = raw_dimensions(node, context.tiler.config.pe_array_size[0])
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
    m, n, k = raw_dimensions(node, context.tiler.config.pe_array_size[0])
    native = any(
        n.meta.get("target_invoke")
        and n.meta["target_invoke"].allocation_owner == "native"
        for n in graph.graph.nodes
    )
    dim = context.tiler.config.pe_array_size[0]
    bw = context.tiler.config.connection_bytes_per_cycle("dram_sram")
    traffic = m * k + k * n + 4 * m * n
    microtiles = (m // dim) * (n // dim) * (k // dim)
    commands = (
        10
        if native
        else 2 * (m * k + k * n) // (dim * dim)
        + 2 * microtiles
        + (m * n) // (dim * dim)
        + 2
    )
    return Cost(microtiles * dim + traffic / bw + commands, traffic)


def estimate_deployment(node, graph, context, parameters):
    kind, p, _ = facts(node)
    result = node.meta.get("tile_result")
    return (
        result.cost
        if result is not None
        else cost(
            kind,
            p,
            parameters["tile"],
            parameters["mode"],
            context.tiler.config,
        )
    )


@dataclass(frozen=True)
class NativePlacement(Placement):
    requirements: tuple = ()


class NativeMatrixModel:
    """Interstellar factors native tiles; this adapter owns legality and price.

    Levels are array, private on-chip arenas, DRAM. Outer order is fixed to
    the command expansion, so the model cannot claim unimplemented retention.
    Estimates combine resource work with inherited Lean16 service calibration; they are not new hardware measurements.
    """

    def __init__(self, kind, params, mode, config):
        self.kind, self.params, self.mode, self.config = (
            kind,
            dict(params),
            mode,
            config,
        )
        self.dim = dim = config.pe_array_size[0]
        le = interstellar.le
        from voyager_compiler.artifacts import hardware_digest

        self.key = (
            kind,
            tuple(sorted(params.items())),
            mode,
            hardware_digest(config),
            MODEL_REVISION,
        )
        self.layer = interstellar.Layer(
            nifm=ceildiv(params["k" if kind == "matmul" else "ci"], dim) * dim,
            nofm=ceildiv(params["n" if kind == "matmul" else "co"], dim) * dim,
            wofm=(
                ceildiv(params["m"], dim) * dim
                if kind == "matmul"
                else params["ow"]
            ),
            hofm=1 if kind == "matmul" else params["oh"],
            wfil=1 if kind == "matmul" else params["kernel"],
            hfil=1 if kind == "matmul" else params["kernel"],
            nimg=1 if kind == "matmul" else params["batch"],
            wstd=1 if kind == "matmul" else params["stride"],
            hstd=1 if kind == "matmul" else params["stride"],
        )
        onchip, onchip_partitions = memory_level(
            config, ("scratchpad", "accumulator", "scratchpad")
        )
        dram, dram_partitions = memory_level(config, ("dram",) * 3)
        resource = interstellar.Resource(
            buf_capacity_list=[[1] * 3, onchip, dram],
            buf_access_cost_list=[[0] * 3, [0] * 2, [1]],
            buf_unit_static_cost_list=[[0] * 3, [0] * 2, [0]],
            para_count_list=[dim * dim, 1, 1],
            memory_partitions=[[0, 1, 2], onchip_partitions, dram_partitions],
            invalid_underutilized=False,
        )
        # fit() checks actual byte capacities including padded rows and private
        # ping/pong contexts. Generic element capacities would double-count or
        # mis-size INT32 accumulation. DRAM allocation is checked by the planner.
        hints = {}
        for loop in range(le.NUM):
            spatial = dim if loop in (le.IC, le.OC) else 1
            middle = (
                self.layer.sizes[loop]
                if loop in (le.FX, le.FY)
                else 1 if loop == le.ON else None
            )
            hints[loop] = [
                [None, None if spatial > 1 else 1, spatial],
                [None, middle, 1],
                [None, None, 1],
            ]
        outer = (le.IC, le.OC, le.OX, le.OY, le.ON, le.FX, le.FY)
        schedule = interstellar.Schedule(
            hints, order_priority={level: outer for level in range(3)}
        )
        self.space = MatrixSpace(resource, schedule, fit_level=1)

    def fit(self, mapping):
        tile = selected_tile(self.kind, self.params, mapping)
        # Interior matrix tiles fill array blocks. The last tile may be short;
        # padding search extents keeps useful tiles available for prime M/N/K.
        if self.kind == "matmul" and any(
            size % self.dim and size != self.params[dimension]
            for size, dimension in zip(tile, ("m", "n", "k"))
        ):
            return None
        try:
            requirements = footprint(
                self.kind, self.params, tile, self.mode, self.config
            )
        except CandidateRejected:
            return None
        return NativePlacement(requirements=requirements)

    def cost(self, mapping, placement):
        return cost(
            self.kind,
            self.params,
            selected_tile(self.kind, self.params, mapping),
            self.mode,
            self.config,
        )


def search_native(kind, params, mode, context):
    model = NativeMatrixModel(kind, params, mode, context.tiler.config)
    return search_matrix(
        model,
        model.layer,
        runtime_tolerance=context.tiler.runtime_tolerance,
        name=f"gemmini.{kind}.{mode}",
    )
