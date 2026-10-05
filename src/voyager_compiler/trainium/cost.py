"""Instruction-shaped service estimates for the actual collateral converter.

Times are ns internally, returned as TensorE-clock cycles for Interstellar.
The model is conservative and uncalibrated: documented throughput/latency is
not a proof of achievable issue rate. No paper roofline constants are used.
"""

import math
from dataclasses import dataclass, replace

from interstellar import loop_enum as le
from voyager_compiler.codegen.transform.tiling.cost import _step_classes


@dataclass(frozen=True)
class Service:
    payload_ns: float = 0
    startup_ns: float = 0
    tensor_ns: float = 0
    vector_ns: float = 0
    commands: int = 0
    bytes: int = 0

    @property
    def dma_ns(self):
        return self.payload_ns + self.startup_ns


def clocks(config):
    # Documented engine-specific clocks: cycles from different engines must
    # never be summed before conversion into common time units.
    return config.frequency, (1.12 if config.name == "trainium-v2" else 0.96)


def matmul_ns(config, m, n, k, bits):
    if not (0 < m <= 512 and 0 < n <= 128 and 0 < k <= 128):
        raise ValueError("Invalid TensorE instruction shape")
    # Stationary free axis is logical N; moving free axis is logical M.
    return (4 if bits == 32 else 1) * max(min(64, n), m) / config.frequency


def copy_ns(config, partitions, free):
    return max(64, free) / clocks(config)[1]


def transpose_ns(config, partitions, free):
    # TensorE transpose + PSUM->SBUF copy, matching nl.transpose's large-tile
    # route. Tiny tile engine auto-selection remains a conservative estimate.
    return max(partitions, min(64, free)) / config.frequency, copy_ns(
        config, free, partitions
    )


def dma_service(
    config, rows, cols, bits, row_block=128, transpose=True, store=False
):
    """Match converter.copy's <=128x128 DMA subdivision, including tails.

    Each active DMA engine services eight partitions. We price the busiest
    engine, not bytes divided by full-core bandwidth for narrow transfers.
    The 1.3 us cross-engine delay is charged per instruction conservatively;
    queued overlap of that delay is unknown and reported separately.
    """
    payload = tensor = vector = 0.0
    commands = 0
    per_engine = config.dram_bandwidth / 16
    for row in range(0, rows, min(128, row_block)):
        p = min(128, row_block, rows - row)
        for col in range(0, cols, 128):
            f = min(128, cols - col)
            payload += min(8, p) * f * (bits / 8) / per_engine
            if transpose:
                t, v = (
                    transpose_ns(config, f, p)
                    if store
                    else transpose_ns(config, p, f)
                )
                tensor += t
                vector += v
            commands += 1
    return Service(
        payload,
        commands * config.dram_access_latency,
        tensor,
        vector,
        commands,
        rows * cols * bits // 8,
    )


def sweep(loads, store, output_steps, steps, compute):
    """Shared double-buffer recurrence, including first loads and last store.

    Like Gemmini's DMA sweep, this uses the builder's operand reload counts.
    The aggregate read/write DMA service is shared, not independent full-rate
    links. Within one compute task TensorE/vector stages are conservatively
    serialized; independent tasks overlap DMA using explicit buffer slots.
    """
    transfers = [(1, output_steps)] + [
        (1 << (i + 1), count) for i, (_, count) in enumerate(loads)
    ]

    def service(mask):
        return sum(
            s.dma_ns for i, (s, _) in enumerate(loads) if mask & (1 << (i + 1))
        ) + (store.dma_ns if mask & 1 else 0)

    all_loads = sum(bit for bit, _ in transfers[1:])
    prologue = service(all_loads)
    if steps == 1:
        return prologue + compute + store.dma_ns
    total = sum(
        count * max(service(int(mask)), compute)
        for mask, count in _step_classes(transfers, steps)
    )
    recurring = sum(bit for bit, count in transfers if count == steps)
    return (
        total
        - max(service(all_loads | 1), compute)
        - max(service(recurring), compute)
        + prologue
        + max(service(recurring & ~1), compute)
        + max(store.dma_ns, compute)
        + store.dma_ns
    )


class TrainiumCostModel:
    def __init__(self, shared, config, conv=False, batch_tiles=1):
        self.shared, self.config, self.conv = shared, config, conv
        self.batch_tiles = batch_tiles
        self.estimate = {}

    def calculate_runtime(self, architecture, layer, mapping):
        from .mapping import extent, make_plan

        rc, hw = self.shared, self.config
        m = extent(mapping, le.OX) * extent(mapping, le.OY)
        n, k = extent(mapping, le.OC), extent(mapping, le.IC)
        fx, fy = extent(mapping, le.FX), extent(mapping, le.FY)
        taps = fx * fy
        blocks = rc._l3_blocks(mapping) * rc.batch
        reductions = mapping.loop_blockings[le.IC][3]
        outputs = blocks // reductions
        aloads = rc._batch_loads(
            mapping, (le.IC, le.OX, le.OY, le.ON), rc.batch
        )
        bloads = rc._batch_loads(
            mapping, (le.IC, le.OC, le.FX, le.FY), rc.weight_batch
        )
        ih = (extent(mapping, le.OY) - 1) * layer.hstd + fy
        iw = (extent(mapping, le.OX) - 1) * layer.wstd + fx
        a = dma_service(
            hw,
            ih * iw if self.conv else m,
            k,
            rc.input_dtype_width,
            min(iw, 128) if self.conv else 128,
        )
        # Existing shared builder may transpose weights during DMA. The
        # canonical converter then stores [K,N] physically for that layout.
        b = dma_service(
            hw,
            taps * k if self.conv else (k if rc.weight_hbm_ck else n),
            n if self.conv or rc.weight_hbm_ck else k,
            rc.weight_dtype_width,
            min(k, 128) if self.conv else 128,
            transpose=not rc.weight_transposed,
        )
        out = dma_service(
            hw,
            m,
            n,
            rc.output_dtype_width,
            min(extent(mapping, le.OX), 128) if self.conv else 128,
            store=True,
        )
        bias = (
            dma_service(hw, 1, n, rc.bias_width, 1)
            if rc.bias_width
            else Service()
        )
        biasloads = (
            rc._batch_loads(mapping, (le.OC,), 1) if rc.bias_width else 0
        )
        loads = [(a, aloads), (b, bloads)]
        if biasloads:
            loads.append((bias, biasloads))
        tensor = vector = 0.0
        instructions = 0
        for mi in range(0, m, 512):
            mm = min(512, m - mi)
            for ni in range(0, n, 128):
                nn = min(128, n - ni)
                for ki in range(0, k, 128):
                    kk = min(128, k - ki)
                    tensor += taps * matmul_ns(
                        hw, mm, nn, kk, rc.input_dtype_width
                    )
                    instructions += taps
                    if self.conv or not rc.weight_transposed:
                        t, v = transpose_ns(hw, nn, kk)
                        tensor += taps * t
                        vector += taps * v
                    # Strided local panels must be gathered into instruction
                    # operands when a software tile exceeds one ISA tile.
                    if m > 512 or n > 128 or k > 128 or self.conv:
                        vector += taps * (
                            copy_ns(hw, kk, mm)
                            + (0 if self.conv else copy_ns(hw, nn, kk))
                        )
                vector += copy_ns(hw, nn, mm)  # final PSUM eviction
        output_pass = copy_ns(hw, min(n, 128), m * math.ceil(n / 128))
        tdma = (
            sum(s.dma_ns * count for s, count in loads) + out.dma_ns * outputs
        )
        ttensor = (
            tensor * blocks
            + sum(s.tensor_ns * count for s, count in loads)
            + out.tensor_ns * outputs
        )
        tvector = (
            vector * blocks
            + sum(s.vector_ns * count for s, count in loads)
            + out.vector_ns * outputs
            + output_pass * (blocks - outputs)  # combine partial sums
            + output_pass
            * outputs
            * (int(rc.has_tail) + int(bool(rc.bias_width)))
        )
        # Layout work also occupies TensorE/VectorE and cannot overlap matmul
        # on that same engine. Spread reload-only layout work across the sweep.
        compute = (ttensor + tvector) / blocks
        predicted = max(
            sweep(loads, out, outputs, blocks, compute),
            tdma,
            ttensor,
            tvector,
        )
        # Sensitivity scenario, not an achievable lower-bound guarantee:
        # let all command-startup costs overlap while preserving payload/stages.
        queued = max(
            sweep(
                [(replace(s, startup_ns=0), count) for s, count in loads],
                replace(out, startup_ns=0),
                outputs,
                blocks,
                compute,
            ),
            sum(s.payload_ns * count for s, count in loads)
            + out.payload_ns * outputs,
            ttensor,
            tvector,
        )
        traffic = (
            sum(s.bytes * count for s, count in loads) + out.bytes * outputs
        )
        self.plan = make_plan(mapping, self.batch_tiles, rc.batch)
        scheduled_flops = 2 * m * n * k * taps * blocks
        peak_flops_per_ns = (
            2
            * 128
            * 128
            * hw.frequency
            / (4 if rc.input_dtype_width == 32 else 1)
        )
        self.estimate = dict(
            objective="minimum predicted latency; useful throughput maximized at fixed work",
            predicted_ns=predicted,
            predicted_cycles=predicted * hw.frequency,
            scheduled_flops=scheduled_flops,
            scheduled_tflops=scheduled_flops / predicted / 1000,
            scheduled_tensor_peak_fraction=scheduled_flops
            / predicted
            / peak_flops_per_ns,
            hbm_peak_fraction=traffic / predicted / hw.dram_bandwidth,
            software_tile=dict(M=m, N=n, K=k),
            tensor_instructions=instructions * blocks,
            dma_commands=sum(s.commands * count for s, count in loads)
            + out.commands * outputs,
            dma_startup_ns=sum(s.startup_ns * count for s, count in loads)
            + out.startup_ns * outputs,
            zero_startup_sensitivity_ns=queued,
            scope="Scheduled matrix kernel; excludes graph-boundary pad/slice/permute kernels",
            dma_payload_ns=sum(s.payload_ns * count for s, count in loads)
            + out.payload_ns * outputs,
            hbm_bytes=traffic,
            service_ns=dict(DMA=tdma, TensorE=ttensor, VectorE=tvector),
            estimated_service_utilization=dict(
                DMA=tdma / predicted,
                TensorE=ttensor / predicted,
                VectorE=tvector / predicted,
            ),
            limitations="Not profiler active time. Conservative serialized per-DMA startup and on-chip task stages; NKI instruction combining, exact queue issue, bank conflicts and spills uncalibrated.",
        )
        return predicted * hw.frequency

    def calculate_memory_cost(self, architecture, layer, mapping):
        # Diagnostic only; speed_only prevents memory/energy tie-breaking.
        return self.estimate["hbm_bytes"]


def vector_candidate(config, node, tile_sizes, shapes, tiling):
    """Target capacity and latency for the shared vector/pool enumeration.

    Instruction count is a conservative primitive-based estimate. Special
    reductions and hardware-selected engine routes require calibration.
    """
    from voyager_compiler.codegen.node_info import (
        _pair,
        get_anchor_node,
        get_arg_value,
        require_allocation,
    )
    from voyager_compiler.codegen.transform.tiling.cost import get_dtype_width

    from .mapping import slot_bytes

    anchor = get_anchor_node(node)
    name = str(anchor.target)
    storage = traffic = 0
    dma = tensor = vector = scalar = 0.0
    for operand, shape in shapes.items():
        if shape is None or (
            operand is not node and not require_allocation(operand)
        ):
            continue
        if not all(isinstance(x, int) for x in shape):
            return None
        elements = math.prod(shape)
        if not shape or elements / min(128, shape[-1]) > 4096:
            return None
        if (
            "pool" in name
            and len(shape) == 4
            and (shape[0] != 1 or shape[-1] > 128)
        ):
            return None
        bits = get_dtype_width(operand.meta.get("dtype") or operand.value.dtype)
        storage += slot_bytes(elements, shape[-1], bits)
        service = dma_service(
            config,
            elements // shape[-1],
            shape[-1],
            bits,
            min(shape[-2], 128) if len(shape) > 2 else 128,
            store=operand is node,
        )
        traffic += service.bytes
        dma += service.dma_ns
        tensor += service.tensor_ns
        vector += service.vector_ns
    output = shapes[node]
    free = math.prod(output) / min(128, output[-1])
    sub = node.meta.get("submodule")
    primitives = (
        [n for n in sub.graph.nodes if n.op == "call_function"]
        if sub
        else [anchor]
    )
    for primitive in primitives:
        op = str(primitive.target)
        if any(x in op for x in ("sigmoid", "silu", "exp", "tanh")):
            scalar += max(64, free) / (config.frequency / 2)
            if "silu" in op:
                vector += copy_ns(config, 128, free)
        elif "max_pool2d" in op:
            vector += (
                math.prod(_pair(get_arg_value(primitive, 1, "kernel_size"))) - 1
            ) * copy_ns(config, 128, free)
        elif "adaptive_avg_pool" in op:
            vector += copy_ns(config, 128, math.prod(anchor.args[0].shape[1:3]))
        else:
            vector += copy_ns(config, 128, free)
    steps = math.prod(tiling)
    # Prologue/drain for a two-buffer pipeline; single-tile path is serial.
    compute = tensor + vector + scalar
    predicted = dma + compute + (steps - 1) * max(dma, compute)
    return storage, predicted * config.frequency, traffic * steps
