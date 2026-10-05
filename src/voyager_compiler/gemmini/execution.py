"""Compact WS traversal facts shared by estimation and explicit ISA lowering."""

import math
from dataclasses import dataclass
from interstellar import loop_enum as le


@dataclass(frozen=True)
class ArrayWork:
    microtiles: int
    fresh_preloads: int
    retained_weights: int
    row_cycles: float


def instruction_rows(rows, fresh_preload, dim, minimum_rows):
    """ExecuteController.total_rows: fresh D preload always feeds a full block."""
    return dim if fresh_preload else max(minimum_rows, rows)


def retains_weight(previous, current):
    return previous == current


def array_work(mapping, batch, dim, minimum_rows):
    """Summarize the same L1/L2 order serialized by _build_tiling.

    Weight coordinates depend on IC/OC/FX/FY. Consecutive inner spatial
    iterations retain the same weight address. The converter visits batch
    images inside each convolution coordinate, so batch also extends a run.
    """
    extent = lambda d: math.prod(mapping.loop_blockings[d][:3]) * math.prod(
        mapping.loop_partitionings[d][:3]
    )
    x, y, n, k, fx, fy = (
        extent(d) for d in (le.OX, le.OY, le.OC, le.IC, le.FX, le.FY)
    )
    groups = math.ceil(x / dim)
    microtiles = (
        batch * y * groups * math.ceil(n / dim) * math.ceil(k / dim) * fx * fy
    )
    run_x = run_y = 1
    stopped = False
    for level in (1, 2):
        for d in sorted(
            range(le.NUM), key=lambda d: mapping.loop_orders[d][level]
        ):
            bound = mapping.loop_blockings[d][level]
            if bound <= 1:
                continue
            if d in (le.IC, le.OC, le.FX, le.FY):
                stopped = True
                break
            if d == le.OX:
                run_x *= bound
            if d in (le.OY, le.ON):
                run_y *= bound
        if stopped:
            break
    reuse_run = batch * math.ceil(run_x / dim) * run_y
    fresh = math.ceil(microtiles / reuse_run)
    # The final short X group executes shortened only with retained weights.
    average_rows = (
        (x // dim) * dim + (max(minimum_rows, x % dim) if x % dim else 0)
    ) / groups
    row_cycles = microtiles * average_rows + fresh * (dim - average_rows)
    return ArrayWork(microtiles, fresh, microtiles - fresh, row_cycles)


from copy import copy
from voyager_compiler.codegen.transform.tiling.cost import _step_classes
from voyager_compiler.codegen.transform.tiling.transfers import (
    dma_service,
    dma_sweep_cycles,
    concurrent_dma_cycles,
)
from .hardware import lean_config
from .scheduling import QueueGeometry
from .constraints import matrix_storage
from voyager_compiler.codegen.transform.tiling.contracts import (
    CandidateEvaluation,
    resources_fit,
)


class GemminiCostModel:
    def __init__(self, shared, batch=1, config=None, tuning=None):
        from .mapping import GemminiTuning

        self.tuning = tuning or GemminiTuning()
        self.shared, self.batch = shared, batch
        self.config = config or lean_config()
        self.buffer_slots = {}

    def evaluate(self, architecture, layer, mapping):
        """Return the winning buffering decision; no hidden last-plan state."""
        from voyager_compiler.codegen.transform.bufferize.plan import (
            batch_candidates,
        )

        return min(
            (
                self._runtime(layer, mapping, batch)
                for batch in batch_candidates(self.batch)
            ),
            key=lambda candidate: candidate.cycles,
        )

    def calculate_runtime(self, architecture, layer, mapping):
        # Compatibility for reports/older callers. Search consumes evaluate().
        result = self.evaluate(architecture, layer, mapping)
        self.plan = result.buffer_plan
        self.breakdown = dict(result.diagnostics)
        self.dram_bytes = self.breakdown.get("dram_bytes", {})
        self.buffer_slots = dict(
            input=self.plan.input_slots, weight=self.plan.weight_slots
        )
        return result.cycles

    def _runtime(self, layer, mapping, tile_batch):
        from .mapping import make_plan

        rc, hw = copy(self.shared), self.config
        rc.batch *= self.batch // tile_batch
        extent = lambda d: rc._extent(mapping, d, 2)
        x, y, n, k = (extent(d) for d in (le.OX, le.OY, le.OC, le.IC))
        fx, fy = extent(le.FX), extent(le.FY)
        dim = hw.pe_array_size[0]
        timing = hw.compute_unit("matrix").modes[0].timing
        work = array_work(mapping, tile_batch, dim, timing.occupancy_cycles)
        blocks = rc._l3_blocks(mapping) * rc.batch
        reductions = mapping.loop_blockings[le.IC][3]
        input_steps = rc._batch_loads(
            mapping, (le.IC, le.OX, le.OY, le.ON), rc.batch
        )
        weight_steps = rc._batch_loads(
            mapping, (le.IC, le.OC, le.FX, le.FY), rc.weight_batch
        )
        output_steps = blocks // reductions
        inp = (
            tile_batch
            * ((y - 1) * layer.hstd + fy)
            * ((x - 1) * layer.wstd + fx)
            * k
        )
        weight, out = fx * fy * k * n, tile_batch * y * x * n
        dram_bytes = dict(
            input=inp * input_steps,
            weight=weight * weight_steps,
            output=out * output_steps,
        )
        plan = make_plan(
            mapping, tile_batch, batch_tiles=self.batch // tile_batch
        )
        storage = matrix_storage(hw, inp, weight, out, plan, self.tuning)
        if not resources_fit(hw, storage):
            return CandidateEvaluation(
                math.inf, plan, storage, (("dram_bytes", dram_bytes),)
            )
        # Command issue and array rows are concurrent resources. A fresh
        # preload can pipeline with the preceding multiply; adding its issue
        # time to every microtile falsely rewards retaining full-row weights.
        compute = (
            max(
                work.row_cycles,
                2 * work.microtiles * timing.issue_interval_cycles,
            )
            + timing.startup_cycles
        )
        load_geometry = hw.connection("dram_dma").transfer_geometry
        store_geometry = hw.connection("acc_to_dram").transfer_geometry
        input_commands = load_geometry.command_count(inp // k, k)
        weight_commands = load_geometry.command_count(weight // n, n)
        store_commands = store_geometry.command_count(out // n, n)
        input_dma = dma_service(hw, "dram_dma", inp, input_commands)
        weight_dma = dma_service(hw, "dram_dma", weight, weight_commands)
        store_dma = dma_service(hw, "acc_to_dram", out, store_commands)
        # A bufferized async-compute invocation is a software tile. Its
        # pipeline startup must be amortized by that tile, not charged once
        # for the whole layer. This is a conservative task-boundary estimate;
        # a converter may recover part of it by pipelining adjacent tasks.
        runtime = dma_sweep_cycles(
            [(input_dma, input_steps), (weight_dma, weight_steps)],
            store_dma,
            output_steps,
            blocks,
            compute,
        )
        queues = QueueGeometry.from_hardware(hw)
        exposed = 0
        for mask, count in _step_classes(
            [(1, output_steps), (2, input_steps), (4, weight_steps)], blocks
        ):
            mask = int(mask)
            load_work = ([input_dma] if mask & 2 else []) + (
                [weight_dma] if mask & 4 else []
            )
            loads = sum(dma.cycles for dma in load_work)
            stores = store_dma.cycles if mask & 1 else 0
            tail = submission_tail(
                work.microtiles,
                input_commands if mask & 2 else 0,
                weight_commands if mask & 4 else 0,
                store_commands if mask & 1 else 0,
                loads,
                stores,
                work.row_cycles,
                queues,
                self.tuning.submission,
                sum(dma.payload_cycles for dma in load_work),
                store_dma.payload_cycles if mask & 1 else 0,
            )
            # The ordinary resource sweep already prices DMA throughput excess.
            dma_cycles = concurrent_dma_cycles(
                load_work, store_dma if mask & 1 else None
            )
            exposed += count * max(0, tail - max(0, dma_cycles - compute))
        runtime += exposed
        if not self.tuning.separate_accumulator_banks:
            # Packing is legal, but this policy cannot promise read-port
            # independence. Conservatively expose output service instead.
            runtime += store_dma.cycles * output_steps
        breakdown = dict(
            runtime=runtime,
            submission_tail_cycles=exposed,
            compute_cycles=blocks * compute,
            fresh_preloads=blocks * work.fresh_preloads,
            weight_reuses=blocks * work.retained_weights,
            array_microtiles=blocks * work.microtiles,
            dram_bytes=dram_bytes,
        )
        breakdown.update(
            dma_commands=dict(
                input=input_commands * input_steps,
                weight=weight_commands * weight_steps,
                output=store_commands * output_steps,
            ),
            dma_payload_cycles=input_dma.payload_cycles * input_steps
            + weight_dma.payload_cycles * weight_steps
            + store_dma.payload_cycles * output_steps,
            dma_command_cycles=input_dma.command_cycles * input_steps
            + weight_dma.command_cycles * weight_steps
            + store_dma.command_cycles * output_steps,
            store_endpoint_cycles=store_dma.endpoint_cycles * output_steps,
        )
        return CandidateEvaluation(
            runtime, plan, storage, tuple(breakdown.items())
        )

    def calculate_memory_cost(self, architecture, layer, mapping):
        result = self.evaluate(architecture, layer, mapping)
        return sum(dict(result.diagnostics)["dram_bytes"].values())


def submission_tail(
    microtiles,
    input_commands,
    weight_commands,
    store_commands,
    load_cycles,
    store_cycles,
    row_cycles,
    queues,
    policy,
    load_payload_cycles=None,
    store_payload_cycles=None,
):
    """Exposed DMA after the converter exhausts a compute task's submission.

    Mirror AsyncSchedule's bounded bursts: one ready store burst precedes
    execute, then load/store bursts alternate between execute chunks. A load
    which completes early gives its remaining opportunities to stores. Only
    admitted execute work is credited past that boundary, including reservation
    entries. Operand readiness is already priced by the DMA sweep. Counting
    only controller entries discards useful work waiting in the reservation
    station. This is a service estimate, not a per-tile queue-size penalty.
    """
    loads = input_commands + weight_commands
    stores = store_commands
    original_loads, original_stores = loads, stores
    stores -= min(stores, policy.store_quantum)
    # No transfer is injected after the final execute chunk.
    opportunities = max(
        0, math.ceil(2 * microtiles / policy.execute_quantum) - 1
    )
    load_chunks = math.ceil(loads / policy.load_quantum)
    store_chunks = math.ceil(stores / policy.store_quantum)
    used = min(opportunities, load_chunks + store_chunks)
    issued_loads = min(load_chunks, used - min(store_chunks, used // 2))
    issued_stores = min(store_chunks, used - issued_loads)
    loads = max(0, loads - issued_loads * policy.load_quantum)
    stores = max(0, stores - issued_stores * policy.store_quantum)
    load_fraction = loads / original_loads if original_loads else 0
    store_fraction = stores / original_stores if original_stores else 0
    if load_payload_cycles is None or store_payload_cycles is None:
        remaining = load_cycles * load_fraction + store_cycles * store_fraction
    else:
        remaining = max(
            load_cycles * load_fraction,
            store_cycles * store_fraction,
            load_payload_cycles * load_fraction
            + store_payload_cycles * store_fraction,
        )
    admitted_work = (
        (queues.controller_execute + queues.reservation_execute)
        / 2
        * row_cycles
        / max(1, microtiles)
    )
    return max(0, remaining - admitted_work)
