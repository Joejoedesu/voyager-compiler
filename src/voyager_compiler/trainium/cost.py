"""Instruction-shaped service estimates for the actual collateral converter.

Times are ns internally, returned as TensorE-clock cycles for Interstellar.
The model is analytical and uncalibrated: documented throughput/latency is
not a proof of achievable issue rate. No paper roofline constants are used.
"""

import math
from collections import Counter
from . import isa
from dataclasses import dataclass, replace
from .execution import (
    TrainiumTuning,
    dma_panels,
    matmul_panels,
    matrix_storage,
    EngineEvent,
    schedule_events,
)

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
    transpose_commands: int = 0
    scalar_ns: float = 0
    panels: tuple = ()

    @property
    def dma_ns(self):
        return self.payload_ns


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
    config,
    rows,
    cols,
    bits,
    row_block=None,
    transpose=True,
    store=False,
    tuning=None,
):
    """Price exactly the converter's panels, with completion latency separate.

    A cross-engine completion delay is not a serialized engine initiation
    interval. The transfer occupancy includes payload only; the enclosing
    dependency model accounts for the delay on exposed edges.
    """
    tuning = tuning or TrainiumTuning()
    payload = tensor = vector = scalar = 0.0
    commands = transposes = 0
    from .movement import TransferPanel

    panels = []
    per_engine = config.dram_bandwidth / 16
    for row, col, p, f, direct in dma_panels(
        rows,
        cols,
        row_block or rows,
        transpose=transpose,
        store=store,
        tuning=tuning,
    ):
        # Each DMA engine owns eight consecutive SBUF partitions.
        # DMA transpose destination partitions are source columns.
        partitions, free = (f, p) if direct else (p, f)
        panel_payload = min(8, partitions) * free * (bits / 8) / per_engine
        before = tensor, vector, scalar
        payload += panel_payload
        if transpose and not direct:
            transposes += 1
            t, v = (
                transpose_ns(config, f, p)
                if store
                else transpose_ns(config, p, f)
            )
            if tuning.isa_lowering:
                expansion = (
                    isa.transpose(f, p) if store else isa.transpose(p, f)
                )
                t = expansion.tensor_cycles / config.frequency
                if store and tuning.copy_policy == "balanced":
                    v = expansion.scalar_cycles / clocks(config)[1]
                else:
                    v = 0
                    scalar += expansion.scalar_cycles / (
                        1.4 if config.name == "trainium-v2" else 1.2
                    )
            tensor += t
            vector += v
        commands += 1
        panels.append(
            TransferPanel(
                row,
                col,
                p,
                f,
                bits,
                store,
                transpose,
                direct,
                panel_payload,
                tensor - before[0],
                vector - before[1],
                scalar - before[2],
            )
        )
    return Service(
        payload,
        config.dram_access_latency if commands else 0,
        tensor,
        vector,
        commands,
        rows * cols * bits // 8,
        transposes,
        scalar,
        tuple(panels),
    )


def sweep(loads, store, output_steps, steps, compute):
    """Shared double-buffer recurrence, including first loads and last store.

    Like Gemmini's DMA sweep, this uses the builder's operand reload counts.
    The aggregate read/write DMA service is shared, not independent full-rate
    links. Compute duration comes from the dependency-aware engine estimate;
    independent tasks can overlap DMA according to the selected buffer plan.
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
    def __init__(self, shared, config, conv=False, batch_tiles=1, tuning=None):
        from .execution import TrainiumTuning

        self.tuning = tuning or TrainiumTuning()
        self.shared, self.config, self.conv = shared, config, conv
        self.batch_tiles = batch_tiles
        self.estimate = {}
        self.execution_plan = None
        from collections import OrderedDict

        self._runtime_cache = OrderedDict()
        self._runtime_cache_hits = 0

    def calculate_runtime(self, architecture, layer, mapping):
        # Compare supported pipeline depths for every mapping. Return the
        # selected plan explicitly through evaluate(), including capacity.
        candidates = []
        for depth in range(
            self.tuning.min_buffer_depth, self.tuning.max_buffer_depth + 1
        ):
            cycles = self._runtime(architecture, layer, mapping, depth)
            candidates.append(
                (
                    cycles,
                    self.plan,
                    self.storage,
                    self.estimate,  # _runtime creates a fresh diagnostics dict
                    self.execution_plan,
                )
            )
        (
            cycles,
            self.plan,
            self.storage,
            self.estimate,
            self.execution_plan,
        ) = min(candidates, key=lambda x: x[0])
        return cycles

    def _runtime(self, architecture, layer, mapping, depth):
        from .mapping import extent, make_plan

        self.execution_plan = None
        rc, hw = self.shared, self.config
        dma_tuning = (
            replace(self.tuning, dma_transpose=False)
            if self.conv
            else self.tuning
        )
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
        self.plan = make_plan(mapping, self.batch_tiles, rc.batch, depth)
        biasloads = (
            rc._batch_loads(mapping, (le.OC,), 1) if rc.bias_width else 0
        )
        # Loop permutations with the same realized extents, recurrence and
        # storage contract have identical costs. Cache that complete result,
        # including selected diagnostics, rather than rebuilding panel graphs.
        cache_key = (
            m,
            n,
            k,
            fx,
            fy,
            blocks,
            reductions,
            aloads,
            bloads,
            biasloads,
            ih,
            iw,
            extent(mapping, le.OX),
            self.plan,
            getattr(self, "_final_evaluation", False),
            tuple(
                getattr(rc, key, None)
                for key in (
                    "input_dtype_width",
                    "weight_dtype_width",
                    "output_dtype_width",
                    "input_dtype_name",
                    "output_dtype_name",
                    "weight_transposed",
                    "weight_hbm_ck",
                    "has_tail",
                    "bias_width",
                )
            ),
        )
        if cache_key in self._runtime_cache:
            self._runtime_cache_hits += 1
            self._runtime_cache.move_to_end(cache_key)
            cycles, self.storage, self.estimate, self.execution_plan = (
                self._runtime_cache[cache_key]
            )
            return cycles
        self.estimate = {}
        self.storage = matrix_storage(
            hw,
            ih * iw * k if self.conv else m * k,
            taps * k * n,
            m * n,
            k,
            n,
            rc.input_dtype_width,
            rc.weight_dtype_width,
            rc.output_dtype_width,
            self.plan,
            self.tuning,
            weight_partitions=k if rc.weight_transposed else n,
        )
        from voyager_compiler.codegen.transform.tiling.contracts import (
            resources_fit,
            StorageRequirement,
        )

        if rc.bias_width:
            from .execution import slot_bytes

            self.storage += (
                StorageRequirement(
                    "SBUF", slot_bytes(n, n, rc.bias_width), 2, 2048
                ),
            )
        if not resources_fit(hw, self.storage):
            return math.inf
        a = dma_service(
            hw,
            ih * iw if self.conv else m,
            k,
            rc.input_dtype_width,
            iw if self.conv else m,
            tuning=dma_tuning,
        )
        # Existing shared builder may transpose weights during DMA. The
        # canonical converter then stores [K,N] physically for that layout.
        b = dma_service(
            hw,
            taps * k if self.conv else (k if rc.weight_hbm_ck else n),
            n if self.conv or rc.weight_hbm_ck else k,
            rc.weight_dtype_width,
            k if self.conv else (k if rc.weight_hbm_ck else n),
            tuning=dma_tuning,
            transpose=not rc.weight_transposed,
        )
        out = dma_service(
            hw,
            m,
            n,
            rc.output_dtype_width,
            extent(mapping, le.OX) if self.conv else m,
            tuning=dma_tuning,
            store=True,
        )
        bias = (
            dma_service(hw, 1, n, rc.bias_width, 1, tuning=dma_tuning)
            if rc.bias_width
            else Service()
        )
        biasloads = (
            rc._batch_loads(mapping, (le.OC,), 1) if rc.bias_width else 0
        )
        loads = [(a, aloads), (b, bloads)]
        if biasloads:
            loads.append((bias, biasloads))
        events = []
        instructions = 0
        expansions = Counter()
        compute_transposes = 0

        def event(engine, ns, dependencies=()):
            events.append(EngineEvent(engine, ns, tuple(dependencies)))
            return len(events) - 1

        for mi, ni, mm, nn, kpanels in matmul_panels(m, n, k):
            accumulated = None
            for ki, kk in kpanels:
                deps = []
                if m > 512 or n > 128 or k > 128 or self.conv:
                    deps.append(event("VectorE", taps * copy_ns(hw, kk, mm)))
                    if not self.conv:
                        deps.append(event("VectorE", copy_ns(hw, nn, kk)))
                if self.conv or not rc.weight_transposed:
                    compute_transposes += taps
                    t, v = transpose_ns(hw, nn, kk)
                    trans = event("TensorE", taps * t, deps)
                    if (
                        self.tuning.isa_lowering
                        and self.tuning.copy_policy == "scalar"
                    ):
                        v = isa.transpose(nn, kk).scalar_cycles / (
                            1.4 if hw.name == "trainium-v2" else 1.2
                        )
                    deps = [
                        event(
                            (
                                "ScalarE"
                                if self.tuning.isa_lowering
                                and self.tuning.copy_policy == "scalar"
                                else "VectorE"
                            ),
                            taps * v,
                            (trans,),
                        )
                    ]
                if accumulated is not None:
                    deps.append(accumulated)
                expansion = isa.matmul(mm, nn, kk, rc.input_dtype_width)
                if self.tuning.isa_lowering:
                    expansions.update(
                        {
                            key: count * taps
                            for key, count in expansion.instructions
                        }
                    )
                accumulated = event(
                    "TensorE",
                    taps * expansion.tensor_cycles / hw.frequency,
                    deps,
                )
                instructions += taps
            eviction = event(
                (
                    "ScalarE"
                    if self.tuning.isa_lowering
                    and self.tuning.copy_policy == "scalar"
                    else "VectorE"
                ),
                (
                    max(64, mm) / (1.4 if hw.name == "trainium-v2" else 1.2)
                    if self.tuning.isa_lowering
                    and self.tuning.copy_policy == "scalar"
                    else copy_ns(hw, nn, mm)
                ),
                (accumulated,),
            )
            # ISA results remain independent panels. Binding a result to a
            # logical output slot is an alias, not a full-result SBUF copy.
            if not self.tuning.isa_lowering or self.conv:
                event("VectorE", copy_ns(hw, nn, mm), (eviction,))
        compute_path, active = schedule_events(events)
        graph_compute = None
        if self.tuning.isa_lowering and not self.conv:
            from .dependencies import compute_graph, evaluate_graph

            graph_compute, compute_counts = compute_graph(
                hw,
                m,
                n,
                k,
                rc.input_dtype_width,
                rc.weight_transposed,
                self.tuning,
                getattr(rc, "input_dtype_name", None),
                getattr(rc, "output_dtype_name", None),
            )
            compute_transposes = compute_counts.get("MATMUL_TRANSPOSE", 0)
            # Expansion counts must follow the chosen operand geometry.
            # Transpose LDWEIGHTS are added with the boundary transposes below.
            expansions = Counter(compute_counts)
            expansions.pop("MATMUL_TRANSPOSE", None)
            expansions["LDWEIGHTS"] -= compute_transposes
            instructions = sum(n.name.startswith("matmul_") for n in graph_compute.nodes)
            compute_timing = evaluate_graph(graph_compute)
            compute_path, active = compute_timing.duration_ns, dict(
                compute_timing.service_ns
            )
        tensor, vector = active.get("TensorE", 0), active.get("VectorE", 0)
        scalar = active.get("ScalarE", 0)
        tscalar = (
            scalar * blocks
            + sum(s.scalar_ns * count for s, count in loads)
            + out.scalar_ns * outputs
        )
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
        # Independent gathers/copies can overlap TensorE, with dependencies
        # represented by events. Add boundary layout work conservatively.
        compute = (
            compute_path
            + (
                ttensor
                - tensor * blocks
                + tvector
                - vector * blocks
                + tscalar
                - scalar * blocks
            )
            / blocks
        )
        self.plan = make_plan(mapping, self.batch_tiles, rc.batch, depth)
        # Exposed first-load / final-store delays. No per-command serialization
        # of completion latency; queues/descriptor issue remain uncalibrated.
        exposed = (
            max((s.startup_ns for s, _ in loads), default=0) + out.startup_ns
        )
        if (
            max(
                self.plan.input_slots,
                self.plan.weight_slots,
                self.plan.output_slots,
            )
            == 1
        ):
            queued = tdma + compute * blocks
        else:
            queued = max(
                sweep(loads, out, outputs, blocks, compute),
                tdma,
                ttensor,
                tvector,
                tscalar,
            )
        constant_bytes = 0
        transpose_count = (
            compute_transposes * blocks
            + sum(s.transpose_commands * count for s, count in loads)
            + out.transpose_commands * outputs
        )
        if self.tuning.isa_lowering and transpose_count:
            constant_bytes = isa.transpose(
                128, 128, dtype=getattr(rc, "input_dtype_name", "float32")
            ).shared_constant_bytes
            if not self.tuning.strict_realization:
                # Native nc_transpose's pinned SDK identity is uint8 in HBM.
                constant_bytes = 128 * 128
            # One shared constant load per kernel, not per transpose. Account
            # payload; first-load completion is already in the prologue.
            constant_service = constant_bytes / hw.dram_bandwidth
            tdma += constant_service
            queued += constant_service
        predicted = queued + exposed
        traffic = (
            sum(s.bytes * count for s, count in loads)
            + out.bytes * outputs
            + constant_bytes
        )
        self.plan = make_plan(mapping, self.batch_tiles, rc.batch, depth)
        scheduled_flops = 2 * m * n * k * taps * blocks
        peak_flops_per_ns = (
            2
            * 128
            * 128
            * hw.frequency
            / (4 if rc.input_dtype_width == 32 else 1)
        )
        if self.tuning.isa_lowering:
            expansions = Counter(
                {key: count * blocks for key, count in expansions.items()}
            )
            expansions.update(
                {
                    "LDWEIGHTS": transpose_count,
                    "MATMUL_TRANSPOSE": transpose_count,
                }
            )
        graph_timing = None
        previous_predicted = predicted
        if graph_compute is not None:
            from .dependencies import (
                kernel_plan,
                evaluate_graph,
                estimate_graph,
            )

            self.execution_plan = kernel_plan(
                hw,
                graph_compute,
                loads,
                out,
                blocks=blocks,
                outputs=outputs,
                buffers=self.plan,
                output_pass=output_pass,
                has_tail=rc.has_tail,
                bias=rc.bias_width,
                counts=expansions,
                identity_bytes=constant_bytes,
                compiler_allocated=self.tuning.buffer_allocation == "compiler",
            )
            graph_timing = (
                evaluate_graph
                if getattr(self, "_final_evaluation", False)
                else estimate_graph
            )(self.execution_plan.graph)
            predicted = (
                graph_timing.duration_ns + hw.timing_profile.fixed_kernel_ns
            )
            services = dict(graph_timing.service_ns)
            tdma = services.get("DMA", 0)
            ttensor, tvector, tscalar = (
                services.get(name, 0)
                for name in ("TensorE", "VectorE", "ScalarE")
            )
            queued = predicted - exposed
        self.estimate = dict(
            execution_contract=(
                "nki-isa-v1" if self.tuning.isa_lowering else "legacy-nki"
            ),
            expanded_isa=dict(expansions),
            shared_constant_bytes=constant_bytes,
            timing_basis=(
                "Panel operation graph: request admission, payload service, semaphore readiness, layout and primitive completion; timing provenance is hardware-owned"
                if graph_timing is not None
                else "Documented steady-state engine service plus exposed DMA completion; SDK startup and short-instruction completion remain unmodeled"
            ),
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
            + out.commands * outputs
            + int(bool(constant_bytes)),
            exposed_dma_completion_ns=(
                exposed if graph_timing is None else None
            ),
            dma_startup_ns=exposed if graph_timing is None else None,
            zero_startup_sensitivity_ns=(
                queued if graph_timing is None else None
            ),
            scope="Scheduled matrix kernel including measured fixed device overhead; boundary operations reported separately",
            timing_profile=hw.timing_profile.name,
            fixed_kernel_ns=(
                hw.timing_profile.fixed_kernel_ns if graph_timing else 0
            ),
            buffer_allocation=(
                "direct"
                if self.tuning.isa_lowering
                else self.tuning.buffer_allocation
            ),
            candidate_retirement_policy=self.tuning.buffer_allocation,
            buffer_depth_speed_credit=self.tuning.buffer_allocation
            != "compiler",
            dma_issue_service_ns=(
                services.get("DMAIssue", 0) if graph_timing else None
            ),
            dma_payload_ns=sum(s.payload_ns * count for s, count in loads)
            + out.payload_ns * outputs,
            hbm_bytes=traffic,
            service_ns=dict(
                DMA=tdma, TensorE=ttensor, VectorE=tvector, ScalarE=tscalar
            ),
            estimated_service_utilization=dict(
                DMA=tdma / predicted,
                TensorE=ttensor / predicted,
                VectorE=tvector / predicted,
                ScalarE=tscalar / predicted,
            ),
            limitations="Characterized primitive completion and DMA service with documented compute throughput; dependency-ready scheduling assumes available queues. Backend command ordering, physical bank contention, throttling and spills are not predicted. Service fractions are not profiler-active utilization.",
            buffer_depth=depth,
            instruction_modes=dict(
                dma_transpose=self.tuning.dma_transpose,
                explicit_isa=self.tuning.explicit_isa,
                isa_lowering=self.tuning.isa_lowering,
                copy_policy=self.tuning.copy_policy,
                matmul_operands=self.tuning.matmul_operands,
            ),
            compute_dependency_ns=compute_path,
            service_sweep_comparator_ns=previous_predicted,
            dependency_model=(
                None
                if graph_timing is None
                else dict(
                    version=1,
                    evaluation=getattr(
                        graph_timing, "evaluation", "exact_finite_graph"
                    ),
                    sampled_repetitions=getattr(
                        graph_timing, "sample_repetitions", blocks
                    ),
                    steady_state_stable=getattr(graph_timing, "stable", True),
                    window_validation_relative_error=getattr(
                        graph_timing, "validation_relative_error", 0.0
                    ),
                    scope=self.execution_plan.scope,
                    template_nodes=len(self.execution_plan.graph.nodes),
                    repetitions=blocks,
                    unknown_completion=list(graph_timing.unknown_latency),
                    last_iteration_spacing_ns=graph_timing.last_period_ns,
                    resource_service_bound_ns=graph_timing.resource_bound_ns,
                    timing_complete=not bool(graph_timing.unknown_latency),
                    scheduling="dependency-ready ASAP with shared DMA request service; final physical reuse is audited in the selected ISA DAG; finite backend queues not assumed",
                    read_retirement="operand retirement at consumer completion; characterized FP32 geometry uses accumulator forwarding, other forwarding remains unknown",
                )
            ),
        )
        if self.tuning.isa_lowering and not self.conv:
            from .orientation import matrix_choice
            self.estimate["matrix_choice"] = matrix_choice(
                hw, m, n, k, rc.input_dtype_width, rc.weight_transposed,
                self.tuning, getattr(rc, "input_dtype_name", None),
                getattr(rc, "output_dtype_name", None),
            )
        cycles = predicted * hw.frequency
        self._runtime_cache[cache_key] = (
            cycles,
            self.storage,
            self.estimate,
            self.execution_plan,
        )
        if len(self._runtime_cache) > 512:
            self._runtime_cache.popitem(last=False)
        return cycles

    def evaluate(self, architecture, layer, mapping):
        from copy import deepcopy
        from voyager_compiler.codegen.transform.tiling.contracts import (
            CandidateEvaluation,
        )

        self._final_evaluation = True
        try:
            cycles = self.calculate_runtime(architecture, layer, mapping)
        finally:
            self._final_evaluation = False
        return CandidateEvaluation(
            cycles,
            self.plan,
            self.storage,
            diagnostics=(("estimate", deepcopy(self.estimate)),),
            execution_plan=self.execution_plan,
        )

    def calculate_memory_cost(self, architecture, layer, mapping):
        # Diagnostic only; speed_only prevents memory/energy tie-breaking.
        # Rejected capacity candidates have no traffic estimate.
        return self.estimate.get("hbm_bytes", math.inf)


def vector_candidate(config, node, tile_sizes, shapes, tiling, tuning=None):
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
    from .lowering import reduction_workspace

    anchor = get_anchor_node(node)
    name = str(anchor.target)
    from .lowering import RECIPES, operation_name

    row_layout = operation_name(anchor.target) in RECIPES
    output_shape = shapes.get(node)
    spatial_pool = (
        "max_pool2d" in name
        and output_shape
        and len(output_shape) == 4
        and output_shape[0] == output_shape[-1] == 1
    )
    if spatial_pool:
        from .lowering import spatial_pool_graph
        from .dependencies import estimate_graph
        from voyager_compiler.codegen.transform.tiling.execution import (
            RepeatedGraph,
        )

        operands = [
            s
            for n, s in shapes.items()
            if n is not node and s and require_allocation(n)
        ]
        if len(operands) != 1 or output_shape[1] > 128:
            return None
        source = operands[0]
        kernel = _pair(get_arg_value(anchor, 1, "kernel_size"))
        step = _pair(get_arg_value(anchor, 2, "stride", kernel) or kernel)
        if step != (1, 1):
            return None
        storage = 128 * (kernel[0] * source[2] + output_shape[2]) * 4
        graph = spatial_pool_graph(
            config,
            output_shape,
            source,
            tuning or TrainiumTuning(),
            math.prod(kernel),
        )
        steps = math.prod(tiling)
        timed = estimate_graph(RepeatedGraph(graph.nodes, steps))
        traffic = (
            (kernel[0] * output_shape[1] * source[2] + math.prod(output_shape))
            * 4
            * steps
        )
        return (
            storage,
            (timed.duration_ns + config.timing_profile.fixed_kernel_ns)
            * config.frequency,
            traffic,
        )
    storage = traffic = retained_traffic = 0
    dma = tensor = vector = scalar = 0.0
    for operand, shape in shapes.items():
        if shape is None or (
            operand is not node and not require_allocation(operand)
        ):
            continue
        if not all(isinstance(x, int) for x in shape):
            return None
        elements = math.prod(shape)
        if not shape or (
            math.prod(shape[:-1]) > 128
            if row_layout
            else elements / min(128, shape[-1]) > 4096
        ):
            return None
        if (
            "pool" in name
            and len(shape) == 4
            and (shape[0] != 1 or shape[-1] > 128)
        ):
            return None
        bits = get_dtype_width(operand.meta.get("dtype") or operand.value.dtype)
        storage += (
            128 * math.ceil(shape[-1] * bits / 8 / 16) * 16
            if row_layout
            else slot_bytes(elements, shape[-1], bits)
        )
        service = dma_service(
            config,
            elements // shape[-1],
            shape[-1],
            bits,
            min(shape[-2], 128) if len(shape) > 2 else elements // shape[-1],
            store=operand is node,
            tuning=(
                replace(tuning or TrainiumTuning(), dma_transpose=False)
                if len(shape) > 2
                else tuning
            ),
        )
        traffic += service.bytes
        if row_layout and operand is not node and len(shape) == 1:
            retained_traffic += service.bytes
        dma += service.dma_ns
        tensor += service.tensor_ns
        vector += service.vector_ns
    output = shapes[node]
    from .lowering import RECIPES, operation_name, reduction_graph
    from .dependencies import evaluate_graph

    reduction_name = operation_name(anchor.target)
    if reduction_name in RECIPES:
        rows, width = math.prod(output[:-1]), output[-1]
        parameters = (
            sum(
                get_arg_value(anchor, index, key, None) is not None
                for index, key in ((2, "weight"), (3, "bias"))
            )
            if reduction_name == "layer_norm"
            else int(
                reduction_name == "rms_norm"
                and get_arg_value(anchor, 2, "weight", None) is not None
            )
        )
        # The same expansion is priced before bufferization and emitted later.
        timing = evaluate_graph(
            reduction_graph(config, reduction_name, min(128, rows), width)
        )
        groups = math.ceil(rows / 128)
        layout_panels = math.ceil(width / 128) * groups * (2 + parameters)
        t, c = transpose_ns(config, min(rows, 128), min(width, 128))
        tensor += layout_panels * t
        scalar += layout_panels * c
        vector += groups * timing.duration_ns
        # Extra row-major ISA workspace is checked in addition to named scratch.
        workspace = reduction_workspace(reduction_name, width, parameters, tuning=tuning)
        if row_layout:
            storage = max(storage, workspace + parameters * 128 * width * 4)
        # Workspace is shared across pipeline slots (nonmatrix_slot_size).
    free = math.prod(output) / min(128, output[-1])
    sub = node.meta.get("submodule")
    primitives = (
        [n for n in sub.graph.nodes if n.op == "call_function"]
        if sub
        else [anchor]
    )
    for primitive in primitives:
        op = str(primitive.target)
        if reduction_name in RECIPES:
            continue
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
    if (tuning or TrainiumTuning()).isa_lowering:
        from .lowering import tile_graph
        from voyager_compiler.codegen.transform.tiling.execution import (
            RepeatedGraph,
        )

        operations = tuple(operation_name(p.target) for p in primitives)
        operands = tuple(
            tuple(shape)
            for operand, shape in shapes.items()
            if shape and operand is not node and require_allocation(operand)
        )
        window = (
            math.prod(_pair(get_arg_value(anchor, 1, "kernel_size")))
            if "max_pool2d" in name
            else 1
        )
        graph = tile_graph(
            config,
            operations,
            tuple(output),
            operands,
            tuning or TrainiumTuning(),
            window,
            repetitions=steps,
        )
        # The shared compiler owns the tile repetitions; no new search path.
        from .dependencies import estimate_graph

        timed = estimate_graph(RepeatedGraph(graph.nodes, steps))
        predicted = timed.duration_ns + config.timing_profile.fixed_kernel_ns
        return (
            storage,
            predicted * config.frequency,
            traffic * steps - retained_traffic * (steps - 1),
        )
    # Prologue/drain for a two-buffer pipeline; single-tile path is serial.
    compute = tensor + vector + scalar
    predicted = (
        dma
        + compute
        + (steps - 1) * max(dma, compute)
        + 2 * config.dram_access_latency
    )
    return storage, predicted * config.frequency, traffic * steps
