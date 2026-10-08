"""Trainium binding of operation contracts to the shared dependency evaluator."""

from dataclasses import dataclass, replace
from functools import lru_cache
from collections import Counter

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    OperationEvent,
    RepeatedGraph,
    ExecutionPlan,
    evaluate_graph as _evaluate_graph,
)
from .execution import matmul_panels
from . import isa


@lru_cache(maxsize=512)
def evaluate_graph(graph):
    """Reuse immutable graph evaluations across equivalent search candidates."""
    return _evaluate_graph(graph)


def engine_clock(config, resource):
    return {
        "TensorE": config.frequency,
        "VectorE": 1.12 if config.name == "trainium-v2" else 0.96,
        "ScalarE": 1.4 if config.name == "trainium-v2" else 1.2,
    }.get(resource, config.frequency)


def timed_event(
    config,
    name,
    engine,
    service,
    dependencies=(),
    *,
    implementation="",
    step=None,
    timing_implementation="",
    timing_override=None,
    **schedule,
):
    timing = None
    if implementation and step is not None:
        contract = config.operation_implementation(implementation)
        definition = contract.steps[step]
        values = {v.name: v for v in contract.values}
        bindings = dict(definition.bindings)
        unit = config.compute_unit(definition.unit)
        capability = next(
            c
            for c in unit.mode(definition.mode).operations
            if c.operation == definition.operation
            and c.matches(
                {key: values[value].dtype for key, value in bindings.items()},
                {key: values[value].memory for key, value in bindings.items()},
            )
        )
        timing = unit.operation_timing(definition.mode, capability)
    clock = engine_clock(config, engine)
    law_name = timing_implementation or implementation
    if step is not None and step > 0 and definition.operation == "copy":
        law_name = (
            f'nki.copy.PSUM.{values[bindings["result"]].dtype.name}.{engine}'
        )
    law = config.timing_profile.operation(law_name)
    occupancy, completion = law.evaluate(service) if law else (service, None)
    if timing_override is not None:
        occupancy, completion, forward = timing_override
        schedule.setdefault("forward_ns", forward)
    return OperationEvent(
        name,
        engine,
        (
            occupancy
            if timing is None or timing.issue_interval_cycles is None
            else timing.issue_interval_cycles / clock
        ),
        (
            occupancy
            if timing is None or timing.occupancy_cycles is None
            else timing.occupancy_cycles / clock
        ),
        (
            completion
            if timing is None or timing.latency_cycles is None
            else timing.latency_cycles / clock
        ),
        dependencies=tuple(dependencies),
        implementation=implementation,
        **schedule,
    )


def compute_graph(
    config,
    m,
    n,
    k,
    bits,
    transposed,
    tuning,
    dtype=None,
    output_dtype=None,
    input_row=False,
    output_row=False,
    weight_layout="generic",
):
    from .orientation import matrix_choice, orientation_graph

    choice = matrix_choice(
        config,
        m,
        n,
        k,
        bits,
        transposed,
        tuning,
        dtype,
        output_dtype,
        input_row,
        output_row,
        weight_layout,
    )
    return orientation_graph(
        config,
        m,
        n,
        k,
        bits,
        transposed,
        replace(tuning, matmul_orientation=choice["orientation"]),
        dtype,
        output_dtype,
        input_row,
        output_row,
        weight_layout,
    )


@lru_cache(maxsize=2048)
def _weights_compute_graph(
    config, m, n, k, bits, transposed, tuning, dtype=None, output_dtype=None
):
    """One software GEMM tile, sharing the converter's panel/ISA definitions."""
    dtype = dtype or ("float32" if bits == 32 else "bfloat16")
    output_dtype = output_dtype or dtype
    nodes, counts = [], Counter()
    last_tensor_matmul = False
    last_tensor_index = -1

    def emit(
        engine,
        service,
        deps=(),
        implementation="",
        step=None,
        role="copy",
        timing_implementation="",
        timing_override=None,
    ):
        nonlocal last_tensor_matmul, last_tensor_index
        if engine == "TensorE":
            last_tensor_matmul = implementation.startswith("nki.matmul.")
            last_tensor_index = len(nodes)
        index = len(nodes)
        nodes.append(
            timed_event(
                config,
                f"{role}_{index}",
                engine,
                service,
                deps,
                implementation=implementation,
                step=step,
                timing_implementation=timing_implementation,
                timing_override=timing_override,
            )
        )
        return index

    weight_panels = {}
    for mi, ni, mm, nn, kpanels in matmul_panels(m, n, k):
        acc = None
        for ki, kk in kpanels:
            if tuning.matmul_operands == "staged":
                deps = []
                if m > 512 or n > 128 or k > 128:
                    deps = [
                        Dependency(
                            emit(
                                "VectorE",
                                max(64, mm) / engine_clock(config, "VectorE"),
                                implementation=f"nki.copy.SBUF.{dtype}.VectorE",
                                step=0,
                            )
                        ),
                        Dependency(
                            emit(
                                "VectorE",
                                max(64, kk) / engine_clock(config, "VectorE"),
                                implementation=f"nki.copy.SBUF.{dtype}.VectorE",
                                step=0,
                            )
                        ),
                    ]
                if not transposed:
                    engine = (
                        "ScalarE"
                        if tuning.copy_policy == "scalar"
                        else "VectorE"
                    )
                    expansion = isa.transpose(nn, kk, config, engine, dtype)
                    counts.update(
                        {
                            name: count
                            for name, count in expansion.instructions
                            if name in ("LDWEIGHTS", "MATMUL_TRANSPOSE")
                        }
                    )
                    clear = emit(
                        "VectorE",
                        max(64, nn) / engine_clock(config, "VectorE"),
                        deps,
                        implementation="nki.isa.memset.VectorE",
                        role="psum_clear",
                    )
                    trans = emit(
                        "TensorE",
                        expansion.tensor_cycles / config.frequency,
                        (Dependency(clear),),
                        expansion.implementation,
                        0,
                        "transpose",
                    )
                    copied = emit(
                        engine,
                        expansion.scalar_cycles / engine_clock(config, engine),
                        (Dependency(trans),),
                        expansion.implementation,
                        1,
                    )
                    deps = [Dependency(copied)]
            else:
                deps = []
                weight_key = (ni, ki)
                reuse = tuning.matmul_operands == "reuse"
                if reuse and weight_key in weight_panels:
                    weight_ready = weight_panels[weight_key]
                elif not transposed:
                    engine = (
                        "ScalarE"
                        if tuning.copy_policy == "scalar"
                        else "VectorE"
                    )
                    expansion = isa.transpose(nn, kk, config, engine, dtype)
                    counts.update(
                        {
                            name: count
                            for name, count in expansion.instructions
                            if name in ("LDWEIGHTS", "MATMUL_TRANSPOSE")
                        }
                    )
                    clear = emit(
                        "VectorE",
                        max(64, nn) / engine_clock(config, "VectorE"),
                        implementation="nki.isa.memset.VectorE",
                        role="psum_clear",
                    )
                    trans = emit(
                        "TensorE",
                        expansion.tensor_cycles / config.frequency,
                        (Dependency(clear),),
                        expansion.implementation,
                        0,
                        "transpose",
                    )
                    weight_ready = emit(
                        engine,
                        expansion.scalar_cycles / engine_clock(config, engine),
                        (Dependency(trans),),
                        expansion.implementation,
                        1,
                    )
                    if reuse:
                        weight_panels[weight_key] = weight_ready
                else:
                    weight_ready = None
                if weight_ready is not None:
                    deps.append(Dependency(weight_ready))
            if acc is not None:
                deps.append(Dependency(acc, milestone="forward"))
            else:
                clear = emit(
                    "VectorE",
                    max(64, mm) / engine_clock(config, "VectorE"),
                    implementation="nki.isa.memset.VectorE",
                    role="psum_clear",
                )
                deps.append(Dependency(clear))
            staged = tuning.matmul_operands == "staged" and (
                m > 512 or n > 128 or k > 128
            )
            expansion = isa.matmul(
                mm,
                nn,
                kk,
                bits,
                config,
                dtype,
                moving_stride=1 if staged else (k + 127) // 128,
                stationary_stride=(
                    (k + 127) // 128 if transposed and not staged else 1
                ),
                streaming=last_tensor_matmul
                and not any(d.source > last_tensor_index for d in deps),
            )
            counts.update(dict(expansion.instructions))
            acc = emit(
                "TensorE",
                expansion.tensor_cycles / config.frequency,
                deps,
                expansion.implementation,
                0,
                "matmul",
                timing_implementation=expansion.timing_implementation,
                timing_override=expansion.timing_override,
            )
        engine = "ScalarE" if tuning.copy_policy == "scalar" else "VectorE"
        emit(
            engine,
            max(64, mm) / engine_clock(config, engine),
            (Dependency(acc),),
            implementation=f"nki.copy.PSUM.{output_dtype}.{engine}",
            step=0,
            role="evict",
        )
    return RepeatedGraph(tuple(nodes)), counts


def kernel_plan(
    config,
    compute,
    loads,
    store,
    *,
    blocks,
    outputs,
    buffers,
    output_pass,
    has_tail,
    bias,
    counts,
    compiler_allocated=True,
    identity_bytes=65536,
):
    """Reuse identical immutable plans across shared mapping permutations."""
    return _kernel_plan(
        config,
        compute,
        tuple(loads),
        store,
        blocks=blocks,
        outputs=outputs,
        buffers=(
            buffers.input_slots,
            buffers.weight_slots,
            buffers.output_slots,
        ),
        output_pass=output_pass,
        has_tail=has_tail,
        bias=bias,
        counts=tuple(sorted(counts.items())),
        compiler_allocated=compiler_allocated,
        identity_bytes=identity_bytes,
    )


@lru_cache(maxsize=2048)
def _kernel_plan(
    config,
    compute,
    loads,
    store,
    *,
    blocks,
    outputs,
    buffers,
    output_pass,
    has_tail,
    bias,
    counts,
    compiler_allocated=True,
    identity_bytes=65536,
):
    """Compact periodic load/compute/store graph for shared matrix traversal.

    Transfer recurrence is the shared traversal's reload period. Last-consumer
    completion conservatively retires input slots; no unproven early-read or
    PSUM-bank independence is credited. Backend resource scheduling remains an
    explicit dependency-ASAP assumption, not a physical allocation guarantee.
    """
    reductions = blocks // outputs
    if any(blocks % count for _, count in loads):
        raise ValueError("Nonintegral load recurrence cannot be represented")
    nodes = []
    roots, load_starts = [], []

    def add(name, resource, service, deps=(), period=1, phase=0, latency=None):
        index = len(nodes)
        nodes.append(
            OperationEvent(
                name,
                resource,
                service,
                service,
                latency,
                dependencies=tuple(deps),
                period=period,
                phase=phase,
            )
        )
        return index

    from .movement import transfer_graph

    def append_transfer(name, service, deps=(), period=1, phase=0):
        graph = transfer_graph(config, service.panels)
        offset = len(nodes)
        starts = []
        for node in graph.nodes:
            edges = tuple(
                replace(d, source=d.source + offset) for d in node.dependencies
            )
            if not edges:
                starts.append(len(nodes))
                edges = tuple(Dependency(d) for d in deps)
            nodes.append(
                replace(
                    node,
                    name=f"{name}_{node.name}",
                    dependencies=edges,
                    period=period,
                    phase=phase,
                )
            )
        end = add(
            name,
            "control",
            0,
            tuple(Dependency(offset + j) for j in range(len(graph.nodes))),
            period,
            phase,
            0,
        )
        return end, starts

    identity = None
    if any(name == "MATMUL_TRANSPOSE" and count for name, count in counts):
        law = config.timing_profile.load
        issue = law.issue_ns if law else identity_bytes / config.dram_bandwidth
        request = add(
            "identity_request",
            "DMAIssue",
            issue,
            period=blocks,
            latency=law.dispatch_ns if law else 0,
        )
        # Explicit identity DMA preserves the selected operand dtype.
        # There is no hidden INT8 identity or backend cast in this realization.
        payload = (
            law.payload(identity_bytes / config.dram_bandwidth)
            if law
            else identity_bytes / config.dram_bandwidth
        )
        identity = add(
            "identity_payload",
            "DMA",
            identity_bytes / config.dram_bandwidth,
            (Dependency(request),),
            blocks,
            latency=payload
            + (law.notification_ns if law else config.dram_access_latency),
        )
    for i, (load, count) in enumerate(loads):
        period = blocks // count
        last, starts = append_transfer(f"load_{i}", load, period=period)
        roots.append(last)
        load_starts.append((starts, period))
    offset = len(nodes)
    for node in compute.nodes:
        deps = tuple(
            replace(d, source=d.source + offset) for d in node.dependencies
        )
        if not deps:
            deps = tuple(Dependency(root) for root in roots)
        if identity is not None and node.resource == "TensorE":
            deps += (Dependency(identity),)
        nodes.append(replace(node, dependencies=deps))
    compute_end = add(
        "compute_done",
        "control",
        0,
        tuple(Dependency(offset + i) for i in range(len(compute.nodes))),
        latency=0,
    )
    last = compute_end
    if reductions > 1:
        combine = OperationEvent(
            "combine",
            "VectorE",
            output_pass,
            output_pass,
            dependencies=(
                Dependency(compute_end),
                Dependency(compute_end, 1),
            ),
            period=reductions,
            phase=0,
            except_phase=True,
        )
        last = len(nodes)
        nodes.append(combine)
        # The shared split-K accumulator is one resident reduction context.
        nodes[last] = replace(
            nodes[last],
            dependencies=nodes[last].dependencies + (Dependency(last, 1),),
        )
    # Join the current compute result with any previous partial-sum updates.
    finish = add(
        "result_ready",
        "control",
        0,
        tuple(Dependency(i) for i in sorted({compute_end, last})),
        latency=0,
    )
    if reductions > 1:
        for i, node in enumerate(compute.nodes):
            if not node.dependencies:
                j = offset + i
                nodes[j] = replace(
                    nodes[j],
                    dependencies=nodes[j].dependencies
                    + (Dependency(finish, 1),),
                )
    if has_tail or bias:
        finish = add(
            "epilogue",
            "VectorE",
            output_pass * (int(bool(has_tail)) + int(bool(bias))),
            (Dependency(finish),),
            reductions,
            reductions - 1,
        )
    store_index, _ = append_transfer(
        "store", store, (finish,), reductions, reductions - 1
    )
    if not compiler_allocated:
        for i, (starts, period) in enumerate(load_starts):
            slots = (buffers[0], buffers[1], 1)[i] or 1
            for first in starts:
                nodes[first] = replace(
                    nodes[first],
                    dependencies=nodes[first].dependencies
                    + (
                        Dependency(
                            compute_end, (slots - 1) * period + 1, "read"
                        ),
                    ),
                )
        distance = ((buffers[2] or 1) - 1) * reductions + 1
        for i, node in enumerate(compute.nodes):
            if not node.dependencies:
                j = offset + i
                nodes[j] = replace(
                    nodes[j],
                    dependencies=nodes[j].dependencies
                    + (Dependency(store_index, distance),),
                )
    graph = RepeatedGraph(tuple(nodes), blocks)
    implementations = tuple(
        sorted({n.implementation for n in nodes if n.implementation})
    )
    return ExecutionPlan(
        graph,
        implementations,
        counts,
        tuple(
            (name, count or 1)
            for name, count in zip(
                ("input_slots", "weight_slots", "output_slots"), buffers
            )
        ),
        (
            "Panel load/layout/compute/reduction/store graph; no logical-depth speed credit; exact physical reuse audited after selection"
            if compiler_allocated
            else "Panel graph with legacy logical-slot retirement assumptions"
        ),
    )


def audit_realization(plans, converter):
    """Check selected expansion/compute dependencies before source publication.

    This authenticates the modeled compute regions against the regions actually
    consumed by GEMM lowering. It does not certify NKI's physical bank placement,
    DMA deduplication, or its hardware instruction schedule.
    """
    if not plans:
        return dict(
            status="not_applicable", scope="No selected matrix execution plan"
        )
    from collections import Counter
    import json
    from copy import deepcopy

    expected_counts = Counter()
    expected_graphs = Counter()
    expected_bindings = Counter()
    expected_loads = Counter()
    expected_implementations = set()
    for plan in plans:
        expected_counts.update(dict(plan["instruction_counts"]))
        expected_implementations.update(plan["implementations"])
        graph = deepcopy(plan["graph"])
        compute = [
            node
            for node in graph["nodes"]
            if node["name"].startswith(
                ("copy_", "transpose_", "matmul_", "evict_", "psum_clear_")
            )
        ]
        if not compute:
            raise ValueError(
                "Selected matrix plan has no compute implementation"
            )
        offset = next(
            i for i, node in enumerate(graph["nodes"]) if node is compute[0]
        )
        for node in compute:
            node["dependencies"] = [
                dict(edge, source=edge["source"] - offset)
                for edge in node["dependencies"]
                if offset <= edge["source"] < offset + len(compute)
                and edge["distance"] == 0
            ]
        key = json.dumps(dict(nodes=compute, repetitions=1), sort_keys=True)
        expected_graphs[key] += graph["repetitions"]
        periods = {n["name"]: n["period"] for n in graph["nodes"]}
        expected_loads[
            key,
            graph["repetitions"] // periods["load_0"],
            graph["repetitions"] // periods["load_1"],
        ] += 1
        slots = dict(plan["buffer_slots"])
        expected_bindings[
            key, slots["input_slots"], slots["weight_slots"]
        ] += graph["repetitions"]
    actual_graphs = Counter(
        json.dumps(graph, sort_keys=True)
        for graph in converter.realized_compute_graphs
    )
    if expected_graphs != actual_graphs:
        raise ValueError(
            "Realized matrix dependency templates differ from the selected execution plan"
        )
    actual_bindings = Counter(
        (
            json.dumps(graph, sort_keys=True),
            slots["input_slots"],
            slots["weight_slots"],
        )
        for graph, slots in zip(
            converter.realized_compute_graphs,
            converter.matrix_buffer_bindings,
        )
    )
    if actual_bindings != expected_bindings:
        raise ValueError(
            "Realized input/weight buffer slots differ from the selected execution plan"
        )
    loaded = Counter(
        event["dst"] for event in converter.events if event["kind"] == "copy"
    )
    regions = {
        (
            json.dumps(graph, sort_keys=True),
            slots["input_name"],
            slots["weight_name"],
        )
        for graph, slots in zip(
            converter.realized_compute_graphs,
            converter.matrix_buffer_bindings,
        )
    }
    actual_loads = Counter((key, loaded[a], loaded[b]) for key, a, b in regions)
    if actual_loads != expected_loads:
        raise ValueError(
            "Realized input/weight load recurrence differs from the selected execution plan"
        )
    if not expected_implementations <= converter.used_implementations:
        raise ValueError(
            "Realized operation implementations differ from the selected execution plan"
        )
    actual_counts = {
        name: converter.expanded_isa.get(name, 0) for name in expected_counts
    }
    # Boundary kernels (padding/transposition) can add operations outside the
    # matrix model. Exact whole-program counts are only asserted for pure matrix
    # programs; the per-matrix graph audit above remains exact in either case.
    exact_counts = actual_counts == dict(expected_counts)
    movement_reselected = (
        getattr(converter, "movement_selector", None) is not None
    )
    replaced_movement_counts = {
        "LDWEIGHTS",
        "MATMUL_TRANSPOSE",
        "COPY_VECTOR",
        "COPY_SCALAR",
    }
    if any(
        actual_counts[name] < count
        for name, count in expected_counts.items()
        if not (movement_reselected and name in replaced_movement_counts)
    ):
        raise ValueError(
            "Realized instruction expansion is smaller than the selected plan"
        )
    # A selected layout route may legitimately remove Tensor transpose and its
    # LDWEIGHTS/copy chain. The shared software compute/load recurrence checks
    # above still apply; the replacement is validated against endpoint requests
    # and the final typed ISA graph, not the superseded movement counts.
    if movement_reselected and not converter.movement_selector.requests:
        raise ValueError("Movement reselection has no endpoint certificates")
    return dict(
        status="passed",
        compute_regions=sum(actual_graphs.values()),
        implementations=sorted(expected_implementations),
        exact_whole_program_tensor_counts=exact_counts,
        movement_reselected=movement_reselected,
        movement_count_contract=(
            "selected endpoint chains"
            if movement_reselected
            else "original dependency template"
        ),
        scope="Matrix implementations, logical buffer declarations, load recurrence and expansion counts; template agreement is not an independent emitted dependency audit",
        limitations="Boundary DMA readiness and slot retirement are modeled from KernelBufferPlan; physical scheduling/bank allocation require execution profiling",
    )


def _active_count(node, repetitions):
    count = max(0, (repetitions - 1 - node.phase) // node.period + 1)
    return repetitions - count if node.except_phase else count


@dataclass(frozen=True)
class SteadyTiming:
    duration_ns: float
    service_ns: tuple
    unknown_latency: tuple
    resource_bound_ns: float
    last_period_ns: float
    sample_repetitions: int
    period: int
    stable: bool
    validation_relative_error: float = 0.0
    evaluation: str = "startup_steady_tail"


def _compressed_cadence(graph):
    """Shorten repeated interiors while preserving all long-cadence boundaries.

    For example, 16 runs of 4096 iterations become 16 runs of 4, 8, 16 and
    32 iterations. All 16 reloads and the actual prefix/drain remain explicit.
    Only aligned first/last phases and exactly scalable distances are supported.
    """
    import math

    total = graph.repetitions
    periods = sorted({1, *(n.period for n in graph.nodes)})
    gaps = [
        (large // small, small, large)
        for small, large in zip(periods, periods[1:])
        if large >= 32 * small and large < total
    ]
    if not gaps or any(n.phase not in (0, n.period - 1) for n in graph.nodes):
        return None
    _, small, large = max(gaps)
    small_period = math.lcm(*(p for p in periods if p <= small))
    distances = {d.distance for n in graph.nodes for d in n.dependencies}
    small_period = math.lcm(
        small_period, *(d for d in distances if 0 < d < large)
    )
    scalable = [
        total,
        *(p for p in periods if p >= large),
        *(d for d in distances if d >= large),
    ]
    divisor = math.gcd(*scalable)
    factor = 1
    while (
        divisor % (2 * factor) == 0
        and large // (2 * factor) >= 4 * small_period
    ):
        factor *= 2
    if factor < 8:
        return None

    def sample(f):
        nodes = tuple(
            replace(
                node,
                period=(
                    node.period // f if node.period >= large else node.period
                ),
                phase=(
                    (node.period // f - 1 if node.phase else 0)
                    if node.period >= large
                    else node.phase
                ),
                dependencies=tuple(
                    (
                        replace(d, distance=d.distance // f)
                        if d.distance >= large
                        else d
                    )
                    for d in node.dependencies
                ),
            )
            for node in graph.nodes
        )
        count = total // f
        return count, evaluate_graph(RepeatedGraph(nodes, count)).duration_ns

    lengths, durations = zip(
        *(sample(factor // scale) for scale in (1, 2, 4, 8))
    )
    slopes = [
        (durations[i + 1] - durations[i]) / (lengths[i + 1] - lengths[i])
        for i in range(3)
    ]
    mean = sum(slopes) / len(slopes)
    if any(abs(s - mean) > 0.01 * max(1, abs(mean)) for s in slopes):
        return None
    predicted = durations[-2] + (lengths[-1] - lengths[-2]) * slopes[-2]
    error = abs(durations[-1] - predicted) / max(1, durations[-1])
    if error > 0.002:
        return None
    service = Counter()
    for node in graph.nodes:
        service[node.resource] += _active_count(node, total) * node.occupancy_ns
    bound = max(service.values(), default=0)
    duration = max(bound, durations[-1] + (total - lengths[-1]) * slopes[-1])
    return SteadyTiming(
        duration,
        tuple(sorted(service.items())),
        tuple(n.name for n in graph.nodes if n.latency_ns is None),
        bound,
        slopes[-1],
        lengths[-1],
        small_period,
        True,
        error,
        "startup_steady_tail_compressed_cadence",
    )


@lru_cache(maxsize=128)
def estimate_graph(graph):
    """Estimate a long graph from startup, neighboring periods and drain.

    T(N) = T(W) + (N-W)*II, where T(W) includes the real prefix and tail.
    Periodic transfers, reduction phases and reuse-distance dependencies survive
    in the window. Three consistent increments and a doubled-window check establish
    observed convergence, not a proof of exactness. Short, irregular or unstable
    graphs fall back to finite replay; selected matrix plans always use replay.
    """
    import math

    total = graph.repetitions
    if sum(_active_count(n, total) for n in graph.nodes) <= 20000:
        return evaluate_graph(graph)
    # An interior one-time event is not a stationary prefix or drain.
    if any(
        n.period >= total and n.phase not in (0, total - 1) for n in graph.nodes
    ):
        return evaluate_graph(graph)
    compressed = _compressed_cadence(graph)
    if compressed is not None:
        return compressed
    periods = [n.period for n in graph.nodes if n.period < total]
    distances = [
        d.distance for n in graph.nodes for d in n.dependencies if d.distance
    ]
    period = math.lcm(*periods, *distances) if periods or distances else 1
    if 12 * period >= total:
        return evaluate_graph(graph)
    remainder = total % period

    def sample(repetitions):
        nodes = tuple(
            (
                replace(n, period=repetitions, phase=repetitions - 1)
                if n.period >= total and n.phase == total - 1
                else n
            )
            for n in graph.nodes
        )
        return evaluate_graph(RepeatedGraph(nodes, repetitions)).duration_ns

    def close(a, b, tolerance):
        return abs(a - b) <= tolerance * max(1, abs(a), abs(b))

    # A transient cannot be steady if its interval is below the sustained
    # demand of any exclusive resource (including issue bandwidth).
    sustained = Counter()
    for node in graph.nodes:
        fraction = 1 / node.period if node.period < total else 0
        if node.except_phase:
            fraction = 1 - fraction
        sustained[node.resource] += fraction * max(
            node.issue_ns, node.occupancy_ns
        )
    minimum_interval = max(sustained.values(), default=0)
    length = 2 * period + remainder
    previous = sample(length)
    stable = False
    # Enlarge the neighboring window when startup lasts longer or resource
    # interleaving jitters. Five levels cover up to 190 base periods.
    for level in range(5):
        step = period * 2**level
        slopes = []
        for _ in range(3):
            if length + step >= total:
                return evaluate_graph(graph)
            length += step
            current = sample(length)
            slopes.append((current - previous) / step)
            previous = current
        slope = sum(slopes) / len(slopes)
        if slope < minimum_interval * (1 - 0.002):
            continue
        if not all(close(s, slope, 0.01) for s in slopes):
            continue
        check_length = 2 * length - remainder
        if check_length >= total:
            break
        checked = sample(check_length)
        predicted = current + (check_length - length) * slope
        if close(checked, predicted, 0.002):
            error = abs(checked - predicted) / max(1, checked)
            slope = (checked - current) / (check_length - length)
            length, current = check_length, checked
            stable = True
            break
    if not stable:
        return evaluate_graph(graph)
    service = Counter()
    for node in graph.nodes:
        service[node.resource] += _active_count(node, total) * node.occupancy_ns
    bound = max(service.values(), default=0)
    duration = max(bound, current + (total - length) * slope)
    return SteadyTiming(
        duration,
        tuple(sorted(service.items())),
        tuple(n.name for n in graph.nodes if n.latency_ns is None),
        bound,
        slope,
        length,
        period,
        True,
        error,
    )
