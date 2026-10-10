"""Independent source-call audit and whole-executable boundary accounting.

This is not a certificate of Neuron allocation or instruction scheduling. A
compiled-profile audit checks those separately. Source-only counts are named
accordingly: the backend can fold assembly copies and add control instructions.
"""

import ast
from collections import Counter
from dataclasses import asdict
import math
from .movement import TransferPanel, transfer_graph
from voyager_compiler.codegen.transform.tiling.execution import (
    evaluate_graph,
    OperationEvent,
    Dependency,
    RepeatedGraph,
)
from dataclasses import replace


def matmul_free_stride(checker, expression):
    """Read the free-axis stride from an actual regular SBUF operand view."""
    import numpy as np
    from .movement_search import coordinates

    root, axes = coordinates(checker, expression, (slice(0, 2), slice(None)))
    if len(axes) != 2 or checker.program.tensors[root].memory != "SBUF":
        return None
    if axes[1].shape[1] < 2 or not np.all(np.diff(axes[0], axis=1) == 0):
        return None
    delta = np.diff(axes[1], axis=1)
    return abs(int(delta[0, 0])) if np.all(delta == delta[0, 0]) else None


def analyze_selected(
    program,
    hardware,
    *,
    execution_model="baseline",
    graph_observer=None,
    reorder_window=16,
):
    """Replay the actual selected ISA DAG, including physical reuse edges.

    This final audit uses the same primitive timing laws as candidate templates.
    It exposes disagreement with the compact search model instead of silently
    dropping explicit temporaries, clears or retained-operand initialization.
    """
    dependency_audit = {}
    if execution_model == "scheduled-ready":
        from .region_dependencies import refine

        program, dependency_audit = refine(program)
    from .instruction_plan import Builder, Tensor, BITS
    from .dependencies import engine_clock
    from . import isa

    checker = Builder.__new__(Builder)
    checker.program = program
    checker.constants = {}
    for n, e in program.indices.items():
        checker.constants[n] = checker.value(e)
    from .timing import StreamTransposeTiming

    stream_context = StreamTransposeTiming()
    nodes = []
    descriptors = {}
    contextual = execution_model in (
        "scheduled-ready",
        "primitives",
        "context",
        "context-ready",
        "pipeline",
        "pipeline-ready",
        "pipeline-startup",
        "pipeline-startup-ready",
    )
    collect = contextual or graph_observer is not None
    from functools import lru_cache

    @lru_cache(maxsize=32768)
    def stride(expression):
        return matmul_free_stride(checker, expression)

    completion = []
    last_tensor_matmul = False
    last_tensor_index = -1
    geometry_counts = Counter()
    reads = writes = 0
    native_identity = any(
        i.opcode == "nisa.nc_transpose"
        and program.contracts[i.implementation]["engine"] == "TensorE"
        for i in program.instructions
    )
    identity_completion = None
    if native_identity:
        # Pinned SDK expansion: one shared uint8 128x128 HBM identity,
        # converted to the operand type in SBUF by the native DMA.
        reads = 128 * 128
        panel = TransferPanel(
            0,
            0,
            128,
            128,
            8,
            False,
            False,
            False,
            8 * 128 / (hardware.dram_bandwidth / 16),
        )
        nodes.extend(transfer_graph(hardware, (panel,)).nodes)
        identity_completion = len(nodes) - 1
    for index, ins in enumerate(program.instructions):
        args = [checker.value(e) for e in ins.args]
        kw = {k: checker.value(e) for k, e in ins.kwargs}
        dst = checker.value(ins.destination)
        dependencies = tuple(
            Dependency(completion[d]) for d in ins.dependencies
        )
        engine = program.contracts[ins.implementation]["engine"]
        free = math.prod(dst.shape[1:])
        partitions = dst.shape[0]
        data = next(
            (v for v in (*args, *kw.values()) if isinstance(v, Tensor)), None
        )
        opcode = ins.opcode.removeprefix("nisa.")
        law_name = ""
        timing_override = None
        if opcode == "dma_copy":
            src = kw.get("src", args[0] if args else None)
            store = dst.memory == "HBM"
            local = src if store else dst
            byte_count = math.prod(local.shape) * BITS[local.dtype] // 8
            if store:
                writes += byte_count
            else:
                reads += byte_count
            # The same busiest-engine law as movement.partition_payload_ns.
            # Keep this expression self-contained for replay of saved plans.
            ideal = (
                min(8, local.shape[0])
                * math.prod(local.shape[1:])
                * (BITS[local.dtype] / 8)
                / (hardware.dram_bandwidth / 16)
            )
            # Match transfer_graph's request, dispatch, payload and completion
            # nodes while retaining all selected predecessors.
            panel = TransferPanel(
                0,
                0,
                local.shape[0],
                math.prod(local.shape[1:]),
                BITS[local.dtype],
                store,
                False,
                False,
                ideal,
            )
            offset = len(nodes)
            for n in transfer_graph(hardware, (panel,)).nodes:
                edges = (
                    tuple(
                        replace(d, source=d.source + offset)
                        for d in n.dependencies
                    )
                    or dependencies
                )
                nodes.append(
                    replace(n, name=f"i{index}_{n.name}", dependencies=edges)
                )
            completion.append(len(nodes) - 1)
            continue
        clock = engine_clock(hardware, engine)
        service = max(64, free) / clock
        if opcode == "nc_matmul":
            a, b = args
            if kw.get("is_transpose"):
                expansion = isa.transpose(
                    a.shape[0], a.shape[1], hardware, "ScalarE", a.dtype
                )
            else:
                operand_roots = {program.root(name) for name in ins.reads} - {
                    program.root(name) for name in ins.writes
                }
                fresh_operands = any(
                    d > last_tensor_index
                    and operand_roots.intersection(
                        program.root(name)
                        for name in program.instructions[d].writes
                    )
                    for d in ins.dependencies
                )
                streaming = last_tensor_matmul and not fresh_operands
                expansion = isa.matmul(
                    b.shape[1],
                    a.shape[1],
                    a.shape[0],
                    BITS[a.dtype],
                    hardware,
                    a.dtype,
                    moving_stride=matmul_free_stride(checker, ins.args[1]),
                    stationary_stride=matmul_free_stride(checker, ins.args[0]),
                    streaming=streaming,
                )
                timing_override = expansion.timing_override
                if timing_override is not None:
                    geometry_counts["steady" if streaming else "cold"] += 1
            service = expansion.tensor_cycles / hardware.frequency
            law_name = (
                expansion.timing_implementation or expansion.implementation
            )
        elif opcode == "nc_transpose" and engine == "TensorE":
            dependencies += (Dependency(identity_completion),)
            expansion = isa.transpose(
                data.shape[0], data.shape[1], hardware, "ScalarE", data.dtype
            )
            service = expansion.tensor_cycles / hardware.frequency
            law_name = (
                expansion.timing_implementation or expansion.implementation
            )
        elif opcode == "nc_transpose":
            from .movement_search import coordinates

            def address(expr):
                root, axes = coordinates(checker, expr)
                placement = program.placements[root]
                if len(axes) != 2:
                    raise ValueError(
                        "Stream timing requires a two-dimensional local view"
                    )
                import numpy as np

                if not (
                    np.array_equal(
                        axes[0],
                        np.broadcast_to(
                            axes[0][0, 0] + np.arange(32)[:, None], (32, 32)
                        ),
                    )
                    and np.array_equal(
                        axes[1],
                        np.broadcast_to(
                            axes[1][0, 0] + np.arange(32)[None, :], (32, 32)
                        ),
                    )
                ):
                    raise ValueError("Uncharacterized stream-transpose strides")
                return (
                    (placement.start_partition + int(axes[0][0, 0])) * 262144
                    + placement.byte_address
                    + int(axes[1][0, 0]) * BITS[data.dtype] // 8
                )

            law_name = stream_context.select(
                dtype=dst.dtype,
                source_dtype=data.dtype,
                engine=engine,
                partitions=partitions,
                source_free=math.prod(data.shape[1:]),
                destination_free=free,
                source_address=address(ins.args[0]),
                destination_address=address(ins.destination),
                strides=(1, 1),
            )
            if law_name is None:
                raise ValueError("Uncharacterized selected stream transpose")
        elif opcode == "tensor_copy":
            law_name = f"nki.copy.{data.memory}.{dst.dtype}.{engine}"
        elif opcode == "tensor_tensor":
            service = max(64, 2 * free) / clock
            law_name = f"nki.binary.{dst.dtype}.{engine}"
        elif opcode == "tensor_reduce":
            service = max(64, math.prod(data.shape[1:])) / clock
            law_name = f"nki.reduce.{dst.dtype}.{engine}"
        elif opcode == "reciprocal":
            service = max(64, 8 * free) / clock
            law_name = f"nki.reciprocal.{dst.dtype}.{engine}"
        else:
            kind = {
                "tensor_scalar": "scalar",
                "activation": "activation",
                "memset": "memset",
            }[opcode]
            law_name = f"nki.{kind}.{dst.dtype}.{engine}"
        if engine == "TensorE":
            last_tensor_index = index
            last_tensor_matmul = opcode == "nc_matmul" and not kw.get(
                "is_transpose", False
            )
        if engine == "VectorE" and opcode != "nc_transpose":
            stream_context.reset()
        law = hardware.timing_profile.operation(law_name)
        occupancy, latency = law.evaluate(service) if law else (service, None)
        forward = None
        if timing_override is not None:
            occupancy, latency, forward = timing_override
            if ins.accumulate:
                dependencies = tuple(
                    Dependency(
                        completion[d],
                        milestone=(
                            "forward"
                            if (
                                program.instructions[d].opcode
                                == "nisa.nc_matmul"
                                and set(program.instructions[d].writes)
                                & set(ins.writes)
                            )
                            else "result"
                        ),
                    )
                    for d in ins.dependencies
                )
        if collect:
            desc = dict(
                opcode=opcode,
                engine=engine,
                dtype=dst.dtype,
                partitions=partitions,
                free=free,
                source_free=math.prod(data.shape[1:]) if data else 0,
                source_memory=data.memory if data else None,
                source_dtype=data.dtype if data else None,
                source_stride=None,
                destination_stride=None,
                function=str(
                    kw.get(
                        "op",
                        (
                            args[0]
                            if opcode in ("activation", "tensor_reduce")
                            and args
                            else ""
                        ),
                    )
                ),
                transpose=opcode == "nc_transpose"
                or bool(kw.get("is_transpose")),
            )
            if opcode in ("tensor_copy", "nc_transpose", "nc_matmul"):
                expressions = (*ins.args, *(v for _, v in ins.kwargs))
                values = (*args, *kw.values())
                source_expr = next(
                    (
                        e
                        for e, v in zip(expressions, values)
                        if isinstance(v, Tensor)
                    ),
                    None,
                )
                if source_expr is not None and data.memory == "SBUF":
                    desc["source_stride"] = stride(source_expr)
                if dst.memory == "SBUF":
                    desc["destination_stride"] = stride(ins.destination)
            if opcode == "nc_matmul" and not desc["transpose"]:
                desc.update(
                    moving=args[1].shape[1],
                    stationary=args[0].shape[1],
                    contraction=args[0].shape[0],
                    moving_stride=stride(ins.args[1]),
                    stationary_stride=stride(ins.args[0]),
                    streaming=streaming,
                )
            desc["forward_sources"] = [
                completion[d]
                for d in ins.dependencies
                if ins.accumulate
                and program.instructions[d].opcode == "nisa.nc_matmul"
                and set(program.instructions[d].writes) & set(ins.writes)
            ]
            descriptors[len(nodes)] = desc
            if contextual:
                from .calibrated_isa import evaluate

                override = evaluate(desc, occupancy, latency, forward)
                if override is not None:
                    occupancy, latency, forward = override
                    if forward is not None and ins.accumulate:
                        dependencies = tuple(
                            Dependency(
                                completion[d],
                                milestone=(
                                    "forward"
                                    if program.instructions[d].opcode
                                    == "nisa.nc_matmul"
                                    and set(program.instructions[d].writes)
                                    & set(ins.writes)
                                    else "result"
                                ),
                            )
                            for d in ins.dependencies
                        )
        nodes.append(
            OperationEvent(
                f"i{index}_{opcode}",
                engine,
                occupancy,
                occupancy,
                latency,
                dependencies=dependencies,
                implementation=ins.implementation,
                forward_ns=forward,
            )
        )
        completion.append(len(nodes) - 1)
    if graph_observer is not None:
        graph_observer(RepeatedGraph(tuple(nodes)), descriptors)
    from .physical_context import transform, identity

    graph, ordering = transform(
        RepeatedGraph(tuple(nodes)),
        execution_model,
        reorder_window=reorder_window,
    )
    if execution_model in (
        "pipeline",
        "pipeline-ready",
        "pipeline-startup",
        "pipeline-startup-ready",
    ):
        from .calibrated_isa import pipeline_timing

        callback = pipeline_timing(
            {nodes[i].name: d for i, d in descriptors.items()},
            startup_scenario="startup" in execution_model,
        )
        result = evaluate_graph(graph, event_timing=callback)
    else:
        result = evaluate_graph(graph)
    return dict(
        prediction_ns=result.duration_ns
        + hardware.timing_profile.fixed_kernel_ns,
        instruction_count=len(program.instructions),
        matmul_geometry_events=dict(geometry_counts),
        event_count=len(graph.nodes),
        physical_model=execution_model,
        physical_model_record=identity(
            execution_model, reorder_window=reorder_window
        ),
        dependency_refinement=dependency_audit,
        ordering=ordering,
        unknown_completion=list(result.unknown_latency),
        service_ns=dict(result.service_ns),
        hbm_read_bytes=reads,
        hbm_write_bytes=writes,
        hbm_bytes=reads + writes,
        scope=(
            "Selected logical ISA dependencies; native physical allocation/reuse/spills are unknown"
            if program.encoding_storage == "compiler"
            else "Exact selected ISA dependencies and physical reuse; backend issue scheduling remains modeled"
        ),
    )


def analyze_program(source, converter, record):
    calls = Counter()
    weighted = []

    def visit(node, weight=1):
        if isinstance(node, ast.For):
            call = node.iter
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
                or call.func.attr != "sequential_range"
                or len(call.args) != 1
                or not isinstance(call.args[0], ast.Constant)
            ):
                raise ValueError("Unrecorded source loop")
            for child in node.body:
                visit(child, weight * call.args[0].value)
            return
        if isinstance(node, ast.Call):
            weighted.append((node, weight))
        for child in ast.iter_child_nodes(node):
            visit(child, weight)

    visit(ast.parse(source))
    for node, weight in weighted:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            if node.func.value.id in ("nisa", "nl"):
                calls[f"{node.func.value.id}.{node.func.attr}"] += weight
    if converter.tuning.isa_lowering:
        expected = converter.stats["tensor_instructions"]
        explicit_transposes = sum(
            weight
            for n, weight in weighted
            if isinstance(n.func, ast.Attribute)
            and n.func.attr == "nc_matmul"
            and any(k.arg == "is_transpose" for k in n.keywords)
        )
        if calls["nisa.nc_matmul"] - explicit_transposes != expected:
            raise ValueError(
                "Emitted matmul source differs from realized instruction record"
            )
        if (
            calls["nisa.nc_transpose"] + explicit_transposes
            != converter.stats["isa_transposes"]
        ):
            raise ValueError(
                "Emitted transpose source differs from realized instruction record"
            )
        if (
            calls["nisa.dma_copy"] + calls["nisa.dma_transpose"]
            != converter.stats["isa_dma_panels"]
        ):
            raise ValueError(
                "Emitted DMA source differs from realized panel record"
            )
    boundary_estimates = []
    hw = converter.hardware
    for b in converter.boundary_records:
        bits = 16 if b["dtype"] in ("float16", "bfloat16") else 32
        nodes = []
        known = b.get("implementation") == "isa.rectangular_pad_slice"
        panels = b.get("panels", [])
        if not known:
            rows, cols = (
                math.prod(b["output_shape"][:-1]),
                b["output_shape"][-1],
            )
            panels = [
                dict(
                    rows=min(128, rows - r),
                    columns=min(128, cols - c),
                    valid_rows=min(128, rows - r),
                    valid_columns=min(128, cols - c),
                    fill=True,
                )
                for r in range(0, rows, 128)
                for c in range(0, cols, 128)
            ]
        for i, p in enumerate(panels):
            previous = None
            if p["fill"]:
                service = max(64, p["columns"]) / (
                    1.4 if hw.name == "trainium-v2" else 0.96
                )
                law = hw.timing_profile.operation(
                    f'nki.memset.{b["dtype"]}.VectorE'
                )
                occupancy, latency = (
                    law.evaluate(service) if law else (service, None)
                )
                previous = len(nodes)
                nodes.append(
                    OperationEvent(
                        f"fill_{i}", "VectorE", occupancy, occupancy, latency
                    )
                )
            for store, rows, cols in [
                (False, p["valid_rows"], p["valid_columns"]),
                (True, p["rows"], p["columns"]),
            ]:
                if not rows or not cols:
                    continue
                ideal = (
                    min(8, rows) * cols * (bits / 8) / (hw.dram_bandwidth / 16)
                )
                graph = transfer_graph(
                    hw,
                    (
                        TransferPanel(
                            0, 0, rows, cols, bits, store, False, False, ideal
                        ),
                    ),
                )
                offset = len(nodes)
                for j, node in enumerate(graph.nodes):
                    deps = tuple(
                        replace(d, source=d.source + offset)
                        for d in node.dependencies
                    )
                    if not deps and previous is not None:
                        deps = (Dependency(previous),)
                    nodes.append(
                        replace(
                            node,
                            name=f"boundary_{i}_{store}_{j}",
                            dependencies=deps,
                        )
                    )
                previous = len(nodes) - 1
        timing = evaluate_graph(RepeatedGraph(tuple(nodes)))
        boundary_estimates.append(
            dict(
                **b,
                predicted_ns=timing.duration_ns,
                dma_resource_lower_bound_ns=timing.resource_bound_ns,
                timing_complete=known and not timing.unknown_latency,
                unmodeled=(
                    [] if known else ["boundary language instruction expansion"]
                ),
                unknown_completion=list(timing.unknown_latency),
            )
        )
    scheduled_reads = sum(x["read_bytes"] for x in converter.transfer_records)
    scheduled_writes = sum(x["write_bytes"] for x in converter.transfer_records)
    boundary_reads = sum(x["read_bytes"] for x in converter.boundary_records)
    boundary_writes = sum(x["write_bytes"] for x in converter.boundary_records)
    constant = 16384 if calls["nisa.nc_transpose"] else 0
    if hasattr(converter, "builder"):
        from .instruction_plan import BITS

        program = converter.builder.program
        constant = sum(
            math.prod(t.shape) * BITS[t.dtype] // 8
            for t in program.tensors.values()
            if t.constant
        )
        if any(
            i.opcode == "nisa.nc_transpose"
            and program.contracts[i.implementation]["engine"] == "TensorE"
            for i in program.instructions
        ):
            constant += 128 * 128
    estimates = record.get("estimates", [])
    fixed = hw.timing_profile.fixed_kernel_ns
    matrix_ns = sum(
        x["predicted_ns"] - x.get("fixed_kernel_ns", 0) for x in estimates
    )
    boundary_bound = sum(
        x["dma_resource_lower_bound_ns"] for x in boundary_estimates
    )
    boundary_ns = sum(x["predicted_ns"] for x in boundary_estimates)
    from .lowering import tile_graph

    groups = Counter(
        (
            x["name"],
            tuple(x["operations"]),
            tuple(x["shape"]),
            tuple(tuple(s) for s in x["inputs"]),
            x["pool_window"],
        )
        for x in converter.vector_records
    )
    vectors = []
    for (
        name,
        operations,
        shape,
        inputs,
        window,
    ), repetitions in groups.items():
        graph = tile_graph(
            hw,
            operations,
            shape,
            inputs,
            converter.tuning,
            window,
            repetitions=repetitions,
        )
        timed = evaluate_graph(RepeatedGraph(graph.nodes, repetitions))
        vectors.append(
            dict(
                name=name,
                operations=operations,
                tile=shape,
                repetitions=repetitions,
                predicted_ns=timed.duration_ns,
                unknown_completion=timed.unknown_latency,
                timing_complete=not bool(timed.unknown_latency),
            )
        )
    vector_ns = sum(x["predicted_ns"] for x in vectors)
    return dict(
        version=3,
        vector_regions=vectors,
        vector_prediction_ns=vector_ns,
        buffer_allocation=converter.tuning.buffer_allocation,
        physical_slot_depth_enforced=False,
        logical_depth_speed_credit=converter.tuning.buffer_allocation
        != "compiler",
        source_calls=dict(calls),
        source_audit="Parsed actual emitted source independently; matmul, transpose and DMA call counts checked",
        physical_audit="Requires NEFF/NTFF; source declarations do not certify physical buffer reuse",
        transfers=converter.transfer_records,
        boundaries=boundary_estimates,
        hbm=dict(
            scheduled_read_bytes=scheduled_reads,
            scheduled_write_bytes=scheduled_writes,
            boundary_read_bytes=boundary_reads,
            boundary_write_bytes=boundary_writes,
            constant_read_bytes=constant,
            total_bytes=scheduled_reads
            + scheduled_writes
            + boundary_reads
            + boundary_writes
            + constant,
            excluded="backend spills, scalar constants, runtime metadata traffic",
        ),
        matrix_prediction_ns=matrix_ns + fixed if estimates else None,
        boundary_dma_resource_lower_bound_ns=boundary_bound,
        whole_program_prediction_ns=(
            matrix_ns + vector_ns + fixed + boundary_ns
            if estimates or vectors
            else None
        ),
        boundary_prediction_ns=boundary_ns,
        boundary_composition="Sequential boundary regions around matrix work; no credit for cross-region overlap",
        whole_program_timing_complete=bool(estimates or vectors)
        and all(x["timing_complete"] for x in vectors)
        and all(x["timing_complete"] for x in boundary_estimates)
        and all(
            (x.get("dependency_model") or {}).get("timing_complete", False)
            for x in estimates
        ),
        timing_scope="One fixed device term plus matrix bodies and explicit boundary panel graphs; unknown timings are reported",
    )
