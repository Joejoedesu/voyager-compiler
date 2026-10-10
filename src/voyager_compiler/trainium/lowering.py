"""Early ISA recipes shared by tiling, bufferization costing and realization.

Reduction regions remain atomic for tiling: splitting their reduction axis would
change semantics. Their explicit local SSA steps describe temporary lifetimes;
the converter only supplies the tile's concrete addresses and emits these steps.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    name: str
    instruction: str
    inputs: tuple[str, ...]
    shape: str = "tile"
    op: str = ""
    scalar: str = ""


RECIPES = {
    "softmax": (
        Step("maximum", "reduce", ("input",), "row", "max"),
        Step("shifted", "scalar", ("input", "maximum"), op="subtract"),
        Step("exponential", "activation", ("shifted",), op="exp"),
        Step("sum", "reduce", ("exponential",), "row", "add"),
        Step("inverse", "reciprocal", ("sum",), "row"),
        Step("result", "scalar", ("exponential", "inverse"), op="multiply"),
    ),
    "rms_norm": (
        Step("square", "binary", ("input", "input"), op="multiply"),
        Step("sum", "reduce", ("square",), "row", "add"),
        Step("variance", "scale_epsilon", ("sum",), "row"),
        Step("inverse", "activation", ("variance",), "row", "rsqrt"),
        Step("normalized", "scalar", ("input", "inverse"), op="multiply"),
        Step("result", "parameter", ("normalized", "weight"), op="multiply"),
    ),
    "layer_norm": (
        Step("sum", "reduce", ("input",), "row", "add"),
        Step("mean", "scalar", ("sum",), "row", "multiply", "inverse_width"),
        Step("centered", "scalar", ("input", "mean"), op="subtract"),
        Step("square", "binary", ("centered", "centered"), op="multiply"),
        Step("variance_sum", "reduce", ("square",), "row", "add"),
        Step("variance", "scale_epsilon", ("variance_sum",), "row"),
        Step("inverse", "activation", ("variance",), "row", "rsqrt"),
        Step("normalized", "scalar", ("centered", "inverse"), op="multiply"),
        Step("scaled", "parameter", ("normalized", "weight"), op="multiply"),
        Step("result", "parameter", ("scaled", "bias"), op="add"),
    ),
}


def recipe_for(name, tuning=None):
    if name != "layer_norm" or tuning is None:
        return RECIPES[name]
    from .normalization import layernorm_recipe

    return layernorm_recipe(
        Step,
        algorithm=tuning.layernorm_algorithm,
        fused=tuning.layernorm_fused,
        square_engine=tuning.layernorm_square_engine,
    )


def operation_name(target):
    parts = str(target).replace("::", ".").split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def prepare_graph(model, hardware):
    """Bind before fusion/tiling; expose SiLU's two real ISA operations early."""
    import torch

    for node in list(model.graph.nodes):
        if node.op != "call_function":
            continue
        name = operation_name(node.target)
        if name in RECIPES:
            impl = hardware.operation_implementation(f"nki.{name}.float32")
            node.meta["operation_implementation"] = impl.name
            # A row-partition reduction consumes the exact logical feature
            # extent. SIMD width is a service fact, not a padding requirement.
            node.meta["vector_padding_alignment"] = 1
        if node.target == torch.ops.aten.silu.default:
            with model.graph.inserting_before(node):
                sigmoid = model.graph.call_function(
                    torch.ops.aten.sigmoid.default, (node.args[0],)
                )
                product = model.graph.call_function(
                    torch.ops.aten.mul.Tensor, (node.args[0], sigmoid)
                )
            node.replace_all_uses_with(product)
            model.graph.erase_node(node)
    from voyager_compiler.shape_prop import propagate_shape

    for node in model.graph.nodes:
        if node.op == "call_function" and not hasattr(node, "value"):
            propagate_shape(node, model)
    model.graph.lint()
    model.recompile()


def expression(step, values, width, epsilon):
    args = [values[x] for x in step.inputs]
    if step.instruction == "first":
        return f"nisa.tensor_copy({args[0]}[:,0:1], engine=nisa.vector_engine)"
    if step.instruction == "difference_epsilon":
        return f"nisa.tensor_scalar({args[0]}, op0=nl.subtract, operand0={args[1]}, op1=nl.add, operand1={epsilon}, engine=nisa.vector_engine)"
    if step.instruction == "center_scale":
        return f"nisa.tensor_scalar({args[0]}, op0=nl.subtract, operand0={args[1]}, op1=nl.multiply, operand1={args[2]}, engine=nisa.vector_engine)"
    if step.instruction == "reduce":
        return f"nisa.tensor_reduce(nl.{step.op}, {args[0]}, axis=[1], keepdims=True, dtype=nl.float32)"
    if step.instruction == "binary":
        return f"nisa.tensor_tensor({args[0]}, {args[1]}, op=nl.{step.op}, engine=nisa.vector_engine)"
    if step.instruction == "scalar":
        operand = repr(1.0 / width) if step.scalar else args[1]
        return f"nisa.tensor_scalar({args[0]}, op0=nl.{step.op}, operand0={operand}, engine=nisa.vector_engine)"
    if step.instruction == "scale_epsilon":
        return f"nisa.tensor_scalar({args[0]}, op0=nl.multiply, operand0={1.0/width}, op1=nl.add, operand1={epsilon}, engine=nisa.vector_engine)"
    if step.instruction == "activation":
        return f"nisa.activation(op=nl.{step.op}, data={args[0]})"
    if step.instruction == "reciprocal":
        return f"nisa.reciprocal({args[0]})"
    raise ValueError(step)


def reduction_graph(hardware, name, rows, width, tuning=None):
    """Concrete instruction dependencies and service; absent completion stays unknown."""
    from voyager_compiler.codegen.transform.tiling.execution import (
        OperationEvent,
        Dependency,
        RepeatedGraph,
    )
    import math

    nodes, values = [], {}
    for step in recipe_for(name, tuning):
        engine = "ScalarE" if step.instruction == "activation" else "VectorE"
        free = (
            width if step.shape == "tile" or step.instruction == "reduce" else 1
        )
        cycles = (
            math.ceil(width / 128) * max(64, rows)
            if step.instruction == "parameter"
            else max(
                64,
                (
                    8
                    if step.instruction == "reciprocal"
                    else 2 if step.instruction == "binary" else 1
                )
                * free,
            )
        )
        service = cycles / (1.2 if engine == "ScalarE" else 0.96)
        instruction = {
            "first": "copy",
            "difference_epsilon": "scalar",
            "center_scale": "scalar",
        }.get(step.instruction, step.instruction)
        impl = f"nki.{instruction}.float32.{engine}"
        law = hardware.timing_profile.operation(impl)
        occupancy, latency = law.evaluate(service) if law else (service, None)
        dependencies = tuple(
            Dependency(values[x])
            for x in dict.fromkeys(step.inputs)
            if x in values
        )
        values[step.name] = len(nodes)
        nodes.append(
            OperationEvent(
                step.name,
                engine,
                occupancy,
                occupancy,
                latency,
                dependencies=dependencies,
                implementation=impl,
            )
        )
    return RepeatedGraph(tuple(nodes))


def realize_reduction(converter, name, kwargs, destination):
    """Realize the early recipe within the shared bufferizer's selected tile."""
    import math
    from .planning import PanelValue, strides

    c = converter
    shape = kwargs["input"].shape
    width, rows = shape[-1], math.prod(shape[:-1])
    if (
        name == "softmax"
        and kwargs.get("dim", -1) % len(shape) != len(shape) - 1
    ):
        raise NotImplementedError(
            "Trainium reduction must use the last logical axis"
        )
    if name != "softmax" and tuple(kwargs["normalized_shape"]) != (width,):
        raise NotImplementedError(
            "Trainium normalization requires one full, unpadded reduction axis"
        )
    if destination.dtype != "float32":
        raise NotImplementedError(
            "Reduction ISA recipe is currently validated for FP32"
        )
    c.used_implementations.add(f"nki.{name}.float32")
    if hasattr(c, "row_buffers") and kwargs["input"].name in c.row_buffers:
        values = {"input": c.read(kwargs["input"])}
        for step in recipe_for(name, c.tuning):
            if step.instruction == "parameter":
                parameter = kwargs.get(step.inputs[1])
                if parameter is None:
                    values[step.name] = values[step.inputs[0]]
                else:
                    expanded = c.replicate_parameter(parameter, rows, width)
                    values[step.name] = c.tmp(
                        f"nisa.tensor_tensor({values[step.inputs[0]]}, {expanded}, op=nl.{step.op}, engine=nisa.vector_engine)"
                    )
            else:
                values[step.name] = c.tmp(
                    expression(step, values, width, kwargs.get("eps", 1e-5))
                )
        c.reduction_records.append(
            dict(
                operation=name,
                rows=rows,
                width=width,
                parameters=sum(
                    kwargs.get(x) is not None for x in ("weight", "bias")
                ),
                layout="row_partition",
            )
        )
        return values["result"]
    panels = []
    for row in range(0, rows, 128):
        count = min(128, rows - row)
        values = {}
        for argument in ("input",):
            ref = kwargs.get(argument)
            if ref is None:
                if argument == "input":
                    raise ValueError("Missing reduction input")
                continue
            local = c.tmp(
                f"nl.ndarray(({count}, {width}), dtype=nl.float32, buffer=nl.sbuf)"
            )
            for col in range(0, width, 128):
                extent = min(128, width - col)
                ip = c.tmp(f"nl.arange({extent})[:, None]")
                jf = c.tmp(f"nl.arange({count})[None, :]")
                if argument == "input":
                    linear = f"(({jf}+{row})*{width}+{ip}+{col})"
                    coords = [
                        f"(({linear}//{stride})%{extent})"
                        for stride, extent in zip(strides(shape), shape)
                    ]
                else:
                    coords = [f"({ip}+{col})"]
                tile = c.local_copy(c.tmp(c.index(ref, coords)))
                if argument != "input" and count != 1:
                    tile = c.local_copy(
                        f"{tile}.broadcast_to(({extent}, {count}))"
                    )
                tile = c.transpose(tile)
                c.emit(
                    f"{local}[:, {col}:{col+extent}] = nisa.tensor_copy({tile}, engine=nisa.vector_engine)"
                )
            values[argument] = local
        for step in recipe_for(name, c.tuning):
            if step.instruction == "parameter":
                values[step.name] = values[step.inputs[0]]
            else:
                values[step.name] = c.tmp(
                    expression(step, values, width, kwargs.get("eps", 1e-5))
                )
        for col in range(0, width, 128):
            extent = min(128, width - col)
            tile = c.tmp(f"{values['result']}[:, {col}:{col+extent}]")
            tile = c.local_copy(tile)
            tile = c.transpose(tile)
            for step in recipe_for(name, c.tuning):
                if step.instruction != "parameter":
                    continue
                parameter = kwargs.get(step.inputs[1])
                if parameter is not None:
                    ip = c.tmp(f"nl.arange({extent})[:, None]")
                    scalar = c.local_copy(
                        c.tmp(c.index(parameter, [f"({ip}+{col})"]))
                    )
                    tile = c.tmp(
                        f"nisa.tensor_scalar({tile}, op0=nl.{step.op}, operand0={scalar}, engine=nisa.vector_engine)"
                    )
            panels.append((row, col, count, extent, tile))
        c.reduction_records.append(
            dict(
                operation=name,
                rows=count,
                width=width,
                parameters=sum(
                    kwargs.get(x) is not None for x in ("weight", "bias")
                ),
            )
        )
    return PanelValue(rows, width, panels)


def reduction_workspace(name, width, parameters=0, tuning=None):
    """Per-partition allocation envelope for the early recipe's live SSA values.

    All 128 partitions are reserved conservatively, including partial row tiles.
    Reuse begins only after the last reader; input/output conversion temporaries
    add two 128-wide panels. This is workspace, not fictitious hardware capacity.
    """
    recipe = recipe_for(name, tuning)
    last = {
        value: max(i for i, step in enumerate(recipe) if value in step.inputs)
        for value in {x for step in recipe for x in step.inputs}
    }
    live = {"input": width}
    peak = sum(live.values())
    for i, step in enumerate(recipe):
        if step.instruction == "parameter":
            continue
        live[step.name] = width if step.shape == "tile" else 1
        peak = max(peak, sum(live.values()))
        for value in tuple(live):
            if last.get(value, len(recipe)) == i:
                del live[value]
    return 128 * (16 * ((4 * peak + 15) // 16) + 2 * 128 * 4)


def tile_graph(
    hardware, operations, shape, inputs, tuning, pool_window=1, repetitions=1
):
    """Instruction expansion used during tile selection, before bufferization.

    ``inputs`` are scheduled (not broadcast) operand shapes. The template owns
    loads, local layout, dependent ISA operations, and stores. Repetitions are
    supplied by the shared tiler. Queue admission is the existing DMA contract.
    """
    import math
    from dataclasses import replace
    from .cost import dma_service
    from .movement import transfer_graph
    from voyager_compiler.codegen.transform.tiling.execution import (
        OperationEvent,
        Dependency,
        RepeatedGraph,
    )

    if len(operations) == 1 and operations[0] in RECIPES:
        return row_reduction_graph(
            hardware, operations[0], shape, inputs, tuning, repetitions
        )
    if (
        operations == ("max_pool2d",)
        and len(shape) == 4
        and shape[0] == shape[-1] == 1
    ):
        return spatial_pool_graph(
            hardware, shape, inputs[0], tuning, pool_window
        )
    nodes = []

    def append(graph, dependencies=()):
        offset = len(nodes)
        for node in graph.nodes:
            deps = tuple(
                replace(d, source=d.source + offset) for d in node.dependencies
            )
            if not deps:
                deps = tuple(Dependency(i) for i in dependencies)
            nodes.append(
                replace(
                    node, name=f"n{len(nodes)}_{node.name}", dependencies=deps
                )
            )
        if len(nodes) == offset:
            return tuple(dependencies)
        # One explicit completion join avoids duplicating every predecessor
        # edge on every layout command (quadratic graph size).
        end = len(nodes)
        nodes.append(
            OperationEvent(
                f"join_{end}",
                "control",
                0,
                0,
                0,
                dependencies=tuple(Dependency(i) for i in range(offset, end)),
            )
        )
        return (end,)

    ready = []
    for operand in inputs:
        service = dma_service(
            hardware,
            math.prod(operand[:-1]),
            operand[-1],
            32,
            (
                min(128, operand[-2])
                if len(operand) > 2
                else math.prod(operand[:-1])
            ),
            tuning=tuning,
        )
        ready.extend(append(transfer_graph(hardware, service.panels)))
    for name in operations:
        if name in RECIPES:
            rows, width = math.prod(shape[:-1]), shape[-1]
            parameters = (
                2 if name == "layer_norm" else 1 if name == "rms_norm" else 0
            )
            # Local partition-to-free transpose and assembly are explicit work.
            for row in range(0, rows, 128):
                count = min(128, rows - row)
                layouts = []
                for _ in range(1):
                    for col in range(0, width, 128):
                        layouts.extend(
                            append(
                                layout_graph(
                                    hardware, min(128, width - col), count
                                ),
                                ready,
                            )
                        )
                computed = append(
                    reduction_graph(hardware, name, count, width, tuning),
                    layouts,
                )
                returned = []
                for col in range(0, width, 128):
                    returned.extend(
                        append(
                            layout_graph(
                                hardware, count, min(128, width - col)
                            ),
                            computed,
                        )
                    )
                # Each row group is independent; resource occupancy couples it.
                ready_for_store = (
                    returned if row == 0 else ready_for_store + returned
                )
            ready = ready_for_store
        else:
            free = math.ceil(math.prod(shape) / min(128, shape[-1]))
            engine = (
                "ScalarE"
                if name in ("exp", "sigmoid", "relu", "tanh")
                else "VectorE"
            )
            count = max(0, pool_window - 1) if name == "max_pool2d" else 1
            for i in range(count):
                service = max(64, free if engine == "ScalarE" else 2 * free) / (
                    1.2 if engine == "ScalarE" else 0.96
                )
                impl = (
                    f"nki.activation.float32.{engine}"
                    if engine == "ScalarE"
                    else "nki.binary.float32.VectorE"
                )
                law = hardware.timing_profile.operation(impl)
                occupancy, latency = (
                    law.evaluate(service) if law else (service, None)
                )
                ready = append(
                    RepeatedGraph(
                        (
                            OperationEvent(
                                name,
                                engine,
                                occupancy,
                                occupancy,
                                latency,
                                implementation=impl,
                            ),
                        )
                    ),
                    ready,
                )
    output = dma_service(
        hardware,
        math.prod(shape[:-1]),
        shape[-1],
        32,
        min(128, shape[-2]) if len(shape) > 2 else math.prod(shape[:-1]),
        store=True,
        tuning=tuning,
    )
    append(transfer_graph(hardware, output.panels), ready)
    return RepeatedGraph(tuple(nodes))


def row_reduction_graph(hardware, name, shape, inputs, tuning, repetitions):
    """Row-partition realization and retained operands, shared with selection."""
    import math
    from dataclasses import replace
    from .movement import TransferPanel, transfer_graph, partition_payload_ns
    from voyager_compiler.codegen.transform.tiling.execution import (
        OperationEvent,
        Dependency,
        RepeatedGraph,
    )

    nodes = []

    def transfer(operand, store=False, deps=(), period=1):
        rows, width = math.prod(operand[:-1]), operand[-1]
        boundary = operand[-2] if len(operand) > 2 else rows
        panels = tuple(
            TransferPanel(
                r,
                c,
                min(boundary, rows - r),
                min(tuning.dma_columns, width - c),
                32,
                store,
                False,
                False,
                partition_payload_ns(
                    hardware,
                    min(boundary, rows - r),
                    min(tuning.dma_columns, width - c),
                    32,
                ),
            )
            for r in range(0, rows, boundary)
            for c in range(0, width, tuning.dma_columns)
        )
        graph = transfer_graph(hardware, panels)
        start = len(nodes)
        for n in graph.nodes:
            dependencies = tuple(
                replace(d, source=d.source + start) for d in n.dependencies
            ) or tuple(Dependency(x) for x in deps)
            nodes.append(
                replace(
                    n,
                    name=f"n{len(nodes)}_{n.name}",
                    dependencies=dependencies,
                    period=period,
                )
            )
        return tuple(range(start, len(nodes)))

    ready = []
    for operand in inputs:
        loaded = transfer(
            operand, period=repetitions if len(operand) == 1 else 1
        )
        if len(operand) == 1:
            # Explicit ones x retained row parameter, followed by PSUM copy.
            rows = math.prod(shape[:-1])
            ones = len(nodes)
            nodes.append(
                OperationEvent(
                    f"parameter_ones_{ones}",
                    "VectorE",
                    max(64, rows) / 0.96,
                    max(64, rows) / 0.96,
                    None,
                    period=repetitions,
                    implementation="nki.isa.memset.VectorE",
                )
            )
            previous = (*loaded, ones)
            for col in range(0, operand[-1], 512):
                extent = min(512, operand[-1] - col)
                for engine, service, impl in (
                    (
                        "VectorE",
                        max(64, extent) / 0.96,
                        "nki.memset.float32.VectorE",
                    ),
                    (
                        "TensorE",
                        4 * max(min(64, rows), extent) / hardware.frequency,
                        "nki.matmul.float32",
                    ),
                    (
                        "ScalarE",
                        max(64, extent) / 1.2,
                        "nki.copy.PSUM.float32.ScalarE",
                    ),
                ):
                    law = hardware.timing_profile.operation(impl)
                    occupancy, latency = (
                        law.evaluate(service) if law else (service, None)
                    )
                    nodes.append(
                        OperationEvent(
                            f"parameter_{len(nodes)}",
                            engine,
                            occupancy,
                            occupancy,
                            latency,
                            dependencies=tuple(Dependency(x) for x in previous),
                            period=repetitions,
                            implementation=impl,
                        )
                    )
                    previous = (len(nodes) - 1,)
            ready.extend(previous)
        else:
            ready.extend(loaded)
    start = len(nodes)
    graph = reduction_graph(
        hardware, name, math.prod(shape[:-1]), shape[-1], tuning
    )
    for step, n in zip(recipe_for(name, tuning), graph.nodes):
        dependencies = tuple(
            replace(d, source=d.source + start) for d in n.dependencies
        )
        dependencies += (
            tuple(Dependency(i) for i in ready)
            if not dependencies or step.instruction == "parameter"
            else ()
        )
        if step.instruction == "parameter":
            service = max(64, 2 * shape[-1]) / 0.96
            law = hardware.timing_profile.operation(
                "nki.binary.float32.VectorE"
            )
            occupancy, latency = (
                law.evaluate(service) if law else (service, None)
            )
            n = replace(
                n,
                occupancy_ns=occupancy,
                issue_ns=occupancy,
                latency_ns=latency,
            )
        nodes.append(
            replace(
                n, name=f"n{len(nodes)}_{n.name}", dependencies=dependencies
            )
        )
    transfer(shape, store=True, deps=(len(nodes) - 1,))
    return RepeatedGraph(tuple(nodes))


def spatial_pool_graph(hardware, shape, operand, tuning, window):
    import math
    from dataclasses import replace
    from .movement import TransferPanel, transfer_graph, partition_payload_ns
    from voyager_compiler.codegen.transform.tiling.execution import (
        OperationEvent,
        Dependency,
        RepeatedGraph,
    )
    from .operations import pool_reduction_steps

    rows, width = shape[1:3]
    kh = operand[1] - rows + 1
    nodes = []

    def transfer(free, count, store=False, deps=()):
        panels = tuple(
            TransferPanel(
                r,
                c,
                rows,
                min(tuning.dma_columns, free - c),
                32,
                store,
                False,
                False,
                partition_payload_ns(
                    hardware, rows, min(tuning.dma_columns, free - c), 32
                ),
            )
            for r in range(count)
            for c in range(0, free, tuning.dma_columns)
        )
        start = len(nodes)
        for n in transfer_graph(hardware, panels).nodes:
            edges = tuple(
                replace(d, source=d.source + start) for d in n.dependencies
            ) or tuple(Dependency(i) for i in deps)
            nodes.append(
                replace(n, name=f"n{len(nodes)}_{n.name}", dependencies=edges)
            )
        return tuple(range(start, len(nodes)))

    ready = transfer(operand[2], kh)
    kw = operand[2] - width + 1
    if kh * kw != window:
        raise ValueError("Pool halo differs from selected window")
    for i, (_, _, _, free) in enumerate(
        pool_reduction_steps(kh, kw, operand[2])
    ):
        service = max(64, 2 * free) / 0.96
        law = hardware.timing_profile.operation("nki.binary.float32.VectorE")
        occupancy, latency = law.evaluate(service) if law else (service, None)
        nodes.append(
            OperationEvent(
                f"maximum{i}",
                "VectorE",
                occupancy,
                occupancy,
                latency,
                dependencies=tuple(Dependency(j) for j in ready),
                implementation="nki.binary.float32.VectorE",
            )
        )
        ready = (len(nodes) - 1,)
    stored = transfer(width, 1, True, ready)
    # Selected physical generations reuse the tile arena. A following halo
    # load cannot overwrite it until the preceding store has consumed output.
    # Do not credit independent iterations with unlimited local storage.
    for i, n in enumerate(nodes):
        if n.resource == "DMAIssue" and not n.dependencies:
            nodes[i] = replace(
                n, dependencies=(Dependency(stored[-1], distance=1),)
            )
    return RepeatedGraph(tuple(nodes))


def layout_graph(hardware, partitions, free):
    """Gather, TensorE transpose, PSUM eviction, and SBUF assembly."""
    from voyager_compiler.codegen.transform.tiling.execution import (
        OperationEvent,
        Dependency,
        RepeatedGraph,
    )

    values = (
        ("VectorE", max(64, free) / 0.96, "nki.gather.float32.VectorE"),
        (
            "TensorE",
            max(partitions, min(64, free)) / hardware.frequency,
            "nki.transpose_copy.float32.ScalarE",
        ),
        ("ScalarE", max(64, partitions) / 1.2, "nki.copy.PSUM.float32.ScalarE"),
        (
            "VectorE",
            max(64, partitions) / 0.96,
            "nki.copy.SBUF.float32.VectorE",
        ),
    )
    nodes = []
    for i, (engine, service, impl) in enumerate(values):
        law = hardware.timing_profile.operation(impl)
        occupancy, latency = law.evaluate(service) if law else (service, None)
        nodes.append(
            OperationEvent(
                f"layout_{i}",
                engine,
                occupancy,
                occupancy,
                latency,
                dependencies=() if not i else (Dependency(i - 1),),
                implementation=impl,
            )
        )
    return RepeatedGraph(tuple(nodes))
