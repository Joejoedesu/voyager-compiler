"""Recipe builders and pre-allocation storage refinement.

Build only the candidate passed by the shared planner. Declare execution and
storage ownership here; final allocation and selection stay in shared stages.
"""

import torch

from voyager_compiler.compiler.bufferize.invocation import build_native
from voyager_compiler.compiler.bufferize.residency import build_whole
from voyager_compiler.ir.buffer import TargetInvoke

from .scheduling import cost, footprint, raw_requirements
from .semantics import facts, raw_dimensions


def build_raw_explicit(node, context, parameters):
    raw_requirements(node, context, False)
    graph = build_whole(node, context)
    compute = next(n for n in graph.graph.nodes if n.target == node.target)
    compute.meta["target_invoke"] = TargetInvoke(
        "gemmini.explicit",
        "ws_matmul",
        (),
        (),
        (),
        resources=("matrix",),
        scheduling_owner="compiler",
    )
    return graph


def build_raw_loop(node, context, parameters):
    demand = raw_requirements(node, context, True)
    m, n, k = raw_dimensions(node)
    invocation = TargetInvoke(
        "gemmini.hardware_loop",
        "loop_ws",
        (),
        (),
        (),
        resources=("matrix", "loop_contexts"),
        allocation_owner="native",
        scheduling_owner="hardware",
        implicit_traffic_bytes=m * k + k * n + 4 * m * n,
        native_requirements=demand,
    )
    return build_native(node, invocation)


def build_native_kernel(node, context, parameters):
    kind, p, _ = facts(node)
    tile, mode = parameters["tile"], parameters["mode"]
    invocation = TargetInvoke(
        f"gemmini.{kind}.{mode}",
        kind,
        (),
        (),
        (),
        resources=(
            ("matrix", "scratchpad", "accumulator")
            if kind in ("matmul", "conv2d")
            else ("dma", "accumulator")
        ),
        allocation_owner="native",
        scheduling_owner="hardware" if mode == "hardware_loop" else "compiler",
        implicit_traffic_bytes=cost(kind, p, tile, mode).dram_bytes,
        native_requirements=footprint(
            kind, p, tile, mode, context.tiler.config
        ),
    )
    return build_native(node, invocation)


def refine(program, config):
    from voyager_compiler.compiler.allocate import _walk

    for node in _walk(program.graph):
        if (
            node.target == torch.ops.voyager.alloc.default
            and node.meta.get("space") == "Scratchpad"
        ):
            node.meta["memory_instance"] = (
                "accumulator"
                if node.value.dtype == torch.int32
                else "scratchpad"
            )


def lower_operations(prepared, config, options):
    """Lower normalized Q/DQ boundaries into native INT8 operations.

    Scales and packed weight layouts come from shared PT2E/normalization. This
    pass neither observes tensors nor chooses new scales or tiles. The independent
    converted graph supplies the numerical reference before any target rewriting.
    """
    from torch.fx import Graph, GraphModule
    from torch.utils._pytree import tree_flatten, tree_unflatten

    from voyager_compiler.codegen.transform.rewrites import (
        extract_input_preprocessor,
    )
    from voyager_compiler.shape_prop import fetch_attr

    from .semantics import OPS

    aten, qops, native = (
        torch.ops.aten,
        torch.ops.quantized_ops,
        torch.ops.gemmini,
    )
    gm = prepared.graph
    if any(n.target in (aten._int_mm.default, *OPS) for n in gm.graph.nodes):
        return
    for node in gm.graph.nodes:
        if node.target not in (aten.relu_.default, aten.add_.Tensor):
            continue
        source = node.args[0]
        while True:
            if source.op == "placeholder" or len(source.users) != 1:
                raise NotImplementedError(
                    "Gemmini in-place operations require exclusive temporaries"
                )
            if source.target not in (
                qops.dequantize.default,
                qops.quantize_symmetric.default,
            ):
                break
            source = source.args[0]
    with torch.no_grad():
        reference, _ = tree_flatten(gm(*prepared.args, **prepared.kwargs))
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    flat_args, _ = tree_flatten((prepared.args, prepared.kwargs))
    input_contract, inputs = [], []
    # The shared helper extracts a single, unbranched boundary preprocessing chain.
    for old, value in zip(placeholders, flat_args):
        cursor = old
        while len(cursor.users) == 1:
            user = next(iter(cursor.users))
            if user.target == qops.quantize_symmetric.default:
                break
            if user.target not in (
                aten.permute.default,
                aten.reshape.default,
                aten.view.default,
            ):
                raise NotImplementedError(
                    "Gemmini input requires an explicit INT8 quantization boundary"
                )
            cursor = user
        else:
            raise NotImplementedError(
                "Gemmini input preprocessing requires one quantized input domain"
            )
        if tuple(user.args[2:]) != (-128, 127):
            raise NotImplementedError(
                "Gemmini inputs require signed INT8 boundaries"
            )
        scale = float(fetch_attr(gm, user.args[1].target))
        gm, preprocessor = extract_input_preprocessor(
            gm, old.target, additional_ops=(qops.quantize_symmetric.default,)
        )
        packed = preprocessor(value).to(torch.int8).contiguous()
        inputs.append(packed)
        input_contract.append(
            dict(
                name=old.name,
                scale=scale,
                logical_shape=list(value.shape),
                physical_shape=list(packed.shape),
            )
        )
    graph, root = Graph(), torch.nn.Module()
    cache, constants = {}, {}
    packed = {}
    for node, value in zip(
        (n for n in gm.graph.nodes if n.op == "placeholder"), inputs
    ):
        packed[node] = graph.placeholder(node.name)

    def scalar(node):
        value = fetch_attr(gm, node.target)
        if (
            value.numel() != 1
            or not torch.isfinite(value).all()
            or value.item() <= 0
        ):
            raise NotImplementedError(
                "Gemmini requires positive finite per-tensor scales"
            )
        return float(value)

    def constant(node, dtype):
        key = (node, dtype)
        if key not in constants:
            value = fetch_attr(gm, node.target).detach()
            bounds = torch.iinfo(dtype)
            if (
                not torch.isfinite(value).all()
                or not torch.equal(value, value.round())
                or value.min() < bounds.min
                or value.max() > bounds.max
            ):
                raise ValueError(
                    f"Gemmini constant {node.name} does not fit {dtype}"
                )
            name = f"constant_{len(constants)}"
            root.register_buffer(name, value.to(dtype).contiguous())
            constants[key] = graph.get_attr(name)
        return constants[key]

    def parameters(node):
        p = {a.name: a.default_value for a in node.target._schema.arguments}
        p.update(zip(p, node.args))
        p.update(node.kwargs)
        return p

    def pair(value):
        v = (value, value) if isinstance(value, int) else tuple(value)
        if len(v) != 2 or v[0] != v[1]:
            raise NotImplementedError(
                "Gemmini requires equal spatial parameters"
            )
        return v[0]

    transparent = (
        aten.reshape.default,
        aten.view.default,
        aten.flatten.using_ints,
        aten.permute.default,
        aten.clone.default,
        aten.detach.default,
        aten.alias.default,
    )

    def nonnegative(node):
        if node.target in (aten.relu.default, aten.relu_.default):
            return True
        if node.target in (
            *transparent,
            qops.dequantize.default,
            qops.quantize_symmetric.default,
        ):
            return nonnegative(node.args[0])
        return False

    def domain(node):
        if node.target == qops.dequantize.default:
            return scalar(node.args[1]) * domain(node.args[0])
        if node.target == qops.quantize_symmetric.default or node in packed:
            return 1.0
        if node.target in transparent:
            return domain(node.args[0])
        raise NotImplementedError(
            f"Gemmini requires an explicit quantization boundary at {node.name}"
        )

    def emit(node, scale):
        """Encode node's real value in the requested domain, preserving every Q."""
        key = (node, scale)
        if key in cache:
            return cache[key]
        if node in packed:
            result = packed[node]
            if scale != 1:
                result = graph.call_function(
                    native.requantize.default, (result, 1 / scale)
                )
        elif node.target == qops.dequantize.default:
            if any(v is not None for v in node.args[2:]):
                raise NotImplementedError(
                    "Gemmini requires symmetric direct integer conversion"
                )
            result = emit(node.args[0], scale / scalar(node.args[1]))
        elif node.target == qops.quantize_symmetric.default:
            if tuple(node.args[2:]) != (-128, 127):
                raise NotImplementedError(
                    "Gemmini deployment requires signed INT8 boundaries"
                )
            result = emit(node.args[0], scalar(node.args[1]))
            if scale != 1:
                result = graph.call_function(
                    native.requantize.default, (result, 1 / scale)
                )
        elif node.target in (qops.linear.default, qops.conv2d.default):
            p = parameters(node)
            x = emit(node.args[0], 1.0)
            w = constant(node.args[1], torch.int8)
            if len(node.args) > 2 and node.args[2] is not None:
                b = constant(node.args[2], torch.int32)
            else:
                name = node.name + "_zero_bias"
                width = fetch_attr(gm, node.args[1].target).shape[-1]
                root.register_buffer(
                    name, torch.zeros(width, dtype=torch.int32)
                )
                b = graph.get_attr(name)
            if node.target == qops.linear.default:
                if p["layout"] != "xc,ck" or len(node.args[0].value.shape) != 2:
                    raise NotImplementedError(
                        "Gemmini linear requires shared xc,ck layout and rank-2 activation"
                    )
                result = graph.call_function(
                    native.matmul.default, (x, w, b, 1 / scale)
                )
            else:
                weight = fetch_attr(gm, node.args[1].target)
                if (
                    p["groups"] != 1
                    or pair(p["dilation"]) != 1
                    or weight.shape[0] != weight.shape[1]
                ):
                    raise NotImplementedError(
                        "Gemmini requires square dense undilated convolution"
                    )
                result = graph.call_function(
                    native.conv2d.default,
                    (x, w, b, pair(p["stride"]), pair(p["padding"]), 1 / scale),
                )
        elif node.target in (aten.relu.default, aten.relu_.default):
            result = graph.call_function(
                native.requantize.default,
                (emit(node.args[0], scale), 1.0, True),
            )
        elif node.target in (aten.add.Tensor, aten.add_.Tensor):
            p = parameters(node)
            if (
                p["alpha"] != 1
                or node.args[0].value.shape != node.args[1].value.shape
            ):
                raise NotImplementedError(
                    "Gemmini residual requires equal shapes and alpha=1"
                )
            if any(domain(n) != scale for n in node.args[:2]):
                raise NotImplementedError(
                    "Gemmini residual requires a shared input/output quantization domain; configure shared observers"
                )
            result = graph.call_function(
                native.add.default,
                (
                    emit(node.args[0], scale),
                    emit(node.args[1], scale),
                    1.0,
                    1.0,
                ),
            )
        elif node.target == qops.adaptive_avg_pool2d.default:
            if pair(node.args[1]) != 1 or domain(node.args[0]) != scale:
                raise NotImplementedError(
                    "Gemmini global average requires a preserved quantization domain"
                )
            result = graph.call_function(
                native.average.default, (emit(node.args[0], scale),)
            )
        elif node.target == qops.max_pool2d.default:
            p = parameters(node)
            src = node.args[0]
            if p["ceil_mode"] or pair(p["dilation"]) != 1:
                raise NotImplementedError(
                    "Gemmini maxpool requires floor geometry and dilation=1"
                )
            k, pad = pair(p["kernel_size"]), pair(p["padding"])
            if pad and not nonnegative(src):
                raise NotImplementedError(
                    "Gemmini zero-padded maxpool requires proven nonnegative input"
                )
            result = graph.call_function(
                native.maxpool.default,
                (
                    emit(src, scale),
                    k,
                    pair(p["stride"]) if p["stride"] else k,
                    pad,
                    nonnegative(src),
                ),
            )
        elif node.target in transparent:
            src = emit(node.args[0], scale)
            result = graph.node_copy(
                node,
                lambda n: (
                    src if n is node.args[0] else constant(n, torch.int64)
                ),
            )
        else:
            raise NotImplementedError(
                f"Unsupported Gemmini operation: {node.format_node()}"
            )
        cache[key] = result
        return result

    output = next(n for n in gm.graph.nodes if n.op == "output")
    leaves, spec = tree_flatten(output.args[0])
    scales = [domain(n) for n in leaves]
    lowered = [emit(n, s) for n, s in zip(leaves, scales)]
    graph.output(tree_unflatten(lowered, spec))
    result = GraphModule(root, graph)
    expected = tree_unflatten(
        [(v / s).round().to(torch.int8) for v, s in zip(reference, scales)],
        spec,
    )
    with torch.no_grad():
        torch.testing.assert_close(result(*inputs), expected, rtol=0, atol=0)
    prepared.graph, prepared.args, prepared.kwargs = result, tuple(inputs), {}
    prepared.reference = expected
    prepared.provenance["gemmini_quantization"] = dict(
        inputs=input_contract,
        outputs=[dict(name=n.name, scale=s) for n, s in zip(leaves, scales)],
        comparison="independent shared converted graph; exact integer output comparison",
    )
