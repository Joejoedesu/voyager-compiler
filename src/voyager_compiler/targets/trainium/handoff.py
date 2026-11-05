"""Decode attention/projection residency through the shared bufferized stages.

Opt-in while hardware qualification is pending. No emitted-source rewriting,
weight prefetch, private allocation or model-family dispatch.
"""

import copy
import math
from dataclasses import replace

import torch
from torch.fx import Graph, GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value, query_kmajor
from voyager_compiler.codegen.subgraph import (
    create_and_insert_subgraph,
    replace_node_with_graph_module,
)
from voyager_compiler.compiler.bufferize.attention_v3 import build_attention_fa3
from voyager_compiler.compiler.bufferize.pipeline import (
    _gemm_plan,
    build_pipelined_buffers,
)
from voyager_compiler.compiler.bufferize.regions import (
    build_sequential_region,
    estimate_sequential_region,
)
from voyager_compiler.compiler.bufferize.residency import (
    compose_resident_stages,
)
from voyager_compiler.compiler.bufferize.ops import commit
from voyager_compiler.compiler.bufferize.utils import outline_dps_ops
from voyager_compiler.compiler.search.tiler import Tiler
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.ir.plan import Cost, ImplementationChoice
from voyager_compiler.shape_prop import ShapeProp, fake_like, set_node_value
from .attention import TrainiumAttention, TrainiumFlashAttention, SDPA_KMAJOR
from .gemm import GemmSteps, TrainiumGemv, window, get_slot, alloc_placed
from .model import TrainiumModel
from .patterns import INSTRUCTION_PATTERNS
from .utils import SRAM, LOCAL
from voyager_compiler.codegen.transform import fuse_patterns

TAG = "trainium_resident_handoff"
PRODUCTS = {
    torch.ops.aten.matmul.default,
    torch.ops.quantized_ops.matmul.default,
    torch.ops.quantized_ops.linear.default,
}
VIEWS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
}


def _plain_projection(module, call):
    """Expose a layout-only wrapper around one bias-free projection.

    Normalization can put a weight transpose and linear in a submodule. Do not
    flatten activation epilogues, extra arithmetic, effects or nested regions.
    """
    region = call.meta.get("submodule") if call.op == "call_module" else None
    if not isinstance(region, GraphModule):
        return call
    products = [n for n in region.graph.nodes if n.target in PRODUCTS]
    if len(products) != 1:
        return call
    product = products[0]
    if (
        product.args[0].op != "placeholder"
        or product.kwargs.get("layout") != "xc,ck"
        or (
            product.target is torch.ops.quantized_ops.linear.default
            and get_arg_value(product, 2, "bias", None) is not None
        )
    ):
        return call
    if any(
        n.op not in ("placeholder", "output")
        and n is not product
        and n.target not in VIEWS
        for n in region.graph.nodes
    ):
        return call
    output = next(n for n in region.graph.nodes if n.op == "output")
    if output.args[0] is not product and output.args[0] != (product,):
        return call
    region = copy.deepcopy(region)
    output = next(n for n in region.graph.nodes if n.op == "output")
    if isinstance(output.args[0], Node):
        output.args = ((output.args[0],),)
    (result,) = replace_node_with_graph_module(
        module, call, region, propagate=True
    )
    module.graph.erase_node(call)
    return result


def outline(module, mode):
    if mode not in ("off", "auto", "sbuf", "hbm"):
        raise ValueError(f"unknown resident handoff mode: {mode}")
    if mode == "off":
        return
    for attention in list(module.graph.nodes):
        if attention.target is not SDPA_KMAJOR:
            continue
        q, k, v = attention.args[:3]
        # The shared FA3 builder does not implement stochastic dropout. Keep
        # unqualified mask formats outside this optional region as well.
        mask = get_arg_value(attention, 3, "attn_mask", None)
        if get_arg_value(attention, 4, "dropout_p", 0.0) != 0.0 or (
            mask is not None and mask.value.dtype != torch.bfloat16
        ):
            continue
        shape = tuple(attention.value.shape)
        if (
            len(shape) != 4
            or shape[0] != 1
            or shape[2:] != (1, 128)
            or attention.value.dtype != torch.bfloat16
            or attention.kwargs.get("transpose_output", False)
            or any(n.value.dtype != torch.bfloat16 for n in (q, k, v))
            or k.value.shape[1] == 0
            or shape[1] % k.value.shape[1]
            or k.value.shape[1] != v.value.shape[1]
            or (
                shape[1] != k.value.shape[1]
                and not attention.kwargs.get("enable_gqa", False)
            )
        ):
            continue
        chain, current = [attention], attention
        while len(current.users) == 1:
            nxt = next(iter(current.users))
            if nxt.target not in VIEWS:
                break
            chain.append(nxt)
            current = nxt
        if len(current.users) != 1:
            continue
        projection = _plain_projection(module, next(iter(current.users)))
        if (
            projection.target not in PRODUCTS
            or (
                projection.target is torch.ops.quantized_ops.linear.default
                and get_arg_value(projection, 2, "bias", None) is not None
            )
            or projection.args[0] is not current
            or tuple(current.value.shape) != (1, 1, shape[1] * 128)
            or tuple(projection.args[1].value.shape)[:1] != (shape[1] * 128,)
            or projection.args[1].value.ndim != 2
            or projection.args[1].value.dtype != torch.bfloat16
            or projection.kwargs.get(
                "layout",
                (
                    "xc,kc"
                    if projection.target
                    is torch.ops.quantized_ops.linear.default
                    else "xc,ck"
                ),
            )
            != "xc,ck"
            or projection.value.dtype != torch.bfloat16
        ):
            continue
        # Prove the intervening views preserve head-major channel order.
        coordinates = torch.arange(math.prod(shape)).reshape(shape)
        for view in chain[1:]:
            coordinates = view.target(
                coordinates, *view.args[1:], **view.kwargs
            )
        if not torch.equal(
            coordinates.reshape(-1), torch.arange(math.prod(shape))
        ):
            continue
        call = create_and_insert_subgraph(
            chain + [projection], module, results=[projection]
        )
        call.meta[TAG] = mode
    module.graph.lint()
    module.recompile()


class ResidentAttention(TrainiumAttention):
    resident_output = True

    def buffers(self, tiling):
        buffers = super().buffers(tiling)
        return (
            *buffers,
            ("handoff_transpose", (128, 512), torch.bfloat16, 16, 1, LOCAL),
        )


class ResidentGemv(TrainiumGemv):
    resident_input = True

    def __init__(self, node, model):
        super().__init__(node, model)
        self.operands = (None, *self.operands[1:])

    def fit(self, tiling):
        if tiling.tile_sizes[0] != 1 or tiling.tile_sizes[1] % 128:
            return None
        return super().fit(tiling)

    def transfers(self, shapes, tiling):
        return [
            m
            for m in super().transfers(shapes, tiling)
            if m[0] is not self.node.args[0]
        ]

    def placement(self, tiling, shapes, moves):
        placement = super().placement(tiling, shapes, moves)
        if placement is None:
            return None
        depth = self.node.args[0].value.shape[-1]
        return replace(
            placement,
            buffers=(
                *placement.buffers,
                (
                    "resident_activation",
                    (128, depth // 128),
                    torch.bfloat16,
                    16,
                    1,
                    SRAM,
                ),
            ),
        )


class ResidentModel(TrainiumModel):
    def prepare_attention(self, node, acc_dtype, bool_mask):
        return ResidentAttention(node, self, acc_dtype, bool_mask)

    def prepare(self, node):
        return (
            ResidentGemv(node, self)
            if node.meta.get("resident_input")
            else super().prepare(node)
        )


class ResidentAttentionPipeline(TrainiumFlashAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handoff_heads = math.prod(self.grid[:-2]) * self.g_head
        self.out_unfold_shape = (128, self.handoff_heads)

    def _allocate_output(self):
        return alloc_placed((128, self.handoff_heads), self.out_dtype, 1, SRAM)

    def _store_output(self, b, coords):
        tile, out, sem = get_slot(b.out_slots, 0), b.out, b.out_sem
        head = coords[1] * self.g_head + coords[self.gq] * self.tq

        # Scratch is passed explicitly through the ordinary commit region.
        def store(tile, out, local):
            source = tile.reshape(self.tq, 128)
            temporary = window(local, (0, 0), (128, self.tq))
            torch.ops.voyager.insert(torch.transpose(source, 0, 1), temporary)
            dst = window(out, (0, head), (128, self.tq))
            torch.ops.voyager.insert(temporary.clone(), dst)

        commit(store, [tile, out, b.handoff_transpose], post=get_slot(sem, 0))


class ResidentGemmSteps(GemmSteps):
    """Read stationary columns directly; reuse shared traversal/accumulation buffers."""

    def __init__(self, node, plan, model):
        super().__init__(node, plan, model)
        self.kwargs["layout"] = "cx,ck"
        i = self.names.index("resident_activation")
        self.names.pop(i)
        self.scratch_specs.pop(i)

    def _step(self, tiles, outs, boxes, step, first, last):
        # Express a single offset in the frozen IR, rather than nesting a
        # runtime tile view and a per-head subview of the same dimension.
        dim = self.plan.grid_dims[2]
        k = (step // math.prod(self.plan.grid[dim + 1 :])) % self.plan.num_k
        boxes = dict(boxes, resident_k_offset=k * (self.plan.tile_k // 128))
        super()._step(tiles, outs, boxes, step, first, last)

    def _accumulate(
        self, tiles, boxes, activation, weight, chunk, j, w, first, psum
    ):
        # One physical column is one head's128 contraction coordinates. Preserve
        # the same128-wide product order the ordinary GEMM translator emits.
        for block in range(self.plan.tile_k // 128):
            act = window(
                activation, (0, boxes["resident_k_offset"] + block), (128, 1)
            )
            moving = window(weight, (block * 128, j), (128, w))
            product = self.op(act, moving, **self.kwargs)
            torch.ops.voyager.insert(
                product if first and block == 0 else product + psum, psum
            )


def implementations(components):
    def build(node, context, parameters):
        region = copy.deepcopy(node.meta["submodule"])
        ShapeProp(region).propagate(
            *(fake_like(n.value) for n in node.all_input_nodes)
        )
        output = next(n for n in region.graph.nodes if n.op == "output")
        if isinstance(output.args[0], Node):
            output.args = ((output.args[0],),)
            region.recompile()
        if parameters["storage"] == "hbm":
            result = build_sequential_region(region, context, components)
            # Region extraction can rename placeholder Nodes without renaming
            # their Python argument targets. Synchronize this private copy to
            # prevent an alias assignment from overwriting another argument.
            for module in result.modules():
                if isinstance(module, GraphModule):
                    for argument in module.graph.nodes:
                        if argument.op == "placeholder":
                            argument.target = argument.name
                    module.recompile()
            return result
        attn = next(n for n in region.graph.nodes if n.target is SDPA_KMAJOR)
        product = next(n for n in region.graph.nodes if n.target in PRODUCTS)
        heads = attn.value.shape[1]
        if (
            attn.args[1].value.shape[1] == 0
            or heads % attn.args[1].value.shape[1]
        ):
            raise CandidateRejected("invalid grouped heads")
        model = ResidentModel(context.tiler.config, context.accumulate_fp32)
        ctx = replace(
            context, tiler=Tiler(model, context.tiler.runtime_tolerance)
        )
        producer = build_attention_fa3(
            attn,
            ctx,
            pipeline=ResidentAttentionPipeline,
            query_kmajor=query_kmajor(attn),
            transpose_output=False,
        )
        if producer is None:
            raise CandidateRejected("unsupported resident attention")
        # The search uses the ordinary logical GEMV shape; its backing is packed
        # [head_dim,heads]. The consumer reads views directly, without DMA.
        graph = Graph()
        a = graph.placeholder("activation")
        w = graph.placeholder("weight")
        set_node_value(a, fake_like(attn.value).reshape(1, heads * 128))
        set_node_value(w, fake_like(product.args[1].value))
        gemm = graph.call_function(
            torch.ops.quantized_ops.matmul.default,
            (a, w),
            {"layout": "xc,ck", "out_dtype": torch.bfloat16},
        )
        set_node_value(gemm, fake_like(product.value).reshape(1, -1))
        gemm.meta["resident_input"] = True
        plan = _gemm_plan(gemm, ctx.tiler)
        if plan is None:
            raise CandidateRejected("no resident projection mapping")
        steps = ResidentGemmSteps(gemm, plan, model)
        activation_index = plan.activation_idx[0]
        steps.in_specs[activation_index] = None
        inputs = list(plan.inputs)
        inputs[activation_index] = fake_like(attn.value).reshape(128, heads)
        consumer = build_pipelined_buffers(
            steps.kernel,
            plan.grid,
            steps.in_specs,
            steps.out_specs,
            tuple(inputs),
            scratch_specs=steps.scratch_specs,
            num_slots=ctx.num_slots,
        )
        fuse_patterns(consumer, INSTRUCTION_PATTERNS)
        outline_dps_ops(consumer)
        original_inputs = list(region.graph.nodes)
        ph = [n for n in original_inputs if n.op == "placeholder"]
        producer_args = [ph.index(n) for n in attn.all_input_nodes]
        consumer_args = [(0, 0), ph.index(product.args[1])]
        joined = compose_resident_stages(
            [(producer, producer_args), (consumer, consumer_args)],
            [fake_like(n.value) for n in ph],
            output_shape=tuple(product.value.shape),
        )
        ac = attn.meta["tile_result"]
        gc = gemm.meta["tile_result"]
        joined.meta["region_choices"] = (
            (
                "attention",
                ImplementationChoice(
                    "trainium.attention.resident", (), ac, ac.cost
                ),
            ),
            (
                "projection",
                ImplementationChoice(
                    "trainium.projection.resident", (), gc, gc.cost
                ),
            ),
        )
        joined.meta["region_cost"] = Cost(
            ac.cost.latency + gc.cost.latency,
            (
                ac.cost.dram_bytes + gc.cost.dram_bytes
                if gc.cost.dram_bytes is not None
                else None
            ),
        )
        return joined

    return tuple(
        ImplementationSpec(
            f"trainium.attention_projection.{storage}",
            lambda n, a, storage=storage: n.meta.get(TAG) in ("auto", storage),
            build,
            candidates=lambda n, c, storage=storage: ({"storage": storage},),
            estimate=estimate_sequential_region,
            cache_by_shape=True,
            description="Shared bufferized attention/projection with "
            + storage
            + " boundary",
        )
        for storage in ("hbm", "sbuf")
    )
