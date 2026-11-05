"""High/residual products in the existing shared matrix pipeline.

This is an explicitly enabled numerical alternative, not a new native kernel.
The input ABI stays FP32; all splitting and products execute on the device.
"""
import functools
import math
from collections import Counter
from dataclasses import dataclass, replace

import torch

from voyager_compiler.codegen.node_info import gemm_layout, is_gemm_op, weight_transforms
from voyager_compiler.codegen.transform import fuse_patterns
from voyager_compiler.compiler.bufferize.pipeline import (
    _gemm_plan, _stamp_anchor_meta, build_pipelined_buffers, spec_tiled_dims,
)
from voyager_compiler.compiler.bufferize.utils import effect_cond, outline_dps_ops
from voyager_compiler.compiler.search.traversal import (
    INPUT_DIMS, OUTPUT_DIMS, WEIGHT_DIMS, batch_loads, l3_steps, total_extent,
)
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec

from .gemm import GemmSteps, TrainiumGemm, gemm_step_busy, lands_on_tensor_engine, le
from .model import TrainiumModel
from .patterns import INSTRUCTION_PATTERNS
from .utils import SRAM

voyager = torch.ops.voyager
ROLES = ('compensated_act_hi', 'compensated_act_lo',
         'compensated_weight_hi', 'compensated_weight_lo')
DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16}


def matches(node, anchor):
    if not node.meta.get('trainium_compensated_matmul', False) or not is_gemm_op(anchor):
        return False
    if gemm_layout(anchor).weight != 'ck':
        return False
    if any(n.value.dtype != torch.float32 for n in anchor.args[:2]):
        return False
    if anchor.value.dtype != torch.float32 or len(anchor.args) != 2:
        return False
    # Prologue/tail legality and capacity are owned by the ordinary matrix
    # adapter. Numerical opt-in applies to the contraction, not a fused spelling.
    return True


@dataclass(frozen=True)
class CompensatedGemm(TrainiumGemm):
    split_fractions: tuple = (1.0, 1.0)

    @classmethod
    def from_node(cls, node, model):
        op = super().from_node(node, model)
        return replace(op, key=(*op.key, 'three_product', model.split_format))

    def fit(self, mapping):
        if (self.weight_k_major == self.weight_loaded or self.bias_bits
                or self.out_transposed or self.block_size != 1
                or total_extent(mapping, le.ON, level=2) != 1):
            return None
        return super().fit(mapping)

    def representation_buffers(self, mapping, buffers):
        rows, depth, cols = self._tile(functools.partial(total_extent, mapping, level=2))
        # A row-major activation is prepared/transposed one row chunk at a time
        # by GemmSteps, including its optional normalization prologue.
        act_rows = rows if self.act_k_major != self.act_loaded else min(rows, self.model.config.partitions)
        dtype = DTYPES[self.model.split_format]
        shapes = ((depth, act_rows), (depth, act_rows), (depth, cols), (depth, cols))
        return buffers + tuple((role, shape, dtype, 16, 1, SRAM)
                               for role, shape in zip(ROLES, shapes))

    def cost(self, mapping, placement):
        steps = l3_steps(mapping) * self.batch
        fractions = (1.0 if self.act_k_major == self.act_loaded else
                     batch_loads(mapping, INPUT_DIMS, self.batch, self.batch)/steps,
                     batch_loads(mapping, WEIGHT_DIMS, self.batch, self.weight_batch)/steps)
        return TrainiumGemm.cost(replace(self, split_fractions=fractions), mapping, placement)

    def _step_ns(self, rows, depth, cols, on, **kwargs):
        copies = Counter()
        prologue = kwargs.pop('prologue_engines', ())
        busy = gemm_step_busy(self.model, rows, depth, cols, (16, 16, 32), True,
                             act_k_major=True, distance=None, copy_busy=copies, **kwargs)
        # Three half-precision products, but only one final eviction/tail.
        busy['tensor'] *= 3
        busy.update(self.model.passes_busy([(e, 1.0, 32) for e in prologue], rows, depth))
        for width, fraction in zip((rows, cols), self.split_fractions):
            for engine, ns in self.model.passes_busy([('vector', 2.0, 32)], depth, width).items():
                busy[engine] += fraction * ns
        # Input relayout is FP32 work done once per loaded tile, not once per
        # half-precision product. Reuse the ordinary transpose service model.
        plain = gemm_step_busy(self.model, rows, depth, cols, (32, 32, 32), True,
                              act_k_major=True, distance=None, tail_copies=False,
                              copy_busy=Counter())
        for index, (loaded, width) in enumerate(((self.act_k_major == self.act_loaded or self.act_loaded, rows), (self.weight_loaded, cols))):
            if not loaded or not lands_on_tensor_engine(1, width):
                continue
            relayout_copies = Counter()
            relayout = gemm_step_busy(self.model, rows, depth, cols, (32, 32, 32), index != 1,
                                     act_k_major=index != 0, distance=None, tail_copies=False,
                                     copy_busy=relayout_copies)
            for engine, ns in relayout.items():
                busy[engine] += self.split_fractions[index] * (ns - plain[engine])
            for engine, ns in relayout_copies.items():
                copies[engine] += self.split_fractions[index] * ns
        return max(self.model.balance_copy_busy(busy, copies).values())


class CompensatedModel(TrainiumModel):
    def __init__(self, config, accumulate_fp32, split_format):
        super().__init__(config, accumulate_fp32)
        self.split_format = split_format

    def prepare(self, node):
        return CompensatedGemm.from_node(node, self)


class CompensatedSteps(GemmSteps):
    """Specialize operand preparation/products; reuse all matrix control and tails."""
    def __init__(self, node, plan, model):
        super().__init__(node, plan, model)
        weight_node = weight_transforms(plan.anchor.args[1])[0]
        sub = node.meta.get('submodule')
        if sub is not None:
            placeholders = [n for n in sub.graph.nodes if n.op == 'placeholder']
            weight_node = node.args[placeholders.index(weight_node)]
        self.weight_index = list(node.all_input_nodes).index(weight_node)
        self.act_index = plan.activation_idx[0] if self.transposes is None else None
        self.split_shapes, self.split_holds = [], []
        for index, role in ((self.act_index, ROLES[0]), (self.weight_index, ROLES[2])):
            if index is None:
                self.split_shapes.append(self.buffers[role].shape)
                self.split_holds.append(1)
                continue
            spec = plan.in_specs[index]
            self.split_shapes.append(tuple(spec.tile_sizes))
            dims = list(spec_tiled_dims(spec, plan.grid))
            if not dims:
                hold = math.prod(plan.grid)
            else:
                inner = max(g for _, g, _ in dims)
                repeat = next(r for _, g, r in dims if g == inner)
                hold = repeat * math.prod(plan.grid[inner+1:])
            self.split_holds.append(hold)

    def _split(self, value, hi, lo):
        voyager.insert(value.clone(), hi)
        voyager.insert(value - hi, lo)

    def _prepare_step(self, tiles, boxes, step):
        for i, index in enumerate((self.act_index, self.weight_index)):
            if index is None:
                continue
            hi = boxes[ROLES[2*i]].view(self.split_shapes[i])
            lo = boxes[ROLES[2*i+1]].view(self.split_shapes[i])
            def decompose(value=tiles[index], hi=hi, lo=lo):
                self._split(value, hi, lo)
            if self.split_holds[i] == 1:
                decompose()
            else:
                effect_cond(step % self.split_holds[i] == 0, decompose)

    def _accumulate(self, tiles, boxes, activation, weight, chunk, j, w, first, psum):
        if self.transposes is not None:
            shape = activation.shape
            act = [boxes[role][:, :shape[-1]] for role in ROLES[:2]]
            if j == 0:
                self._split(activation, *act)
        else:
            act = [boxes[role].view(self.split_shapes[0]) for role in ROLES[:2]]
        weights = [boxes[role].view(self.split_shapes[1]) for role in ROLES[2:]]
        for term, (a, b) in enumerate(((0, 0), (0, 1), (1, 0))):
            product = self._product(tiles, boxes, act[a], weights[b], chunk, j, w, first and term == 0)
            voyager.insert(product if first and term == 0 else product + psum, psum)


def build(node, context, parameters):
    plan = _gemm_plan(node, context.tiler)
    if plan is None:
        raise CandidateRejected('no legal shared compensated-matrix mapping')
    steps = CompensatedSteps(node, plan, context.tiler.model)
    graph = build_pipelined_buffers(steps.kernel, plan.grid, steps.in_specs,
        steps.out_specs, plan.inputs, scratch_specs=steps.scratch_specs,
        num_slots=context.num_slots)
    fuse_patterns(graph, INSTRUCTION_PATTERNS)
    outline_dps_ops(graph)
    _stamp_anchor_meta(graph, plan.anchor)
    return graph


IMPLEMENTATION = ImplementationSpec('trainium.gemm.compensated', matches, build,
    candidates=lambda n, c: ({'format': f} for f in DTYPES),
    model=lambda config, accumulate_fp32, parameters:
        CompensatedModel(config, accumulate_fp32, parameters['format']),
    description='Explicit high/residual three-product arithmetic in the shared GEMM pipeline')
