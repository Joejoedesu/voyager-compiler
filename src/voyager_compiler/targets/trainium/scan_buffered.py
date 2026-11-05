"""Integrating-factor recurrence on the shared buffer pipeline.

The common builder owns all HBM movement, input/output slots and explicit state.
Only cumulative-sum and scaled-exp tile primitives are target-specific.
"""
import math
import torch

from voyager_compiler.compiler.bufferize.pipeline import build_pipelined_buffers
from voyager_compiler.compiler.bufferize.utils import _InputSpec, _OutputSpec, _ScratchSpec, effect_cond, outline_dps_ops
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.ir.plan import Cost
from .model import TrainiumModel
from .scan import OP, _validate, traffic_bytes


@torch.library.custom_op('trainium_ops::prefix_sum', mutates_args=(), device_types='cpu')
def prefix_sum(data: torch.Tensor, initial: torch.Tensor | None) -> torch.Tensor:
    result = torch.cumsum(data, -1)
    return result if initial is None else result + initial


@prefix_sum.register_fake
def _prefix_fake(data, initial):
    return torch.empty_like(data)


@torch.library.custom_op('trainium_ops::scaled_exp', mutates_args=(), device_types='cpu')
def scaled_exp(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.exp(data * scale)


@scaled_exp.register_fake
def _exp_fake(data, scale):
    return torch.empty_like(data)


EXECUTION_TAG = 'trainium_scan_execution'
PRIMITIVES = (torch.ops.trainium_ops.prefix_sum.default, torch.ops.trainium_ops.scaled_exp.default)


def geometry(channels, length, states, width, slots):
    if channels % 128 or width <= 0 or length % width or slots not in (2,3):
        raise CandidateRejected('bufferized scan requires complete channel/time tiles and 2/3 slots')
    # delta/u, B/C and output slots, five panel temporaries, persistent carry.
    panels = length//width
    stream_slots = min(slots, channels//128*panels)
    bc_slots = min(slots, panels*states)
    a_slots = min(slots, channels//128)
    demand = 512*((3*stream_slots+2*bc_slots+5)*width + (a_slots+2)*states)
    return demand


def candidates(node, context):
    channels,length=node.args[0].shape
    states=node.args[2].shape[1]
    for pieces in (1,2,4,8,16,32,64):
        if length%pieces: continue
        width=length//pieces
        if width < min(128,length): continue
        for slots in (2,3):
            if geometry(channels,length,states,width,slots) <= context.tiler.config.usable_scratchpad_size:
                yield dict(sequence_tile=width,slots=slots,algorithm='integrating_factor')


def build(node, context, parameters):
    inputs=tuple(n.value for n in node.args)
    _validate(*inputs)
    if node.meta.get('trainium_scan_algorithm') != 'integrating_factor':
        raise CandidateRejected('bufferized scan requires integrating_factor numerical policy')
    channels,length=inputs[0].shape; states=inputs[2].shape[1]
    width,slots=parameters['sequence_tile'],parameters['slots']
    if geometry(channels,length,states,width,slots)>context.tiler.config.usable_scratchpad_size:
        raise CandidateRejected('bufferized scan exceeds usable SBUF')
    v=torch.ops.voyager

    def kernel(index,d,u,a,b,c,out,carry,negative_a,prefix,du,positive,negative,drive):
        channel,time,state=index
        def initialize_carry():
            v.insert(torch.zeros_like(carry),carry)
            v.insert(-a,negative_a)
        effect_cond((time==0)&(state==0),initialize_carry)
        def prepare_panel():
            v.insert(prefix_sum(d,None),prefix)
            v.insert(d*u,du)
            v.insert(torch.zeros_like(out),out)
        effect_cond(state==0,prepare_panel)
        a_column=v.subview(a,[0,state],[128,1],[1,1])
        negative_column=v.subview(negative_a,[0,state],[128,1],[1,1])
        carry_column=v.subview(carry,[0,state],[128,1],[1,1])
        v.insert(scaled_exp(prefix,a_column),positive)
        v.insert(scaled_exp(prefix,negative_column),negative)
        v.insert(du*b,drive)
        v.insert(drive*negative,drive)
        v.insert(prefix_sum(drive,carry_column),drive)
        v.insert(drive*positive,positive)
        v.insert(v.subview(positive,[0,width-1],[128,1],[1,1]).clone(),carry_column)
        v.insert(positive*c,positive)
        v.insert(out+positive,out)

    graph=build_pipelined_buffers(kernel,(channels//128,length//width,states),
        [_InputSpec((128,width),(0,1),(False,False)),
         _InputSpec((128,width),(0,1),(False,False)),
         _InputSpec((128,states),(0,None),(False,False)),
         _InputSpec((1,width),(2,1),(False,False)),
         _InputSpec((1,width),(2,1),(False,False))],
        [_OutputSpec((channels,length),(128,width),(0,1),torch.float32)],inputs,
        scratch_specs=[_ScratchSpec((128,states),torch.float32) for _ in range(2)]
                      +[_ScratchSpec((128,width),torch.float32) for _ in range(5)],
        num_slots=slots)
    outline_dps_ops(graph)
    return graph


def estimate(node, graph, context, parameters):
    channels,length=node.args[0].shape; states=node.args[2].shape[1]
    width=parameters['sequence_tile']; model=TrainiumModel(context.tiler.config,False)
    blocks=channels//128; panels=length//width; steps=blocks*panels*states
    vector=model.engine_ns('vector',width,32)
    scalar=model.engine_ns('scalar',width,32)
    # Actual emitted body: five products/adds and cumulative sum.
    # Panel prefix/du/reset and persistent-state initialization are explicit.
    per_step=max(6*vector,2*scalar+model.engine_ns('scalar',1,32))
    per_step += 3*vector/states
    dmas=[(model.dma_ns(128,width,32,load=True),blocks*panels)]*2
    dmas += [(model.dma_ns(128,states,32,load=True),blocks)]
    dmas += [(model.dma_ns(128,width,32,load=True),steps)]*2
    dmas += [(model.dma_ns(128,width,32),blocks*panels)]
    ns=model.sweep_ns(dmas,steps,per_step)+2*blocks*model.engine_ns('vector',states,32)
    return Cost(ns*context.tiler.config.frequency,traffic_bytes(channels,length,states))


IMPLEMENTATION=ImplementationSpec('trainium.selective_scan.bufferized',
    lambda n,a:a.target is OP and n.meta.get(EXECUTION_TAG)=='bufferized',
    build,candidates=candidates,estimate=estimate,
    description='Explicit recurrent state and shared pipelined HBM movement')
