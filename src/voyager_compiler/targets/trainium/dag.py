"""Dtype-preserving pointwise DAGs in the shared buffer pipeline.

Every intermediate is an ordinary SRAM buffer with an explicit write/read.
The target chooses physical tile width; the shared scheduler owns movement,
slots, waits and loops. No whole-kernel native invocation is introduced.
"""
import inspect
import math
from collections import Counter

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.subgraph import create_and_insert_subgraph, copy_graph_module
from voyager_compiler.compiler.bufferize.pipeline import build_pipelined_buffers
from voyager_compiler.compiler.bufferize.utils import _InputSpec, _OutputSpec, _ScratchSpec
from voyager_compiler.compiler.search.planner import DEFAULT_CANDIDATE_BUDGET
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.ir.buffer import TargetInvoke
from voyager_compiler.ir.plan import Cost
from voyager_compiler.shape_prop import ShapeProp

from .model import TrainiumModel
from .utils import chain_ops, chain_tiles, scratch_regions, target_engines

ATEN = torch.ops.aten
OPS = {ATEN.add.Tensor, ATEN.sub.Tensor, ATEN.mul.Tensor, ATEN.div.Tensor,
       ATEN.sqrt.default, ATEN.rsqrt.default, ATEN.reciprocal.default,
       ATEN.neg.default, ATEN.exp.default, ATEN.silu.default}
TAG = 'trainium_elementwise_dag'
DTYPES = (torch.float32, torch.bfloat16)


def suffix_broadcast(value, shape):
    """A contiguous suffix, repeated over leading dimensions only.

    These operands can be tiled without expansion or a packing copy. Other
    broadcast geometries remain available through the component implementation.
    """
    if value.ndim > len(shape) or not value.is_contiguous():
        return False
    padded = (1,) * (len(shape) - value.ndim) + tuple(value.shape)
    first = next((i for i, d in enumerate(padded) if d != 1), len(shape))
    return padded[first:] == tuple(shape)[first:]


def eligible(node):
    value = getattr(node, 'value', None)
    return (node.op == 'call_function' and node.target in OPS
            and isinstance(value, torch.Tensor) and value.dtype in DTYPES
            and value.numel() >= 128 and value.numel() % 128 == 0
            and node.kwargs.get('alpha', 1) == 1
            and all(isinstance(getattr(n, 'value', None), torch.Tensor)
                    and n.value.dtype == value.dtype
                    and suffix_broadcast(n.value, value.shape)
                    and n.value.numel() >= 128
                    for n in node.all_input_nodes))


def outline(module, *, approximate_reciprocal=False):
    """Group connected equal-shape pointwise work, preserving every external use."""
    # Expose existing short instruction/graph chains before grouping the DAG.
    for node in list(module.graph.nodes):
        if node.op != 'call_module':
            continue
        sub = node.meta.get('submodule')
        if not isinstance(sub, GraphModule):
            continue
        ops = [n for n in sub.graph.nodes if n.op == 'call_function']
        if not ops or not all(eligible(n) for n in ops):
            continue
        output = next(n for n in sub.graph.nodes if n.op == 'output').args[0]
        if not isinstance(output, Node):
            continue
        # all_input_nodes deduplicates operands: f(x, x) still binds two
        # placeholders. Bind the actual call, including keyword/default args.
        bound = inspect.signature(sub.forward).bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        remap = {n: bound.arguments[n.target]
                 for n in sub.graph.nodes if n.op == 'placeholder'}
        with module.graph.inserting_before(node):
            for n in ops:
                remap[n] = module.graph.node_copy(n, remap.__getitem__)
                remap[n].value, remap[n].shape = n.value, n.shape
        node.replace_all_uses_with(remap[output])
        module.graph.erase_node(node)
    available = {n for n in module.graph.nodes if eligible(n)}
    order = {n: i for i, n in enumerate(module.graph.nodes)}
    while available:
        seed = min(available, key=order.__getitem__)
        group, stack = set(), [seed]
        while stack:
            n = stack.pop()
            if n not in available:
                continue
            available.remove(n); group.add(n)
            stack.extend(x for x in (*n.all_input_nodes, *n.users)
                         if x in available and x.value.shape == seed.value.shape
                         and x.value.dtype == seed.value.dtype)
            stack.extend(u for arg in n.all_input_nodes for u in arg.users
                         if u in available and u.value.shape == seed.value.shape
                         and u.value.dtype == seed.value.dtype)
        # Outlining at the last op must not move a result past an external user.
        end = max(order[n] for n in group)
        if any(order[u] < end for n in group for u in n.users if u not in group):
            continue
        results = sorted((n for n in group if any(u not in group for u in n.users)), key=order.__getitem__)
        if not results:
            continue
        fused = create_and_insert_subgraph(list(group), module, results=results)
        fused.meta[TAG] = True
        fused.meta['trainium_approximate_reciprocal'] = approximate_reciprocal
    module.graph.lint(); module.recompile()


def description(node):
    region = node.meta['submodule']
    inputs = [n for n in region.graph.nodes if n.op == 'placeholder']
    ops = [n for n in region.graph.nodes if n.op == 'call_function']
    out = next(n for n in region.graph.nodes if n.op == 'output').args[0]
    outputs = list(out) if isinstance(out, (tuple, list)) else [out]
    return region, inputs, ops, outputs


def colors(ops, outputs):
    """Reuse an intermediate only after its final read, never in that same op."""
    positions = {n: i for i, n in enumerate(ops)}
    last = {n: max((positions[u] for u in n.users if u in positions), default=positions[n]) for n in ops}
    assigned, ends = {}, []
    for i, n in enumerate(ops):
        if n in outputs:
            continue
        slot = next((j for j, end in enumerate(ends) if end < i), len(ends))
        if slot == len(ends):
            ends.append(last[n])
        else:
            ends[slot] = last[n]
        assigned[n] = slot
    return assigned, len(ends)


def domain(node):
    """Logical size, contiguous inner sweep, and the semantic dtype.

    With broadcasting the grid visits heads (the outer extent) inside each
    suffix tile. The shared pipeline therefore loads a broadcast tile once
    and reuses it, rather than materializing a full-size expanded operand.
    """
    _, inputs, ops, outputs = description(node)
    value = outputs[0].value
    if value.dtype not in DTYPES or any(
            n.value.dtype != value.dtype or n.value.shape != value.shape
            for n in (*ops, *outputs)) or any(
                n.value.dtype != value.dtype or
                not suffix_broadcast(n.value, value.shape) for n in inputs):
        raise CandidateRejected('DAG requires one supported dtype and compatible suffix operands')
    size = value.numel()
    inner = min(n.value.numel() for n in inputs)
    if inner < 128 or size % inner or any(n.value.numel() not in (inner, size) for n in inputs):
        raise CandidateRejected('DAG supports one broadcast suffix domain')
    return size, inner, value


def geometry(node, parameters):
    region, inputs, ops, outputs = description(node)
    size, inner, value = domain(node)
    width, slots = parameters['width'], parameters['slots']
    if type(width) is not int or not 1 <= width <= 8192 or slots not in (2, 3):
        raise CandidateRejected('DAG width must be1..8192 and pipeline slots2 or3')
    tile = 128 * width
    if inner % tile:
        raise CandidateRejected('DAG tile must divide the contiguous suffix')
    assigned, count = colors(ops, outputs)
    # Per-op auxiliary scratch is shared; each SSA intermediate is separately
    # represented by the colored buffers, so chain scratch is unnecessary.
    tiles = set()
    for op in ops:
        tiles |= chain_tiles(chain_ops([op]), width)
    scratch = scratch_regions('op', tiles)
    auxiliary = sum(math.prod(shape) * 4 * nslots for _, shape, _, nslots, _ in scratch)
    demand = ((len(inputs) + len(outputs)) * (1 if size == tile else slots) + count) * tile * value.element_size() + auxiliary
    return region, inputs, ops, outputs, assigned, count, scratch, demand


def _candidate_parameters(node, context):
    try:
        _, size, _ = domain(node)
    except CandidateRejected:
        return
    # Bounded search lattice, not an ISA limit. Include non-power-of-two
    # divisors, which can avoid unnecessarily narrow DMA/compute tiles.
    widths = sorted({min(size // 128, 128), *range(128, 8193, 128)})
    scalar_modes = (False, True) if any(scalar_literal(n) for n in description(node)[2]) else (False,)
    approximate = node.meta.get('trainium_approximate_reciprocal', False) and any(
        n.target == ATEN.reciprocal.default or
        (n.target == ATEN.div.Tensor and isinstance(n.args[1], Node))
        for n in description(node)[2])
    for width in widths:
        if size % (128 * width) == 0:
            for slots in (2, 3):
                for scalar_literals in scalar_modes:
                    parameters = dict(width=width, slots=slots, scalar_literals=scalar_literals)
                    if geometry(node, parameters)[-1] > context.tiler.config.usable_scratchpad_size:
                        continue
                    yield dict(parameters, activation_reciprocal=False)
                    # Three bounded policies: automatic literals, scalar
                    # literals, then scalar literals + approximate reciprocal.
                    if approximate and scalar_literals == scalar_modes[-1]:
                        yield dict(parameters, activation_reciprocal=True)


def candidates(node, context):
    """Shortlist only oversized searches using the existing resource cost.

    Reserve one shared-planner trial for the materialized comparator. Small
    searches retain their order; large searches retain the analytical winner
    and next-best alternatives for allocation refinement. All DAG candidates
    have the same external traffic, so stable latency ordering preserves the
    planner's tie behavior. No engine/traffic estimates are changed.
    """
    parameters = list(_candidate_parameters(node, context))
    limit = DEFAULT_CANDIDATE_BUDGET - 1
    if len(parameters) > limit:
        parameters.sort(key=lambda p: estimate(node, None, context, p).latency)
        parameters = parameters[:limit]
    return iter(parameters)


def scalar_literal(node):
    # Trn2 ScalarE tensor_scalar supports a single add or multiply, not sub.
    return (node.target in {ATEN.add.Tensor, ATEN.mul.Tensor}
            and len(node.args) > 1 and type(node.args[1]) in (int, float))


def build(node, context, parameters):
    if parameters.get('activation_reciprocal', False) and not node.meta.get('trainium_approximate_reciprocal', False):
        raise CandidateRejected('activation reciprocal requires explicit numerical-policy opt-in')
    region, inputs, ops, outputs, assigned, count, scratch, demand = geometry(node, parameters)
    if demand > context.tiler.config.usable_scratchpad_size:
        raise CandidateRejected(f'flat DAG needs {demand} SBUF bytes')
    width, slots = parameters['width'], parameters['slots']
    size, inner, value = domain(node)
    tile = 128 * width
    broadcast = inner != size
    shape = (size // inner, inner) if broadcast else (size,)
    tile_shape = (1, tile) if broadcast else (tile,)
    grid = (inner // tile, size // inner) if broadcast else (size // tile,)
    index_map = (1, 0) if broadcast else (0,)
    # A tile body with explicit destinations. No FX objects reach Dynamo.
    graph = torch.fx.Graph()
    loaded = [graph.placeholder(f'in_{i}') for i in range(len(inputs))]
    stores = [graph.placeholder(f'out_{i}') for i in range(len(outputs))]
    temps = [graph.placeholder(f'tmp_{i}') for i in range(count)]
    aux = [graph.placeholder(f'aux_{i}') for i in range(len(scratch))]
    remap = dict(zip(inputs, loaded))
    destination = dict(zip(outputs, stores))
    for n, slot in assigned.items():
        destination[n] = temps[slot]
    for n in ops:
        computed = graph.node_copy(n, remap.__getitem__)
        graph.call_function(torch.ops.voyager.insert.default, (computed, destination[n]))
        remap[n] = destination[n]
    graph.output(())
    body = GraphModule(torch.nn.Module(), graph)
    # Scratch is attached per primitive during export, not assigned by emit.
    # The body presently needs auxiliary scratch only for tensor division.
    if scratch:
        for n in list(body.graph.nodes):
            if n.op != 'call_function' or n.target not in OPS:
                continue
            subgraph = torch.fx.Graph()
            args = [subgraph.placeholder(f'a{i}') for i in range(len(n.all_input_nodes))]
            mapping = dict(zip(n.all_input_nodes, args))
            result = subgraph.node_copy(n, mapping.__getitem__)
            subgraph.output(result)
            sub = GraphModule(torch.nn.Module(), subgraph)
            name = f'operation_{n.name}'; body.add_module(name, sub)
            with body.graph.inserting_before(n):
                attr = body.graph.get_attr(name)
                wrapped = body.graph.call_function(torch.ops.higher_order.with_scratch,
                    (attr, *n.all_input_nodes), {'scratch':dict(zip((s[0] for s in scratch), aux))})
            n.replace_all_uses_with(wrapped); body.graph.erase_node(n)
        body.graph.lint(); body.recompile()

    def kernel(grid_index, *refs):
        body(*refs)

    shapes = [tuple(n.value.shape) for n in outputs]
    class Flatten(torch.nn.Module):
        def __init__(self, pipeline):
            super().__init__(); self.pipeline = pipeline

        def forward(self, *args):
            result = self.pipeline(*(x.reshape(-1, inner) if broadcast else x.reshape(-1)
                                     for x in args))
            if not isinstance(result, (tuple, list)):
                result = (result,)
            return tuple(value.reshape(shape) for value, shape in zip(result, shapes))

    result = build_pipelined_buffers(kernel, grid,
        [_InputSpec(tile_shape, index_map,
                    (n.value.numel() == inner, False) if broadcast else (False,))
         for n in inputs],
        [_OutputSpec(shape, tile_shape, index_map, n.value.dtype) for n in outputs],
        # Disjoint example tensors avoid export's input-alias rejection for
        # disjoint views such as rotary halves. This is tracing-only: no clone
        # is inserted into the executable and the external inputs stay read-only.
        tuple(n.value.clone() for n in node.all_input_nodes),
        scratch_specs=[_ScratchSpec(tile_shape, value.dtype) for _ in range(count)]
                      + [_ScratchSpec(*s[1:]) for s in scratch],
        num_slots=slots, wrapper=Flatten)
    # Keep the external allocation's semantic shape in the saved ABI. The
    # flattened view is only a movement map; it must not leak into the output
    # signature when the emitter elides metadata-only result reshapes.
    returned = next(n for n in result.graph.nodes if n.op == 'output').args[0]
    for returned_value, semantic_shape in zip(returned, shapes):
        allocation = returned_value
        while allocation.target in (ATEN.view.default, ATEN.reshape.default):
            allocation = allocation.args[0]
        if allocation.target != torch.ops.voyager.alloc.default:
            raise ValueError('DAG result does not resolve to its explicit output allocation')
        users = list(allocation.users)
        allocation.args = (list(semantic_shape), *allocation.args[1:])
        with result.graph.inserting_after(allocation):
            flat = result.graph.call_function(ATEN.view.default, (allocation, list(shape)))
        for user in users:
            user.replace_input_with(allocation, flat)
    result.graph.lint(); result.recompile()
    ShapeProp(result, recurse=True).propagate(*(n.value.clone() for n in node.all_input_nodes))
    result.meta['dag_sbuf_bytes'] = demand
    for module in result.modules():
        if isinstance(module, GraphModule):
            for operation in module.graph.nodes:
                if operation.op == 'call_function' and operation.target in OPS:
                    scalar = int(parameters.get('scalar_literals', False) and scalar_literal(operation))
                    reciprocal = int(parameters.get('activation_reciprocal', False))
                    operation.meta['target_invoke'] = TargetInvoke(
                        'trainium.elementwise_dag', f'elementwise_chunk.{width}.{scalar}.{reciprocal}', (), (), (),
                        resources=('scalar', 'vector'), scheduling_owner='hardware')
    return result


def estimate(node, graph, context, parameters):
    _, inputs, ops, outputs, assigned, _, _, _ = geometry(node, parameters)
    config = context.tiler.config
    model = TrainiumModel(config, context.accumulate_fp32)
    size, inner, value = domain(node)
    width = parameters['width']; steps = size // (128 * width)
    bits = value.element_size() * 8
    busy = Counter()
    finished, last_read, last_write = {}, Counter(), Counter()
    for op in ops:
        engines = ('scalar',) if parameters.get('scalar_literals', False) and scalar_literal(op) else target_engines(op.target)
        inverse = op.target == ATEN.reciprocal.default
        division = op.target == ATEN.div.Tensor and isinstance(op.args[1], Node)
        reciprocal_engine = 'scalar' if parameters.get('activation_reciprocal', False) else 'vector'
        if inverse:
            engines = (reciprocal_engine,)
        duration = 0
        for engine in engines:
            # Neuron2.31 documents vector reciprocal as max(MIN_II,8*N)
            # engine cycles versus max(MIN_II,N) for ScalarE activation.
            elements = 8 * width if inverse and engine == 'vector' else width
            service = model.engine_ns(engine, elements, 32 if division else bits)
            busy[engine] += service
            duration += service
        if division:
            elements = width * (8 if reciprocal_engine == 'vector' else 1)
            service = model.engine_ns(reciprocal_engine, elements, 32)
            busy[reciprocal_engine] += service
            duration += service
        ready = max((finished.get(n, 0) for n in op.all_input_nodes), default=0)
        if op in assigned:
            slot = assigned[op]
            ready = max(ready, last_read[slot], last_write[slot])
            last_write[slot] = ready + duration
        finished[op] = ready + duration
        for n in op.all_input_nodes:
            if n in assigned:
                slot = assigned[n]
                last_read[slot] = max(last_read[slot], finished[op])
    busy['scalar'] += model.store_issue_ns(len(outputs) * steps, steps)
    dmas = [(model.dma_ns(128, width, bits), steps) for _ in outputs]
    dmas += [(model.dma_ns(128, width, bits, load=True),
              n.value.numel() // (128 * width)) for n in inputs]
    # Engine totals alone assume independent work. A branched DAG must also
    # respect producer/consumer edges and reuse of its single-slot scratch.
    # This is a dependency floor, not an exact downstream scheduling model.
    compute = max(max(busy.values()), max(finished.values()))
    ns = model.sweep_ns(dmas, steps, compute)
    moved = sum(n.value.numel() for n in (*inputs, *outputs)) * (bits // 8)
    return Cost(ns * config.frequency, moved)


IMPLEMENTATION = ImplementationSpec('trainium.elementwise_dag',
    lambda n, a: n.meta.get(TAG, False), build, candidates=candidates, estimate=estimate,
    description='Dtype-preserving equal-shape DAG with explicit live SRAM intermediates in the shared pipeline')


def materialized(components):
    from voyager_compiler.compiler.bufferize.regions import build_sequential_region, estimate_sequential_region

    def build_region(node, context, parameters):
        region = copy_graph_module(node.meta['submodule'])
        out = next(n for n in region.graph.nodes if n.op == 'output')
        if isinstance(out.args[0], Node):
            out.args = ((out.args[0],),); region.recompile()
        return build_sequential_region(region, context, components)

    return ImplementationSpec('trainium.elementwise_dag.materialized',
        lambda n, a: n.meta.get(TAG, False), build_region, estimate=estimate_sequential_region,
        description='Sequential component comparator for an equal-shape DAG')
