"""Native whole-kernel selective state-space scan implementations.

The semantic op preserves a serial FP32 recurrence. A registered recipe
selects the sequence tile, declares its private SBUF demand and cost, and
freezes that tile in TargetInvoke. NKI realizes that selected recipe using
tensor_tensor_scan by default. The explicit integrating-factor option changes
floating-point evaluation order and uses cumulative sums instead. Emission
consumes the saved algorithm and tile; it never selects them.
"""
import numpy as np
import torch

from voyager_compiler.compiler.bufferize.invocation import build_native
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.ir.buffer import TargetInvoke
from voyager_compiler.ir.plan import BufferRequirement, Cost

from .model import TrainiumModel


def _validate(delta, u, a, b, c):
    if any(x.dtype != torch.float32 for x in (delta, u, a, b, c)):
        raise ValueError("selective_scan currently requires FP32 tensors")
    if delta.ndim != 2 or a.ndim != 2:
        raise ValueError("selective_scan expects [channels, time] and [channels, states]")
    channels, length = delta.shape
    if u.shape != delta.shape or a.shape[0] != channels:
        raise ValueError("selective_scan channel/time extents differ")
    if b.shape != (a.shape[1], length) or c.shape != b.shape:
        raise ValueError("selective_scan expects B/C [states, time]")
    if min(channels, length, a.shape[1]) <= 0:
        raise ValueError("selective_scan requires nonempty dimensions")


@torch.library.custom_op("trainium_ops::selective_scan", mutates_args=(), device_types="cpu")
def selective_scan(delta: torch.Tensor, u: torch.Tensor, a: torch.Tensor,
                   b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Unbatched SSM: h[t]=exp(delta[t]*A)*h[t-1]+delta[t]*B[t]*u[t].

    The CPU reference follows NumPy's FP32 exp and serial recurrence. This
    avoids treating the different libm rounding of torch.exp, compounded
    over a long sequence, as a change to the input workload's semantics.
    """
    _validate(delta, u, a, b, c)
    d, values, weights, b_values, c_values = (
        x.detach().numpy() for x in (delta, u, a, b, c)
    )
    alpha = np.exp(d[:, None, :] * weights[:, :, None])
    update = d[:, None, :] * b_values[None, :, :] * values[:, None, :]
    state = np.zeros(weights.shape, dtype=np.float32)
    output = np.empty(d.shape, dtype=np.float32)
    for position in range(d.shape[1]):
        state = alpha[..., position] * state + update[..., position]
        output[:, position] = np.sum(c_values[None, :, position] * state, axis=-1)
    return torch.from_numpy(output)


@selective_scan.register_fake
def _selective_scan_fake(delta, u, a, b, c):
    _validate(delta, u, a, b, c)
    return torch.empty_like(delta)


OP = torch.ops.trainium_ops.selective_scan.default
PREFIX = "selective_scan."
ALGORITHM_TAG = "trainium_scan_algorithm"


def candidates(node, context):
    length = node.args[0].shape[1]
    if node.meta.get(ALGORITHM_TAG, "direct") == "integrating_factor":
        # Equal time panels, including non-power-of-two sequence lengths.
        widths = {length // pieces for pieces in (1, 2, 4, 8, 16, 32, 64)
                  if length % pieces == 0 and length // pieces >= min(128, length)}
        return tuple({"sequence_tile": width, "algorithm": "integrating_factor"}
                     for width in sorted(widths))
    return tuple(
        {"sequence_tile": width}
        for width in sorted({min(length, n) for n in (128, 256, 512, 1024, 2048, 4096, length)})
        if length % width == 0
    )


def footprint(channels, length, width):
    # Retain delta/u across all states, alongside the full output, six
    # sequence tiles, and the A/previous-state columns. Reserve the reuse
    # explicitly rather than depending on lower-compiler load elimination.
    return 128 * (3 * length + 6 * width + 2) * 4


def traffic_bytes(channels, length, states):
    # delta/u once, B/C once per state (partition-broadcast DMA really
    # reads their replicated bytes), A once, and the complete output.
    return channels * (length * (3 + 2 * states) + states) * 4


def integrating_footprint(states, width):
    # Ten time-panel arrays; A, -A and the boundary state occupy three columns
    # per state. These are native-private buffers, not shared allocator aliases.
    return 128 * (10 * width + 3 * states) * 4


def integrating_cost_breakdown(config, channels, length, states, width):
    """Existing resource rates applied to the integrating-factor instruction work.

    No endpoint-fitted multiplier. Matching NTFFs show that this cumulative
    instruction takes one vector service, unlike tensor_tensor_scan's two-ALU
    pricing. Surviving boundary copies run on ScalarE; the final panel's carry
    is dead. As in the direct model, this is a resource bound plus exposed
    fill/drain, not an exact model of recurrence dependencies or queue overlap.
    """
    model = TrainiumModel(config, False)
    blocks, chunks = channels // 128, length // width
    v = model.engine_ns("vector", width, 32)
    cumulative = v
    vector = model.engine_ns("vector", states, 32) + chunks * (
        2 * v + cumulative + states * (
            5 * v + cumulative))
    scalar = model.engine_ns("scalar", states, 32) + chunks * states * (
        2 * model.engine_ns("scalar", width, 32))
    scalar += max(0,chunks-1)*states*model.engine_ns('scalar',1,32)
    parameters = model.dma_ns(128, states, 32, load=True)
    pair = 2 * model.dma_ns(128, width, 32, load=True)
    store = model.dma_ns(128, width, 32)
    dma = parameters + chunks * (pair + states * pair + store)
    ends = parameters + 2 * pair + cumulative + v + 2 * model.engine_ns("scalar", width, 32) + store
    return dict(vector_ns=blocks * vector, scalar_ns=blocks * scalar,
                dma_ns=blocks * dma, exposed_ns=blocks * ends,
                latency_ns=blocks * (max(vector, scalar, dma) + ends))


def cost_breakdown(config, channels, length, states, width):
    """Resource bounds plus exposed fill/drain, in nanoseconds.

    NTFF and the backend instruction stream establish cross-engine overlap,
    one hardware scan instruction with two ALUs, and scalar-width carry
    copies that may be eliminated. Existing engine/transfer constants are
    reused unchanged. This is a throughput model, not a fitted instruction
    scheduler; dependency stalls can still make a short scan slower.
    """
    model = TrainiumModel(config, False)
    blocks, chunks = channels // 128, length // width
    scalar = states * chunks * model.engine_ns("scalar", width, 32)
    # Four pointwise instructions and one scan with two ALUs. The scan
    # pays one instruction overhead, and the carry is one element, not
    # an entire sequence tile. Conservatively charge it even if aliased.
    scan = 2 * model.engine_ns("vector", width, 32) - config.vector_overhead_ns
    vector = model.engine_ns("vector", length, 32) + states * (
        model.engine_ns("vector", 1, 32) + chunks * (
            4 * model.engine_ns("vector", width, 32) + scan
            + model.engine_ns("vector", 1, 32)
        )
    )
    retained_loads = 2 * model.dma_ns(128, length, 32, load=True)
    column = model.dma_ns(128, 1, 32, load=True)
    bc_loads = 2 * model.dma_ns(128, width, 32, load=True)
    store = model.dma_ns(128, length, 32)
    dma = retained_loads + states * (column + chunks * bc_loads) + store
    # First input/parameter loads and activation cannot hide beneath earlier
    # work in this channel block. The final store waits for its result.
    ends = retained_loads + column + bc_loads + model.engine_ns("scalar", width, 32) + store
    return dict(vector_ns=blocks * vector, scalar_ns=blocks * scalar,
                dma_ns=blocks * dma, exposed_ns=blocks * ends,
                latency_ns=blocks * (max(vector, scalar, dma) + ends))


def build(node, context, parameters):
    delta, u, a, b, c = (n.value for n in node.args)
    _validate(delta, u, a, b, c)
    channels, length = delta.shape
    width = parameters["sequence_tile"]
    algorithm = parameters.get("algorithm", "direct")
    if algorithm not in ("direct", "integrating_factor"):
        raise ValueError(f"unknown selective_scan algorithm: {algorithm}")
    if width <= 0 or length % width:
        raise CandidateRejected("selective_scan requires complete sequence panels")
    if channels % 128:
        raise CandidateRejected("selective_scan channel tiles require a multiple of 128")
    states = a.shape[1]
    demand = (integrating_footprint(states, width) if algorithm == "integrating_factor"
              else footprint(channels, length, width))
    if demand > context.tiler.config.usable_scratchpad_size:
        raise CandidateRejected("selective_scan private buffers exceed usable SBUF")
    traffic = traffic_bytes(channels, length, states)
    opcode = PREFIX + str(width)
    if algorithm != "direct":
        opcode += "." + algorithm
    invocation = TargetInvoke(
        "trainium.selective_scan", opcode, (), (), (),
        resources=("scalar", "vector", "dma"), allocation_owner="native",
        scheduling_owner="hardware", implicit_traffic_bytes=traffic,
        native_requirements=(BufferRequirement("scan_sbuf", "scratchpad", demand),),
    )
    return build_native(node, invocation)


def estimate(node, graph, context, parameters):
    config = context.tiler.config
    channels, length = node.args[0].shape
    states = node.args[2].shape[1]
    width = parameters["sequence_tile"]
    cost = (integrating_cost_breakdown if parameters.get("algorithm") == "integrating_factor"
            else cost_breakdown)
    ns = cost(config, channels, length, states, width)["latency_ns"]
    traffic = next(n.meta["target_invoke"].implicit_traffic_bytes
                   for n in graph.graph.nodes if "target_invoke" in n.meta)
    return Cost(ns * config.frequency, dram_bytes=traffic)


IMPLEMENTATION = ImplementationSpec(
    "trainium.selective_scan", lambda n, anchor: anchor.target is OP,
    build, candidates=candidates, estimate=estimate,
    description="FP32 selective scan with a selected sequence tile and native private SBUF",
)


def body(inputs, output, shape, states, width, algorithm="direct"):
    """NKI for the saved native invocation. No layout or tile decisions."""
    channels, length = shape
    if channels % 128 or width <= 0 or length % width:
        raise ValueError("invalid frozen selective_scan tile")
    if algorithm == "integrating_factor":
        return integrating_body(inputs, output, shape, states, width)
    if algorithm != "direct":
        raise ValueError(f"unknown frozen selective_scan algorithm: {algorithm}")
    delta, u, a, b, c = inputs
    lines = [
        f"scan_acc = nl.ndarray((128, {length}), dtype=nl.float32, buffer=nl.sbuf)",
        f"scan_delta = nl.ndarray((128, {length}), dtype=nl.float32, buffer=nl.sbuf)",
        f"scan_u = nl.ndarray((128, {length}), dtype=nl.float32, buffer=nl.sbuf)",
        f"scan_a = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"scan_previous = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)",
    ]
    for name in ("b", "c", "alpha", "update", "state", "weighted"):
        lines.append(f"scan_{name} = nl.ndarray((128, {width}), dtype=nl.float32, buffer=nl.sbuf)")
    lines += [
        f"for scan_channel in nl.affine_range({channels // 128}):",
        "    nisa.memset(dst=scan_acc, value=0.0)",
        f"    nisa.dma_copy(dst=scan_delta, src={delta}[scan_channel*128:(scan_channel+1)*128, :])",
        f"    nisa.dma_copy(dst=scan_u, src={u}[scan_channel*128:(scan_channel+1)*128, :])",
        f"    for scan_s in nl.affine_range({states}):",
        f"        nisa.dma_copy(dst=scan_a, src={a}[scan_channel*128:(scan_channel+1)*128, scan_s:scan_s+1])",
        "        nisa.memset(dst=scan_previous, value=0.0)",
        f"        for scan_t in nl.affine_range({length // width}):",
    ]
    for name, value in (("b", b), ("c", c)):
        lines.append(f"            nisa.dma_copy(dst=scan_{name}, src={value}.ap(pattern=[[0, 128], [1, {width}]], offset=scan_s*{length}+scan_t*{width}))")
    lines += [
        f"            nisa.activation(dst=scan_alpha, op=nl.exp, data=scan_delta[:, scan_t*{width}:(scan_t+1)*{width}], scale=scan_a)",
        f"            nisa.tensor_tensor(dst=scan_update, data1=scan_delta[:, scan_t*{width}:(scan_t+1)*{width}], data2=scan_b, op=nl.multiply)",
        f"            nisa.tensor_tensor(dst=scan_update, data1=scan_update, data2=scan_u[:, scan_t*{width}:(scan_t+1)*{width}], op=nl.multiply)",
        "            nisa.tensor_tensor_scan(dst=scan_state, data0=scan_alpha, data1=scan_update, initial=scan_previous, op0=nl.multiply, op1=nl.add)",
        f"            nisa.tensor_copy(dst=scan_previous, src=scan_state[:, {width-1}:{width}])",
        "            nisa.tensor_tensor(dst=scan_weighted, data1=scan_state, data2=scan_c, op=nl.multiply)",
        f"            nisa.tensor_tensor(dst=scan_acc[:, scan_t*{width}:(scan_t+1)*{width}], data1=scan_acc[:, scan_t*{width}:(scan_t+1)*{width}], data2=scan_weighted, op=nl.add)",
        f"    nisa.dma_copy(dst={output}[scan_channel*128:(scan_channel+1)*128, :], src=scan_acc)",
    ]
    return lines


def integrating_body(inputs, output, shape, states, width):
    """Realize h=exp(A*prefix(delta))*(h_in+cumsum(drive/exp(...))).

    Tile-local prefix factors avoid repeatedly multiplying rounded exp values.
    This is an opt-in numerical identity: extreme prefix exponents can overflow
    even when direct recurrence intermediates remain finite. No input values or
    benchmark IDs influence selection. All state is computed from runtime inputs.
    """
    channels, length = shape
    delta, u, a, b, c = inputs
    lines = []
    for name in ("a", "negative_a", "carry"):
        lines.append(f"scan_{name} = nl.ndarray((128, {states}), dtype=nl.float32, buffer=nl.sbuf)")
    for name in ("prefix", "du", "acc", "b", "c", "positive", "negative", "drive", "state", "weighted"):
        lines.append(f"scan_{name} = nl.ndarray((128, {width}), dtype=nl.float32, buffer=nl.sbuf)")
    lines += [
        f"for scan_channel in nl.sequential_range({channels // 128}):",
        f"    nisa.dma_copy(dst=scan_a, src={a}[scan_channel*128:(scan_channel+1)*128, :])",
        "    nisa.tensor_scalar(dst=scan_negative_a, data=scan_a, op0=nl.multiply, operand0=-1.0, engine=nisa.engine.scalar)",
        "    nisa.memset(dst=scan_carry, value=0.0)",
        f"    for scan_t in nl.sequential_range({length // width}):",
        f"        nisa.dma_copy(dst=scan_prefix, src={delta}[scan_channel*128:(scan_channel+1)*128, scan_t*{width}:(scan_t+1)*{width}])",
        f"        nisa.dma_copy(dst=scan_du, src={u}[scan_channel*128:(scan_channel+1)*128, scan_t*{width}:(scan_t+1)*{width}])",
        "        nisa.tensor_tensor(dst=scan_du, data1=scan_prefix, data2=scan_du, op=nl.multiply)",
        "        nisa.tensor_scalar_cumulative(dst=scan_prefix, src=scan_prefix, op0=nl.add, imm0=0.0, op1=nl.add, reduce_cmd=nisa.reduce_cmd.reset_reduce)",
        "        nisa.memset(dst=scan_acc, value=0.0)",
        f"        for scan_s in nl.sequential_range({states}):",
    ]
    for name, value in (("b", b), ("c", c)):
        lines.append(f"            nisa.dma_copy(dst=scan_{name}, src={value}.ap(pattern=[[0, 128], [1, {width}]], offset=scan_s*{length}+scan_t*{width}))")
    lines += [
        "            nisa.activation(dst=scan_positive, op=nl.exp, data=scan_prefix, scale=scan_a[:, scan_s:scan_s+1])",
        "            nisa.activation(dst=scan_negative, op=nl.exp, data=scan_prefix, scale=scan_negative_a[:, scan_s:scan_s+1])",
        "            nisa.tensor_tensor(dst=scan_drive, data1=scan_du, data2=scan_b, op=nl.multiply)",
        "            nisa.tensor_tensor(dst=scan_drive, data1=scan_drive, data2=scan_negative, op=nl.multiply)",
        "            nisa.tensor_scalar_cumulative(dst=scan_state, src=scan_drive, op0=nl.add, imm0=0.0, op1=nl.add, imm1=scan_carry[:, scan_s:scan_s+1], reduce_cmd=nisa.reduce_cmd.load_reduce)",
        "            nisa.tensor_tensor(dst=scan_weighted, data1=scan_state, data2=scan_positive, op=nl.multiply)",
        f"            nisa.tensor_copy(dst=scan_carry[:, scan_s:scan_s+1], src=scan_weighted[:, {width-1}:{width}])",
        "            nisa.tensor_tensor(dst=scan_weighted, data1=scan_weighted, data2=scan_c, op=nl.multiply)",
        "            nisa.tensor_tensor(dst=scan_acc, data1=scan_acc, data2=scan_weighted, op=nl.add)",
        f"        nisa.dma_copy(dst={output}[scan_channel*128:(scan_channel+1)*128, scan_t*{width}:(scan_t+1)*{width}], src=scan_acc)",
    ]
    return lines
