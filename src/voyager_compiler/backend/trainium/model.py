"""Trainium's runtime and fit model, ``TrainiumModel``.

What the shared tile search asks about one NeuronCore, answered from its
``TrainiumConfig``.  A tile lives in SBUF with its rows on the partitions and
its last dim free.  ``TrainiumModel.prepare`` hands each op to its op class,
whose fit places the kernel's buffers and whose cost prices its steps the way
the builders and the translator run them: ``TrainiumGemm`` and
``TrainiumGemv`` (``gemm.py``), ``TrainiumAttention`` (``attention.py``),
``TrainiumVector`` and ``TrainiumPool`` (``vector.py``).  The model prices
what they share:

- one VectorE or ScalarE instruction, an op's passes over a tile and a
  ``quantize_mx``, which transposes its tile on TensorE 128 columns at a
  time before quantizing each chunk;
- a DMA, which moves a tile in as few ``dma_copy`` instructions as access
  patterns walk (``_dma_count``), a load of ``STREAM_BYTES`` or more in one
  per piece of its 128-row blocks (``stream_blocks``): each its issue, then
  the rows' bytes, packet overhead included, at the core's DMA bandwidth;
- an MX (microscaling) tile, stored contraction-major: 4 consecutive
  contraction elements per 32-bit word, a 512-element chunk per 128
  partitions, chunks side by side on the free dim, and one E8M0 scale byte
  per 32 elements in a tile of the same layout, filled only on each
  quadrant's first 4 partitions;
- a sweep: the engines run concurrently, so a step costs its busiest
  engine, and the shared schedule overlaps a step's DMAs with the compute
  around it (``sweep_cycles``).

Times are nanoseconds here and TensorE cycles to the search.
"""

import math
from collections import Counter

import torch

import interstellar
from voyager_compiler.codegen.node_info import (
    get_anchor_node,
    is_conv2d,
    is_fully_connected,
    is_gemm_op,
    is_pooling,
    reads_kmajor,
)

# The bufferizer's own rules for which nodes become kernels and which reuse
# another's build.
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    _bufferize_key,
    _is_compute,
)
from voyager_compiler.codegen.transform.tiling.runtime_model import (
    AttentionSpace,
    GridSpace,
    MatrixSpace,
    RuntimeModel,
)
from voyager_compiler.codegen.transform.tiling.traversal import sweep_cycles

from .attention import TrainiumAttention
from .gemm import TrainiumGemm, TrainiumGemv
from .utils import (
    MX_PACK,
    QUADRANT,
    QUADRANT_SCALES,
    extents,
    output_buffers,
    rows_cols,
    sbuf_bytes,
)
from .vector import TrainiumPool, TrainiumVector

MAX_POOL2D = torch.ops.aten.max_pool2d.default


# A load of at least ``STREAM_BYTES`` moves in pieces of whole 128-row
# blocks: each piece the fewest blocks that hold ``PIECE_BYTES``, the last
# one the blocks left over, so an op reading a block waits for its piece
# alone; a shorter load lasts about a transfer's fixed latency.
STREAM_BYTES = 1 << 20
PIECE_BYTES = 256 << 10


def stream_blocks(blocks, block_bytes):
    """The 128-row blocks, ``block_bytes`` each, that one ``dma_copy`` of a
    load of ``blocks`` moves: all of them, or for a load of
    ``STREAM_BYTES`` or more the fewest that hold ``PIECE_BYTES``."""
    if blocks * block_bytes >= STREAM_BYTES:
        return math.ceil(PIECE_BYTES / block_bytes)
    return blocks


def _dma_count(tile, full, partitions):
    """``dma_copy`` instructions the translator moves a ``tile`` of a
    tensor ``full`` in: one when its rows lie evenly apart across every
    partition and row block, else one per run of evenly spaced rows -- rows
    along the second-to-last dim, continuing into the dim above while the
    dims below it are whole -- or per partition block."""
    rows = math.prod(tile[:-1])
    run = tile[-2] if len(tile) > 1 else 1
    full = (1,) * (len(tile) - len(full)) + tuple(full)[-len(tile) :]
    for d in range(len(tile) - 2, 0, -1):
        if tile[d] != full[d]:
            break
        run *= tile[d - 1]
    if run >= rows or run % partitions == 0:
        return 1
    if run < partitions:
        return math.ceil(rows / run)
    return math.ceil(rows / partitions)


class TrainiumModel(RuntimeModel):
    """One NeuronCore: TensorE, VectorE and ScalarE beside SBUF and PSUM,
    and DMA to HBM.

    NKI places SBUF and PSUM buffers itself, so a tiling that fits takes a
    ``TrainiumPlacement`` of the buffers its kernel computes in, at no
    address.

    Attributes:
        matrix_space: How a GEMM is mapped: a 128 x 128 array, SBUF as one
            byte pool and HBM.  L1 holds no tile of its own -- the GEMM
            builder cuts each step into instructions itself -- so its
            blockings are pinned to 1 and the search maps the PE partition,
            the SBUF tile and the HBM grid.  The reduction stays innermost at
            L3, where the builders accumulate it.
        grid_space: How a vector op or a matrix-vector GEMM is cut: lanes of
            1, since a tile's rows go on the partitions and its last dim is
            free, and every tiling that fits is priced.
        attention_space: How a flash attention is cut: query and key tiles
            in whole partitions, the head unpadded, since it lies on the
            partitions QKᵀ contracts over or is P·V's moving free dim.
    """

    def __init__(self, config, accumulate_fp32):
        super().__init__(config, accumulate_fp32)
        p, q = config.pe_array_size
        architecture = interstellar.Resource(
            buf_capacity_list=[
                [1, 1, 1],
                # Unbounded: L1 only passes the PE tile through.
                [math.inf, math.inf, math.inf],
                [config.usable_scratchpad_size],
                [config.dram_size * 1024**3],  # GB -> bytes
            ],
            buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100], [1000]],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            para_count_list=[p * q, 1, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            mac_capacity=0,
            partition_mode=[0, 0, 0, 0],
            invalid_underutilized=False,
            bank_size_list=[None, None, None, None],
        )
        unit = {"blocking_size": 1, "partitioning_size": 1}
        hints = {
            loop: {"level1": dict(unit)} for loop in ("OX", "OY", "OC", "ON")
        }
        hints["IC"] = {
            "level0": {"order": 1, "partitioning_size": p},
            "level1": dict(unit),
            "level3": {"order": 0},
        }
        hints["OC"]["level0"] = {"order": 0, "partitioning_size": q}
        for loop in ("FX", "FY"):
            hints[loop] = {f"level{i}": dict(unit) for i in range(4)}
        schedule_data = interstellar.extract_input.extract_schedule_info(
            {"schedule_hint": hints}, 4
        )
        schedule = interstellar.Schedule(
            schedule_data["schedule_hint"], schedule_data["partition_loops"]
        )
        self.matrix_space = MatrixSpace(architecture, schedule)
        self.grid_space = GridSpace(lanes=1, ranked=True)
        self.attention_space = AttentionSpace(
            unit=config.partitions, head_multiple=1
        )

    def prepare(self, node):
        """The model of ``node``, bare or fused: a ``TrainiumGemm`` for a
        GEMM, a ``TrainiumGemv`` for a matrix-vector one, which the grid
        search tiles, a ``TrainiumPool`` for an NCHW 2-D max pool and a
        ``TrainiumVector`` for a vector op.

        Raises:
            NotImplementedError: ``node`` is a convolution or another pool,
                which do not lower to Trainium.
        """
        anchor = get_anchor_node(node)
        if anchor.target is MAX_POOL2D:
            return TrainiumPool(node, self)
        if is_conv2d(anchor) or is_pooling(anchor):
            raise NotImplementedError(f"{anchor}: not lowered on Trainium")
        if not is_gemm_op(anchor):
            return TrainiumVector(node, self)
        if is_fully_connected(anchor):
            return TrainiumGemv(node, self)
        return TrainiumGemm.from_node(node, self)

    def prepare_attention(self, node, acc_dtype, bool_mask):
        """A ``TrainiumAttention`` for the flash-attention ``node``.

        Raises:
            NotImplementedError: ``node`` is not K-major (``sdpa_kmajor``):
                a row-major or MX attention does not lower to Trainium.
        """
        if not reads_kmajor(node):
            raise NotImplementedError(f"{node}: not lowered on Trainium")
        return TrainiumAttention(node, self, acc_dtype, bool_mask)

    def engine_ns(self, engine, elements, bits):
        """One instruction on ``engine`` over ``elements`` per partition, its
        widest operand ``bits`` wide."""
        config = self.config
        if engine == "scalar":
            rate, ghz = config.scalar_elements, config.scalar_ghz
            fixed = config.scalar_overhead_ns
        else:
            rate = (
                config.vector_elements_16
                if bits <= 16
                else config.vector_elements_32
            )
            ghz, fixed = config.vector_ghz, config.vector_overhead_ns
        return math.ceil(elements / rate) / ghz + fixed

    def dma_ns(self, rows, cols, bits, count=1, load=False):
        """A ``rows x cols`` tile of ``bits``-wide elements moved in
        ``count`` ``dma_copy`` instructions; a ``load`` moved in one is cut
        into one per piece of its 128-row blocks (``stream_blocks``)."""
        config = self.config
        row_bytes = math.ceil(cols * bits / 8)
        if load and count == 1:
            parts = min(rows, config.partitions)
            blocks = math.ceil(rows / parts)
            count = math.ceil(blocks / stream_blocks(blocks, parts * row_bytes))
        moved = rows * (row_bytes + config.dma_row_overhead_bytes)
        return count * config.dma_issue_ns + moved / config.dram_bandwidth

    def _mx_layout(self, depth, other):
        """``(partitions, words)`` of an MX tile contracting ``depth`` beside
        ``other`` elements: a word per 4 contraction elements, the first 512
        on the partitions and every further chunk of 512 beside them."""
        p = self.config.partitions
        chunks = math.ceil(depth / (p * MX_PACK))
        return min(p, math.ceil(depth / MX_PACK)), chunks * other

    def mx_sbuf(self, depth, other, scales):
        """SBUF bytes of an MX codes tile, or of its ``scales``: one byte per
        word's place, though only each quadrant's first 4 partitions hold
        one."""
        partitions, words = self._mx_layout(depth, other)
        bits = 8 if scales else 32
        return sbuf_bytes(partitions, words, bits, self.config.partitions)

    def mx_dma_ns(self, depth, other, scales):
        """One transfer of an MX codes tile in one ``dma_copy``, or of its
        ``scales`` in one per quadrant."""
        partitions, words = self._mx_layout(depth, other)
        if not scales:
            return self.dma_ns(partitions, words, 32)
        quadrants = math.ceil(partitions / QUADRANT)
        return quadrants * self.dma_ns(QUADRANT_SCALES, words, 8)

    def buffer_sbuf(self, n, shape, mx):
        """SBUF bytes of buffer ``n``'s tile ``shape``: one array per output,
        rows on the partitions, or the MX layout ``mx``
        (``TrainiumGrid._mx_tiles``) gives it."""
        if n in mx:
            return sum(self.mx_sbuf(*array) for array in mx[n])
        p = self.config.partitions
        return sum(
            sbuf_bytes(*rows_cols(s), bits, p)
            for s, bits in output_buffers(shape, n)
        )

    def buffer_dma_ns(self, n, shape, mx, load):
        """One transfer of buffer ``n``'s tile ``shape``, a ``load`` or a
        store: each output's ``dma_copy`` instructions (``_dma_count``), or
        the MX layout's transfers."""
        if n in mx:
            return sum(self.mx_dma_ns(*array) for array in mx[n])
        values = n.value if isinstance(n.value, (tuple, list)) else [n.value]
        p = self.config.partitions
        return sum(
            self.dma_ns(
                *rows_cols(s),
                bits,
                _dma_count(tuple(s), value.shape, p),
                load,
            )
            for (s, bits), value in zip(output_buffers(shape, n), values)
        )

    def quantize_mx_busy(self, rows, cols):
        """Each engine's busy time quantizing a ``rows x cols`` tile, rows on
        the partitions, to MX along its columns: per 128 rows and 512
        columns, TensorE transposes every 4th column into PSUM 4 times,
        VectorE copies each out into a staging tile and runs one
        ``quantize_mx`` over it."""
        config = self.config
        overhead = config.tensor_overhead_cycles
        chunks = math.ceil(cols / (config.partitions * MX_PACK))
        busy = Counter()
        for r in extents(rows, config.partitions):
            transposes = MX_PACK * chunks
            busy["tensor"] += transposes * (r + overhead) / config.frequency
            busy["vector"] += transposes * self.engine_ns("vector", r, 16)
            busy["vector"] += chunks * self.engine_ns("vector", MX_PACK * r, 16)
        return busy

    def passes_busy(self, passes, rows, cols):
        """Each engine's busy time making ``passes`` -- ``(engine, share,
        bits)`` -- over a ``rows x cols`` tile, 128 rows per instruction."""
        row_chunks = math.ceil(rows / self.config.partitions)
        busy = Counter()
        for engine, share, bits in passes:
            elements = math.ceil(cols * share)
            busy[engine] += row_chunks * self.engine_ns(engine, elements, bits)
        return busy

    def sweep_ns(self, dmas, steps, compute):
        """A sweep of ``steps`` grid steps of ``compute`` each, moving
        ``dmas`` -- ``(ns, transfers)`` per buffer, the output first --
        overlapped by the shared schedule when double-buffered; the first
        DMA's latency and the kernel's fixed cost come on top."""
        config = self.config
        if config.double_buffered_l2:
            sweep = sweep_cycles(dmas, steps, compute)
        else:
            sweep = sum(ns * count for ns, count in dmas) + steps * compute
        return sweep + config.dram_access_latency + config.kernel_overhead_ns


def kernel_nodes(gm):
    """The nodes the bufferizer turns into kernels, one each and in order,
    with the key under which it reuses the build of an identical one.  Taken
    before ``compile``, which replaces them."""
    return [(n, _bufferize_key(n)) for n in gm.graph.nodes if _is_compute(n)]


def predicted_seconds(nodes, config):
    """The latency the tile search predicted for each kernel of ``nodes``
    (``kernel_nodes``): the cost of its winning tiling, stamped as
    ``meta["tile_result"]`` on a GEMM's anchor and on a GEMV or vector op
    itself.  A kernel whose build was reused takes the prediction of the one
    it reuses.

    Returns:
        Seconds per kernel, in order.
    """
    priced, predicted = {}, []
    for node, key in nodes:
        anchor = get_anchor_node(node) or node
        result = anchor.meta.get("tile_result") or node.meta.get("tile_result")
        if result is not None:
            seconds = result.cost.latency / (config.frequency * 1e9)
            priced.setdefault(key, seconds)
        predicted.append(priced[key])
    return predicted
