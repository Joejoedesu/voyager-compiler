"""One Trainium NeuronCore, described to the shared compiler.

``TrainiumConfig`` is an ``AcceleratorConfig`` -- the fields the shared passes,
tile search and memory planner read -- plus what Trainium's runtime model
(``TrainiumModel``) needs: the other engines' clocks and rates, the matmul's
moving-operand limit and DMA overheads.  ``TRN2`` and ``TRN3`` are AWS's
documented figures for one core, and the Trn2 profiles' where those are
missing or differ:

- NKI architecture guides (``nki/guides/architecture/trainium2_arch``):
  SBUF 128 partitions x 224 KiB (28 MiB) on NeuronCore-v3, TensorE 128 x 128
  at 2.4 GHz with FP8 at twice the BF16 rate and FP32 at a quarter of it (20
  vs 79 dense TFLOPS), a moving operand of at most 512 columns (one 2 KiB
  PSUM bank of float32, 8 banks a partition), VectorE at 0.96 GHz and
  ScalarE at 1.2 GHz.
- A Trn2 device profile: per partition, VectorE moves 2 16-bit elements a
  cycle and 1 when any operand is 32-bit (a PSUM accumulator is float32),
  ScalarE 1, and an instruction costs about 160 ns on VectorE and 270 ns on
  ScalarE beyond its elements.
- NKI DMA bandwidth guide: 16 DMA engines per core at 23 B/ns (TRN2) or
  33 B/ns (TRN3), about 1300 ns from semaphore to transfer start, and
  transfers under 2 KiB per partition bound by per-packet overhead.
- Trn2 profiles: back to back, a matmul or transpose streams its moving
  columns plus ``tensor_overhead_cycles``; GpSimd issues a ``dma_copy`` in
  about 650 ns; each of the 16 DMA rings moves about 27 B/ns, a row of a
  transfer costing ``dma_row_overhead_bytes`` more and a copy
  ``dma_copy_ns`` more, and idles between copies (``dma_copy_idle_ns``); a
  copy on 64 partitions or fewer moves at about half the rate (190-199
  GB/s against 438 for 342 and 171 KB copies on 32 and 64 partitions);
  a store issues from ScalarE's queue, in about 550 ns of ScalarE's time
  (a prefill step's attention, ``o_proj`` and ``gate_proj``).
- A step's compile log: neuronx-cc pins ``STEP_PINNED_SBUF`` bytes of each
  partition for its own DMA scratch, and starts a kernel's loads under the
  kernel before it, spilling them when the two do not fit
  (``HOISTED_SBUF``).

The shared compiler sees a 128 x 128 array clocked at TensorE's rate, SBUF as
an unbanked scratchpad less what the step keeps (``scratchpad_offset``), since
every buffer a kernel uses is in its tile result's placement and the plan,
and lanes of 1, since a tile's rows go on the partitions and its last dim is
free.
"""

from dataclasses import dataclass, fields, replace

from voyager_compiler.codegen.node_info import is_linear, is_matmul
from voyager_compiler.compiler.bufferize import (
    BuilderRule,
)
from voyager_compiler.hardware import (
    AcceleratorConfig,
    Bandwidth,
    BandwidthUnit,
    CapacityUnit,
    ComputationUnit,
    Connection,
    DataType,
    MemoryHierarchy,
    MemoryInstance,
    MemoryLevel,
    MemorySize,
    SpatialUnrolling,
    StorageTarget,
)
from voyager_compiler.ir.plan import ExecutionOptions

from .attention import SDPA_KMAJOR, build_attention
from .gemm import build_gemm
from .model import TrainiumModel
from .utils import HOISTED_SBUF, PARTITIONS

# Bytes of each SBUF partition a step's neuronx-cc compile pins for itself:
# its dynamic DMA scratch (``--dynamic-dma-scratch-size-per-partition``,
# 16 KiB) and 8 more.  A kernel's buffers beside them would be spilled.
STEP_PINNED_SBUF = 16392

# A dense GEMM -- ``linear``, ``matmul`` and their MX forms -- and a K-major
# flash attention lower by Trainium's own builders.
GEMM_RULE = BuilderRule(
    "trainium_gemm",
    lambda node, anchor: is_linear(anchor) or is_matmul(anchor),
    build_gemm,
)
ATTENTION_RULE = BuilderRule(
    "trainium_attention",
    lambda node, anchor: anchor.target is SDPA_KMAJOR,
    build_attention,
)


@dataclass(frozen=True)
class TrainiumCalibration:
    """Versioned seed parameters for the performance model, not capacities.

    Values come from the compiler-2 Trn2 profile. Offline fitting produces a
    new profile; it does not rewrite the physical device graph.
    """

    version: str = "compiler-2-b850a39"

    vector_elements_16: int = 2
    vector_elements_32: int = 1
    scalar_elements: int = 1
    tensor_overhead_cycles: int = 6
    vector_overhead_ns: float = 160.0
    scalar_overhead_ns: float = 270.0
    fp8_matmul_speedup: float = 2.0
    fp32_matmul_rate: float = 0.25
    fp32_transpose_rate: float = 0.5
    dma_issue_ns: float = 650.0
    dma_row_overhead_bytes: int = 200
    dma_copy_ns: float = 31.0
    dma_copy_idle_ns: float = 80.0
    dma_half_rate_partitions: int = 64
    dma_store_issue_ns: float = 550.0
    kernel_overhead_ns: float = 0.0


@dataclass(frozen=True, init=False)
class TrainiumConfig(AcceleratorConfig):
    """Trainium physical graph with a separate performance-profile identity."""

    profile: TrainiumCalibration

    def __init__(
        self,
        *,
        target="trn2",
        profile=None,
        pe_array_size=(128, 128),
        vector_unit_width=1,
        frequency=2.4,
        scratchpad_size=28 << 20,
        scratchpad_offset=None,
        double_buffered_l2=True,
        async_pipeline=False,
        dram_size=12.0,
        dram_bandwidth=432.0,
        dram_access_latency=1300.0,
        psum_banks=8,
        psum_bank_bytes=2048,
        vector_ghz=0.96,
        scalar_ghz=1.2,
        **graph_or_profile,
    ):
        profile = profile or TrainiumCalibration()
        profile_names = {f.name for f in fields(TrainiumCalibration)}
        updates = {
            k: graph_or_profile.pop(k)
            for k in tuple(graph_or_profile)
            if k in profile_names
        }
        if updates:
            profile = replace(profile, **updates)
        if "computation_units" in graph_or_profile:
            # dataclasses.replace preserves the graph and changes profile fields.
            super().__init__(frequency=frequency, **graph_or_profile)
        else:
            if graph_or_profile:
                raise TypeError(
                    f"unknown Trainium configuration fields: {sorted(graph_or_profile)}"
                )
            partitions, columns = pe_array_size
            if scratchpad_offset is None:
                scratchpad_offset = partitions * (
                    STEP_PINNED_SBUF + HOISTED_SBUF
                )
            dtypes = (
                DataType("bfloat16", 16),
                DataType("float32", 32),
                DataType("fp8_e4m3", 8),
            )
            units = (
                ComputationUnit(
                    "matrix",
                    dtypes,
                    frozenset(("matmul", "transpose")),
                    (
                        SpatialUnrolling("IC", partitions),
                        SpatialUnrolling("OC", columns),
                    ),
                    frequency_ghz=frequency,
                ),
                ComputationUnit(
                    "vector",
                    dtypes,
                    frozenset(("elementwise", "reduction")),
                    (
                        SpatialUnrolling("lanes", vector_unit_width),
                        SpatialUnrolling("pool_channels", vector_unit_width),
                    ),
                    frequency_ghz=vector_ghz,
                ),
                ComputationUnit(
                    "scalar",
                    dtypes,
                    frozenset(("scalar",)),
                    frequency_ghz=scalar_ghz,
                ),
            )
            shared = frozenset(StorageTarget)
            memories = MemoryHierarchy(
                (
                    MemoryLevel(
                        "local",
                        (
                            MemoryInstance(
                                "psum",
                                MemorySize(
                                    partitions * psum_banks * psum_bank_bytes,
                                    CapacityUnit.BYTES,
                                ),
                                frozenset((StorageTarget.PSUM,)),
                                partitions=partitions,
                                banks=psum_banks,
                                row_bytes=psum_bank_bytes,
                            ),
                        ),
                    ),
                    MemoryLevel(
                        "on_chip",
                        (
                            MemoryInstance(
                                "scratchpad",
                                MemorySize(scratchpad_size, CapacityUnit.BYTES),
                                shared,
                                partitions=partitions,
                                reserved_bytes=scratchpad_offset,
                            ),
                        ),
                    ),
                    MemoryLevel(
                        "off_chip",
                        (
                            MemoryInstance(
                                "dram",
                                MemorySize(
                                    dram_size * 1024**3, CapacityUnit.BYTES
                                ),
                                shared,
                            ),
                        ),
                    ),
                )
            )
            edges = (
                Connection(
                    "dram_sram",
                    "dram",
                    "scratchpad",
                    Bandwidth(dram_bandwidth, BandwidthUnit.GB_PER_SECOND),
                    latency_ns=dram_access_latency,
                ),
                Connection(
                    "sbuf_tensor",
                    "scratchpad",
                    "matrix",
                    Bandwidth(None, BandwidthUnit.BYTES_PER_CYCLE),
                ),
                Connection(
                    "psum_tensor",
                    "psum",
                    "matrix",
                    Bandwidth(None, BandwidthUnit.BYTES_PER_CYCLE),
                ),
                Connection(
                    "sbuf_vector",
                    "scratchpad",
                    "vector",
                    Bandwidth(None, BandwidthUnit.BYTES_PER_CYCLE),
                ),
                Connection(
                    "psum_vector",
                    "psum",
                    "vector",
                    Bandwidth(None, BandwidthUnit.BYTES_PER_CYCLE),
                ),
                Connection(
                    "sbuf_scalar",
                    "scratchpad",
                    "scalar",
                    Bandwidth(None, BandwidthUnit.BYTES_PER_CYCLE),
                ),
            )
            super().__init__(
                name=target,
                target_name="trainium",
                frequency=frequency,
                computation_units=units,
                memory=memories,
                connections=edges,
                execution=ExecutionOptions(
                    2 if double_buffered_l2 else 1, async_pipeline
                ),
            )
        object.__setattr__(self, "profile", profile)

    def __getattr__(self, name):
        profile = object.__getattribute__(self, "profile")
        try:
            return getattr(profile, name)
        except AttributeError:
            raise AttributeError(name) from None

    @property
    def target(self):
        return self.name

    @property
    def psum_banks(self):
        return self.memory_instance("psum").banks

    @property
    def psum_bank_bytes(self):
        return self.memory_instance("psum").row_bytes

    @property
    def vector_ghz(self):
        return self.compute_unit("vector").frequency_ghz

    @property
    def scalar_ghz(self):
        return self.compute_unit("scalar").frequency_ghz

    @property
    def partitions(self) -> int:
        """SBUF partitions: TensorE's rows, the matmul contraction."""
        return self.pe_array_size[0]

    @property
    def moving_free_max(self) -> int:
        """Columns of a matmul's moving operand: a PSUM bank of float32."""
        return self.psum_bank_bytes // 4

    @property
    def psum_bytes(self) -> int:
        """PSUM bytes per partition."""
        return self.psum_banks * self.psum_bank_bytes

    def runtime_model(self, accumulate_fp32):
        """This core's ``TrainiumModel``.

        Args:
            accumulate_fp32: A GEMM split along its reduction accumulates in
                float32.
        """
        return TrainiumModel(self, accumulate_fp32)

    def target_spec(self):
        from .target import TARGET

        return TARGET


def _core(target, sbuf_bytes, dram_bandwidth):
    """A core of ``target``: ``sbuf_bytes`` of SBUF, less ``STEP_PINNED_SBUF``
    and ``HOISTED_SBUF`` of each partition, and ``dram_bandwidth`` GB/s of
    DMA."""
    return TrainiumConfig(
        target=target,
        pe_array_size=(PARTITIONS, 128),
        vector_unit_width=1,
        frequency=2.4,
        scratchpad_size=sbuf_bytes,
        scratchpad_offset=PARTITIONS * (STEP_PINNED_SBUF + HOISTED_SBUF),
        double_buffered_l2=True,
        async_pipeline=False,
        dram_size=12.0,
        dram_bandwidth=dram_bandwidth,
        dram_access_latency=1300.0,
    )


# The ``transform`` layouts: every GEMM weight contraction-major (``"ck"``,
# [in, out]; K is the output dim), so a weight tile lands with the
# contraction on the partitions TensorE contracts over, the activations
# between matrix-matrix GEMMs contraction-major (``"cx"``) for the same
# reason, and feature maps NCHW, so a pool tile has H on the partitions and W
# on the free axis.
LAYOUT = {
    "tensor_layout": "nchw",
    "gemm_layout": "cx,ck",
    "gemv_layout": "xc,ck",
}

TRN2 = _core("trn2", 28 << 20, 16 * 27.0)
TRN3 = _core("trn3", 30 << 20, 16 * 33.0)
