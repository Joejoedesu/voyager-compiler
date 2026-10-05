# Trainium mapping and analytical cost

The default configuration models **one physical NeuronCore**. For small NKI
transfers, use the per-engine DMA model below, not device HBM bandwidth or
AccelOpt's roofline normalization. These estimates rank supported mappings;
they are not calibrated device timings. No hardware measurements are available.

| Quantity | NeuronCore-v2 | NeuronCore-v3 | Basis |
|---|---:|---:|---|
| SBUF | 24 MiB, 128 partitions | 28 MiB, 128 partitions | AWS architecture |
| PSUM | 2 MiB, 8 banks | 2 MiB, 8 banks | AWS architecture |
| Tensor / Vector / Scalar clock | 2.8 / 1.12 / 1.4 GHz | 2.4 / .96 / 1.2 GHz | AWS architecture |
| DMA engines | 16 | 16 | AWS DMA guide |
| Bandwidth per DMA engine | 17 GB/s | 23 GB/s | AWS DMA guide |
| Aggregate modeled DMA payload bandwidth | 272 GB/s | 368 GB/s | 16 engines, one core |
| Per-command startup charge | 1300 ns | 1300 ns | Conservative modeling assumption |
| SBUF reservation | 4 MiB | 4 MiB | Software policy for NKI temporaries |

Sources: [v2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html),
[v3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html),
[DMA guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/deep-dives/nki-dma-bandwidth-guide.html).
The last source describes approximately 1300 ns of cross-engine delay. Charging
that delay serially for every emitted command is an assumption, **not a
documented DMA initiation interval**. `zero_startup_sensitivity_ns` reprices
the same mapping without that charge; neither scenario is a measured bound.
The HBM connection's `latency_ns` and bandwidth are the model inputs, so
`dataclasses.replace` can vary them for sensitivity or future calibration.

## Shared flow and policy ownership

The Gemmini 9-30 checkout supplied the shared `VoyagerMappingPolicy`,
`KernelBufferPlan`, and runtime-only Interstellar selector hooks. Its DMA sweep
informed the prologue/drain recurrence. Gemmini's physical constants, systolic
execution rules and queue depths are not imported.

`trainium/mapping.py` owns legality and capacity; `trainium/cost.py` owns
candidate service time. `tiler.py` still prepares and runs Interstellar. Its
winning immutable buffer plan is consumed by the existing convolution/GEMM
builders. The same shared allocator and collateral emitter remain in use.
`trainium/converter.py` decomposes the scheduled software tiles into NKI ISA
panels. It does not choose new HBM windows or an independent loop schedule.

| Rule | Default Voyager | Trainium |
|---|---|---|
| L1 stores | Separate operand capacities | Disabled as physical capacities |
| L2 operand allocation | Voyager size/bank policy | Unified SBUF, partition pitch, actual slot counts |
| L2 IC order | Innermost | Unrestricted |
| External IC reduction | Consecutive | Retained: builder requirement |
| Convolution filter tiling | Whole filter within software tile | Retained: builder requirement |
| Inner loop order | Voyager representative order | Retained to avoid duplicate software tiles; converter fixes ISA order |
| Whole GEMM tile <=128 | Previous Trainium restriction | Removed; split into legal ISA panels |
| Objective | Runtime tolerance then energy/traffic | Strict minimum modeled latency; first exact tie |
| Fully connected/GEMV | Vector route available | TensorE route included in shared matrix search |
| Vector/pool sizing | Voyager banking and timing | Partition-aware capacity and target service estimate |

The shape transform still pads dimensions using the 128-wide configuration.
This limits tail efficiency and search granularity. Disabling padding requires
extending the shared search's spatial partition factors; simply turning off
padding while retaining a fixed 128-way partition would make tails unmappable.
Resident execution and alternate convolution layouts are not supported here.

## What is charged

**DMA.** The converter currently subdivides each collateral copy into panels
of at most 128 rows by 128 columns; higher-rank windows preserve the penultimate
dimension's row boundaries. Charge the busiest DMA engine's payload, up to
eight active partitions per engine, plus startup for each panel. Narrow
transfers do not receive full-core bandwidth. Read and write traffic share
the modeled DMA service budget. Bias loads, reload counts, transpose staging
and PSUM-to-SBUF copies are included. Halos with masked-out padding are priced
conservatively at the full scheduled extent.

**TensorE.** For a panel of logical `(M,N,K)`, B is stationary and A is moving.
Require M<=512, N<=128, K<=128. Use `max(min(64,N),M)` TensorE cycles for
BF16/FP16 and four times that for FP32. Short K does not reduce those cycles.
Software panels larger than an instruction incur local operand gathers.
See [nc_matmul](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.isa.nc_matmul.html).

**Layout and vector work.** Tensor transpose is estimated as
`max(P,min(64,F))` TensorE cycles; its PSUM eviction and explicit copies use
`max(64,F)` VectorE cycles. These use different engine clocks before summing
nanoseconds. Split-K additions and final bias/activation passes are charged.
Tiny transpose engine selection, compiler copy elimination and implicit
copies are not precisely modeled. See
[nc_transpose](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.isa.nc_transpose.html)
and [tensor_copy](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.isa.tensor_copy.html).
Vector/pool candidates use a simpler primitive count, including ScalarE
activation time; reductions and compound operations need calibration.

**Overlap.** A double-buffer recurrence charges first loads, recurring
compute/transfers, and the final store. Operand reload counts follow the
same external loop order as the builder. On-chip stages within a task are
serialized conservatively. Reload-only layout work is averaged across tasks;
this is not a dependency-exact simulator of the final NEFF. Queue depths,
bank conflicts, instruction issue overlap, launch overhead, compiler spills
and NKI rewrites remain uncalibrated. The matrix estimate excludes standalone
graph-boundary pad/slice/permute kernels; it is not whole-model latency.

## Performance and utilization

For fixed useful work, minimizing predicted latency maximizes useful
throughput. Maximizing raw engine-active time would also reward redundant
transposes, padding and transfers, so it is not the objective.

Reports distinguish scheduled FLOP/s and Tensor peak fraction, HBM payload
fraction, and estimated engine **service-time fractions**. The DMA service
fraction includes assumed startup; it can approach 100% with poor achieved
payload bandwidth. Tensor service includes transposes. Scheduled FLOPs include
padding, so they are not necessarily useful model FLOPs. SBUF occupancy is a
capacity check, not a performance objective.

[AccelOpt Table 4](https://arxiv.org/html/2511.15915v2) uses FP32 roofline
normalizers of 440.2/640 GB/s and 23.75/19.75 TFLOP/s for v2/v3. Keep these
as a separate paper-comparison scenario. They do not replace the DMA guide's
partition-level service model. Matching a roofline percentage by changing its
denominator does not establish accurate timing, and the paper's profiled
engine-active fractions are not the service fractions above.

The biggest present optimization gaps are DMA coalescing beyond 128x128,
keeping a useful partition layout across operations, dependency-aware engine
overlap, persistent PSUM across external K tiles, and unpadded tail search.
The model now exposes several of these costs, but cannot search optimizations
the converter/shared builders do not express. Validate candidate ranking with
future hardware sweeps before treating absolute latency as predictive.
