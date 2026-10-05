# Trainium through Voyager's bufferized pipeline

This checkout starts from `54ecee2a62b46b7bb72a7d33f3a7d4351d0c22da`, the
same committed base as `voyager-trainium` and `voyager-gemmini-9-30`. It does
not import the earlier Trainium semantic-export flow or whole-operation NKI
templates. The Gemmini checkout was consulted for shared lowering hooks and
static collateral consumption. No commits were created.

## Shared stages and target differences

| Stage | Default Voyager | Trainium |
| --- | --- | --- |
| Hardware / precision policy | Voyager graph and family recipes | Physical NeuronCore-v2/v3 graph; native FP32/BF16 recipes |
| Export | `export_model` | Same |
| Transform | `_transform_voyager`: shape propagation, canonicalization, padding, layout normalization, fusion, deduplication | Same function; existing systolic/NHWC layout policy and a restricted fusion pattern list |
| Tiling | `build_interstellar_tiler` and existing search/cost model | Same Interstellar enumeration; Trainium mapping/size policy and instruction/DMA cost, minimum-latency objective |
| Bufferization | `bufferize_graph`, destination-passing kernels, pipelined loops, copy windows, slots, waits, reductions | Same `per_kernel` flow, consuming the buffer-slot plan evaluated by search |
| Allocation | `plan_memory`, alias/lifetime analysis and best-fit placement | Same analysis and best-fit allocator, with partition-aware SBUF allocation sizes/alignment |
| Collateral emission | `emit_program` / `gen_code_bufferized` | Same `model.txt`, `layers.txt`, `tensor_files/`; adds `hardware.json` |
| Target code | Voyager collateral consumer | `trainium.converter` reads the serialized bufferized `model.txt` and emits `nki/program.py` and `nki/plan.json` |
| Machine compilation | Target toolchain | AWS NKI cross-compiles Python to NEFF; CPU simulation checks values |

The conversion does not search for new GEMM tiles. It follows the generated
loops, predicates, DMA windows, software-pipeline slots, reduction order, and
fused primitive operations. Static scalar control is specialized into Python
source; unsupported control or compute raises an error. A source SHA and
conversion events in `plan.json` make this traceable to the input collateral.
DMA windows may be subdivided into legal NKI instructions without changing the
scheduled tensor window. Transposes adapt row-major HBM data to the NKI
partition axis. The converter performs no tensor arithmetic on the CPU.

**NKI owns final physical allocation and engine scheduling.** Voyager buffer
identities and slots are preserved as NKI arrays, and source addresses and
semaphore events are recorded, but absolute SBUF addresses are not imposed on
NKI. Explicit waits are checked for balanced posts; instruction dependencies
enforce order in generated NKI. Voyager's proposed engine overlap is not a
cycle-accurate guarantee of the final NEFF schedule.

## Hardware and constraints

`trainium.hardware.neuron_core(2)` describes 24 MiB SBUF; version 3 describes
28 MiB. Both have 128 SBUF partitions and 2 MiB PSUM in eight banks. The graph
includes TensorE, VectorE, ScalarE, GpSimdE, DMA and SyncE, with documented
connectivity; it does not invent Voyager's physical L1 stores. Source URLs
are recorded in the hardware provenance.

Scheduled GEMM software tiles may contain several legal TensorE instructions.
The converter uses stationary B panels with N,K <=128 and moving A panels
with M<=512. Its output free-axis limit of 4096 is a **software support limit**,
not the physical SBUF capacity. Convolution currently limits scheduled N,K to
128 and spatial M to 512. SBUF sizing uses a 16-byte pitch per partition,
reserves 4 MiB for compiler/instruction temporaries, and shares the existing
lifetime/alias allocator. Retained operands use one slot; changing operands
and multiple output generations use two. Outer batches count as changes.

TensorE instructions accumulate in FP32 PSUM across the K panels within a
software tile. External split-K reductions still round at the bufferized SBUF
boundaries for BF16. NKI owns final physical allocation; the capacity estimate
is not proof that arbitrary programs fit without compiler spills.

The four Interstellar levels are traversal abstractions, not four physical
memories. Trainium disables Voyager's L1 capacities, bank-rounded operand
sizes, L2 IC-innermost rule, SRAM/stream timing equations, and energy/traffic
tradeoffs. It retains consecutive external reductions and unsplit convolution
filters because the shared builders require them. Default dimension padding
and 128-channel search granularity remain implementation limitations; they
are **not** Trainium ISA requirements. The default Voyager policy is unchanged.

See [trainium-cost.md](trainium-cost.md) for rates, formulas, search ownership,
removed constraints, sensitivity and remaining model limitations. `hardware.json`
contains the winning matrix-kernel estimate and its scope. Vector/pool tiling
uses a separate target service estimate inside the shared enumeration; it is
less detailed and currently does not produce a full per-engine report.

## Use

Run Voyager in the existing compiler environment:

```python
import voyager_compiler as vc
from voyager_compiler.trainium.hardware import neuron_core

config = neuron_core(2)  # or 3
graph = vc.export_model(model, inputs)
vc.transform(graph, inputs, config=config)
vc.compile(graph, inputs, config=config, output_dir="artifacts/example")
```

Reconvert previously generated bufferized collateral independently:

```bash
PYTHONPATH=src /home/zhouhua/Research/ML/ml-env/bin/python \
  -m voyager_compiler.trainium artifacts/example --target trainium-v2
```

The independent CPU simulation/cross-compilation runner and generated artifacts
live in `/home/zhouhua/Research/ML/Trainium/bufferized-10-05`; see its `README.md`.
The earlier `bufferized-10-01` results preserve the pre-change comparator.
It uses `/home/zhouhua/Research/ML/Trainium/env`, with pinned Neuron 2.20. The
compiler itself does not import the Neuron SDK. A NEFF is a compiled Neuron
executable, not evidence of device execution or measured performance.

## Validation and present scope

`test/test_trainium.py` verifies that shared tiling, bufferization, memory
planning and emission actually execute; split-K emits the expected compute,
DMA and synchronization operations; serialized reconversion is reproducible;
and malformed collateral/waits and illegal instruction sizes fail.

The standalone cases cover split-K, multiple output tiles, nonaligned tails,
batch-one/vector GEMM, batched GEMM, BF16 GEMM, linear+bias+ReLU, residual+ReLU,
pointwise add/multiply, SiLU, padded convolution+ReLU, max pooling with and
without padding, and global average pooling. References are PyTorch outputs,
with a separate check of the executable bufferized FX graph. FP32 NKI tolerance
is 5e-4 absolute/relative; BF16 requires relative L2 <= 1% and max error <= 0.5.

Supported convolution is NHWC/HWIO, batch-one tiles, groups=1 and dilation=1;
the bufferizer supplies halo padding. Pooling supports batch-one tiles and
global average pooling. Unsupported layouts/operators are rejected. Resident
flow, dynamic tensor-dependent control, quantized Voyager storage, and complete
attention/normalization kernels are not implemented. Static source expansion
has a two-million-operation guard.

These representative tests do **not** validate full ResNet50 or the complete
Autocomp/AccelOpt benchmark shapes in this new flow. In particular, Voyager's
special first RGB convolution path has not been adapted. The older checkout's
analytical reports describe its different generated programs and must not be
attributed to this checkout. No hardware speedup or agreement with paper
latencies is claimed by CPU simulation or NEFF compilation.
