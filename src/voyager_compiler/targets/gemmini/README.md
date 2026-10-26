# Gemmini target

Gemmini uses the shared prepare → shared quantization → normalize → custom operations → recipe search →
bufferize → allocate/freeze → emit flow. This port incorporates the donor's INT8
vision arithmetic and command expansions without importing its separate compiler
pipeline. Start with the shared [roles](../../../../docs/roles.md) and
[binding procedure](../../../../docs/adoption/workload-binding.md); this guide
owns only Gemmini-specific behavior.

## Extension points and implementation owners

[`target.py`](target.py) only assembles `TargetSpec` and its paired recipes.
There is no pass implementation or ISA selection in registration. The shared
[extension catalog](../../../../docs/extensions.md) defines the contracts.

| Public binding | Owner here | Responsibility |
| --- | --- | --- |
| `TargetSpec.hardware` | [hardware.py](hardware.py) | Physical engines, memories and capacities |
| `TargetSpec.operators` | [semantics.py](semantics.py) | Executable arithmetic, accepted shapes/dtypes, operation facts and legality |
| `TargetSpec.quantization` | [quantization.py](quantization.py) | Configurable shared PT2E quantizer; domain-transfer rules and static-observer policy |
| `TargetSpec.lower` | [lowering.py](lowering.py) | Normalized Q/DQ graph → custom INT8 operations and explicit input/output ABI |
| `TargetSpec.normalization` | [target.py](target.py) | Request NHWC/HWIO and xc,ck layouts, retain Q boundaries, use native partial tiles |
| Recipe `candidates`, `estimate`; target `model` | [scheduling.py](scheduling.py) | Propose/prune tiles, declare capacity requirements and estimate costs; shared planner selects |
| Recipe `build`, target `refine` | [lowering.py](lowering.py) | Build the selected buffer/native operation and bind memory requirements before shared allocation |
| `TargetSpec.emit` | [emission/__init__.py](emission/__init__.py) | Consume frozen selection and addresses; emit the complete command artifact |

```text
gemmini/
├── target.py          registration; follow each bound callable from here
├── hardware.py        physical configuration
├── semantics.py       operations, accepted forms and pure facts
├── quantization.py    shared quantizer/domain policy
├── scheduling.py      candidate/capacity/cost callbacks
├── lowering.py        custom operations, buffer builders and storage refinement
└── emission/
    ├── __init__.py    registered emit hook
    ├── _commands.py   private expansion of the selected tile/mode
    ├── _encoding.py   private instruction-word packing
    └── replay.py      explicit simulation collateral adapter
```

Command expansion and instruction packing are internal to emission, not public
extension points. Expansion returns commands only; it does not return another
schedule or select tiles. Emission reads operation facts from semantics, never
from scheduling. Both scheduling and lowering use the same capacity calculation.
The empty `GemminiModel.prepare` fallback is intentional: these recipes provide
whole-operation `estimate` callbacks instead of shared per-tile `OpModel` objects.

No separate `legalize`, `encode`, `runtime` or `characterize` hook is registered.
The replay helper prepares artifacts and does not pretend to execute hardware.
Shared normalization, final allocation, freezing and bundling remain shared.

## Supported numerical contract

Ordinary exported PyTorch models use shared PT2E preparation (including eval
Conv–BatchNorm folding), calibration and conversion. The default policy is static
symmetric per-tensor power-of-two INT8 activation/weight quantization with INT32
bias. Supply `options["calibration"]` as `(args, kwargs)` samples; otherwise the
example is the single sample. Observer finalization includes the last retained
sample. No QAT, new environment or dataset download is required.

`quantization.py` configures the shared quantizer and declares compatible
ReLU/pool domains and common residual input/output domains. Global average keeps
an explicit rounding boundary. It contains no recorder, calibration loop, weight
quantization implementation or layout transform. Per-module/type/operator/order
settings are retained; caller `options["quantizer"]` wins. Unsupported precision
or residual domains reject instead of silently replacing the requested policy.
Direct FP32 integer rounding is an optional shared conversion setting selected
here, avoiding an intermediate bfloat16 lookup-table rounding step.

`lowering.lower_operations` reads the converted scales and shared NHWC/HWIO/CK
layouts. It extracts boundary preprocessing with the shared helper, maps supported
operations, and checks integer outputs against the independent converted graph
before bufferization. It does not recalibrate. This shared-observer default can
choose different scales from the historical donor's per-node calibration; the
old donor-output equality report remains historical, not a new acceptance claim.

Supported operations are dense square undilated convolution, 2D linear/matmul,
ReLU/requantization, equal-shape residual addition, global spatial average,
max pooling, and layout-safe reshape/flatten. Input images become NHWC; conv
weights become HWIO and linear weights K×N. `CompilationResult.args` contains
the quantized inputs for this ABI. Numerical provenance records scales and
logical layouts; returned outputs remain integer. Explicit Gemmini integer ops
already specify their numerical contract and skip float-model calibration.

Output scaling uses float32 multiplication, nearest-even rounding and signed
INT8 saturation. Residual inputs each round/saturate **before** addition. Integer
accumulation rejects overflow; it does not model wraparound. Padded max pooling
requires a proven nonnegative input because the hardware fills padding with zero.
Pooling kernel/stride and accumulator limits are checked by `semantics.py`.
Grouped/dilated convolutions, ceil-mode pooling, general layout-changing flatten,
input mutation, and unsupported tensor functions reject. Exported in-place forms must preserve the shared export/effect contract. There is no hidden host
fallback for unsupported network operations.

## Search and command ownership

`scheduling.py` enumerates capacity-valid tiles, retains up to 24 candidates per
recipe, and provides additive compute/DMA/command estimates. `target.py`
pairs those callbacks with builders in `lowering.py`. Both explicit matmul and hardware-loop matmul
participate in the same search; convolution uses the hardware convolution loop.
Residual DMA tiles and pooling/requantization recipes share the same plan.

Deployment calls reserve private scratchpad/accumulator demand in the shared
allocation plan. Software-expanded explicit commands and hardware-controlled
loops can both own native storage: allocation and scheduling ownership are
independent. Matmul reductions retain their accumulator across selected K tiles.
`emission/_commands.py` expands the **selected** tile; it must not retile or allocate global
addresses. `emission/__init__.py` resolves frozen addresses and emits RoCC words. Required
phase drains preserve residual arithmetic, and each native boundary completes
synchronously. Internal DMA/compute engines may overlap within that boundary.

Matrix tails use instruction dimensions rather than inflating every global
operand. Native convolution handles spatial halo padding. Shared allocation
accounts for global tensors separately from native private storage. See
[padding](../../../../docs/compilation/padding.md) for the cross-stage distinction.

The earlier aligned `aten._int_mm` INT8→INT32 recipes remain available for host
and command-encoding checks. The pinned Lean RTL configuration disables full-width
accumulator readback, so these raw INT32 outputs are **not** covered by the INT8
endpoint validation below.

The cost model is a heuristic, not a fitted queue/overlap model. Cycle observations
below are endpoint measurements, not proof that search selects the fastest tile.
Performance work must inspect transfer placement, reuse and submission together.

## Compile and validate

```python
import torch
from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.targets.gemmini.target import TARGET

model = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU()).eval()
workload = WorkloadSpec("linear", lambda _: model,
                        lambda m, o: ((torch.randn(2, 32),), {}))
result = compile_workload(workload, TARGET, "out/linear", dump_tensors=False)
```

`emission.replay.write_replay(result, directory, commands_path)` creates an initial image
containing only real inputs/parameters and poison, a separate expected image, and
a connected command stream using frozen addresses. Intermediate expected results
are **never** substituted for hardware outputs. The existing Gemmini runner can
consume this directory with `scripts/run.sh REPLAY --out RTL_OUTPUT`.
The helper does not install or invoke a simulator; the generic runtime remains
host replay unless a runtime hook is supplied.

The pinned endpoint has a 16 MiB whole-program aperture. ResNet18 is the
model-level regression and fits that endpoint.
The donor's per-operation staging/reset runtime is not part of this port.
Hardware capacities/ISA assumptions are pinned by `hardware.py` and `semantics.py`;
a different endpoint requires reviewing both, not silently changing constants.

Local checks (use the existing environment):

```bash
PYTHONPATH=src:test /home/zhouhua/Research/ML/ml-env/bin/python -m unittest discover -s tests -v
/home/zhouhua/Research/ML/ml-env/bin/python validation/compare_gemmini.py /tmp/gemmini-comparison
```

The comparison uses seeded local models/inputs, without pretrained weights or
external datasets. The default suite uses Vision and ResNet18 plus three integer kernels.
An explicit `--donor PATH` requests strict comparison with the historical
frontend; it is diagnostic for the changed observer policy, not the default gate.
RTL checks additionally cover both matmul ISA forms with partial tiles and split
reductions, split-channel convolution, connected residual vision and ResNet18.
All compare complete allocated memory images. ResNet tests use 32×32 inputs;
this is numerical/mapping validation, not ImageNet accuracy or throughput.

Dated source hashes, donor comparisons, RTL pins and results are in
[port acceptance](../../../../../reports/voyager-gemmini-port-2026-10-25/acceptance.json).
The [quantization refactor](../../../../../reports/voyager-quantization-refactor-2026-10-25.html)
records shared-frontend validation and the policy change. The [extension review in the walkthrough](../../../../../reports/voyager-compiler-walkthrough.html#extension-review)
records extension ownership; its validation chapter links the dated command-stream checks.
Historical Voyager/Alpha gates retain their own scope and artifacts.
