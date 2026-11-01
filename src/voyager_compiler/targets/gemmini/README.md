# Gemmini target

Gemmini uses the shared prepare → shared quantization → normalize → custom operations → recipe search →
bufferize → allocate/freeze → encode/save → emit flow. This port incorporates the donor's INT8
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
| Recipe `candidates`, `estimate`; target `model` | [scheduling.py](scheduling.py) | Interstellar native matrix adapter, capacity and costs; pointwise candidates; shared planner selects |
| Recipe `build`, target `refine` | [lowering.py](lowering.py) | Build the selected buffer/native operation and bind memory requirements before shared allocation |
| `TargetSpec.encode` | [emission/__init__.py](emission/__init__.py) | Save selected native operations, parameters, allocated addresses and primitive packets as target IR |
| `TargetSpec.emit` | [emission/__init__.py](emission/__init__.py) | Expand saved operations into RoCC commands; no live graph or new schedule choice |

```text
gemmini/
├── target.py          registration; follow each bound callable from here
├── hardware.py        physical configuration
├── semantics.py       operations, accepted forms and pure facts
├── quantization.py    shared quantizer/domain policy
├── scheduling.py      candidate/capacity/cost callbacks
├── lowering.py        custom operations, buffer builders and storage refinement
└── emission/
    ├── __init__.py    registered encode and emit hooks
    ├── _commands.py   private expansion of the selected tile/mode
    ├── _encoding.py   private instruction-word packing
    └── replay.py      explicit simulation collateral adapter
```

Command expansion and instruction packing are internal to emission, not public
extension points. Expansion returns commands only; it does not return another
schedule or select tiles. Emission reads operation facts and mapping projection from semantics. Search and
emission share the matrix storage calculation. The empty `GemminiModel.prepare`
fallback is intentional: native matrix recipes construct their own operation
adapter and call shared `search_matrix`; pointwise recipes use direct estimates.

No separate `legalize`, `runtime` or `characterize` hook is registered.
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

Supported operations are dense and multiplier-one depthwise square undilated convolution, linear/matmul,
ReLU, policy-compatible ReLU6, requantization, equal-shape residual addition, global spatial average,
max pooling, and layout-safe reshape/flatten. Input images become NHWC; conv
dense weights become HWIO, depthwise weights retain C×1×R×R, and linear weights K×N. `CompilationResult.args` contains
the quantized inputs for this ABI. Numerical provenance records scales and
logical layouts; returned outputs remain integer. Explicit Gemmini integer ops
already specify their numerical contract and skip float-model calibration.

Output scaling uses float32 multiplication, nearest-even rounding and signed
INT8 saturation. Residual inputs each round/saturate **before** addition. Integer
accumulation rejects overflow; it does not model wraparound. Padded max pooling
requires a proven nonnegative input because the hardware fills padding with zero.
Pooling kernel/stride and accumulator limits are checked by `semantics.py`.
Other grouped convolutions, depthwise multipliers above one, dilated convolutions, ceil-mode pooling, general layout-changing flatten,
input mutation, and unsupported tensor functions reject. Exported in-place forms must preserve the shared export/effect contract. There is no hidden host
fallback for unsupported network operations.

Depthwise operations bind `gemmini.depthwise_conv2d.hardware_loop`. The target
searches spatial row/column tiles with checked private ping/pong capacities;
channels execute as independent hardware-loop launches. It reuses convolution
halo/command expansion with depthwise addressing and the ISA's depthwise flag.
Its analytical cost counts independent groups, not fictitious dense MACs; it is
not a calibrated performance guarantee.

ReLU6 uses an explicit fixed output quantizer: signed code 127 represents real 6.
The default target quantizer enables this policy; custom `Int8Quantizer` callers
opt in with `relu6_saturation=True`. For an exclusive conv/linear producer,
quantization observes after the activation, and lowering fuses ReLU into the
INT32-accumulator store with multiplier `input_scale * weight_scale / (6/127)`.
There is no intermediate unrestricted INT8 store. Incompatible scales reject;
ReLU alone is not substituted. Compare original-model accuracy separately from
exact execution of this quantized contract.

### Select the quantization scheme in the workload binding

The power-of-two default remains available. `options["gemmini_local_domains"]=True`
selects the existing local-domain policy with non-power-of-two scales and ReLU6
support; an explicit caller quantizer still wins. `Int8Quantizer(local_domains=True)`
is an alternative using the same shared calibration/conversion flow. It observes
an exclusive conv/linear → ReLU at the fused output, keeps residual domains local,
and calibrates global-average output separately. Compatible exclusive matrix
producers store directly in the add's units; a reused identity branch retains its
earlier units and an explicit requantization boundary. Explicit sharing prevents
PT2E's sibling-edge unions from joining distant residual blocks. Per-module/type/
operator overrides, including disabled annotations, still take precedence.

For example, a binding can compare the default with this floating-scale policy:

```python
from voyager_compiler.quantization import get_default_quantizer
from voyager_compiler.targets.gemmini.quantization import Int8Quantizer

spec = "int8,qs=per_tensor_symmetric,qmax=127,pot=0"
common = get_default_quantizer(
    input_activation=spec, output_activation=spec, weight=spec, bias="int32"
)
quantizer = Int8Quantizer(local_domains=True).set_global(
    common.object_type_config[torch.ops.aten.conv2d.default]
)
options = {"quantizer": quantizer, "calibration": samples}
```

Check accuracy against the original float model with fixed assets and thresholds,
then compare hardware cost among passing policies. Exact comparison against the
converted graph checks lowering, not original-model accuracy. A changed scheme is
a binding choice, not a reason to change all workloads' defaults.

The lowering supports a single final multiplier for global average and folds
exclusive scale-preserving ReLU consumers into native conv/matmul/add stores,
including through exclusive views. A residual-input requantization can execute
in its MVIN scaler when that preserves the same INT8 rounding/saturation boundary.
It retains exact comparison with the independently converted graph. Batched
linear inputs flatten their leading dimensions for native GEMM and restore the
result shape. A permitted initial im2col/transpose/pad chain is extracted by the
shared input-preprocessor helper; no internal activation computation moves to the
host. The workload's optional `prepare` hook selects the im2col rewrite **before**
quantization, folding eval Conv–BN first if needed. Native convolution remains an
alternative. The binding must explicitly set `options["allow_input_packing"]=True`
to permit im2col/padding in host preprocessing; selecting a rewrite alone does not
grant that boundary. The helper's optional `reduction_multiple=16` explicitly pads packed
input and weight lanes to the array width; both stored extents are declared.
Without this binding choice, native tail padding remains part of the physical
footprint calculation.

## Search and command ownership

`scheduling.NativeMatrixModel` maps GEMM/conv to three Interstellar levels:
16×16 array, private on-chip storage, DRAM (`fit_level=1`). Matrix recipes each
search factors through the shared engine; both spatial axes and all capacity-valid
channel factors are available. The outer recipe budget does not truncate that
search. Matmul compares explicit and hardware-loop ISA recipes; convolution uses
the hardware convolution loop. Pointwise recipes retain small direct candidates.

The builder preserves `TileResult` in `ImplementationChoice.tile`; emission derives
the native tile from that frozen mapping. Channel search extents and matmul rows are rounded to
16 and projected to logical tails; interior matmul tiles fill 16-element blocks. `semantics.tile_storage` supplies the physical
footprint to both search and emission, including hardware-loop padding and two
private contexts. Named-memory resource capacities come from the shared helper.
Outer order is fixed to the emitted batch → rows → columns → output channels →
input channels traversal. Weight retention across separate spatial tiles is not
implemented; the model must not claim it. Interstellar access counts are logical
accesses, not a measured native DMA trace.

For a 7×7, 512→512, 3×3 convolution, the analytical search now selects
7×7×128×64, producing 32 native launches. The command test decodes actual launch
geometry. This establishes reachability/realization, not measured optimality.

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

`emission.replay.write_replay(result, directory, commands_path)` reuses shared
`collect_replay_images` for bounds, staging and reference separation. It supplies
the INT8/INT32 encoder and the pinned endpoint's command/memory file formats.
It creates an initial image
containing only real inputs/parameters and poison, a separate expected image, and
a connected command stream using frozen addresses. Intermediate expected results
are **never** substituted for hardware outputs. In this workspace, prefer the
existing [VCS simulation queue](../../../../../../vcs-sim-queue/README.md) for
**Gemmini replay only**:

```bash
PYTHONPATH=/home/zhouhua/Research/ML/vcs-sim-queue/src \
/home/zhouhua/Research/ML/ml-env/bin/python -m vcs_queue \
  --config /home/zhouhua/Research/ML/vcs-sim-queue/config.json \
  gemmini REPLAY --out RTL_OUTPUT --host-timeout 3600 --timeout-cycles 25000000
```

Keep compilation, analysis and framework/agent files local. The Gemmini profile
uploads only staged replay commands, manifest and input/reference data. Retain
the job ID and require both completed execution and `result.json`'s `pass` field;
queue submission alone is not validation. Do not change Voyager/Alpha runners.
The existing local fallback is `scripts/run.sh REPLAY --out RTL_OUTPUT` in the
Gemmini runtime checkout; record which backend produced each measurement.
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
All compare complete allocated memory images. The default development ResNet tests use 32×32 inputs;
this is numerical/mapping validation, not ImageNet accuracy or throughput.
The separate [full benchmark comparison](../../../../../reports/validation/maintainability-2026-10-27/performance/summary.json)
uses the supplied pretrained checkpoint and five 224×224 inputs, retaining its
original accuracy thresholds and hardware/memory configuration.

Dated source hashes, donor comparisons, RTL pins and results are in
[port acceptance](../../../../../reports/archive/voyager-gemmini-port-2026-10-25/acceptance.json).
The [quantization refactor](../../../../../reports/archive/voyager-quantization-refactor-2026-10-25.html)
records shared-frontend validation and the policy change. The [extension review in the walkthrough](../../../../../reports/voyager-compiler-walkthrough.html#extension-review)
records extension ownership; its validation chapter links the dated command-stream checks.
Historical Voyager/Alpha gates retain their own scope and artifacts.

## MobileNetV2 binding

[MobileNetV2](../../workloads/mobilenet_v2.py) is a target-agnostic `WorkloadSpec`:
local optional weights and example inputs; no downloads or private compiler.
[Validation](../../../../validation/mobilenet_gemmini.py) prepares depthwise stride
and spatial-tail cases, an inverted-residual block, or the full model plus replay:

```bash
PYTHONPATH=src:test python validation/mobilenet_gemmini.py OUT --case model \
  --weights /path/to/mobilenet_v2.pth --image-size 224 --quantization local
```

Use `--input IMAGE.npy` and repeat `--calibration IMAGE.npy` for supplied NCHW
samples. The default random input is a numerical probe, not an accuracy dataset.
The script checks host execution and prepares queue collateral; only a completed
RTL result proves endpoint agreement. Coverage and exact INT8 execution do not
establish acceptable FP32/model accuracy, which remains a separate binding gate.

## Input-layer and search-space investigation

For a three-channel stem, inspect the shared [input rewrite contract](../../../../docs/compilation/rewrites.md#input-boundary-rewrites)
before adding a new pass. The ATen im2col helper is available, but binding a packed
INT8 GEMM requires preserving scales, rounding/saturation and bias, explicitly
representing packed input/padded weights, and comparing native convolution where
legal. It is not enabled automatically by this documentation change.

Audit output rows and columns, input/output channel tiles and weight retention
against the selected builder. The evaluation's 7×7/512→512 example contrasts
7×7×128×64 tiles (32 launches) with 1×7×32×256 (224 launches); these counts identify
candidate-space questions, not measured speedup or proof both mappings satisfy
this backend's exact resource/numerical contract. Use target-owned candidates or
[the matrix adapter](../../../../docs/compilation/performance-models.md#target-owned-matrix-search-adapters),
and price/realize any retained weights at the correct loop scope.
