<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Gemmini regression plan

The Gemmini regression has two distinct lanes:

- the R lane proves one registered shared-Q/DQ Conv2d from Voyager compilation
  through a public Gemmini helper and complete Spike readback; and
- the W lane qualifies the mapped regions of one frozen ResNet-18 workload as
  independent helper calls.

Neither lane is a connected ResNet-18 program, a classifier-accuracy result,
or target-performance evidence. The R and W contracts own semantic details;
this document owns the maintained commands, required lanes, result accounting,
and failure policy.

## External prerequisites

The compiler and adapter sources live in this repository. Native execution
additionally needs a target workspace containing:

- the qualified Gemmini and gemmini-rocc-tests sources;
- the RISC-V bare-metal compiler and binutils;
- a Gemmini-enabled Spike and `libgemmini.so`; and
- effective helper headers matching the checked-in
  [runtime lock](../../../conversion/gemmini/stack-manifest.json).

The lock is location-independent and validates the supplied stack root. It
does not install the target stack. See the
[injection runbook](injection-flow.md) for the required layout.

Use one Python environment containing Voyager's dependencies. The environment
may be reused. Compiler, conversion, and native output directories must be
absent at the start of every evidence-producing attempt.

## Operation execution anchor

The narrow R anchor isolates the registration path for maintenance. Its retained
fixture contains one ordinary `aten.conv2d` inside the shared fused sequence
`quantize, conv2d, dequantize, quantize` and retains the terminal dequantize.
The adapter converts that canonical package, builds one bare-metal ELF, and runs
it on Spike. A complete same-scope W proof already subsumes R, so this separate
diagnostic is not an additional logical prerequisite for the recorded W result.

The normative graph and numerical contract are in
[injection-contract.md](injection-contract.md), SHA-256
`a36b7a95d5d47bc8531abff4b72fa463731957f92ae5f34705c8cd7b5876d612`.
There is deliberately no standalone
`test/run_gemmini_operation_regression.py` controller or repository-owned
generator for this frozen fixture. The commands below are therefore a
converter/runtime replay from the retained Voyager-generated package, not a
fresh R compiler-generation lane. Exercise the same maintained converter and
runner used by the workload lane:

```bash
export REPO=$PWD
export STACK_ROOT=/absolute/path/to/gemmini-target-workspace
export PYTHON=/absolute/path/to/python
export R_EVIDENCE_ROOT=/path/to/external/gemmini-evidence
export R_MODEL="$R_EVIDENCE_ROOT/compiler-output"
export R_OUT=/path/to/fresh/gemmini-r-regression
export R_CONTRACT="$REPO/docs/targets/gemmini/injection-contract.md"
export R_CONTRACT_SHA=a36b7a95d5d47bc8531abff4b72fa463731957f92ae5f34705c8cd7b5876d612

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO/src" \
"$PYTHON" -B conversion/gemmini/convert.py \
  --model-dir "$R_MODEL" \
  --output-dir "$R_OUT/conversion" \
  --contract "$R_CONTRACT" \
  --contract-sha256 "$R_CONTRACT_SHA"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO/src" \
"$PYTHON" -B conversion/gemmini/run.py \
  --converted-dir "$R_OUT/conversion" \
  --output-dir "$R_OUT/native" \
  --stack-root "$STACK_ROOT" \
  --stack-manifest "$REPO/conversion/gemmini/stack-manifest.json" \
  --contract "$R_CONTRACT" \
  --contract-sha256 "$R_CONTRACT_SHA"
```

The lane passes only when compiler, converter, and native statuses pass; one
helper and one wrapper entry are observed; all 256 output codes are present;
and no value exceeds `rtol=0`, `atol=1`. Exact mismatch count and maximum error
remain required statistics even though exact equality is not required.

## Required workload lanes

The W regression uses
`torchvision.models.resnet18(weights=None).eval()`, seed `20260818`, batch one,
and one `[1, 3, 224, 224]` FLOAT32 input sampled uniformly from `[-1, 1]`.
Exactly two executions of that same input calibrate shared PT2E observers.

| Lane | Required evidence | Acceptance |
| --- | --- | --- |
| compiler/capture | canonical package plus 20 Conv captures | model/capture hash binding |
| adapter | one fresh conversion per mapped region | 30 converted manifests, one helper each |
| native | one fresh ELF and Spike run per converted region | 30 complete ordered readbacks |
| comparison | every signed-INT8 output value | `rtol=0`, `atol=1`, zero failures |
| accounting | both denominators and every helper | 69/69 source, 49/49 post-fold |

BatchNorm is folded into its preceding Conv parameterization because the
qualified helper stack has no BatchNorm helper. ReLU consumes qcodes in the
Conv or residual helper region. The stem MaxPool is a `tiled_conv_auto`
epilogue. Flatten is a layout-only boundary and does not become a helper call.

Capture storage and semantic type are separate. All capture payloads use
little-endian FLOAT32 containers. The manifest must label the stem Conv input
as logical FLOAT32, the other 19 Conv inputs as logical INT8, and every Conv
output as logical INT8. The producer validates integral signed-INT8 range; the
converter must reject a wrong label rather than infer or repair it.

## Workload operation accounting

The current package has 31 top-level canonical operation records. Thirty are
helper-projectable fused regions; one ordinary terminal dequantize record is a
semantic boundary and is not an unsupported target operation.

| Canonical region family | Regions | Native helper | Extra runtime input |
| --- | ---: | --- | --- |
| Conv/ReLU, optional stem MaxPool | 20 | `tiled_conv_auto` | hash-bound runtime capture |
| residual Add/ReLU | 8 | `tiled_resadd_auto` | none |
| global average | 1 | `tiled_global_average_auto` | none |
| Linear | 1 | `tiled_matmul_auto` | none |
| Total | 30 | 30 independent calls | 20 Conv captures |

Every helper call is built and executed independently. A 30-row pass proves
complete mapped-region execution, not connected execution or intermediate
state transfer between helpers.

## Recorded W validation result

The original endpoint campaign generated one canonical package, converted
30/30 regions, and passed 30/30 independent native executions. The compiler
first compared all 1,000 transformed outputs exactly. Native comparison covered
2,635,752 output codes: 2,367,030 exact, 268,722 differing by one code, zero
tolerance failures, and maximum error one.

These mismatch statistics characterize W only. This lane stops at the existing
Gemmini helper/software boundary, which is partial-control, and does not rebind
the same frozen W scope and reference through a full-control point. Gemmini X
has not started.

The compact status and sole preserved Voyager compiler-collateral hash are in
[operations.yaml](operations.yaml). Conversion is not required to reproduce a
previous byte stream; the gate is a fresh 30/30 pass for one newly generated
canonical package.

No promotion-ready package is included; regenerate the complete campaign under
the current contract before promotion. See the
[ResNet-18 report](resnet18.md#golden-ownership-and-promotion) for immutable
promotion policy. Do not combine rows from different canonical packages.

## Maintained W regression recipe

The following recipe mirrors the compiler, conversion, and target-execution
phases used by the Magnet Alpha regression while preserving Gemmini's
per-region interface.

### Common paths

```bash
REPO=$PWD
STACK_ROOT=/absolute/path/to/gemmini-target-workspace
PYTHON=/absolute/path/to/python
CAMPAIGN=/path/to/fresh/gemmini-resnet18-regression
MODEL="$CAMPAIGN/compiler"
RUNTIME_CAPTURE="$MODEL/runtime-capture"
STACK_LOCK="$REPO/conversion/gemmini/stack-manifest.json"
W_CONTRACT="$REPO/docs/targets/gemmini/workload-injection-contract.md"
W_CONTRACT_SHA=9256107f68a65fce92791f2736b660e70dc902496efeacecfc0a89ae2cce4a30
```

`CAMPAIGN` must not exist. All paths must be absolute and available to the
selected execution environment.

### 1. Obtain compiler collateral and runtime capture

The compiler lane must construct the frozen model/input, prepare shared
per-tensor signed-INT8 PT2E quantization with INT32 bias, execute exactly two
calibration steps, finalize calibration, convert PT2E, transform with
`GEMMINI_HARDWARE_CONFIG`, execute the transformed graph once for capture, and
compile that exact graph.

It must retain:

- `model.txt`, `tensor_files/`, `hardware_config.json`, and
  `storage_plan.json`;
- final FX and exact qparams;
- `runtime-capture/runtime-capture.json`; and
- input and output-code files for all 20 Conv regions.

The capture manifest must bind the exact `model.txt` hash. Region input and
output tensors must come from the transformed graph execution, not shape
propagation or reconstructed converter arithmetic.

Run the registered exact-pair adapter through the standard compiler entry
point:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO/src:$REPO" \
"$PYTHON" -B test/test_codegen.py resnet18 \
  --target gemmini \
  --model_output_dir "$MODEL"
```

The adapter freezes model construction, input generation, PTQ policy,
calibration count, hardware profile, and compiler thread count. It also emits
`final_fx.txt`, `resnet18.svg`, `gemmini-resnet18-profile.json`, and a complete
`artifact-inventory.json`. The inventory binds every generated compiler and
capture file except the profile and inventory files themselves.

### 2. Convert all mapped regions

Use these exact fused-region sets from the current workload plan:

```bash
CONV_REGIONS=(
  conv1_fused
  layer1_0_conv1_fused layer1_0_conv2_fused
  layer1_1_conv1_fused layer1_1_conv2_fused
  layer2_0_conv1_fused layer2_0_conv2_fused
  layer2_0_downsample_0_fused
  layer2_1_conv1_fused layer2_1_conv2_fused
  layer3_0_conv1_fused layer3_0_conv2_fused
  layer3_0_downsample_0_fused
  layer3_1_conv1_fused layer3_1_conv2_fused
  layer4_0_conv1_fused layer4_0_conv2_fused
  layer4_0_downsample_0_fused
  layer4_1_conv1_fused layer4_1_conv2_fused
)

ADD_REGIONS=(
  add__fused add__1_fused add__2_fused add__3_fused
  add__4_fused add__5_fused add__6_fused add__7_fused
)

OTHER_REGIONS=(adaptive_avg_pool2d_fused fc_fused)
```

Conv conversion consumes the capture directory in addition to canonical
collateral:

```bash
mkdir -p "$CAMPAIGN/conversion" "$CAMPAIGN/logs"

for region in "${CONV_REGIONS[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  "$PYTHON" -B -m conversion.gemmini.workload convert \
    --model-dir "$MODEL" \
    --output-dir "$CAMPAIGN/conversion/$region" \
    --contract "$W_CONTRACT" \
    --contract-sha256 "$W_CONTRACT_SHA" \
    --fused-region "$region" \
    --runtime-capture "$RUNTIME_CAPTURE" \
    >"$CAMPAIGN/logs/$region.convert.stdout" \
    2>"$CAMPAIGN/logs/$region.convert.stderr"
done
```

Residual, average, and Linear projections consume canonical values directly:

```bash
for region in "${ADD_REGIONS[@]}" "${OTHER_REGIONS[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  "$PYTHON" -B -m conversion.gemmini.workload convert \
    --model-dir "$MODEL" \
    --output-dir "$CAMPAIGN/conversion/$region" \
    --contract "$W_CONTRACT" \
    --contract-sha256 "$W_CONTRACT_SHA" \
    --fused-region "$region" \
    >"$CAMPAIGN/logs/$region.convert.stdout" \
    2>"$CAMPAIGN/logs/$region.convert.stderr"
done
```

Each conversion manifest must report `status: converted`, one emitted helper,
the expected ordinary ATen operation ID, the canonical model hash, and the
contract hash. The generated C must contain exactly one selected `_auto`
helper call. Conversion does not execute Gemmini and does not prove a pass.

### 3. Run every converted region on Spike

The native runner builds and inspects a fresh ELF, executes Gemmini-enabled
Spike, parses the complete output stream, reverses target layout, and performs
the tolerance comparison.

```bash
mkdir -p "$CAMPAIGN/native"

for region in \
  "${CONV_REGIONS[@]}" \
  "${ADD_REGIONS[@]}" \
  "${OTHER_REGIONS[@]}"; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" \
  "$PYTHON" -B -m conversion.gemmini.workload run \
    --converted-dir "$CAMPAIGN/conversion/$region" \
    --output-dir "$CAMPAIGN/native/$region" \
    --stack-root "$STACK_ROOT" \
    --stack-manifest "$STACK_LOCK" \
    --contract "$W_CONTRACT" \
    --contract-sha256 "$W_CONTRACT_SHA" \
    --spike-timeout-seconds 900 \
    >"$CAMPAIGN/logs/$region.run.stdout" \
    2>"$CAMPAIGN/logs/$region.run.stderr"
done
```

Do not enable `--trace-instructions` for the ordinary functional regression.
Instruction traces can grow to multiple gigabytes and are not needed to prove
complete output correctness. Retain concise build logs, ELF inspection,
readback, comparison, and `result.json` for every row.

### 4. Aggregate and promote the campaign

The acceptance contract below requires one aggregate that accounts for every
region, conversion manifest, ELF, native result, readback, and comparison.
No general campaign aggregator is currently checked into this repository.
Therefore a newly generated campaign is not promotion-ready after the per-row
commands alone. Keep it unpromoted until a reviewed aggregate accounts for all
30 rows and the denominators below.

## W acceptance and aggregation

A W campaign passes only when all of the following are true:

- the compiler semantic gate is exact over all 1,000 final outputs;
- the canonical package contains 30 unambiguous mapped fused regions;
- the capture manifest contains all 20 Conv regions and binds `model.txt`;
- capture input logical dtypes split one FLOAT32 and 19 INT8 while all payload
  containers remain FLOAT32;
- the compiler inventory binds all 243 generated package files, including 196
  canonical tensor files and 40 capture payloads;
- 30 conversion manifests pass with the expected helper partition;
- 30 fresh ELFs pass validation and execute on the locked Spike stack;
- every logical output element is present and ordered;
- every row has zero tolerance mismatches at `rtol=0`, `atol=1`; and
- an aggregate accounts for every manifest, ELF, native result, readback, and
  comparison result.

The aggregate must separately report exact matches, one-code differences,
maximum absolute error, and total compared elements. It must also retain the
69-source-occurrence and 49-post-fold-occurrence denominators. Unsupported or
failed rows may not disappear from either denominator.

## Focused source gates

Run these checks before either public lane:

```bash
PYTHONPATH="$REPO/src" "$PYTHON" -m unittest \
  test.unit_tests.test_target_operator_traits \
  test.unit_tests.test_agent_context_routing \
  test.unit_tests.test_shared_test_dependency_boundary

"$PYTHON" -m py_compile \
  conversion/gemmini/convert.py \
  conversion/gemmini/run.py \
  conversion/gemmini/workload.py \
  conversion/gemmini/_workload_helpers.py
```

Run the target-owned fail-closed, capture-contract, and quantization-domain
legalization suite:

```bash
PYTHONPATH="$REPO/src:$REPO" "$PYTHON" -m pytest -q \
  conversion/gemmini/test_convert.py \
  conversion/gemmini/test_workload.py \
  conversion/gemmini/test_compiler_quantization.py
```

`test_workload.py` checks the distinction between FLOAT32 storage containers
and logical INT8 code payloads, including fractional, out-of-range, mislabeled,
and model-hash failure cases. `test_compiler_quantization.py` checks that only
bit-identical Q/DQ/ReLU domains are erased and that the frozen workload aligns
all 17 helper-owned ReLU boundaries without changing transformed output.

## Failure ownership

| Failure | Owning boundary |
| --- | --- |
| contract or model hash differs | compiler/converter interface |
| capture lacks a Conv region or binds another model | compiler/capture lane |
| region is missing or ambiguous | compiler serialization or caller |
| qparams or helper fields are unrepresentable | target adapter/helper ABI |
| stack-lock validation fails | target checkout/toolchain setup |
| ELF build or extension load fails | native build/runtime setup |
| output is missing, unordered, or outside tolerance | native projection or semantics |

Functional Spike wall time includes simulation overhead. It is not target
latency, and this plan makes no cycle, frequency, traffic, utilization, energy,
speedup, RTL, connected-execution, or classifier-accuracy claim.

## Breadth evidence policy

The former monolithic registration-probe campaign is historical discovery
evidence, not a maintained regression command. New operation coverage should
use small tests at the shared semantic owner, fail-closed adapter tests, and a
focused compiler-to-Spike positive lane. Do not recreate one controller that
duplicates model construction, compiler orchestration, artifact hashing, and
all target evidence.
