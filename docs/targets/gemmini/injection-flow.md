<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Gemmini injection, conversion, and runtime flow

The flow has three clean boundaries:

```text
PyTorch + shared PT2E Q/DQ
  -> Gemmini semantic matching, grouping, and final-FX validation
  -> model.txt + tensor files + hardware/storage metadata
  -> conversion/gemmini adapter
  -> runner.c + packed payloads + conversion manifest
  -> RISC-V ELF -> Spike --extension=gemmini -> complete readback
```

Voyager owns semantics and canonical tensors. The adapter validates those
facts, performs target layout/representation projection, and chooses a public
Gemmini helper. The helper owns tiling, DMA, local storage, and RoCC commands.
Spike executes the linked ELF. The current W path runs independent regions;
it is not a connected ResNet-18 runtime.

The target hardware pipeline is the `OpMatcher` view of linear PyTorch
semantics. The required public Conv grouping is derived from that pipeline;
capture-only leading input quantization is added at the injection boundary.
The linear matcher cannot represent the residual DAG, so rich W forms are
grouped by target-owned rewrites that preserve ingress and DAG context.
Target lowering validates exact member order and dependencies, serialization
validates qmaps, and the adapter independently validates shapes, qparams, and
helper-ABI legality.
Residual Add is a two-input DAG that the generic sequential matcher cannot
represent. The adapter separately projects an admitted canonical region to one
public helper call; that call is not one hardware instruction.

The compiler registry admits the FX `with-flatten` global-average variant.
Canonical serialization omits that shape-only Flatten while preserving its
tensor identity and shape; the adapter validates the resulting three-member
`model.txt` representation at its own boundary.

## 1. Prepare Voyager and the target stack

Use a Voyager checkout plus a target workspace with this relative layout:

```text
<stack-root>/
  gemmini/
  gemmini-stack/
    libgemmini/
    runtime-prefix/bin/spike
    runtime-prefix/lib/libgemmini.so
    spike-src/
    toolchain-env/riscv-tools/bin/riscv64-unknown-elf-gcc
```

Initialize the Gemmini submodules and build/install the RISC-V toolchain,
Gemmini-enabled Spike extension, and libgemmini into that layout. The checked-
in [runtime lock](../../../conversion/gemmini/stack-manifest.json)
contains relative paths, source revisions, and effective file/tool hashes. It
does not install software and deliberately contains no machine-specific root
or semantic-contract digest.

The repository is source-self-contained at the compiler/adapter boundary, but
not binary-self-contained: a target checkout alone does not include the built
RISC-V toolchain, Spike extension, or shared library. A public target revision
is usable only when its submodules and effective libgemmini files reproduce the
checked-in lock; no converter-side repair substitutes for that setup.

Set explicit roots from the Voyager repository:

```bash
export REPO=$PWD
export STACK_ROOT=/absolute/path/to/gemmini-target-workspace
export PYTHON=/absolute/path/to/python
export STACK_LOCK="$REPO/conversion/gemmini/stack-manifest.json"
export W_CONTRACT="$REPO/docs/targets/gemmini/workload-injection-contract.md"
export W_CONTRACT_SHA=9256107f68a65fce92791f2736b660e70dc902496efeacecfc0a89ae2cce4a30
export ARTIFACT_ROOT=/path/to/fresh/gemmini-attempt

test -x "$PYTHON"
test -d "$STACK_ROOT/gemmini"
test -x "$STACK_ROOT/gemmini-stack/runtime-prefix/bin/spike"
test -s "$STACK_LOCK"
test ! -e "$ARTIFACT_ROOT"
```

The Python environment must contain Voyager's dependencies; use
`PYTHONPATH=$REPO/src`. It may be reused. Freshness applies to output
directories, not to the environment.

## 2. Compile and capture a region

Select `GEMMINI_HARDWARE_CONFIG` when preparing, transforming, and compiling
the model. Use shared PT2E Q/DQ and ordinary PyTorch operations. Canonical
serialization must contain semantic fused regions, not Gemmini helper
arguments or schedules.

For a W workload, retain:

- `model.txt`, `tensor_files/`, `hardware_config.json`, and
  `storage_plan.json`;
- `final_fx.txt` and exact qparams;
- a workload-specific compile profile, any optional graph visualization, and
  `artifact-inventory.json`; and
- one runtime-capture manifest and directory with per-region input and
  independent expected-output files bound by hash.

The maintained ResNet-18 instance names its optional visualization
`resnet18.svg` and its compile profile `gemmini-resnet18-profile.json`. Those
names are workload-regression evidence, not requirements of the general
Gemmini injection contract. Generate that instance through its registered
adapter:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$REPO/src:$REPO" \
"$PYTHON" -B test/test_codegen.py resnet18 \
  --target gemmini \
  --model_output_dir "$ARTIFACT_ROOT/compiler"
```

The profile records the fixed model, input, PTQ/calibration policy, region
partition, environment versions, and exact compiler semantic comparison. The
inventory hashes the complete compiler/capture package; `model.txt` alone is
not a golden identity because qparam and payload changes need not alter its
structural text.

The ResNet-18-specific capture procedure and current mapping results are in
[resnet18.md](resnet18.md). The reusable adapter is
`conversion.gemmini.workload`; no ResNet-specific converter is required.

## 3. Convert one canonical region

Use the fused region name from `model.txt`:

```bash
PYTHONPATH="$REPO/src" "$PYTHON" -B -m conversion.gemmini.workload convert \
  --model-dir "$ARTIFACT_ROOT/compiler" \
  --output-dir "$ARTIFACT_ROOT/conversion" \
  --contract "$W_CONTRACT" \
  --contract-sha256 "$W_CONTRACT_SHA" \
  --fused-region <fused-op-name> \
  --runtime-capture "$ARTIFACT_ROOT/compiler/runtime-capture"
```

Conversion consumes only canonical semantic facts and the explicit runtime
capture. It validates qparams and helper representability, packs payloads,
and emits one wrapper/helper call. It does not tile the operation or generate
RoCC commands itself; the selected Gemmini `_auto` helper does that at
runtime.

Supported projections currently include qualified Conv2d, residual add,
global average, and linear regions. A converter rejection means that this
canonical region cannot be represented by the qualified adapter/helper ABI;
it does not by itself prove hardware non-support.

## 4. Build and execute the converted collateral

```bash
PYTHONPATH="$REPO/src" "$PYTHON" -B -m conversion.gemmini.workload run \
  --converted-dir "$ARTIFACT_ROOT/conversion" \
  --output-dir "$ARTIFACT_ROOT/native" \
  --stack-root "$STACK_ROOT" \
  --stack-manifest "$STACK_LOCK" \
  --contract "$W_CONTRACT" \
  --contract-sha256 "$W_CONTRACT_SHA" \
  --spike-timeout-seconds 900
```

The runner validates the lock against `STACK_ROOT`, compiles and inspects a
fresh bare-metal ELF, invokes the Gemmini Spike extension, parses every output
record, reverses target layout, and compares the complete logical result.
Instruction tracing is opt-in because Spike `-l` can create multi-gigabyte
logs; ordinary functional validation retains concise stdout/stderr, static
RoCC sites, complete readback, and comparison results.

## 5. Interpret the evidence

The converted manifest answers what canonical fields were consumed and which
helper call was emitted. The native result answers whether the ELF ran and the
complete target output passed its tolerance. A generated helper call is not
execution evidence; a Spike exit code without complete readback is not a
semantic pass.

Functional Spike does not provide target-cycle, traffic, utilization, energy,
or RTL evidence. Host wall time includes simulation overhead and must not be
reported as Gemmini performance.

Common failures are owned as follows:

| Failure | Owning boundary |
| --- | --- |
| contract hash changed | compiler/converter interface must be rebound |
| runtime-lock hash or revision mismatch | target stack differs |
| output directory exists | caller violated fresh-evidence policy |
| fused region is absent or ambiguous | compiler serialization/caller |
| scale or helper form is unrepresentable | adapter/helper boundary |
| ELF or extension load fails | toolchain/runtime setup |
| readback exceeds tolerance | semantic/native projection |
