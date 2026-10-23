<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager dump to Magnet Alpha C-model collateral

`convert.py` is a standalone, fail-closed postprocessor. It reads compiler
artifacts and emits layer-level Alpha C-model schedules and tensors; it does
not import Voyager or make compiler decisions.

## Inputs and outputs

The converter consumes:

- `model.txt`: graph targets and fused sequences, shapes, named schedule
  factors, physical-copy facts, explicit qparams, and encoded scratchpad
  addresses;
- schema-v1 `hardware_config.json`: the exact compiled hardware profile; and
- `tensor_files/*.bin`: payloads and quantization metadata.

The default output is `<model>/cmod`. Every supported executable record
receives schedule and `1_*.txt` files. A `model.txt` may also contain a
compiler-classified external boundary serialized with `op: "unsupported"`;
only those explicit records may be skipped. An unknown ordinary target or
unrecognized target sequence is a conversion error. `layers.txt` preserves
supported graph order. `conversion_report.json` records emitted, historically
named `fused`, skipped, unsupported, and invalid records, plus supported
execution islands and their externally materialized boundaries. Here `fused`
means an absorbed graph record, commonly Q/DQ; it is not a hardware-fusion
claim. A conversion that emits no target layer, contains a malformed record, or
contains an invalid supported layer publishes no partial output and writes
`<output>.failure.json` without replacing an existing result.

Use a fresh Magnet Alpha dump compiled with AoV's fixed mixed W4A4/W8A8 policy.
Default-W8/W8A8 dumps are not current conversion evidence.

## Convert

```bash
python3 conversion/magnet-alpha/convert.py \
  models/yolo5-face-nv \
  --force
```

Run repository-relative commands from the Voyager repository root. Set an
environment-specific Python executable explicitly when `python3` is not the
configured Voyager environment.

The input may be a model name under `models/`, a model directory, or
`model.txt`. Use `--output PATH` to choose the destination and `--no-tensors`
to omit text tensor emission while retaining validation. Regenerate dumps
that lack schema-v1 `hardware_config.json`.

Use `--require-all-supported` when conversion is expected to encounter no
compiler-classified unsupported execution records, as in the current YOLOv5
lane. This does not upgrade absorbed Q/DQ semantics or host-output-layout work
to verified Alpha support. Add
`--require-connected-program` when the supported records must also form one
connected Alpha dataflow component. These are conversion acceptance gates;
they do not by themselves prove end-to-end scratchpad residency or runtime
execution.

## Conversion contract

Final FX and `model.txt` own canonical operation semantics. The current CMOD
interface also requires schedule, storage, address, and quantization facts.
This converter owns their deterministic Alpha projection: external route and
disposition, recognized Q/DQ absorption, descriptor and epilogue selection,
skip reason, and CMOD-only tensor encoding. It must never
select a schedule, allocate or relocate storage, infer padding, invent a
missing epilogue, derive replacement quantization arithmetic, or repair an
unsupported compiler record.

For GEMM, the converter consumes the immutable `schedule_factor_*` projection
selected by either Alpha scheduler provider. It derives only the CMOD counters
that are deterministic functions of those factors, then independently checks
their register ranges and weight-memory capacity. It never changes the
selected factors or generates a replacement mapping.

Only supported Alpha work becomes a layer: GEMM convolution, depthwise
convolution, channel shuffle, physical shuffle copy, exact max-pool, and
nearest upsample. The converter selects those builders from canonical
`OpOverload.target` values and fused target sequences. Structural graph
records remain connectivity evidence. The current converter recognizes fixed
sequences; it does not expand an arbitrary composite into several CMOD layers.
Such a sequence fails until an explicit supported translation is added.

Every operation receives one converter-owned external disposition:

- `supported`: emit the CMOD execution kind selected for its target form;
- `fused`: historical name for an operation absorbed into a supported record;
  report its target owner, but emit no separate layer;
- `non_executable`: skip only for an explicit structural/addressing reason;
- `unsupported`: skip only an explicit compiler-classified external boundary
  serialized with `op: "unsupported"`, report its dataflow boundaries, and
  emit no layer. A canonical target or target sequence without a registered
  Alpha CMOD builder fails conversion instead of acquiring this disposition.

`status: "complete"` means every supported record was emitted and validated.
At least one CMOD layer must have been emitted. Complete does not mean every
model operation is Alpha-supported: explicit compiler-classified boundaries
may remain. Read the report's `claims` and `execution_scope` fields before
making a full-program claim.

Compiler support does not imply converter support. A canonical operation with
no CMOD builder is a converter error even if Voyager can schedule and map it.
Legacy `cmod_*` fields may be ignored or checked for agreement, but are not
authoritative; fresh compiler dumps contain none.

Each supported record must already contain:

- positive named factors (`batch_size`, `c0..c3`, `k0..k3`, `p0..p3`,
  `p2s`, `q0..q3`, `q2s`, `r`, and `s` for GEMM; `c1`, `k1`, `p1`, and `q1`
  for NMP);
- final padding, split, concat, shuffle, copy, and NMP extension fields;
- exact scratchpad instance and encoded addresses;
- resident payload references and canonical nonresident tensor values;
- target-owned fixed-point multipliers, shifts, bias, activation, precision,
  and scale operands; and
- rank, shape, dtype, lane, and memory-access facts consistent with
  `hardware_config.json`.

The converter consumes `scratchpad`, not host `memory`, and validates the
compiler-selected memory instance. It projects companion scales and packed O4
payloads into CMOD's required traversal and lane order without changing their
canonical values or assuming neutral codes.

GEMM coverage must match resident `N/K/P/Q` and the named factors. INT8 weight
payload C stride is `ceil(resident C / 4) * 4`; other supported precisions use
resident C. Weight K equals resident K. Contradictory or wider unrepresentable
payloads fail.

NMP factors are mode-specific assertions, not generic GEMM loops. Depthwise,
pool, shuffle, copy, and upsample validate against their own input/output
domains and hardware lane width. Only exact 2D max-pool maps to max-pool.

`utils.py` owns shared parsing and target-file vocabulary used by `convert.py`.
Route and absorption policy stays in the converter unless another conversion
tool has a concrete need for the same pure helper.

## Fidelity boundary

Successful conversion proves that emitted layer collateral is structurally
C-model-compatible and internally consistent. Conversion does not itself
prove target execution, whole-network accuracy, independently generated AoV
parity, latency, energy, or accelerator call counts.

Remaining distinct boundaries include packed whole-memory images and load
sequencing, drain/NMS controls, calibrated energy, and end-to-end accuracy.
Preserve all conversion warnings in any conclusion.
