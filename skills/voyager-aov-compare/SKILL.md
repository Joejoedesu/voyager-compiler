---
name: voyager-aov-compare
description: Run, validate, and diagnose Voyager compiler flows against user-supplied AoV and Alpha collateral.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager AoV Compare

Validate one explicit equivalence level at a time. Never report AoV parity from
only a successful compile or matching raw operation count.

Read [references/current-environment.md](references/current-environment.md)
before the first run in a session or whenever external inputs may have changed.

## Establish the baseline

1. Locate the Voyager repository and read its contributor instructions.
2. Record `git status --short` and preserve unrelated worktree changes.
3. Verify every user-supplied checkpoint, source tree, collateral root, Python,
   and output directory.
4. Check modification times instead of trusting an old `model.txt` or report.
5. Keep external collateral read-only. Do not run generation or snapshot
   scripts unless the user explicitly requests regeneration.

## Select the validation level

Choose the smallest level that answers the request, then state what it proves:

1. **Focused behavior:** run the affected unit test.
2. **Compiler smoke:** compile with zero calibration to validate graph plumbing.
3. **Rewrite validation:** inspect every converted and transformed output and
   require transformed and mapped outputs to agree exactly.
4. **Structural collateral:** compare logical layers, shapes, operand dtypes,
   scale shapes, and normalized operation classes.
5. **Numerical collateral:** compare aligned payloads, scales, accumulators,
   bias stages, and outputs against supplied reference collateral.
6. **Accuracy:** use calibrated data and a task metric. Never treat a
   zero-calibration statistic as an accuracy result.

## Run the mixed-policy Magnet Alpha model

Magnet Alpha YOLO compilation uses the adapter's fixed mixed W4A4/W8A8 policy.
Do not add global activation, weight, bias, or output-activation overrides
unless the experiment intentionally changes that policy.

```bash
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"

MPLCONFIGDIR=/tmp/matplotlib \
"$PYTHON" test/test_codegen.py yolo5-face-nv \
  --pt2e \
  --target magnet-alpha \
  --conv2d_im2col \
  --scratchpad_mapping end_to_end \
  --execution_order dp \
  --calibration_steps 0 \
  --local_scheduler magnet_alpha_rule_based \
  --model_name_or_path "$YOLO_CHECKPOINT" \
  --model_output_dir "$OUTPUT_ROOT" \
  --dump_tensors
```

Zero calibration is only a deterministic compiler smoke test. Calibrated or
accuracy experiments require a separately supplied dataset.

Treat the run as failed if any output is missing, non-finite,
shape-incompatible, or fails its configured equality gate. Confirm the process
exit status instead of relying on a printed message.

## Compare structural collateral

No private comparator or AoV collateral is bundled. When the user supplies an
external read-only comparator, compare normalized semantic classes rather than
raw directory counts. Separate logical compute, physical movement, explicit
quantization helpers, serialized addressing records, and output packaging.
Normalize layout, padding, element width, signedness, packed-int4 order, and
scale encoding before classifying a mismatch.

## Compare numerical collateral

Align the logical layer, test ID, tensor stage, quantization domain, layout,
padding, signedness, packing, and scale encoding. Compare in first-divergence
order:

1. quantized input and weight payloads;
2. input and weight fine scales;
3. coarse scales or fixed-point multiplier and shift;
4. integer partial sums and post-scale accumulators;
5. bias representation and application order; and
6. producer output and downstream requantization.

Report exact equality for integer data. For floating references, report the
tolerance, maximum error, normalized RMSE, cosine similarity, and SQNR when
appropriate. Stop at the first unexplained divergent stage.

## Report the result

Include the repository state, exact command, checkpoint checksum, calibration
mode, target, output directory, optional reference-collateral provenance,
validation levels completed, mismatched layer names, known representation
differences, and the strongest justified conclusion.
