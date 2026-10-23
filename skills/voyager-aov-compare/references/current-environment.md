<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager and external collateral setup

Treat every input as user-supplied and verify it before a run. This repository
does not package private AoV collateral, site-local checkpoints, or simulator
installations.

## Portable variables

From the repository root, define the paths needed by the requested validation:

```bash
export REPO="$PWD"
export PYTHON=python
export VOYAGER_YOLO_ROOT=/path/to/yolov5-face
export YOLO_CHECKPOINT=/path/to/yolov5n-0.5.pt
export OUTPUT_ROOT=/tmp/voyager-yolo-magnet-alpha
```

The compiler entry point is `test/test_codegen.py`; the combined YOLO and
Magnet Alpha adapter is
`test/utils/models/yolo5_face_nv_magnet_alpha.py`. Set `MPLCONFIGDIR` to a
writable temporary directory if Matplotlib cannot create its normal cache.

## Reference inputs

For a compiler smoke test, use a conventional YOLO checkpoint. The adapter
projects the first convolution to grayscale, rewrites SiLU to ReLU, and folds
BatchNorm while loading it. AoV comparison is a distinct, optional validation
level and requires the user to supply a read-only collateral root and describe
its provenance.

Never run a collateral-generation or snapshot script during a comparison
unless the user explicitly requests regeneration.

## Collateral naming

AoV-style collateral commonly converts dots in model names to underscores,
for example `model.1.branch2.5` to `model_1_branch2_5`. Typical layer files are
`schedule1.txt`, `1_inputs.txt`, `1_weights.txt`, `1_outputs.txt`, and their
scale companions. Read schedule metadata rather than inferring precision from
file names.

## Generated evidence

Keep fresh reports outside the source tree. Record the source revision,
checkpoint checksum, command, calibration mode, output directory, comparison
scope, and first unexplained mismatch. Do not cite an untracked site-local
helper or an unavailable scratch path as reproducible evidence.
