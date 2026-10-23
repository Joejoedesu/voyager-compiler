<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# YOLOv5 face detection on Magnet Alpha

The `yolo5-face-nv` adapter accepts a conventional YOLOv5-face checkpoint and
prepares it for the Magnet Alpha compiler path. Model loading performs the
target-required preprocessing before PT2E export:

- project the first RGB convolution onto one grayscale input channel by
  summing its channel weights;
- replace every SiLU activation with ReLU;
- fold BatchNorm parameters into their preceding convolutions; and
- apply the existing Alpha input-padding and depthwise-bias legalization.

No AoV checkpoint, activation-range sidecar, calibration video, or accuracy
dataset is required for a zero-calibration compiler smoke test.

## Compile

Run from the repository root. The YOLO source tree and checkpoint are external
inputs and must be supplied explicitly when they are not installed under
`libraries/`.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export VOYAGER_YOLO_ROOT=/path/to/yolov5-face
export YOLO_CHECKPOINT=/path/to/yolov5n-0.5.pt

python test/test_codegen.py yolo5-face-nv \
  --pt2e \
  --target magnet-alpha \
  --conv2d_im2col \
  --scratchpad_mapping end_to_end \
  --execution_order dp \
  --calibration_steps 0 \
  --local_scheduler magnet_alpha_rule_based \
  --model_name_or_path "$YOLO_CHECKPOINT" \
  --model_output_dir /tmp/yolo5-face-nv-alpha \
  --dump_tensors
```

This smoke test checks compiler plumbing and transformed-to-memory-mapped
reference consistency. It is not an accuracy measurement.

## Convert

Convert the fresh compiler output with the fail-closed Alpha converter:

```bash
python conversion/magnet-alpha/convert.py \
  /tmp/yolo5-face-nv-alpha \
  --output /tmp/yolo5-face-nv-alpha-converted \
  --require-all-supported \
  --require-connected-program
```

The converter emits Alpha schedules and tensors only when every executable
operation is supported and the emitted program is connected. See the
[converter guide](../../conversion/magnet-alpha/README.md) for its structural
acceptance contract. Conversion does not establish target execution.

## Optional QAT tools

`qat.py` and `qat_deploy.py` remain optional training tools. They require an
external annotated dataset and are not part of the compiler/converter sanity
gate above. Their paths are configurable through command-line arguments and
`VOYAGER_YOLO_ROOT`.
