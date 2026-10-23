<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Magnet Alpha regression

The maintained open-source regression boundary covers compiler serialization
and fail-closed collateral conversion. A zero-calibration run is a compiler
smoke test, not an accuracy result.

## YOLO compiler smoke test

Use a conventional `yolov5n-0.5.pt` checkpoint. The adapter converts the
first convolution to one grayscale input channel, replaces SiLU with ReLU,
and folds BatchNorm before PT2E export.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export VOYAGER_YOLO_ROOT=/path/to/yolov5-face
export YOLO_CHECKPOINT=/path/to/yolov5n-0.5.pt
export COMPILER_OUTPUT=/tmp/yolo5-face-nv-alpha

python test/test_codegen.py yolo5-face-nv \
  --pt2e \
  --target magnet-alpha \
  --conv2d_im2col \
  --scratchpad_mapping end_to_end \
  --execution_order dp \
  --calibration_steps 0 \
  --local_scheduler magnet_alpha_rule_based \
  --model_name_or_path "$YOLO_CHECKPOINT" \
  --model_output_dir "$COMPILER_OUTPUT" \
  --dump_tensors
```

Acceptance requires successful serialization and exact
transformed-to-memory-mapped reference comparison. Converted-reference versus
target-transformed statistics are diagnostic in this zero-calibration lane.

## Structural `model.txt` comparison

When a prior snapshot is available, compare input and parameter layouts,
operation target sequences, tiling factors, and hardware configuration.
Report quantization values and scalar-versus-length-one metadata separately;
those expected representation differences must not be described as complete
file equality.

## Fail-closed conversion

```bash
export CONVERTED_OUTPUT=/tmp/yolo5-face-nv-alpha-converted

python conversion/magnet-alpha/convert.py \
  "$COMPILER_OUTPUT" \
  --output "$CONVERTED_OUTPUT" \
  --require-all-supported \
  --require-connected-program
```

Accept conversion only when `conversion_report.json` has no issues and its
claims report complete supported-operation emission and connected Alpha
dataflow. Conversion does not prove target execution.

## Focused maintenance checks

```bash
python -m unittest discover \
  --start-directory conversion/magnet-alpha \
  --pattern 'test_convert.py'
```

See the [converter guide](../../../conversion/magnet-alpha/README.md) for the
file-level contract and failure policy.
