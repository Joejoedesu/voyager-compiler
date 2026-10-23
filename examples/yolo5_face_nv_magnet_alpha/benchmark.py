#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Benchmark AoV YOLO FP32 and calibrated PT2E face detections."""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPOSITORY_ROOT / "test"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Model loading, target policy, and calibration share one exact-pair adapter.
from test.test_codegen import VECTOR_PIPELINE
from test.utils.models import (
    yolo5_face_nv_magnet_alpha as workload_target,
)
from test.utils.models.utils import get_hardware_config
from voyager_compiler import compile as voyager_compile
from voyager_compiler.codegen import (
    END_TO_END_SCRATCHPAD_MAPPING,
    LOCAL_SCHEDULERS,
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
)

DEFAULT_CALIBRATION_STEPS = 30
DEFAULT_EVALUATION_FRAMES = 60
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_NMS_IOU_THRESHOLD = 0.5
DEFAULT_ACCURACY_IOU_THRESHOLD = 0.5
DEFAULT_CPU_THREADS = 8
DEFAULT_MIN_COMPILED_ACCURACY = 0.99
DEFAULT_MIN_COMPILED_MEAN_IOU = 0.65
DEFAULT_MIN_COMPILED_DETECTION_RATE = 1.0
DEFAULT_MIN_COMPILED_AGREEMENT = 1.0
DEFAULT_MIN_COMPILED_AGREEMENT_IOU = 0.70
DEFAULT_GROUND_TRUTH = Path(workload_target.AOV_CALIBRATION_VIDEO).with_name(
    "calibration_video_gt.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "AoV YOLO checkpoint; defaults to the exact-pair adapter's "
            "checkpoint list"
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(workload_target.AOV_CALIBRATION_VIDEO),
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=Path(workload_target.AOV_CAMERA_CALIBRATION),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=DEFAULT_CALIBRATION_STEPS,
    )
    parser.add_argument(
        "--aov-calibration-profile",
        choices=workload_target.AOV_CALIBRATION_PROFILES,
        default=workload_target.AOV_DEFAULT_CALIBRATION_PROFILE,
        help=(
            "Use fresh 192px observer calibration or the released AoV "
            "status/xscale profile"
        ),
    )
    parser.add_argument(
        "--evaluation-frames",
        type=int,
        default=DEFAULT_EVALUATION_FRAMES,
        help="Held-out frame count; use 0 for every remaining frame",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=DEFAULT_NMS_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--accuracy-iou-threshold",
        type=float,
        default=DEFAULT_ACCURACY_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for machine-readable JSON results",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("/tmp/voyager-aov-yolo-accuracy"),
        help="Final compiled dump directory",
    )
    parser.add_argument(
        "--local-scheduler",
        choices=LOCAL_SCHEDULERS,
        default=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    )
    parser.add_argument(
        "--min-compiled-accuracy",
        type=float,
        default=DEFAULT_MIN_COMPILED_ACCURACY,
    )
    parser.add_argument(
        "--min-compiled-mean-iou",
        type=float,
        default=DEFAULT_MIN_COMPILED_MEAN_IOU,
    )
    parser.add_argument(
        "--min-compiled-detection-rate",
        type=float,
        default=DEFAULT_MIN_COMPILED_DETECTION_RATE,
    )
    parser.add_argument(
        "--min-compiled-agreement",
        type=float,
        default=DEFAULT_MIN_COMPILED_AGREEMENT,
    )
    parser.add_argument(
        "--min-compiled-agreement-iou",
        type=float,
        default=DEFAULT_MIN_COMPILED_AGREEMENT_IOU,
    )
    return parser.parse_args()


def _compiler_args(args):
    checkpoint = None if args.checkpoint is None else str(args.checkpoint)
    return SimpleNamespace(
        activation=None,
        bf16=False,
        bias=None,
        calibration_steps=args.calibration_steps,
        force_scale_power_of_two=False,
        model_name_or_path=checkpoint,
        output_activation=None,
        pt2e=True,
        target=workload_target.TARGET,
        weight=None,
        aov_calibration_profile=args.aov_calibration_profile,
    )


def _detect_head_metadata(model):
    detect_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "Detect"
    ]
    if len(detect_modules) != 1:
        raise ValueError(
            f"Expected one AoV Detect module, found {len(detect_modules)}"
        )

    detect = detect_modules[0]
    strides = detect.stride.detach().cpu().float()
    anchors = detect.anchors.detach().cpu().float()
    anchors = anchors * strides.view(-1, 1, 1)
    return anchors, strides


def _flatten_raw_heads(outputs):
    if isinstance(outputs, torch.Tensor):
        return [outputs] if outputs.ndim == 5 else []

    heads = []
    for output in outputs:
        heads.extend(_flatten_raw_heads(output))
    return heads


def _decode_raw_heads(outputs, anchors, strides):
    raw_heads = _flatten_raw_heads(outputs)
    if len(raw_heads) != len(strides):
        raise ValueError(
            f"Expected {len(strides)} raw heads, got {len(raw_heads)}"
        )

    decoded_boxes = []
    decoded_scores = []
    decoded_objectness = []
    for head, head_anchors, stride in zip(raw_heads, anchors, strides):
        head = head.detach().cpu().float()
        if head.shape[0] != 1 or head.shape[-1] < 16:
            raise ValueError(f"Unexpected AoV raw head shape: {head.shape}")

        height, width = head.shape[2:4]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).view(
            1,
            1,
            height,
            width,
            2,
        )
        anchor_grid = head_anchors.view(1, -1, 1, 1, 2)

        box_xy = (head[..., :2].sigmoid() * 2 - 0.5 + grid) * stride
        box_wh = (head[..., 2:4].sigmoid() * 2).square() * anchor_grid
        top_left = box_xy - box_wh / 2
        bottom_right = box_xy + box_wh / 2
        objectness = head[..., 4].sigmoid()
        class_probability = head[..., 15].sigmoid()

        decoded_boxes.append(
            torch.cat((top_left, bottom_right), dim=-1).reshape(-1, 4)
        )
        decoded_scores.append((objectness * class_probability).reshape(-1))
        decoded_objectness.append(objectness.reshape(-1))

    return (
        torch.cat(decoded_boxes),
        torch.cat(decoded_scores),
        torch.cat(decoded_objectness),
    )


def _box_iou(box, boxes):
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    union = box_area + boxes_area - intersection
    return intersection / union.clamp(min=torch.finfo(torch.float32).eps)


def _nms(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    kept = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlaps = _box_iou(boxes[current], boxes[remaining])
        order = remaining[overlaps <= iou_threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long)


def _scale_box_to_frame(box, frame_shape):
    frame_height, frame_width = frame_shape
    image_size = workload_target.AOV_CALIBRATION_IMAGE_SIZE
    gain = min(image_size / frame_height, image_size / frame_width)
    horizontal_pad = (image_size - frame_width * gain) / 2
    vertical_pad = (image_size - frame_height * gain) / 2

    box = box.clone()
    box[[0, 2]] -= horizontal_pad
    box[[1, 3]] -= vertical_pad
    box /= gain
    box[[0, 2]].clamp_(0, frame_width)
    box[[1, 3]].clamp_(0, frame_height)
    return box.round()


def _select_detection(
    outputs,
    anchors,
    strides,
    frame_shape,
    confidence_threshold,
    nms_iou_threshold,
):
    boxes, scores, objectness = _decode_raw_heads(
        outputs,
        anchors,
        strides,
    )
    candidates = (objectness > confidence_threshold) & (
        scores > confidence_threshold
    )
    boxes = boxes[candidates]
    scores = scores[candidates]
    if boxes.numel() == 0:
        return None

    kept = _nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[kept]
    scores = scores[kept]

    # AoV's face detector selects the largest surviving face by default.
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    selected = areas.argmax()
    return {
        "box": _scale_box_to_frame(boxes[selected], frame_shape),
        "confidence": float(scores[selected]),
    }


def _load_baseline_boxes(path):
    with path.open() as ground_truth_file:
        records = yaml.safe_load(ground_truth_file)

    boxes = {}
    for frame_name, frame_record in records.items():
        faces = frame_record.get("detected_faces_0", [])
        if len(faces) != 1:
            raise ValueError(
                f"Expected one baseline face in {frame_name}, got {len(faces)}"
            )
        frame_index = int(frame_name.removeprefix("frame_"))
        boxes[frame_index] = torch.tensor(
            faces[0]["face_location"],
            dtype=torch.float32,
        )
    return boxes


def _load_evaluation_inputs(args, input_channels):
    camera_matrix, distortion = workload_target._load_camera_calibration(
        str(args.camera_calibration)
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not open evaluation video: {args.video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = args.calibration_steps
    available_frames = total_frames - start_frame
    evaluation_frames = (
        available_frames
        if args.evaluation_frames == 0
        else args.evaluation_frames
    )
    if evaluation_frames <= 0 or evaluation_frames > available_frames:
        raise ValueError(
            f"Requested {evaluation_frames} held-out frames; "
            f"{available_frames} are available"
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    inputs = []
    try:
        for offset in range(evaluation_frames):
            decoded, frame = capture.read()
            if not decoded:
                raise ValueError(
                    f"Video ended while reading frame {start_frame + offset}"
                )
            model_input = workload_target._preprocess_calibration_input(
                frame,
                input_channels,
                torch.float32,
                camera_matrix,
                distortion,
            )
            inputs.append((start_frame + offset, model_input, frame.shape[:2]))
    finally:
        capture.release()
    return inputs, total_frames


def _evaluate_model(
    model,
    inputs,
    baseline_boxes,
    anchors,
    strides,
    args,
    label,
):
    ious = []
    confidences = []
    detections = []
    elapsed_seconds = 0.0
    for frame_index, model_input, frame_shape in tqdm(inputs, desc=label):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(model_input)
        elapsed_seconds += time.perf_counter() - start
        detection = _select_detection(
            outputs,
            anchors,
            strides,
            frame_shape,
            args.confidence_threshold,
            args.nms_iou_threshold,
        )

        baseline_box = baseline_boxes[frame_index]
        if detection is None:
            ious.append(0.0)
            detections.append(None)
            continue

        iou = float(_box_iou(detection["box"], baseline_box.view(1, 4)))
        ious.append(iou)
        confidences.append(detection["confidence"])
        detections.append(detection)

    correct = sum(iou >= args.accuracy_iou_threshold for iou in ious)
    return {
        "metrics": {
            "detections": sum(item is not None for item in detections),
            "frames": len(inputs),
            "mean_confidence": (
                float(np.mean(confidences)) if confidences else 0.0
            ),
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "top_face_accuracy": correct / len(ious),
            "mean_cpu_inference_seconds": elapsed_seconds / len(inputs),
        },
        "detections": detections,
    }


def _prediction_agreement(
    reference_result,
    candidate_result,
    threshold,
    reference_label,
    candidate_label,
):
    agreement_ious = []
    for reference, candidate in zip(
        reference_result["detections"],
        candidate_result["detections"],
    ):
        if reference is None or candidate is None:
            agreement_ious.append(0.0)
            continue
        agreement_ious.append(
            float(
                _box_iou(
                    reference["box"],
                    candidate["box"].view(1, 4),
                )
            )
        )
    label = f"{reference_label}_{candidate_label}"
    return {
        f"mean_{label}_box_iou": float(np.mean(agreement_ious)),
        f"{label}_box_agreement": sum(
            iou >= threshold for iou in agreement_ious
        )
        / len(agreement_ious),
    }


def _print_metrics(label, metrics, accuracy_iou_threshold):
    accuracy = 100 * metrics["top_face_accuracy"]
    print(
        f"{label}: accuracy@IoU{accuracy_iou_threshold:g}="
        f"{accuracy:.2f}%, "
        f"mean IoU={metrics['mean_iou']:.4f}, "
        f"detections={metrics['detections']}/{metrics['frames']}, "
        "CPU inference="
        f"{metrics['mean_cpu_inference_seconds']:.3f} s/frame"
    )


def _acceptance(args, compiled_metrics, agreement):
    detection_rate = compiled_metrics["detections"] / compiled_metrics["frames"]
    actual = {
        "compiled_accuracy": compiled_metrics["top_face_accuracy"],
        "compiled_mean_iou": compiled_metrics["mean_iou"],
        "compiled_detection_rate": detection_rate,
        "calibrated_compiled_agreement": agreement[
            "calibrated_compiled_box_agreement"
        ],
        "calibrated_compiled_mean_iou": agreement[
            "mean_calibrated_compiled_box_iou"
        ],
    }
    minimum = {
        "compiled_accuracy": args.min_compiled_accuracy,
        "compiled_mean_iou": args.min_compiled_mean_iou,
        "compiled_detection_rate": args.min_compiled_detection_rate,
        "calibrated_compiled_agreement": args.min_compiled_agreement,
        "calibrated_compiled_mean_iou": (args.min_compiled_agreement_iou),
    }
    failures = [
        f"{name}={actual[name]:.6g} is below {threshold:.6g}"
        for name, threshold in minimum.items()
        if actual[name] < threshold
    ]
    return {
        "passed": not failures,
        "minimum": minimum,
        "actual": actual,
        "failures": failures,
    }


def main():
    args = parse_args()
    if args.calibration_steps <= 0:
        raise ValueError("Calibration steps must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("CPU thread count must be positive")

    torch.set_num_threads(args.cpu_threads)
    compiler_args = _compiler_args(args)
    compiler_args = workload_target._effective_args(compiler_args)
    voyager_fp32_model, example_args = workload_target.load_model(
        compiler_args,
        torch.float32,
    )
    original_model = workload_target._load_checkpoint(
        compiler_args.model_name_or_path
    )
    workload_target._ensure_detect_helpers(original_model)
    workload_target._set_raw_head_export(original_model)
    anchors, strides = _detect_head_metadata(original_model)
    input_channels = example_args[0].shape[1]
    inputs, total_video_frames = _load_evaluation_inputs(
        args,
        input_channels,
    )
    baseline_boxes = _load_baseline_boxes(args.ground_truth)
    for frame_index, _, _ in inputs:
        if frame_index not in baseline_boxes:
            raise ValueError(f"No baseline face box for frame {frame_index}")

    original_result = _evaluate_model(
        original_model,
        inputs,
        baseline_boxes,
        anchors,
        strides,
        args,
        "Evaluating original checkpoint",
    )
    voyager_fp32_result = _evaluate_model(
        voyager_fp32_model,
        inputs,
        baseline_boxes,
        anchors,
        strides,
        args,
        "Evaluating Voyager FP32 reference",
    )

    quantized_model, quantized_example_args = workload_target.load_model(
        compiler_args,
        torch.float32,
    )
    hardware_config = get_hardware_config(compiler_args)
    quantizer = workload_target.configure_quantizer(
        None,
        compiler_args,
    )
    calibration_data = workload_target.retrieve_dataset(compiler_args)
    calibrated_model = workload_target.prepare_calibrated_model(
        quantized_model,
        quantized_example_args,
        quantizer,
        calibration_data,
        compiler_args,
        hardware_config,
    )
    calibration_profile_report = calibrated_model.meta.get(
        "aov_calibration_profile",
        {},
    )
    calibrated_result = _evaluate_model(
        calibrated_model,
        inputs,
        baseline_boxes,
        anchors,
        strides,
        args,
        "Evaluating calibrated PT2E",
    )
    workload_target.transform(
        calibrated_model,
        quantized_example_args,
        patterns=VECTOR_PIPELINE,
        hardware_config=hardware_config,
    )
    transformed_result = _evaluate_model(
        calibrated_model,
        inputs,
        baseline_boxes,
        anchors,
        strides,
        args,
        "Evaluating transformed PT2E",
    )
    voyager_compile(
        calibrated_model,
        quantized_example_args,
        output_dir=str(args.model_output_dir),
        dump_tensors=True,
        hardware_config=hardware_config,
        scheduler_backend=args.local_scheduler,
        scratchpad_mapping=END_TO_END_SCRATCHPAD_MAPPING,
    )
    compiled_result = _evaluate_model(
        calibrated_model,
        inputs,
        baseline_boxes,
        anchors,
        strides,
        args,
        "Evaluating compiled and memory-mapped PT2E",
    )

    original_metrics = original_result["metrics"]
    voyager_fp32_metrics = voyager_fp32_result["metrics"]
    calibrated_metrics = calibrated_result["metrics"]
    transformed_metrics = transformed_result["metrics"]
    compiled_metrics = compiled_result["metrics"]
    agreement = _prediction_agreement(
        voyager_fp32_result,
        calibrated_result,
        args.accuracy_iou_threshold,
        "fp32",
        "calibrated",
    )
    agreement.update(
        _prediction_agreement(
            calibrated_result,
            transformed_result,
            args.accuracy_iou_threshold,
            "calibrated",
            "transformed",
        )
    )
    agreement.update(
        _prediction_agreement(
            calibrated_result,
            compiled_result,
            args.accuracy_iou_threshold,
            "calibrated",
            "compiled",
        )
    )
    acceptance = _acceptance(args, compiled_metrics, agreement)
    result = {
        "benchmark_kind": "AoV recorded-baseline agreement",
        "warning": (
            "The YAML boxes are recorded AoV baseline outputs, not "
            "independent human annotations; this is not WIDER FACE AP."
        ),
        "configuration": {
            "accuracy_iou_threshold": args.accuracy_iou_threshold,
            "calibration_profile": args.aov_calibration_profile,
            "calibration_frame_range": [0, args.calibration_steps - 1],
            "calibration_steps": args.calibration_steps,
            "checkpoint": compiler_args.model_name_or_path,
            "confidence_threshold": args.confidence_threshold,
            "cpu_threads": args.cpu_threads,
            "evaluation_frame_range": [
                inputs[0][0],
                inputs[-1][0],
            ],
            "evaluation_frames": len(inputs),
            "ground_truth": str(args.ground_truth),
            "nms_iou_threshold": args.nms_iou_threshold,
            "local_scheduler": args.local_scheduler,
            "model_output_dir": str(args.model_output_dir),
            "quantization": workload_target.AOV_PRECISION_POLICY,
            "total_video_frames": total_video_frames,
            "video": str(args.video),
        },
        "original_checkpoint": original_metrics,
        "voyager_fp32_reference": voyager_fp32_metrics,
        "calibration_profile": calibration_profile_report,
        "calibrated_pt2e": calibrated_metrics,
        "transformed_pt2e": transformed_metrics,
        "compiled_pt2e": compiled_metrics,
        "delta": {
            "calibrated_minus_original_accuracy_percentage_points": 100
            * (
                calibrated_metrics["top_face_accuracy"]
                - original_metrics["top_face_accuracy"]
            ),
            "calibrated_minus_original_mean_iou": (
                calibrated_metrics["mean_iou"] - original_metrics["mean_iou"]
            ),
            "calibrated_minus_voyager_fp32_accuracy_percentage_points": 100
            * (
                calibrated_metrics["top_face_accuracy"]
                - voyager_fp32_metrics["top_face_accuracy"]
            ),
            "transformed_minus_calibrated_accuracy_percentage_points": 100
            * (
                transformed_metrics["top_face_accuracy"]
                - calibrated_metrics["top_face_accuracy"]
            ),
            "transformed_minus_calibrated_mean_iou": (
                transformed_metrics["mean_iou"] - calibrated_metrics["mean_iou"]
            ),
            "compiled_minus_calibrated_accuracy_percentage_points": 100
            * (
                compiled_metrics["top_face_accuracy"]
                - calibrated_metrics["top_face_accuracy"]
            ),
            "compiled_minus_calibrated_mean_iou": (
                compiled_metrics["mean_iou"] - calibrated_metrics["mean_iou"]
            ),
        },
        "prediction_agreement": agreement,
        "acceptance": acceptance,
    }

    _print_metrics(
        "Original checkpoint",
        original_metrics,
        args.accuracy_iou_threshold,
    )
    _print_metrics(
        "Voyager FP32 reference",
        voyager_fp32_metrics,
        args.accuracy_iou_threshold,
    )
    _print_metrics(
        "Calibrated PT2E",
        calibrated_metrics,
        args.accuracy_iou_threshold,
    )
    _print_metrics(
        "Transformed PT2E",
        transformed_metrics,
        args.accuracy_iou_threshold,
    )
    _print_metrics(
        "Compiled and memory-mapped PT2E",
        compiled_metrics,
        args.accuracy_iou_threshold,
    )
    accuracy_delta = result["delta"][
        "calibrated_minus_original_accuracy_percentage_points"
    ]
    print("Calibrated minus original accuracy: " f"{accuracy_delta:+.2f} pp")
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not acceptance["passed"]:
        raise RuntimeError(
            "Compiled accuracy acceptance failed: "
            + "; ".join(acceptance["failures"])
        )


if __name__ == "__main__":
    main()
