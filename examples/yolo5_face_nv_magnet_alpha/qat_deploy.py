#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Refresh, freeze, compile, and verify a sidecar-free AoV QAT checkpoint."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[1]
TEST_ROOT = REPOSITORY_ROOT / "test"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import benchmark
import qat as qat_train
from test.test_codegen import VECTOR_PIPELINE
from test.utils.models import (
    yolo5_face_nv_magnet_alpha as workload_target,
)
from test.utils.models.utils import get_hardware_config
from test.verify_help import compare_output_statistics
from voyager_compiler import (
    QScheme,
    compile as voyager_compile,
    convert_pt2e,
    expand,
    fetch_attr,
    prepare_qat_pt2e,
)
from voyager_compiler.codegen import (
    END_TO_END_SCRATCHPAD_MAPPING,
    LOCAL_SCHEDULERS,
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
)
from voyager_compiler.fake_quantize import (
    FusedAmaxObsFakeQuantize,
)
from voyager_compiler.codegen.target_backends.magnet_alpha import (
    fixed_point as magnet_alpha_fixed_point,
)

PT2EParameterEMA = workload_target.PT2EParameterEMA


DEFAULT_OUTPUT = Path("runs/yolo5_face_nv_magnet_alpha/deployment")
DEFAULT_MAX_STAGE_HEAD_ERROR = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qat-checkpoint", type=Path, required=True)
    parser.add_argument("--folded-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-steps", type=int, default=30)
    parser.add_argument(
        "--evaluation-start-frame",
        type=int,
        default=None,
        help=("First held-out benchmark frame; defaults to --refresh-steps"),
    )
    parser.add_argument("--evaluation-frames", type=int, default=0)
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
        default=benchmark.DEFAULT_GROUND_TRUTH,
    )
    parser.add_argument(
        "--local-scheduler",
        choices=LOCAL_SCHEDULERS,
        default=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--use-live-weights", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=benchmark.DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=benchmark.DEFAULT_NMS_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--accuracy-iou-threshold",
        type=float,
        default=benchmark.DEFAULT_ACCURACY_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--min-compiled-accuracy",
        type=float,
        default=benchmark.DEFAULT_MIN_COMPILED_ACCURACY,
    )
    parser.add_argument(
        "--min-compiled-mean-iou",
        type=float,
        default=benchmark.DEFAULT_MIN_COMPILED_MEAN_IOU,
    )
    parser.add_argument(
        "--min-compiled-detection-rate",
        type=float,
        default=benchmark.DEFAULT_MIN_COMPILED_DETECTION_RATE,
    )
    parser.add_argument(
        "--min-compiled-agreement",
        type=float,
        default=benchmark.DEFAULT_MIN_COMPILED_AGREEMENT,
    )
    parser.add_argument(
        "--min-compiled-agreement-iou",
        type=float,
        default=benchmark.DEFAULT_MIN_COMPILED_AGREEMENT_IOU,
    )
    parser.add_argument(
        "--max-stage-head-error",
        type=float,
        default=DEFAULT_MAX_STAGE_HEAD_ERROR,
    )
    parser.add_argument(
        "--allow-unqualified-checkpoint",
        action="store_true",
        help=(
            "Allow an exact smoke/debug checkpoint that was not selected "
            "by finite YOLO validation fitness"
        ),
    )
    parser.add_argument("--allow-accuracy-failure", action="store_true")
    return parser.parse_args()


def _compiler_args(args, folded_checkpoint):
    return SimpleNamespace(
        activation=None,
        aov_calibration_profile=(
            workload_target.AOV_FRESH_192_CALIBRATION_PROFILE
        ),
        aov_image_size=workload_target.AOV_CALIBRATION_IMAGE_SIZE,
        bf16=False,
        bias=None,
        calibration_steps=args.refresh_steps,
        force_scale_power_of_two=False,
        model_name_or_path=str(folded_checkpoint),
        output_activation=None,
        pt2e=True,
        target=workload_target.TARGET,
        weight=None,
    )


def _resolve_folded_checkpoint(args, state):
    if args.folded_checkpoint is not None:
        return args.folded_checkpoint
    if (
        workload_target.checkpoint_format(state)
        == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
    ):
        path = state.get("source_checkpoint")
        if path is None:
            raise ValueError(
                "Deployment snapshot has no source checkpoint; pass "
                "--folded-checkpoint"
            )
        path = Path(path)
        if not path.is_absolute():
            path = args.qat_checkpoint.parent / path
        return path.resolve()
    provenance = state.get("provenance", {})
    path = provenance.get("folded_checkpoint")
    if path is None:
        raise ValueError(
            "QAT checkpoint has no folded source; pass --folded-checkpoint"
        )
    return Path(path)


def _validate_deployment_source(args, state, folded_checkpoint):
    checkpoint_type = workload_target.checkpoint_format(state)
    phase = workload_target.checkpoint_phase(state)
    if phase != "target_exact":
        raise ValueError(
            "Deployment requires a target-exact QAT checkpoint, got "
            f"{phase!r}"
        )
    provenance = state.get("provenance", {})
    if provenance.get("released_weight_sidecars") is not False:
        raise ValueError(
            "Deployment requires sidecar-free folded-live QAT provenance"
        )
    if provenance.get("first_layer_input_mode", "padding") != "padding":
        raise ValueError(
            "AoV QAT deployment requires first-layer padding preprocessing"
        )
    image_size = provenance.get("image_size")
    if image_size not in workload_target.AOV_SUPPORTED_QAT_IMAGE_SIZES:
        raise ValueError(
            "Deployment requires a supported 192px or 800px QAT "
            "checkpoint, got "
            f"{image_size!r}"
        )
    policy = provenance.get("precision_policy")
    if policy is None:
        raise ValueError("QAT checkpoint has no precision-policy provenance")
    if policy != workload_target.AOV_PRECISION_POLICY:
        raise ValueError(
            "Deployment precision policy differs from QAT checkpoint: "
            f"{workload_target.AOV_PRECISION_POLICY!r} != {policy!r}"
        )
    if checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT:
        best_fitness = provenance.get("best_fitness")
        checkpoint_role = provenance.get("checkpoint_role")
        selection_fitness = provenance.get("selection_fitness")
    else:
        best_fitness = state.get("best_fitness")
        checkpoint_role = state.get("checkpoint_role")
        selection_fitness = state.get("selection_fitness")
    parameter_source = workload_target.checkpoint_parameter_source(
        state,
        use_live_resume_weights=args.use_live_weights,
    )
    ema_updates = (
        int(provenance.get("ema_updates", 0))
        if checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
        else int(state.get("ema", {}).get("updates", 0))
    )
    if (
        checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
        and args.use_live_weights
        and parameter_source != "live"
    ):
        raise ValueError(
            "--use-live-weights cannot replace the selected parameters in "
            f"a deployment snapshot ({parameter_source!r})"
        )
    best_is_finite = False
    normalized_fitness = None
    if best_fitness is not None:
        try:
            normalized_fitness = float(best_fitness)
            best_is_finite = math.isfinite(normalized_fitness)
        except (TypeError, ValueError):
            best_is_finite = False
    normalized_selection = None
    selection_is_finite = False
    if selection_fitness is not None:
        try:
            normalized_selection = float(selection_fitness)
            selection_is_finite = math.isfinite(normalized_selection)
        except (TypeError, ValueError):
            selection_is_finite = False
    qualified = (
        checkpoint_role == "best"
        and best_is_finite
        and selection_is_finite
        and normalized_selection == normalized_fitness
        and parameter_source == "ema"
        and ema_updates > 0
    )
    if not qualified and not args.allow_unqualified_checkpoint:
        raise ValueError(
            "Deployment checkpoint is not the finite-fitness `best` "
            "snapshot; "
            "use --allow-unqualified-checkpoint only for smoke/debug runs"
        )
    if not folded_checkpoint.is_file():
        raise FileNotFoundError(
            f"Folded checkpoint not found: {folded_checkpoint}"
        )
    expected_sha256 = (
        state.get("source_checkpoint_sha256")
        if checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
        else provenance.get("folded_checkpoint_sha256")
    )
    if expected_sha256 is None:
        raise ValueError("QAT checkpoint has no folded-source checksum")
    actual_sha256 = workload_target.checkpoint_sha256(folded_checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Folded checkpoint checksum differs from QAT provenance: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return {
        "best_fitness": normalized_fitness,
        "checkpoint_role": checkpoint_role,
        "selection_fitness": normalized_selection,
        "parameter_source": parameter_source,
        "ema_updates": ema_updates,
        "uses_ema_weights": parameter_source == "ema",
        "checkpoint_format": checkpoint_type,
        "qualified_by_yolo_fitness": qualified,
        "qualification_override": bool(
            args.allow_unqualified_checkpoint and not qualified
        ),
    }


def _load_prepared(args, state, folded_checkpoint):
    checkpoint_type = workload_target.checkpoint_format(state)
    model_checkpoint = (
        args.qat_checkpoint
        if (
            checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
            and args.folded_checkpoint is None
        )
        else folded_checkpoint
    )
    compiler_args = _compiler_args(args, model_checkpoint)
    compiler_args = workload_target._effective_args(compiler_args)
    model, example_args = workload_target.load_model(
        compiler_args,
        torch.float32,
    )
    model.requires_grad_(True)
    hardware = get_hardware_config(compiler_args)
    quantizer = workload_target.configure_quantizer(
        None,
        compiler_args,
    )
    graph = prepare_qat_pt2e(
        model,
        quantizer,
        example_args,
        hardware_config=hardware,
        quantization_region_qspec=(
            workload_target._get_quantization_region_qspec(
                compiler_args,
            )
        ),
    )
    controller = workload_target.PT2EQATController(
        graph,
        hardware,
        expected_exact_boundaries=(
            workload_target.AOV_RELEASED_COLLATERAL_EXACT_BOUNDARY_COUNT
        ),
    )
    if checkpoint_type == workload_target.RESUME_CHECKPOINT_FORMAT:
        controller.load_source_state_dict(state["qat_source"])
        if not args.use_live_weights:
            ema = PT2EParameterEMA(graph)
            ema.load_state_dict(state["ema"], graph)
            ema.copy_to(graph)
    else:
        controller.load_source_state_dict(
            workload_target.deployment_source_state(state)
        )
    controller.enter_dynamic_fake()
    if (
        checkpoint_type == workload_target.DEPLOYMENT_CHECKPOINT_FORMAT
        or not args.use_live_weights
    ):
        controller.reset_weight_statistics()
    return graph, controller, model, example_args, hardware, compiler_args


def _refresh_192(
    graph,
    controller,
    example_args,
    args,
):
    if args.refresh_steps:
        controller.reset_activation_statistics()
        camera_matrix, distortion = workload_target._load_camera_calibration(
            str(args.camera_calibration)
        )
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise ValueError(f"Could not open refresh video: {args.video}")

        channels = example_args[0].shape[1]
        try:
            with torch.no_grad():
                for frame_index in range(args.refresh_steps):
                    decoded, frame = capture.read()
                    if not decoded:
                        raise ValueError(
                            "Refresh video ended after "
                            f"{frame_index} frames; "
                            f"{args.refresh_steps} requested"
                        )
                    inputs = workload_target._preprocess_calibration_input(
                        frame,
                        channels,
                        torch.float32,
                        camera_matrix,
                        distortion,
                    )
                    graph(inputs)
        finally:
            capture.release()

    configured = controller.enter_target_exact()
    controller.before_forward()
    return configured


def _audit_live_w4(graph, controller, expected_count):
    modules = dict(graph.named_modules(remove_duplicate=False))
    source_nodes = {
        str(node.target): node.args[0]
        for node in graph.graph.nodes
        if (
            node.op == "call_module"
            and str(node.target) in controller.roles.weights
            and node.args
        )
    }
    records = []
    for name in controller.roles.weights:
        module = modules[name]
        if (
            not isinstance(module, FusedAmaxObsFakeQuantize)
            or module.qscheme != QScheme.VSQ
        ):
            continue
        source = source_nodes[name]
        if source.op != "get_attr":
            raise ValueError(f"{name}: W4 source is not a live parameter")
        values = fetch_attr(graph, source.target)
        coarse, fine, payload = module.quantize_vsq_components(values)
        physical = magnet_alpha_fixed_point.encode_alpha_vsq_scales(
            fine,
            coarse,
            fine_quant_max=module.scale_quant_max,
            coarse_scale_quant_max=module.coarse_scale_quant_max,
        )
        mantissa = torch.ops.magnet_alpha.vsq_scale_encode.default(
            fine,
            module.scale_quant_max,
            module.coarse_scale_quant_max,
        )
        payload = payload.detach()
        mantissa = mantissa.detach()
        coarse = coarse.detach()
        fine = fine.detach()
        values = values.detach()
        physical_coarse = physical.coarse_scale.detach()
        logical_reconstruction = (
            payload * expand(fine, payload.shape, module.block_size) * coarse
        )
        physical_reconstruction = (
            payload
            * expand(mantissa, payload.shape, module.block_size)
            * physical_coarse
        )
        reconstruction_delta = (
            physical_reconstruction - logical_reconstruction
        ).abs()
        record = {
            "observer": name,
            "parameter": source.target,
            "payload_shape": list(payload.shape),
            "fine_shape": list(fine.shape),
            "coarse_shape": list(coarse.shape),
            "payload_min": float(payload.min()),
            "payload_max": float(payload.max()),
            "payload_integral": bool(
                torch.equal(payload, torch.round(payload))
            ),
            "mantissa_min": int(mantissa.min()),
            "mantissa_max": int(mantissa.max()),
            "mantissa_integral": bool(
                torch.equal(mantissa, torch.round(mantissa))
            ),
            "coarse_positive_finite": bool(
                torch.isfinite(physical_coarse).all()
                and torch.all(physical_coarse > 0)
            ),
            "physical_roundtrip_maximum_error": float(
                reconstruction_delta.max()
            ),
            "live_weight_maximum_error": float(
                (logical_reconstruction - values).abs().max()
            ),
            "static_payload_values": int(module.static_vsq_payload.numel()),
        }
        records.append(record)
    failures = [
        record["observer"]
        for record in records
        if (
            not record["payload_integral"]
            or record["payload_min"] < -7
            or record["payload_max"] > 7
            or not record["mantissa_integral"]
            or record["mantissa_min"] < 0
            or record["mantissa_max"] > 255
            or not record["coarse_positive_finite"]
            or record["physical_roundtrip_maximum_error"] > 1e-6
            or record["static_payload_values"]
        )
    ]
    return {
        "count": len(records),
        "expected_count": expected_count,
        "failures": failures,
        "records": records,
    }


def _capture_raw_heads(model, inputs, controller=None):
    if controller is not None:
        controller.before_forward()
    with torch.no_grad():
        outputs = model(inputs[0][1])
    return [
        value.detach().cpu().clone() for value in qat_train._raw_heads(outputs)
    ]


def _compare_raw_heads(reference, candidate):
    if len(reference) != len(candidate):
        raise RuntimeError(
            "Quantized stage changed the number of detection heads"
        )
    if not reference:
        raise RuntimeError("Quantized stage returned no detection heads")
    records = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"Detection head {index} shape changed from "
                f"{tuple(expected.shape)} to {tuple(actual.shape)}"
            )
        delta = (expected - actual).abs()
        records.append(
            {
                "head": index,
                "shape": list(expected.shape),
                "finite": bool(torch.isfinite(actual).all()),
                "maximum_absolute_error": float(delta.max()),
                "mean_absolute_error": float(delta.mean()),
                "exact_value_fraction": float(
                    torch.eq(expected, actual).float().mean()
                ),
            }
        )
    return {
        "all_finite": all(record["finite"] for record in records),
        "maximum_absolute_error": max(
            record["maximum_absolute_error"] for record in records
        ),
        "heads": records,
    }


def _require_raw_head_equivalence(comparison, label, maximum_error):
    failures = []
    if not comparison["all_finite"]:
        failures.append("non-finite output")
    actual_error = comparison["maximum_absolute_error"]
    if actual_error > maximum_error:
        failures.append(
            f"maximum error {actual_error:.6g} exceeds " f"{maximum_error:.6g}"
        )
    if failures:
        raise RuntimeError(
            f"{label} raw-head equivalence failed: " + "; ".join(failures)
        )


def _write_stage_progress(output_dir, progress):
    path = output_dir / "deployment_stage_progress.json"
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w") as output:
        json.dump(progress, output, indent=2)
    temporary_path.replace(path)


def _compare_raw_head_statistics(reference, candidate, label):
    statistics = compare_output_statistics(
        reference,
        candidate,
        label=label,
    )
    return {
        "comparison_policy": "report_only",
        "requires_matching_shapes_and_finite_values": True,
        "heads": [
            {
                "head": index,
                "shape": list(reference[index].shape),
                **metrics,
            }
            for index, metrics in enumerate(statistics)
        ],
    }


def _write_collateral_manifest(output_dir):
    required = (
        "hardware_config.json",
        "layers.txt",
        "model.txt",
        "performance_estimates.json",
    )
    missing = [
        name
        for name in required
        if not (output_dir / name).is_file()
        or not (output_dir / name).stat().st_size
    ]
    tensor_files = sorted((output_dir / "tensor_files").glob("*.bin"))
    if not tensor_files:
        missing.append("tensor_files/*.bin")
    if missing:
        raise RuntimeError(
            "Compiled collateral is incomplete: " + ", ".join(missing)
        )

    records = []
    for path in sorted(
        item for item in output_dir.rglob("*") if item.is_file()
    ):
        records.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": workload_target.checkpoint_sha256(path),
            }
        )
    manifest = {
        "root": str(output_dir),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "tensor_file_count": len(tensor_files),
        "required_files": list(required),
        "missing": missing,
        "files": records,
    }
    path = output_dir / "collateral_manifest.json"
    with path.open("w") as output:
        json.dump(manifest, output, indent=2)
    return {
        "manifest": str(path),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "tensor_file_count": manifest["tensor_file_count"],
        "missing": manifest["missing"],
    }


def _evaluation_args(args):
    return SimpleNamespace(
        accuracy_iou_threshold=args.accuracy_iou_threshold,
        calibration_steps=(
            args.refresh_steps
            if args.evaluation_start_frame is None
            else args.evaluation_start_frame
        ),
        camera_calibration=args.camera_calibration,
        confidence_threshold=args.confidence_threshold,
        evaluation_frames=args.evaluation_frames,
        ground_truth=args.ground_truth,
        min_compiled_accuracy=args.min_compiled_accuracy,
        min_compiled_agreement=args.min_compiled_agreement,
        min_compiled_agreement_iou=(args.min_compiled_agreement_iou),
        min_compiled_detection_rate=(args.min_compiled_detection_rate),
        min_compiled_mean_iou=args.min_compiled_mean_iou,
        nms_iou_threshold=args.nms_iou_threshold,
        video=args.video,
    )


def main():
    args = parse_args()
    if args.refresh_steps < 0 or args.refresh_steps == 1:
        raise ValueError("Use zero or at least two observer-refresh frames")
    if (
        args.evaluation_start_frame is not None
        and args.evaluation_start_frame < 0
    ):
        raise ValueError("Evaluation start frame must be non-negative")
    if args.cpu_threads <= 0:
        raise ValueError("CPU thread count must be positive")
    if (
        not torch.isfinite(torch.tensor(args.max_stage_head_error))
        or args.max_stage_head_error < 0
    ):
        raise ValueError("Stage head-error threshold must be finite")
    existing_outputs = (
        [path for path in args.output_dir.rglob("*") if path.is_file()]
        if args.output_dir.exists()
        else []
    )
    if existing_outputs:
        raise FileExistsError(
            "Refusing to mix deployment evidence with existing files: "
            + ", ".join(
                str(path.relative_to(args.output_dir))
                for path in existing_outputs[:10]
            )
        )
    torch.set_num_threads(args.cpu_threads)

    state = workload_target.read_checkpoint(args.qat_checkpoint)
    folded_checkpoint = _resolve_folded_checkpoint(args, state)
    qualification = _validate_deployment_source(
        args,
        state,
        folded_checkpoint,
    )
    (
        graph,
        controller,
        source_model,
        example_args,
        hardware,
        compiler_args,
    ) = _load_prepared(args, state, folded_checkpoint)
    configured = _refresh_192(
        graph,
        controller,
        example_args,
        args,
    )
    expected_w4 = workload_target.AOV_RELEASED_W4_WEIGHT_COUNT
    w4_audit = _audit_live_w4(
        graph,
        controller,
        expected_w4,
    )
    if w4_audit["count"] != w4_audit["expected_count"] or w4_audit["failures"]:
        raise RuntimeError("Live W4 source audit failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_provenance = dict(workload_target.checkpoint_provenance(state))
    source_provenance.update(
        {
            "folded_checkpoint": str(folded_checkpoint),
            "folded_checkpoint_sha256": workload_target.checkpoint_sha256(
                folded_checkpoint
            ),
            "deployment_refresh_steps": args.refresh_steps,
        }
    )
    if qualification["qualified_by_yolo_fitness"]:
        workload_target.save_deployment_checkpoint(
            args.output_dir / "deployment_state.pt",
            graph,
            controller,
            provenance=source_provenance,
            checkpoint_role=qualification["checkpoint_role"],
            selection_fitness=qualification["selection_fitness"],
            best_fitness=qualification["best_fitness"],
            parameter_source=qualification["parameter_source"],
            source_state=controller.source_state_dict(),
            ema_updates=qualification["ema_updates"],
            include_observer_state=True,
        )
    else:
        torch.save(
            {
                "format": "voyager.aov.qat.debug-state",
                "format_version": 1,
                "qat_source": controller.source_state_dict(),
                "qat_phase": controller.phase_state_dict(),
                "source_checkpoint": str(args.qat_checkpoint),
                "refresh_steps": args.refresh_steps,
                "parameter_source": qualification["parameter_source"],
            },
            args.output_dir / "deployment_state.pt",
        )

    evaluation_args = _evaluation_args(args)
    anchors, strides = benchmark._detect_head_metadata(source_model)
    inputs, total_frames = benchmark._load_evaluation_inputs(
        evaluation_args,
        example_args[0].shape[1],
    )
    boxes = benchmark._load_baseline_boxes(args.ground_truth)
    missing_boxes = [
        frame_index for frame_index, _, _ in inputs if frame_index not in boxes
    ]
    if missing_boxes:
        raise ValueError(
            "Ground truth has no box for evaluation frames: "
            + ", ".join(str(index) for index in missing_boxes[:10])
        )
    prepared = benchmark._evaluate_model(
        graph,
        inputs,
        boxes,
        anchors,
        strides,
        evaluation_args,
        "Evaluating exact QAT",
    )
    prepared_heads = _capture_raw_heads(
        graph,
        inputs,
        controller,
    )
    stage_progress = {
        "evaluation_start_frame": evaluation_args.evaluation_start_frame,
        "evaluation_frames": args.evaluation_frames,
        "prepared": prepared["metrics"],
    }
    _write_stage_progress(args.output_dir, stage_progress)

    convert_pt2e(
        graph,
        calibration_steps=(
            None if args.refresh_steps == 0 else args.refresh_steps
        ),
        hardware_config=hardware,
    )
    converted = benchmark._evaluate_model(
        graph,
        inputs,
        boxes,
        anchors,
        strides,
        evaluation_args,
        "Evaluating frozen PT2E",
    )
    converted_heads = _capture_raw_heads(graph, inputs)
    prepared_converted = _compare_raw_heads(
        prepared_heads,
        converted_heads,
    )
    stage_progress["converted"] = converted["metrics"]
    stage_progress["prepared_to_converted"] = prepared_converted
    _write_stage_progress(args.output_dir, stage_progress)
    _require_raw_head_equivalence(
        prepared_converted,
        "prepared-to-converted",
        args.max_stage_head_error,
    )
    workload_target.transform(
        graph,
        example_args,
        patterns=VECTOR_PIPELINE,
        hardware_config=hardware,
    )
    transformed = benchmark._evaluate_model(
        graph,
        inputs,
        boxes,
        anchors,
        strides,
        evaluation_args,
        "Evaluating transformed PT2E",
    )
    transformed_heads = _capture_raw_heads(graph, inputs)
    converted_transformed = _compare_raw_head_statistics(
        converted_heads,
        transformed_heads,
        "converted-to-transformed",
    )
    stage_progress["transformed"] = transformed["metrics"]
    stage_progress["converted_to_transformed"] = converted_transformed
    _write_stage_progress(args.output_dir, stage_progress)

    compiled = None
    acceptance = None
    collateral = None
    transformed_compiled = None
    if not args.skip_compile:
        model_output_dir = args.output_dir / "model"
        if model_output_dir.exists() and any(model_output_dir.rglob("*")):
            raise FileExistsError(
                "Refusing to mix compiled collateral with a nonempty "
                f"directory: {model_output_dir}"
            )
        voyager_compile(
            graph,
            example_args,
            output_dir=str(model_output_dir),
            dump_tensors=True,
            hardware_config=hardware,
            scheduler_backend=args.local_scheduler,
            scratchpad_mapping=END_TO_END_SCRATCHPAD_MAPPING,
        )
        collateral = _write_collateral_manifest(model_output_dir)
        compiled = benchmark._evaluate_model(
            graph,
            inputs,
            boxes,
            anchors,
            strides,
            evaluation_args,
            "Evaluating compiled PT2E",
        )
        compiled_heads = _capture_raw_heads(graph, inputs)
        transformed_compiled = _compare_raw_heads(
            transformed_heads,
            compiled_heads,
        )
        stage_progress["compiled"] = compiled["metrics"]
        stage_progress["transformed_to_compiled"] = transformed_compiled
        _write_stage_progress(args.output_dir, stage_progress)
        _require_raw_head_equivalence(
            transformed_compiled,
            "transformed-to-compiled",
            args.max_stage_head_error,
        )
        agreement = benchmark._prediction_agreement(
            prepared,
            compiled,
            args.accuracy_iou_threshold,
            "calibrated",
            "compiled",
        )
        acceptance = benchmark._acceptance(
            evaluation_args,
            compiled["metrics"],
            agreement,
        )
    else:
        agreement = {}

    report = {
        "benchmark_kind": "AoV recorded-baseline agreement",
        "warning": (
            "The YAML boxes are recorded AoV outputs, not independent "
            "WIDER FACE annotations; deployment metrics are not WIDER AP."
        ),
        "profile": {
            "name": "folded-live-qat-192",
            "training_checkpoint": str(args.qat_checkpoint),
            "folded_checkpoint": str(folded_checkpoint),
            "released_weight_sidecars": False,
            "uses_ema": qualification["uses_ema_weights"],
            "parameter_source": qualification["parameter_source"],
            "refresh_steps": args.refresh_steps,
            "evaluation_start_frame": (
                args.refresh_steps
                if args.evaluation_start_frame is None
                else args.evaluation_start_frame
            ),
            "activation_statistics": (
                f"qat-checkpoint-{source_provenance.get('image_size')}px"
                if args.refresh_steps == 0
                else "fresh-192-video-prefix"
            ),
            "exact_boundaries": configured,
            "qat_checkpoint_sha256": workload_target.checkpoint_sha256(
                args.qat_checkpoint
            ),
            "folded_checkpoint_sha256": workload_target.checkpoint_sha256(
                folded_checkpoint
            ),
            "checkpoint_qualification": qualification,
        },
        "video_frames": total_frames,
        "w4_source_audit": w4_audit,
        "prepared": prepared["metrics"],
        "converted": converted["metrics"],
        "transformed": transformed["metrics"],
        "compiled": None if compiled is None else compiled["metrics"],
        "raw_head_equivalence": {
            "strict_maximum_allowed_error": args.max_stage_head_error,
            "prepared_to_converted": prepared_converted,
            "converted_to_transformed": converted_transformed,
            "transformed_to_compiled": transformed_compiled,
        },
        "compiled_collateral": collateral,
        "agreement": agreement,
        "acceptance": acceptance,
    }
    with (args.output_dir / "deployment_report.json").open("w") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))

    if (
        acceptance is not None
        and not acceptance["passed"]
        and not args.allow_accuracy_failure
    ):
        raise SystemExit("AoV QAT deployment accuracy gate failed")


if __name__ == "__main__":
    main()
