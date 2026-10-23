#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Train folded AoV YOLO through dynamic and exact Magnet Alpha QAT."""

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPOSITORY_ROOT / "test"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from test.utils.models import (
    yolo5_face_nv_magnet_alpha as workload_target,
)
from test.utils.models.utils import get_hardware_config
from voyager_compiler import fetch_attr, prepare_qat_pt2e

PT2EParameterEMA = workload_target.PT2EParameterEMA


YOLO_ROOT = Path(
    os.environ.get(
        workload_target.YOLO_ROOT_ENVIRONMENT_VARIABLE,
        workload_target.DEFAULT_YOLO_ROOT,
    )
)
DEFAULT_DATA = YOLO_ROOT / "data" / "widerface_grayscale.yaml"
DEFAULT_HYP = YOLO_ROOT / "data" / "hyp.800.finetune.yaml"
DEFAULT_OUTPUT = Path("runs/yolo5_face_nv_magnet_alpha/qat")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--hyp", type=Path, default=DEFAULT_HYP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=workload_target.AOV_QAT_IMAGE_SIZE,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--static-batch-export",
        action="store_true",
        help=(
            "Export exactly --batch-size examples. This requires every "
            "consumed train/validation batch to have that size."
        ),
    )
    parser.add_argument("--dynamic-epochs", type=int, default=20)
    parser.add_argument("--exact-epochs", type=int, default=20)
    parser.add_argument(
        "--direct-exact-192",
        action="store_true",
        help=(
            "Prime and train target-exact Magnet Alpha QAT directly at 192px; "
            "requires --dynamic-epochs 0"
        ),
    )
    parser.add_argument("--prime-batches", type=int, default=2)
    parser.add_argument(
        "--observer-prime-frames",
        type=int,
        default=0,
        help=(
            "Prime observers from the first N AoV video frames using "
            "deployment preprocessing instead of augmented train batches"
        ),
    )
    parser.add_argument(
        "--observer-video",
        type=Path,
        default=Path(workload_target.AOV_CALIBRATION_VIDEO),
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=Path(workload_target.AOV_CAMERA_CALIBRATION),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adam"),
        default="sgd",
    )
    parser.add_argument("--nominal-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=None,
        help=(
            "Override the YOLO warmup length; use 0 for a short "
            "calibration-sequence fine-tune"
        ),
    )
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def _compiler_args(args):
    checkpoint = None if args.checkpoint is None else str(args.checkpoint)
    return SimpleNamespace(
        activation=None,
        aov_calibration_profile=(
            workload_target.AOV_FRESH_192_CALIBRATION_PROFILE
        ),
        aov_image_size=args.image_size,
        bf16=False,
        bias=None,
        calibration_steps=0,
        force_scale_power_of_two=False,
        model_name_or_path=checkpoint,
        output_activation=None,
        pt2e=True,
        target=workload_target.TARGET,
        weight=None,
    )


def _load_yaml(path: Path):
    with path.open() as source:
        return yaml.safe_load(source)


def _mapping_sha256(values) -> str:
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_dataset(data):
    missing = [
        str(path)
        for path in (Path(data["train"]), Path(data["val"]))
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "AoV QAT dataset paths do not exist: " + ", ".join(missing)
        )


def _load_yolo_training_api():
    datasets = importlib.import_module("utils.face_datasets")
    loss = importlib.import_module("utils.loss")
    metrics = importlib.import_module("utils.metrics")
    validation = importlib.import_module("test")
    return (
        datasets.create_dataloader,
        loss.compute_loss,
        metrics.fitness,
        validation.test,
    )


def _raw_heads(outputs):
    if isinstance(outputs, torch.Tensor):
        return [outputs] if outputs.ndim == 5 else []
    heads = []
    for output in outputs:
        heads.extend(_raw_heads(output))
    return heads


class RawHeadInferenceAdapter(torch.nn.Module):
    """Decode a training-exported PT2E graph for native YOLO validation."""

    def __init__(self, graph, controller, loss_model, names):
        super().__init__()
        self.graph = graph
        self.controller = controller
        object.__setattr__(self, "_loss_model", loss_model)
        detect = loss_model.model[-1]
        self.register_buffer(
            "anchors",
            detect.anchors.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "stride",
            detect.stride.detach().clone(),
            persistent=False,
        )
        self.names = names
        self.hyp = loss_model.hyp
        self.nc = loss_model.nc
        self.gr = loss_model.gr

    @property
    def model(self):
        return object.__getattribute__(self, "_loss_model").model

    def train(self, mode=True):
        self.training = mode
        return self

    def forward(self, inputs, augment=False):
        if augment:
            raise ValueError("AoV QAT validation does not support augmentation")
        self.controller.before_forward()
        outputs = self.graph(inputs)
        heads = _raw_heads(outputs)
        if len(heads) != len(self.stride):
            raise ValueError("Prepared AoV graph returned unexpected heads")

        decoded = []
        for head, anchors, stride in zip(
            heads,
            self.anchors,
            self.stride,
        ):
            batch, anchor_count, height, width, channels = head.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=head.device),
                torch.arange(width, device=head.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).to(head.dtype)
            grid = grid.view(1, 1, height, width, 2)
            anchor_grid = (anchors.to(head.dtype) * stride.to(head.dtype)).view(
                1, anchor_count, 1, 1, 2
            )

            values = torch.cat(
                (
                    head[..., :5].sigmoid(),
                    head[..., 5:15],
                    head[..., 15:].sigmoid(),
                ),
                dim=-1,
            )
            box_xy = (values[..., :2] * 2.0 - 0.5 + grid) * stride
            box_wh = values[..., 2:4].mul(2.0).square() * anchor_grid
            landmarks = [
                values[..., start : start + 2] * anchor_grid + grid * stride
                for start in range(5, 15, 2)
            ]
            values = torch.cat(
                (
                    box_xy,
                    box_wh,
                    values[..., 4:5],
                    *landmarks,
                    values[..., 15:],
                ),
                dim=-1,
            )
            decoded.append(values.view(batch, -1, channels))
        return torch.cat(decoded, dim=1), heads


class _LimitedLoader:
    def __init__(self, loader, maximum):
        self.loader = loader
        self.maximum = min(len(loader), maximum)

    def __len__(self):
        return self.maximum

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.maximum:
                break
            yield batch


class _RequiredBatchLoader:
    def __init__(self, loader, batch_size, label):
        self.loader = loader
        self.batch_size = batch_size
        self.label = label

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            actual = int(batch[0].shape[0])
            if actual != self.batch_size:
                raise ValueError(
                    f"{self.label} produced batch {actual}; static PT2E "
                    f"export requires {self.batch_size}"
                )
            yield batch


def _make_loaders(args, data, hyp, create_dataloader):
    options = SimpleNamespace(single_cls=True)
    train_loader, _ = create_dataloader(
        data["train"],
        args.image_size,
        args.batch_size,
        32,
        options,
        hyp=hyp,
        augment=True,
        cache=False,
        rect=False,
        workers=args.workers,
        grayscale=True,
        prefix="QAT train: ",
    )
    val_loader, _ = create_dataloader(
        data["val"],
        args.image_size,
        args.batch_size,
        32,
        options,
        hyp=hyp,
        augment=False,
        cache=False,
        pad=0.5,
        rect=False,
        workers=args.workers,
        grayscale=True,
        prefix="QAT val: ",
    )
    if args.max_train_batches:
        train_loader = _LimitedLoader(
            train_loader,
            args.max_train_batches,
        )
    if args.max_val_batches:
        val_loader = _LimitedLoader(val_loader, args.max_val_batches)
    if args.static_batch_export:
        train_loader = _RequiredBatchLoader(
            train_loader,
            args.batch_size,
            "QAT training",
        )
        val_loader = _RequiredBatchLoader(
            val_loader,
            args.batch_size,
            "QAT validation",
        )
    return train_loader, val_loader


def _qat_export_inputs(example_args, batch_size, *, dynamic_batch):
    """Make the exported training graph accept every loader batch size."""
    if batch_size == 1:
        return example_args, None
    if len(example_args) != 1 or not isinstance(
        example_args[0],
        torch.Tensor,
    ):
        raise ValueError(
            "AoV QAT expects one tensor positional input for PT2E export"
        )

    example = example_args[0]
    if example.shape[0] != 1:
        raise ValueError(
            "AoV QAT expects a single-image model example before export"
        )
    trace_batch = 2 if dynamic_batch else batch_size
    trace_example = example.expand(
        trace_batch,
        *example.shape[1:],
    ).contiguous()
    if not dynamic_batch:
        return (trace_example,), None

    batch = torch.export.Dim(
        "batch",
        min=1,
        max=batch_size,
    )
    return (trace_example,), ({0: batch},)


def _prepare(args, data, hyp, device):
    compiler_args = _compiler_args(args)
    compiler_args = workload_target._effective_args(compiler_args)
    model, example_args = workload_target.load_model(
        compiler_args,
        torch.float32,
    )
    model.requires_grad_(True)
    model.nc = int(data["nc"])
    model.hyp = hyp
    model.gr = 1.0
    model.names = data["names"]
    loss_model = model

    hardware = get_hardware_config(compiler_args)
    quantizer = workload_target.configure_quantizer(
        None,
        compiler_args,
    )
    export_args, dynamic_shapes = _qat_export_inputs(
        example_args,
        args.batch_size,
        dynamic_batch=not args.static_batch_export,
    )
    graph = prepare_qat_pt2e(
        model,
        quantizer,
        export_args,
        dynamic_shapes=dynamic_shapes,
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
    dynamic_boundaries = controller.enter_dynamic_fake()
    graph.meta["aov_qat_profile"] = {
        "profile": "folded-live-qat",
        "weight_source": compiler_args.model_name_or_path,
        "released_weight_sidecars": False,
        "image_size": int(example_args[0].shape[-1]),
        "dynamic_pe_boundaries": dynamic_boundaries,
        "expected_exact_boundaries": (
            workload_target.AOV_RELEASED_COLLATERAL_EXACT_BOUNDARY_COUNT
        ),
    }
    graph.to(device)
    loss_model.to(device)
    return graph, controller, loss_model, compiler_args


def _weight_parameter_ids(graph, controller):
    identifiers = set()
    for node in graph.graph.nodes:
        if (
            node.op != "call_module"
            or str(node.target) not in controller.roles.weights
            or not node.args
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node) or source.op != "get_attr":
            raise ValueError(
                f"{node.name}: QAT weight source is not a parameter"
            )
        parameter = fetch_attr(graph, source.target)
        if not isinstance(parameter, torch.nn.Parameter):
            raise ValueError(f"{source.target}: QAT weight is not a parameter")
        identifiers.add(id(parameter))
    return identifiers


def _optimizer(args, hyp, graph, controller):
    learning_rate = (
        hyp["lr0"] if args.learning_rate is None else args.learning_rate
    )
    base_weight_decay = (
        hyp["weight_decay"] if args.weight_decay is None else args.weight_decay
    )
    accumulation = (
        max(round(args.nominal_batch_size / args.batch_size), 1)
        if args.gradient_accumulation is None
        else args.gradient_accumulation
    )
    weight_decay = (
        base_weight_decay
        * args.batch_size
        * accumulation
        / args.nominal_batch_size
    )

    weight_ids = _weight_parameter_ids(graph, controller)
    weights = []
    biases = []
    for parameter in graph.parameters():
        (weights if id(parameter) in weight_ids else biases).append(parameter)
    if not weights or not biases:
        raise ValueError(
            "AoV optimizer requires both weight and bias parameter groups"
        )

    parameter_groups = [
        {
            "params": biases,
            "weight_decay": 0.0,
            "is_bias": True,
        },
        {
            "params": weights,
            "weight_decay": weight_decay,
            "is_bias": False,
        },
    ]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=learning_rate,
            betas=(hyp["momentum"], 0.999),
        )
    else:
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=hyp["momentum"],
            nesterov=True,
        )
    return optimizer, {
        "name": args.optimizer,
        "learning_rate": learning_rate,
        "base_weight_decay": base_weight_decay,
        "effective_weight_decay": weight_decay,
        "nominal_batch_size": args.nominal_batch_size,
        "gradient_accumulation": accumulation,
        "bias_parameter_count": len(biases),
        "weight_parameter_count": len(weights),
    }


def _cosine_learning_rate_factor(epoch, total_epochs, final_factor):
    cosine = np.cos(epoch * np.pi / total_epochs)
    return ((1.0 + cosine) / 2.0) * (1.0 - final_factor) + final_factor


def _warmup_optimizer(
    optimizer,
    hyp,
    recipe,
    *,
    integrated_batch,
    warmup_batches,
    epoch,
    total_epochs,
):
    accumulation = recipe["gradient_accumulation"]
    if warmup_batches <= 0 or integrated_batch > warmup_batches:
        return accumulation

    progress = integrated_batch / warmup_batches
    accumulation = max(
        1,
        round(1.0 + progress * (recipe["gradient_accumulation"] - 1.0)),
    )
    target_factor = _cosine_learning_rate_factor(
        epoch,
        total_epochs,
        hyp["lrf"],
    )
    for group in optimizer.param_groups:
        start = hyp["warmup_bias_lr"] if group["is_bias"] else 0.0
        target = group["initial_lr"] * target_factor
        group["lr"] = start + progress * (target - start)
        if "momentum" in group:
            group["momentum"] = hyp["warmup_momentum"] + progress * (
                hyp["momentum"] - hyp["warmup_momentum"]
            )
    return accumulation


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _read_checkpoint(path):
    state = workload_target.read_checkpoint(path)
    if (
        workload_target.checkpoint_format(state)
        != workload_target.RESUME_CHECKPOINT_FORMAT
    ):
        raise ValueError("--resume requires a training-resume checkpoint")
    return state


def _apply_resume_configuration(args, state):
    provenance = state.get("provenance", {})
    if provenance.get("released_weight_sidecars") is not False:
        raise ValueError(
            "Resume requires a sidecar-free folded-live QAT checkpoint"
        )
    folded = provenance.get("folded_checkpoint")
    if args.checkpoint is None:
        if folded is None:
            raise ValueError(
                "Resume checkpoint has no folded source; pass --checkpoint"
            )
        args.checkpoint = Path(folded)

    expected = {
        "image_size": args.image_size,
        "precision_policy": workload_target.AOV_PRECISION_POLICY,
    }
    mismatches = [
        f"{name}={provenance[name]!r}, requested={value!r}"
        for name, value in expected.items()
        if name in provenance and provenance[name] != value
    ]
    for name, requested in (("data", args.data), ("hyp", args.hyp)):
        saved = provenance.get(name)
        if saved is None:
            continue
        if Path(saved).resolve() != requested.resolve():
            mismatches.append(f"{name}={saved!r}, requested={str(requested)!r}")
    if mismatches:
        raise ValueError(
            "Resume configuration differs from checkpoint provenance: "
            + "; ".join(mismatches)
        )


def _guard_output_directory(args):
    existing = (
        list(args.output_dir.rglob("*")) if args.output_dir.exists() else []
    )
    if args.resume is None and existing:
        raise FileExistsError(
            "Refusing to mix a fresh QAT run with existing outputs: "
            + ", ".join(
                str(path.relative_to(args.output_dir)) for path in existing[:10]
            )
        )
    if (
        args.resume is not None
        and args.resume.parent.resolve() != args.output_dir.resolve()
    ):
        raise ValueError(
            "Resume must use the checkpoint directory as --output-dir so "
            "the historical best snapshot remains in scope"
        )


def _save_checkpoint(
    path,
    controller,
    optimizer,
    scheduler,
    scaler,
    ema,
    *,
    phase_epoch,
    best_fitness,
    provenance,
    training_progress=None,
    checkpoint_role="unspecified",
    selection_fitness=None,
):
    workload_target.save_resume_checkpoint(
        path,
        controller,
        optimizer,
        scheduler,
        scaler,
        ema,
        rng_state=_rng_state(),
        phase_epoch=phase_epoch,
        best_fitness=best_fitness,
        provenance=provenance,
        training_progress=training_progress,
        checkpoint_role=checkpoint_role,
        selection_fitness=selection_fitness,
    )


def _load_checkpoint(
    path,
    controller,
    optimizer,
    scheduler,
    scaler,
    ema,
    graph,
    *,
    state=None,
):
    state = _read_checkpoint(path) if state is None else state
    workload_target.restore_resume_checkpoint(
        state,
        controller,
        optimizer,
        scheduler,
        scaler,
        ema,
        graph,
    )
    _restore_rng(state["rng"])
    return state


def _validate_resume_provenance(state, provenance):
    saved = state.get("provenance", {})
    required = (
        "amp",
        "batch_size",
        "data_sha256",
        "dynamic_epochs",
        "effective_hyp_sha256",
        "exact_epochs",
        "folded_checkpoint_sha256",
        "gradient_accumulation",
        "image_size",
        "max_train_batches",
        "max_val_batches",
        "nominal_batch_size",
        "observer_prime_frames",
        "observer_camera_calibration_sha256",
        "observer_source_sha256",
        "optimizer",
        "optimizer_recipe",
        "precision_policy",
        "qat_recipe",
        "released_weight_sidecars",
        "static_batch_export",
        "train_batches_per_epoch",
        "val_every",
        "warmup_batches",
        "workers",
    )
    missing = [name for name in required if name not in saved]
    if missing:
        raise ValueError(
            "Resume checkpoint lacks provenance fields: " + ", ".join(missing)
        )
    mismatches = [
        f"{name}={saved[name]!r}, current={provenance[name]!r}"
        for name in required
        if saved[name] != provenance[name]
    ]
    if mismatches:
        raise ValueError(
            "Resume source differs from checkpoint provenance: "
            + "; ".join(mismatches)
        )


def _validate_training_progress(state, provenance):
    progress = state.get("training_progress")
    if not isinstance(progress, dict):
        raise ValueError("Resume checkpoint has no training progress")
    required = (
        "completed_epochs",
        "integrated_batches",
        "optimizer_steps",
    )
    missing = [name for name in required if name not in progress]
    if missing:
        raise ValueError(
            "Resume checkpoint lacks progress fields: " + ", ".join(missing)
        )
    normalized = {}
    for name in required:
        value = progress[name]
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Resume progress {name} must be a non-negative integer"
            )
        normalized[name] = value

    phase = state["qat_phase"]["phase"]
    phase_epoch = int(state["phase_epoch"])
    completed_epochs = phase_epoch + 1
    if phase == "target_exact":
        completed_epochs += provenance["dynamic_epochs"]
    if normalized["completed_epochs"] != completed_epochs:
        raise ValueError(
            "Resume completed-epoch count disagrees with phase state"
        )
    expected_batches = completed_epochs * provenance["train_batches_per_epoch"]
    if normalized["integrated_batches"] != expected_batches:
        raise ValueError(
            "Resume integrated-batch count disagrees with epoch state"
        )
    if normalized["optimizer_steps"] > normalized["integrated_batches"]:
        raise ValueError("Resume optimizer-step count exceeds batch count")
    return normalized


def _prime_observers(graph, loader, device, batches):
    if batches <= 0:
        raise ValueError("At least one observer-prime batch is required")
    observed = 0
    with torch.no_grad():
        for index, (images, _, _, _) in enumerate(loader):
            if index >= batches:
                break
            graph(images.to(device).float().div_(255.0))
            observed += 1
    if observed < batches:
        raise ValueError(
            f"Needed {batches} observer-prime batches, got {observed}"
        )


def _prime_aov_video_observers(
    graph,
    controller,
    device,
    *,
    frame_count,
    batch_size,
    static_batch,
    video,
    camera_calibration,
):
    """Collect 192px activation ranges on the deployment preprocessing path."""
    if frame_count <= 1:
        raise ValueError("AoV observer priming requires at least two frames")
    if static_batch and frame_count % batch_size:
        raise ValueError(
            "Static PT2E export requires observer-prime frames divisible "
            "by the batch size"
        )

    controller.reset_activation_statistics()
    camera_matrix, distortion = workload_target._load_camera_calibration(
        str(camera_calibration)
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open observer video: {video}")

    pending = []
    observed = 0
    try:
        with torch.no_grad():
            for frame_index in range(frame_count):
                decoded, frame = capture.read()
                if not decoded:
                    raise ValueError(
                        f"Observer video ended at frame {frame_index}"
                    )
                inputs = workload_target._preprocess_calibration_input(
                    frame,
                    1,
                    torch.float32,
                    camera_matrix,
                    distortion,
                )
                pending.append(inputs)
                if len(pending) == batch_size:
                    graph(torch.cat(pending, dim=0).to(device))
                    observed += len(pending)
                    pending.clear()
            if pending:
                graph(torch.cat(pending, dim=0).to(device))
                observed += len(pending)
    finally:
        capture.release()

    if observed != frame_count:
        raise RuntimeError(
            f"Expected {frame_count} observer frames, processed {observed}"
        )


def _verify_gradients(graph):
    missing = []
    nonfinite = []
    zero = []
    for name, parameter in graph.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
        elif not torch.count_nonzero(parameter.grad):
            zero.append(name)
    if missing or nonfinite:
        raise RuntimeError(
            "QAT gradient gate failed: "
            f"missing={missing}, nonfinite={nonfinite}"
        )
    return zero


def _train_epoch(
    graph,
    controller,
    loss_model,
    loader,
    optimizer,
    scaler,
    ema,
    compute_loss,
    device,
    hyp,
    recipe,
    progress,
    warmup_batches,
    total_epochs,
):
    losses = []
    checked_gradients = False
    first_batch_zero_gradients = []
    optimizer.zero_grad(set_to_none=True)
    for images, targets, _, _ in loader:
        integrated_batch = progress["integrated_batches"]
        accumulation = _warmup_optimizer(
            optimizer,
            hyp,
            recipe,
            integrated_batch=integrated_batch,
            warmup_batches=warmup_batches,
            epoch=progress["completed_epochs"],
            total_epochs=total_epochs,
        )
        images = images.to(device).float().div_(255.0)
        targets = targets.to(device)
        controller.before_forward()
        with torch.autocast(
            device_type=device.type,
            enabled=scaler.is_enabled(),
        ):
            predictions = _raw_heads(graph(images))
            loss, items = compute_loss(predictions, targets, loss_model)
        scaler.scale(loss).backward()
        if not checked_gradients:
            first_batch_zero_gradients = _verify_gradients(graph)
            checked_gradients = True
        if integrated_batch % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(graph)
            progress["optimizer_steps"] += 1
        progress["integrated_batches"] += 1
        losses.append(items.detach().cpu())
    return (
        torch.stack(losses).mean(dim=0).tolist(),
        first_batch_zero_gradients,
    )


def _validate(
    args,
    data,
    graph,
    controller,
    loss_model,
    val_loader,
    ema,
    test,
    fitness,
):
    saved_source = controller.source_state_dict()
    adapter = RawHeadInferenceAdapter(
        graph,
        controller,
        loss_model,
        data["names"],
    )
    with ema.averaged_parameters(graph):
        if controller.phase.value == "target_exact":
            controller.reset_weight_statistics()
        result, _, _ = test(
            str(args.data),
            batch_size=args.batch_size,
            imgsz=args.image_size,
            conf_thres=0.001,
            iou_thres=0.6,
            model=adapter,
            dataloader=val_loader,
            save_dir=args.output_dir / "validation",
            plots=False,
        )
    controller.load_source_state_dict(saved_source)
    controller.before_forward()
    validation = np.asarray(result[:4], dtype=np.float64)
    if not np.isfinite(validation).all():
        raise RuntimeError(
            f"YOLO validation returned non-finite metrics: {validation}"
        )
    score = float(fitness(validation[None])[0])
    return {
        "precision": float(result[0]),
        "recall": float(result[1]),
        "map50": float(result[2]),
        "map50_95": float(result[3]),
        "fitness": score,
    }


def main():
    args = parse_args()
    if args.exact_epochs <= 0:
        raise ValueError("Exact epochs must be positive")
    if args.direct_exact_192:
        if args.dynamic_epochs != 0:
            raise ValueError(
                "Direct 192px exact QAT requires --dynamic-epochs 0"
            )
        if args.image_size != workload_target.AOV_CALIBRATION_IMAGE_SIZE:
            raise ValueError(
                "Direct exact QAT requires the 192px deployment resolution"
            )
        if args.observer_prime_frames <= 1:
            raise ValueError(
                "Direct exact QAT requires deployment-path observer frames"
            )
    elif args.dynamic_epochs <= 0:
        raise ValueError("Dynamic epochs must be positive")
    elif args.image_size != workload_target.AOV_QAT_IMAGE_SIZE:
        raise ValueError(
            "The folded-live AoV QAT lifecycle requires 800px training"
        )
    if args.val_every <= 0:
        raise ValueError("Validation interval must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("CPU thread count must be positive")
    if args.batch_size <= 0 or args.nominal_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if (
        args.gradient_accumulation is not None
        and args.gradient_accumulation <= 0
    ):
        raise ValueError("Gradient accumulation must be positive")
    if args.warmup_batches is not None and args.warmup_batches < 0:
        raise ValueError("Warmup batches must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)

    resume_state = None
    if args.resume is not None:
        resume_state = _read_checkpoint(args.resume)
        _apply_resume_configuration(args, resume_state)
    _guard_output_directory(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = _load_yaml(args.data)
    hyp = _load_yaml(args.hyp)
    _require_dataset(data)
    hyp["cls"] *= int(data["nc"]) / 80.0

    graph, controller, loss_model, compiler_args = _prepare(
        args,
        data,
        hyp,
        device,
    )
    (
        create_dataloader,
        compute_loss,
        fitness,
        test,
    ) = _load_yolo_training_api()
    if resume_state is not None:
        # Seed the persistent loader workers from the checkpoint before their
        # iterators are constructed. The checkpoint load below restores the
        # main-process generators again after loader construction.
        _restore_rng(resume_state["rng"])
    train_loader, val_loader = _make_loaders(
        args,
        data,
        hyp,
        create_dataloader,
    )
    optimizer, optimizer_recipe = _optimizer(
        args,
        hyp,
        graph,
        controller,
    )
    total_epochs = args.dynamic_epochs + args.exact_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _cosine_learning_rate_factor(
            epoch,
            total_epochs,
            hyp["lrf"],
        ),
    )
    scheduler.last_epoch = -1
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=args.amp,
    )
    ema = PT2EParameterEMA(graph)

    checkpoint_path = Path(compiler_args.model_name_or_path)
    provenance = {
        "folded_checkpoint": str(checkpoint_path),
        "folded_checkpoint_sha256": workload_target.checkpoint_sha256(
            checkpoint_path
        ),
        "data": str(args.data),
        "hyp": str(args.hyp),
        "image_size": args.image_size,
        "precision_policy": workload_target.AOV_PRECISION_POLICY,
        "released_weight_sidecars": False,
        "qat_recipe": (
            "direct-192-target-exact"
            if args.direct_exact_192
            else "800-dynamic-then-target-exact"
        ),
        "preprocessing_version": 1,
        "first_layer_input_mode": "padding",
        "amp": args.amp,
        "batch_size": args.batch_size,
        "data_sha256": workload_target.checkpoint_sha256(args.data),
        "dynamic_epochs": args.dynamic_epochs,
        "effective_hyp_sha256": _mapping_sha256(hyp),
        "exact_epochs": args.exact_epochs,
        "gradient_accumulation": optimizer_recipe["gradient_accumulation"],
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "nominal_batch_size": args.nominal_batch_size,
        "observer_prime_frames": args.observer_prime_frames,
        "observer_camera_calibration": (
            str(args.camera_calibration) if args.observer_prime_frames else None
        ),
        "observer_camera_calibration_sha256": (
            workload_target.checkpoint_sha256(args.camera_calibration)
            if args.observer_prime_frames
            else None
        ),
        "observer_source": (
            str(args.observer_video) if args.observer_prime_frames else None
        ),
        "observer_source_sha256": (
            workload_target.checkpoint_sha256(args.observer_video)
            if args.observer_prime_frames
            else None
        ),
        "optimizer": optimizer_recipe["name"],
        "optimizer_recipe": optimizer_recipe,
        "static_batch_export": args.static_batch_export,
        "train_batches_per_epoch": len(train_loader),
        "val_every": args.val_every,
        "warmup_batches": args.warmup_batches,
        "workers": args.workers,
    }
    start_phase = "dynamic_fake"
    start_epoch = 0
    best_fitness = float("-inf")
    progress = {
        "completed_epochs": 0,
        "integrated_batches": 0,
        "optimizer_steps": 0,
    }
    if args.resume is not None:
        _validate_resume_provenance(resume_state, provenance)
        progress = _validate_training_progress(
            resume_state,
            provenance,
        )
        state = _load_checkpoint(
            args.resume,
            controller,
            optimizer,
            scheduler,
            scaler,
            ema,
            graph,
            state=resume_state,
        )
        start_phase = state["qat_phase"]["phase"]
        start_epoch = int(state["phase_epoch"]) + 1
        best_fitness = float(state["best_fitness"])
        if ema.updates != progress["optimizer_steps"]:
            raise ValueError(
                "EMA update count disagrees with optimizer progress"
            )
        if scheduler.last_epoch != progress["completed_epochs"] - 1:
            raise ValueError("Scheduler epoch disagrees with training progress")
    elif args.observer_prime_frames:
        _prime_aov_video_observers(
            graph,
            controller,
            device,
            frame_count=args.observer_prime_frames,
            batch_size=args.batch_size,
            static_batch=args.static_batch_export,
            video=args.observer_video,
            camera_calibration=args.camera_calibration,
        )
    else:
        _prime_observers(
            graph,
            train_loader,
            device,
            args.prime_batches,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    phases = (
        ("dynamic_fake", args.dynamic_epochs),
        ("target_exact", args.exact_epochs),
    )
    warmup_batches = (
        max(
            round(hyp["warmup_epochs"] * len(train_loader)),
            1000,
        )
        if args.warmup_batches is None
        else args.warmup_batches
    )
    active = start_phase == "dynamic_fake"
    for phase, epochs in phases:
        if not active:
            active = phase == start_phase
            if not active:
                continue
        phase_start = start_epoch if phase == start_phase else 0
        if phase == "target_exact" and (
            controller.phase.value != "target_exact"
        ):
            if args.dynamic_epochs:
                _save_checkpoint(
                    args.output_dir / "dynamic-final.pt",
                    controller,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    phase_epoch=args.dynamic_epochs - 1,
                    best_fitness=best_fitness,
                    provenance=provenance,
                    training_progress=progress,
                    checkpoint_role="dynamic_final",
                )
            controller.enter_target_exact()
            best_fitness = float("-inf")

        for phase_epoch in range(phase_start, epochs):
            loss, first_batch_zero_gradients = _train_epoch(
                graph,
                controller,
                loss_model,
                train_loader,
                optimizer,
                scaler,
                ema,
                compute_loss,
                device,
                hyp,
                optimizer_recipe,
                progress,
                warmup_batches,
                total_epochs,
            )
            scheduler.step()
            progress["completed_epochs"] += 1
            metrics = None
            if (
                phase_epoch + 1
            ) % args.val_every == 0 or phase_epoch + 1 == epochs:
                metrics = _validate(
                    args,
                    data,
                    graph,
                    controller,
                    loss_model,
                    val_loader,
                    ema,
                    test,
                    fitness,
                )
            record = {
                "phase": phase,
                "phase_epoch": phase_epoch,
                "loss": loss,
                "first_batch_zero_gradients": (first_batch_zero_gradients),
                "validation": metrics,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True))

            latest = args.output_dir / "latest.pt"
            score = float("-inf") if metrics is None else metrics["fitness"]
            _save_checkpoint(
                latest,
                controller,
                optimizer,
                scheduler,
                scaler,
                ema,
                phase_epoch=phase_epoch,
                best_fitness=max(best_fitness, score),
                provenance=provenance,
                training_progress=progress,
                checkpoint_role="latest",
                selection_fitness=(None if metrics is None else score),
            )
            if phase == "target_exact" and score > best_fitness:
                best_fitness = score
                best_resume_path = args.output_dir / "best.pt"
                _save_checkpoint(
                    best_resume_path,
                    controller,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    phase_epoch=phase_epoch,
                    best_fitness=best_fitness,
                    provenance=provenance,
                    training_progress=progress,
                    checkpoint_role="best",
                    selection_fitness=score,
                )
                workload_target.save_deployment_checkpoint(
                    args.output_dir / "best-deploy.pt",
                    graph,
                    controller,
                    provenance=provenance,
                    checkpoint_role="best",
                    selection_fitness=score,
                    best_fitness=best_fitness,
                    parameter_source="ema",
                    ema=ema,
                    include_observer_state=True,
                )
        start_epoch = 0

    if not np.isfinite(best_fitness):
        raise RuntimeError(
            "Exact QAT finished without a finite YOLO validation fitness"
        )
    with (args.output_dir / "training_report.json").open("w") as output:
        json.dump(
            {
                "provenance": provenance,
                "training_progress": progress,
                "best_fitness": best_fitness,
                "best_resume_checkpoint": str(args.output_dir / "best.pt"),
                "best_deployment_checkpoint": str(
                    args.output_dir / "best-deploy.pt"
                ),
                "history": history,
            },
            output,
            indent=2,
        )


if __name__ == "__main__":
    main()
