# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Adapter for the exact resnet18 x Magnet Alpha pairing."""

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import __version__ as torchvision_version
from torchvision import models

from test.verify_help import (
    assert_output_close,
    compare_output_statistics,
)
from voyager_compiler import (
    QuantizationConfig,
    QuantizationSpec,
    compile,
    convert_pt2e,
    export_model,
    finalize_pt2e_calibration,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from voyager_compiler.codegen import (
    MAGNET_ALPHA_COMPILATION_TARGET,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    get_conv_bn_layers,
    serialize_hardware_config,
)
from voyager_compiler.codegen.target_backends import (
    prepare_target_quantization_recalibration,
)
from voyager_compiler.codegen.target_backends.magnet_alpha import (
    AOV_COMPAT_HARDWARE_CONFIG,
)
from voyager_compiler.pt2e_utils import fetch_attr

from .utils import get_compile_args, get_transform_args

WORKLOAD = "resnet18"
TARGET = MAGNET_ALPHA_COMPILATION_TARGET
PROFILE_ID = "resnet18-magnet-alpha-candidate-matrix-v2"
MODEL_SEED = 0
INPUT_SHAPE = (1, 3, 224, 224)
MINIMUM_CALIBRATION_STEPS = 2
A8_QSPEC = "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"
W8_QSPEC = "int8,qs=per_channel_symmetric,ax=0"
A4_VSQ_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_tensor_symmetric,bs=32,"
    "scale=uint8,sqmax=8,csqmax=126"
)
W4_VSQ_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,cax=0,"
    "bs=32,scale=uint8,qmin=-7,qmax=7"
)
BIAS_QSPEC = "int32"
DENSE_CANDIDATE_ID = "alpha-pe-w8a8-o8"
DEFAULT_CANDIDATE_ID = "alpha-pe-w4a4-o8"
CANDIDATE_QSPECS = {
    DENSE_CANDIDATE_ID: {
        "activation": A8_QSPEC,
        "output_activation": A8_QSPEC,
        "weight": W8_QSPEC,
        "bias": BIAS_QSPEC,
    },
    "alpha-pe-w8a8-o4": {
        "activation": A8_QSPEC,
        "output_activation": A4_VSQ_QSPEC,
        "weight": W8_QSPEC,
        "bias": BIAS_QSPEC,
    },
    "alpha-pe-w4a4-o8": {
        "activation": A4_VSQ_QSPEC,
        "output_activation": A8_QSPEC,
        "weight": W4_VSQ_QSPEC,
        "bias": None,
    },
    "alpha-pe-w4a4-o4": {
        "activation": A4_VSQ_QSPEC,
        "output_activation": A4_VSQ_QSPEC,
        "weight": W4_VSQ_QSPEC,
        "bias": None,
    },
}
POST_TARGET_MINIMUM_COSINE = 0.90
POST_TARGET_MINIMUM_SQNR_DB = 5.0
POST_MAPPING_RTOL = 0.0
POST_MAPPING_ATOL = 0.0
CONTRACT_FILENAME = "resnet18_magnet_alpha_contract.json"
INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META = (
    "magnet_alpha_internal_scalar_int8_qdq_summary"
)
MIXED_W4_POLICY_ID = "resnet18-alpha-mixed-w8-w4a4-o8-c512-rescue-v2"
MIXED_W4_O4_AUDIT_POLICY_ID = "resnet18-alpha-mixed-w8-w4a4-o4-c512-audit-v1"
W4_RESCUE_MODULE_PATTERNS = (
    r"^layer4\.0\.conv2$",
    r"^layer4\.1\.conv1$",
)


@dataclass(frozen=True)
class ResNet18MagnetAlphaRunResult:
    """Typed result returned to the neutral CLI dispatcher."""

    workload: str
    target: str
    profile_id: str
    graph_module: Any
    reference_output: Any
    transformed_output: Any
    preprocess_fn: Any
    imprecise_pt2e: bool


class _ImageNetPreprocess(nn.Module):
    """Apply the frozen torchvision ImageNet normalization contract."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, value):
        return (value - self.mean) / self.std


class _LinearAsConv2d(nn.Module):
    """Preserve one Linear exactly as a graph-visible 1x1 Conv2d."""

    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(
            linear.weight.detach().reshape(
                linear.out_features,
                linear.in_features,
                1,
                1,
            )
        )
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())

    def forward(self, value):
        value = value.unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(value, self.weight, self.bias).flatten(1)


def _tensor_bytes(value):
    value = value.detach().cpu().contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_sha256(value):
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(tuple(value.shape)).encode())
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape)).encode())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_export_relus(model):
    """Make torchvision ReLUs out-of-place without replacing modules."""

    normalized = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU) and module.inplace is True:
            module.inplace = False
            normalized += 1
    return normalized


def _deterministic_raw_sample(sample_id):
    """Construct one positive, download-free 224x224 RGB sample."""

    coordinates = torch.linspace(0.0, 1.0, INPUT_SHAPE[-1])
    horizontal = coordinates.reshape(1, -1).expand(
        INPUT_SHAPE[-2],
        INPUT_SHAPE[-1],
    )
    vertical = coordinates.reshape(-1, 1).expand(
        INPUT_SHAPE[-2],
        INPUT_SHAPE[-1],
    )
    raw = torch.stack(
        (
            horizontal,
            vertical,
            (horizontal + vertical) * 0.5,
        )
    ).unsqueeze(0)
    raw = torch.roll(
        raw,
        shifts=(17 * sample_id, 29 * sample_id),
        dims=(-2, -1),
    )
    scale = 0.70 + 0.05 * (sample_id % 5)
    offset = 0.05 + 0.01 * (sample_id % 3)
    return (raw * scale + offset).clamp(0.0, 1.0)


def _sample_record(identity, raw, preprocessed):
    return {
        "identity": identity,
        "shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "raw_sha256": _tensor_sha256(raw),
        "preprocessed_dtype": str(preprocessed.dtype),
        "preprocessed_sha256": _tensor_sha256(preprocessed),
    }


def _shape_and_dtype(node):
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None, None
    return list(value.shape), str(value.dtype)


def _candidate_assignment(target):
    name = str(target)
    if "conv2d" in name:
        return "accelerator", "direct_conv2d"
    if "relu" in name:
        return "accelerator", "ppu_absorption_candidate"
    if "quantize" in name or "dequantize" in name:
        return "accelerator", "boundary_absorption_candidate"
    if "max_pool2d" in name:
        return "accelerator", "direct_max_pool_candidate"
    if any(
        token in name
        for token in (
            "flatten",
            "squeeze",
            "unsqueeze",
            "view",
        )
    ):
        return "accelerator", "address_only_candidate"
    if "add" in name:
        return "unverified", "residual_eadd_gap"
    if "avg_pool" in name or "adaptive_avg_pool" in name:
        return "unverified", "average_pool_gap"
    return "unverified", "target_translation_unverified"


def _operation_inventory(model):
    occurrences = []
    for node in model.graph.nodes:
        if node.op not in ("call_function", "call_module", "call_method"):
            continue
        shape, dtype = _shape_and_dtype(node)
        owner, form = _candidate_assignment(node.target)
        occurrences.append(
            {
                "occurrence_id": node.name,
                "fx_op": node.op,
                "target": str(node.target),
                "shape": shape,
                "dtype": dtype,
                "candidate_owner": owner,
                "candidate_form": form,
            }
        )
    unique_targets = sorted({item["target"] for item in occurrences})
    owner_counts = {}
    for item in occurrences:
        owner = item["candidate_owner"]
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
    return {
        "occurrence_denominator": len(occurrences),
        "unique_operation_denominator": len(unique_targets),
        "unique_targets": unique_targets,
        "candidate_owner_counts": owner_counts,
        "occurrences": occurrences,
    }


def _summarize_tensor(value):
    value = value.detach().cpu()
    flat = value.reshape(-1)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _tensor_sha256(value),
        "element_count": value.numel(),
    }
    if value.numel():
        numeric = flat.to(torch.float64)
        summary["minimum"] = float(numeric.min())
        summary["maximum"] = float(numeric.max())
    if value.numel() <= 16:
        summary["values"] = flat.tolist()
    return summary


def _qparam_inventory(model):
    records = []
    for node in model.graph.nodes:
        target = str(node.target)
        if node.op != "call_function" or not any(
            token in target for token in ("quantize", "dequantize")
        ):
            continue
        operands = []
        for index, argument in enumerate(node.args[1:], start=1):
            record = {"argument_index": index}
            if isinstance(argument, torch.fx.Node):
                record["node"] = argument.name
                if argument.op == "get_attr":
                    value = fetch_attr(model, argument.target)
                    if isinstance(value, torch.Tensor):
                        record["tensor"] = _summarize_tensor(value)
            elif isinstance(argument, torch.Tensor):
                record["tensor"] = _summarize_tensor(argument)
            else:
                record["value"] = argument
            operands.append(record)
        records.append(
            {
                "node": node.name,
                "target": target,
                "input": (
                    node.args[0].name
                    if isinstance(node.args[0], torch.fx.Node)
                    else repr(node.args[0])
                ),
                "qparam_operands": operands,
            }
        )
    return records


class _ContractRecorder:
    def __init__(self, output_dir, payload):
        self.output_dir = Path(output_dir).resolve()
        # The compiler deliberately rejects a nonempty output directory so
        # stale collateral can never be mixed with a fresh run.  Keep this
        # continuously updated workload contract beside that directory,
        # namespaced by the fresh lane, rather than placing it inside the
        # compiler-owned tree before compilation begins.
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir.with_name(
            f"{self.output_dir.name}-{CONTRACT_FILENAME}"
        )
        self.payload = payload
        self.payload["compiler_output_dir"] = str(self.output_dir)
        self.payload["contract_path"] = str(self.path)
        self.write()

    def update(self, **values):
        self.payload.update(values)
        self.write()

    def stage(self, name):
        self.payload["last_completed_stage"] = name
        self.payload.pop("failure", None)
        self.write()

    def fail(self, stage, error):
        self.payload["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        self.write()

    def write(self):
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _uses_quantization(args):
    return bool(
        args.pt2e
        or any(
            getattr(args, name) is not None
            for name in (
                "activation",
                "output_activation",
                "weight",
                "bias",
            )
        )
    )


def _effective_args(args):
    """Resolve the immutable exact-pair policy on a copied namespace."""

    effective = copy.copy(args)
    if effective.model != WORKLOAD or effective.target != TARGET:
        raise ValueError(
            "Adapter identity does not match the requested exact pair: "
            f"{effective.model!r} x {effective.target!r}"
        )
    if not _uses_quantization(effective):
        raise ValueError(
            "Selecting magnet-alpha does not enable PTQ; pass --pt2e or "
            "explicit qspecs"
        )
    defaults = CANDIDATE_QSPECS[DEFAULT_CANDIDATE_ID]
    core_names = ("activation", "output_activation", "weight")
    requested = {
        name: (
            getattr(effective, name)
            if getattr(effective, name) is not None
            else defaults[name]
        )
        for name in core_names
    }
    matches = [
        candidate_id
        for candidate_id, qspecs in CANDIDATE_QSPECS.items()
        if requested == {name: qspecs[name] for name in core_names}
    ]
    if len(matches) != 1:
        raise ValueError(
            "The exact-pair adapter accepts only the four frozen Alpha "
            f"candidate tuples; received {requested!r}"
        )
    candidate_id = matches[0]
    candidate = CANDIDATE_QSPECS[candidate_id]
    supplied_bias = effective.bias
    expected_bias = candidate["bias"]
    if supplied_bias is not None and supplied_bias != expected_bias:
        raise ValueError(
            f"CLI bias qspec conflicts with {candidate_id!r}: "
            f"{supplied_bias!r} != {expected_bias!r}"
        )
    for name, value in candidate.items():
        setattr(effective, name, value)
    effective.magnet_alpha_candidate_id = candidate_id
    if effective.residual is not None:
        raise ValueError(
            "The exact-pair profile owns residual qparams; "
            "--residual is not accepted"
        )
    if effective.calibration_steps == 1:
        raise ValueError(
            "Accepted calibration requires at least two samples; use zero "
            "only for a labeled structural smoke"
        )
    if effective.calibration_steps < 0:
        raise ValueError("calibration_steps cannot be negative")
    if effective.scratchpad_mapping != PER_OPERATION_SCRATCHPAD_MAPPING:
        raise ValueError(
            "The exact-pair profile requires per_operation scratchpad "
            "mapping and rejects end_to_end"
        )
    if effective.execution_order != "fx":
        raise ValueError("The exact-pair profile requires FX execution order")
    for name in ("bf16", "bufferize", "conv2d_im2col", "use_maxpool_2x2"):
        if getattr(effective, name, False):
            raise ValueError(f"The exact-pair profile rejects --{name}")
    if effective.model_name_or_path not in (None, "NONE", "none"):
        raise ValueError(
            "The frozen profile uses seed-0 weights=None and rejects an "
            "external model_name_or_path"
        )
    return effective


def _get_candidate_quantizer(effective):
    """Build one exact candidate and keep non-GEMM boundaries at A8."""

    candidate_quantizer = get_default_quantizer(
        input_activation=effective.activation,
        output_activation=effective.output_activation,
        weight=effective.weight,
        bias=effective.bias,
        force_scale_power_of_two=effective.force_scale_power_of_two,
    )
    if not effective.magnet_alpha_candidate_id.startswith("alpha-pe-w4a4"):
        return candidate_quantizer

    quantizer = get_default_quantizer(
        input_activation=A8_QSPEC,
        output_activation=A8_QSPEC,
        weight=W8_QSPEC,
        bias=BIAS_QSPEC,
        force_scale_power_of_two=effective.force_scale_power_of_two,
    )
    conv2d_target = torch.ops.aten.conv2d.default
    w8_conv_qconfig = quantizer.object_type_config[conv2d_target]
    w4_conv_qconfig = candidate_quantizer.object_type_config[conv2d_target]
    if effective.magnet_alpha_candidate_id == "alpha-pe-w4a4-o8":
        w4_conv_qconfig = replace(
            w4_conv_qconfig,
            output_activation=w8_conv_qconfig.output_activation,
        )
    for pattern in W4_RESCUE_MODULE_PATTERNS:
        quantizer.set_module_name(pattern, w4_conv_qconfig)

    boundary_qspec = QuantizationSpec.from_str(A8_QSPEC)
    boundary_qconfig = QuantizationConfig(
        boundary_qspec,
        boundary_qspec,
        None,
        None,
    )
    for target in (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.adaptive_avg_pool2d.default,
    ):
        quantizer.set_object_type(target, boundary_qconfig)
    return quantizer


def _load_frozen_model(example_input):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = models.resnet18(weights=None).eval()
    base_state_sha256 = _state_sha256(model)
    with torch.no_grad():
        base_output = model(example_input)

    model.fc = _LinearAsConv2d(model.fc)
    modules_to_fuse = get_conv_bn_layers(model)
    if modules_to_fuse:
        torch.ao.quantization.fuse_modules(
            model,
            modules_to_fuse,
            inplace=True,
        )
    with torch.no_grad():
        projected_output = model(example_input)
    assert_output_close(
        base_output,
        projected_output,
        label="Canonical versus projected ResNet-18",
        rtol=1e-5,
        atol=1e-6,
    )
    normalized_relu_count = _normalize_export_relus(model)
    return (
        model,
        base_state_sha256,
        _state_sha256(model),
        normalized_relu_count,
    )


def _calibrate(model, calibration_inputs):
    for value in calibration_inputs:
        with torch.no_grad():
            model(value)


def run(
    args,
    *,
    vector_stages,
    hardware_config,
    scheduler_backend,
):
    """Run the exact resnet18 x Magnet Alpha adapter."""

    effective = _effective_args(args)
    candidate_id = effective.magnet_alpha_candidate_id
    if hardware_config != AOV_COMPAT_HARDWARE_CONFIG:
        raise ValueError(
            "The exact-pair adapter requires the resolved immutable Alpha "
            "compatibility profile"
        )
    if scheduler_backend != "magnet_alpha_rule_based":
        raise ValueError(
            "The initial exact-pair profile requires the existing "
            "magnet_alpha_rule_based scheduler"
        )

    preprocessing = _ImageNetPreprocess()
    raw_input = _deterministic_raw_sample(0)
    example_input = preprocessing(raw_input)
    calibration_raw = [
        _deterministic_raw_sample(index + 1)
        for index in range(effective.calibration_steps)
    ]
    calibration_inputs = [preprocessing(value) for value in calibration_raw]
    (
        model,
        base_hash,
        projected_hash,
        normalized_relu_count,
    ) = _load_frozen_model(example_input)
    hardware_payload = serialize_hardware_config(hardware_config)
    calibration_mode = (
        "skipped_for_structural_smoke"
        if effective.calibration_steps == 0
        else "representative_data"
    )
    recorder = _ContractRecorder(
        effective.model_output_dir,
        {
            "schema_version": 1,
            "workload": WORKLOAD,
            "target": TARGET,
            "profile_id": PROFILE_ID,
            "candidate_id": candidate_id,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision_version,
            "model": {
                "implementation": "torchvision.models.resnet18",
                "weights": None,
                "seed": MODEL_SEED,
                "base_state_sha256": base_hash,
                "projected_state_sha256": projected_hash,
                "projection": "Linear to exactly equivalent 1x1 Conv2d",
                "export_relu_normalization": {
                    "module": "torch.nn.ReLU",
                    "from_inplace": True,
                    "to_inplace": False,
                    "normalized_module_count": normalized_relu_count,
                },
                "max_pool": "canonical 3x3 stride-2 max pool",
            },
            "input": _sample_record(
                "resnet18-input-000",
                raw_input,
                example_input,
            ),
            "preprocessing": {
                "layout": "NCHW",
                "raw_range": [0.0, 1.0],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "calibration": {
                "mode": calibration_mode,
                "steps": effective.calibration_steps,
                "minimum_accepted_steps": MINIMUM_CALIBRATION_STEPS,
                "samples": [
                    _sample_record(
                        f"resnet18-calibration-{index:03d}",
                        raw,
                        processed,
                    )
                    for index, (raw, processed) in enumerate(
                        zip(calibration_raw, calibration_inputs)
                    )
                ],
            },
            "quantization": {
                "scope": (
                    "ptq_mixed_by_source_module"
                    if candidate_id.startswith("alpha-pe-w4a4")
                    else "ptq"
                ),
                "candidate_id": candidate_id,
                "policy_id": (
                    MIXED_W4_POLICY_ID
                    if candidate_id == "alpha-pe-w4a4-o8"
                    else (
                        MIXED_W4_O4_AUDIT_POLICY_ID
                        if candidate_id == "alpha-pe-w4a4-o4"
                        else candidate_id
                    )
                ),
                "candidate_ids_used": (
                    [DENSE_CANDIDATE_ID, candidate_id]
                    if candidate_id.startswith("alpha-pe-w4a4")
                    else [candidate_id]
                ),
                "activation_qspec": effective.activation,
                "output_activation_qspec": effective.output_activation,
                "weight_qspec": effective.weight,
                "default_dense_candidate_id": (
                    DENSE_CANDIDATE_ID
                    if candidate_id.startswith("alpha-pe-w4a4")
                    else candidate_id
                ),
                "mixed_module_override_patterns": (
                    list(W4_RESCUE_MODULE_PATTERNS)
                    if candidate_id.startswith("alpha-pe-w4a4")
                    else []
                ),
                "mixed_output_domain": (
                    "reuse W8 scalar-O8 Conv2d output contract"
                    if candidate_id == "alpha-pe-w4a4-o8"
                    else (
                        "candidate O4/PVS Conv2d output contract"
                        if candidate_id == "alpha-pe-w4a4-o4"
                        else "candidate output contract"
                    )
                ),
                "bias_and_accumulator_qspec": BIAS_QSPEC,
                "effective_bias_qspec": effective.bias,
                "non_gemm_boundary_qspec": A8_QSPEC,
                "zero_point": 0,
                "rounding": "as declared by the selected qspec tuple",
                "saturation": "as declared by the selected qspec tuple",
                "scale_provenance": "calibrated PT2E observers",
                "axis": {
                    "activation": "per-tensor",
                    "weight": 0,
                    "bias": "derived input_scale * weight_scale",
                },
            },
            "hardware_profile": {
                "sha256": _json_sha256(hardware_payload),
                "payload": hardware_payload,
            },
            "execution_policy": {
                "scheduler": scheduler_backend,
                "scratchpad_mapping": effective.scratchpad_mapping,
                "execution_order": effective.execution_order,
                "implicit_cpu_fallback": False,
                "unsupported_operations_are_not_cpu_owned": True,
                "connected_runtime": False,
            },
            "comparison_policy": {
                "float_to_converted": "report every-logit statistics",
                "converted_to_target": {
                    "minimum_cosine": POST_TARGET_MINIMUM_COSINE,
                    "minimum_sqnr_db": POST_TARGET_MINIMUM_SQNR_DB,
                },
                "target_to_mapped": {
                    "rtol": POST_MAPPING_RTOL,
                    "atol": POST_MAPPING_ATOL,
                },
            },
            "accepted_lane": (
                effective.calibration_steps >= MINIMUM_CALIBRATION_STEPS
            ),
            "last_completed_stage": "contract_frozen",
        },
    )

    stage = "full_precision_reference"
    try:
        with torch.no_grad():
            full_precision_output = model(example_input)
        recorder.stage(stage)

        stage = "export"
        graph_module = export_model(model, (example_input,))
        exported_inventory = _operation_inventory(graph_module)
        recorder.update(exported_inventory=exported_inventory)
        recorder.stage(stage)

        stage = "prepare_pt2e"
        quantizer = _get_candidate_quantizer(effective)
        graph_module = prepare_pt2e(
            graph_module,
            quantizer,
            hardware_config=hardware_config,
        )
        observer_count = sum(
            "Observer" in type(module).__name__
            or "FakeQuantize" in type(module).__name__
            for module in graph_module.modules()
        )
        recorder.update(prepared_observer_count=observer_count)
        recorder.stage(stage)

        stage = "calibration"
        _calibrate(graph_module, calibration_inputs)
        configured_boundaries = 0
        if calibration_inputs:
            finalize_pt2e_calibration(graph_module, hardware_config)
            configured_boundaries = prepare_target_quantization_recalibration(
                graph_module,
                TARGET,
                reset_observers=True,
            )
            if configured_boundaries:
                for value in calibration_inputs:
                    prepare_target_quantization_recalibration(
                        graph_module,
                        TARGET,
                        reset_observers=False,
                    )
                    with torch.no_grad():
                        graph_module(value)
        recorder.update(
            configured_target_recalibration_boundaries=(configured_boundaries)
        )
        recorder.stage(stage)

        stage = "convert_pt2e"
        convert_pt2e(
            graph_module,
            effective.bias,
            calibration_steps=effective.calibration_steps,
            hardware_config=hardware_config,
        )
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        with torch.no_grad():
            converted_output = graph_module(example_input)
        float_to_converted = compare_output_statistics(
            full_precision_output,
            converted_output,
            label="Full-precision versus converted ResNet-18",
        )
        recorder.update(
            converted_inventory=_operation_inventory(graph_module),
            qparams=_qparam_inventory(graph_module),
            float_to_converted_statistics=float_to_converted,
        )
        recorder.stage(stage)

        stage = "target_transform"
        transform_args = get_transform_args(
            effective,
            vector_stages,
            hardware_config=hardware_config,
        )
        transform(graph_module, (example_input,), **transform_args)
        qdq_summary = copy.deepcopy(
            graph_module.meta.get(
                INTERNAL_SCALAR_INT8_QDQ_SUMMARY_META,
            )
        )
        if qdq_summary is None:
            raise ValueError(
                "Magnet Alpha transform did not record internal scalar "
                "INT8 QDQ lowering"
            )
        final_fx_inventory = _operation_inventory(graph_module)
        recorder.update(
            final_fx_inventory=final_fx_inventory,
            internal_scalar_int8_qdq_lowering=qdq_summary,
        )
        if effective.calibration_steps >= MINIMUM_CALIBRATION_STEPS:
            recognized_qdq = qdq_summary.get("recognized_count")
            lowered_qdq = qdq_summary.get("lowered_count")
            unencodable_qdq = qdq_summary.get("unencodable_count")
            if recognized_qdq != lowered_qdq or unencodable_qdq != 0:
                raise ValueError(
                    "accepted ResNet-18 Alpha QDQ lowering is incomplete: "
                    f"recognized={recognized_qdq}, lowered={lowered_qdq}, "
                    f"unencodable={unencodable_qdq}"
                )
        with torch.no_grad():
            transformed_output = graph_module(example_input)
        target_statistics = compare_output_statistics(
            converted_output,
            transformed_output,
            label="Converted versus transformed ResNet-18",
            minimum_cosine=POST_TARGET_MINIMUM_COSINE,
            minimum_sqnr_db=POST_TARGET_MINIMUM_SQNR_DB,
        )
        recorder.update(
            converted_to_target_statistics=target_statistics,
        )
        recorder.stage(stage)

        stage = "compile"
        compile_args = get_compile_args(
            effective,
            hardware_config=hardware_config,
            scheduler_backend=scheduler_backend,
        )
        compile(graph_module, (example_input,), **compile_args)
        with torch.no_grad():
            mapped_output = graph_module(example_input)
        assert_output_close(
            transformed_output,
            mapped_output,
            label="Transformed versus mapped ResNet-18",
            rtol=POST_MAPPING_RTOL,
            atol=POST_MAPPING_ATOL,
        )
        recorder.stage(stage)
    except Exception as error:
        recorder.fail(stage, error)
        raise

    return ResNet18MagnetAlphaRunResult(
        workload=WORKLOAD,
        target=TARGET,
        profile_id=candidate_id,
        graph_module=graph_module,
        reference_output=converted_output,
        transformed_output=transformed_output,
        preprocess_fn=preprocessing,
        imprecise_pt2e=True,
    )


__all__ = [
    "A4_VSQ_QSPEC",
    "A8_QSPEC",
    "CANDIDATE_QSPECS",
    "DEFAULT_CANDIDATE_ID",
    "PROFILE_ID",
    "ResNet18MagnetAlphaRunResult",
    "TARGET",
    "WORKLOAD",
    "W4_VSQ_QSPEC",
    "W8_QSPEC",
    "run",
]
