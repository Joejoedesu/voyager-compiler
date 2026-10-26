# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""YOLO5 face n=0.5 workload and its optional Alpha inference recipe.

Build/example inputs are the workload contract. PTQ calibration and immutable
deployment-state ingestion are optional; no trainer, optimizer, EMA or resume
controller is included. Assets are supplied by path and fingerprinted.
"""

import hashlib
import math
import os
import pickle
import re
import sys
import types
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from voyager_compiler.export_utils import export_model, get_conv_bn_layers
from voyager_compiler.extensions import WorkloadSpec, registry
from voyager_compiler.ir.graph import QUANTIZATION_REGION
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)
from voyager_compiler.targets.magnet_alpha.observers import (
    FusedAmaxObsFakeQuantize,
)
from voyager_compiler.targets.magnet_alpha.ptq import (
    derive_bias_qparams_fn,
    finalize_pt2e_calibration,
    get_default_quantizer,
)
from voyager_compiler.targets.magnet_alpha.qspec import (
    DerivedQuantizationSpec,
    QScheme,
    QuantizationSpec,
)

MAGNET_ALPHA_COMPILATION_TARGET = "magnet-alpha"

RESUME_CHECKPOINT_FORMAT = "voyager.aov.qat.resume"

DEPLOYMENT_CHECKPOINT_FORMAT = "voyager.aov.qat.deployment"

CHECKPOINT_FORMAT_VERSION = 1

_WEIGHTED_OPERATORS = {torch.ops.aten.conv2d.default}

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

YOLO_ROOT_ENVIRONMENT_VARIABLE = "VOYAGER_YOLO_ROOT"

YOLO_CHECKPOINT_ENVIRONMENT_VARIABLE = "VOYAGER_YOLO_CHECKPOINT"

DEFAULT_YOLO_ROOT = REPOSITORY_ROOT / "libraries" / "yolov5-face"

DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "libraries" / "weights" / "yolov5n-0.5.pt"
)

AOV_CALIBRATION_VIDEO = str(REPOSITORY_ROOT / "data" / "calibration_video.avi")

AOV_CAMERA_CALIBRATION = str(REPOSITORY_ROOT / "data" / "calib_cam0.pkl")

AOV_CALIBRATION_IMAGE_SIZE = 192

AOV_QAT_IMAGE_SIZE = 800

AOV_SUPPORTED_QAT_IMAGE_SIZES = (
    AOV_CALIBRATION_IMAGE_SIZE,
    AOV_QAT_IMAGE_SIZE,
)

AOV_QAT_DEPLOYMENT_FORMAT = DEPLOYMENT_CHECKPOINT_FORMAT

AOV_QAT_DEPLOYMENT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION

AOV_QAT_DEPLOYMENT_PREPROCESSING_VERSION = 1

AOV_CALIBRATION_PAD_VALUE = 114

AOV_PRECISION_POLICY = "mixed"

AOV_FRESH_192_CALIBRATION_PROFILE = "fresh-192"

AOV_RELEASED_COLLATERAL_CALIBRATION_PROFILE = "released-collateral"

AOV_CALIBRATION_PROFILES = (
    AOV_FRESH_192_CALIBRATION_PROFILE,
    AOV_RELEASED_COLLATERAL_CALIBRATION_PROFILE,
)

AOV_DEFAULT_CALIBRATION_PROFILE = AOV_FRESH_192_CALIBRATION_PROFILE

AOV_RELEASED_COLLATERAL_STATUS = str(
    REPOSITORY_ROOT / "data" / "face_assoc_best_rank_0.pth"
)

AOV_RELEASED_COLLATERAL_ROOT = str(REPOSITORY_ROOT / "data" / "yolov5")

AOV_RELEASED_COLLATERAL_XSCALE = 4.0

AOV_RELEASED_W8_ACTIVATION_DENOMINATOR = 126

AOV_VOYAGER_W8_ACTIVATION_DENOMINATOR = 126

AOV_RELEASED_VSQ_PHYSICAL_COARSE_DENOMINATOR = 8 * 255

AOV_RELEASED_W4_WEIGHT_FINE_ENDPOINT = 7

AOV_RELEASED_W4_WEIGHT_COARSE_ENDPOINT = 127

AOV_RELEASED_W4_WEIGHT_MANTISSA_MAX = 255

AOV_RELEASED_W4_WEIGHT_COUNT = 36

AOV_RELEASED_COLLATERAL_EXPECTED_COUNTS = {
    "activation_observer_count": 115,
    "weight_observer_count": 85,
    "conv_input_observer_count": 77,
    "conv_input_layer_count": 85,
    "head_output_observer_count": 3,
    "quantization_region_count": 10,
}

AOV_RELEASED_COLLATERAL_EXACT_BOUNDARY_COUNT = 85

AOV_FIRST_STEM_CONV = "model.0.stem_1.conv"

POST_REWRITE_OUTPUT_RTOL = 1e-5

POST_REWRITE_OUTPUT_ATOL = 1e-6


@dataclass(frozen=True)
class Yolo5FaceNVTargetPolicy:
    """Immutable policy owned by this exact workload-target adapter."""

    workload: str
    target: str
    profile_id: str
    activation_qspec: str
    output_activation_qspec: str | None
    weight_qspec: str
    bias_qspec: str | None
    w8_activation_qspec: str
    w8_weight_qspec: str
    depthwise_w8_weight_qspec: str
    nongemm_qspec: str
    w8_layers: tuple[str, ...]
    depthwise_w8_layers: tuple[str, ...]
    input_channel_padding: tuple[tuple[str, int], ...]
    output_channel_padding: tuple[tuple[str, int], ...]
    first_layer_input_mode: str
    project_depthwise_scalar_bias: bool
    enable_target_recalibration: bool
    enable_quantization_region_qspec: bool
    default_imprecise_pt2e: bool


WORKLOAD = "yolo5-face-nv"

TARGET = MAGNET_ALPHA_COMPILATION_TARGET

_W4A4_ACTIVATION_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_tensor_symmetric,"
    "bs=32,scale=uint8,sqmax=8,csqmax=126"
)

_W4A4_WEIGHT_QSPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,cax=0,"
    "bs=32,scale=uint8,qmin=-7,qmax=7"
)

_W8A8_ACTIVATION_QSPEC = "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"

_W8_WEIGHT_QSPEC = "int8,qs=per_channel_symmetric,ax=0"

_DWC_W8_WEIGHT_QSPEC = "int8,qs=per_tensor_symmetric"

_NONGEMM_QSPEC = "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"

_INPUT_CHANNEL_PADDING = (
    ("model.1.branch1.2", 32),
    ("model.1.branch2.0", 32),
)

_OUTPUT_CHANNEL_PADDING = (
    ("model.21.m.0", 64),
    ("model.21.m.1", 64),
    ("model.21.m.2", 64),
)

_W8_LAYERS = (
    "model.0.stem_1.conv",
    "model.0.stem_2a.conv",
    "model.0.stem_2b.conv",
    "model.0.stem_3.conv",
    "model.1.branch1.0",
    "model.1.branch1.2",
    "model.1.branch2.0",
    "model.1.branch2.3",
    "model.2.0.branch2.0",
    "model.2.0.branch2.3",
    "model.2.1.branch2.0",
    "model.2.1.branch2.3",
    "model.2.2.branch2.0",
    "model.2.2.branch2.3",
    "model.3.branch1.0",
    "model.3.branch2.0",
    "model.3.branch2.3",
    "model.4.0.branch2.0",
    "model.4.0.branch2.3",
    "model.4.1.branch2.0",
    "model.4.1.branch2.3",
    "model.4.2.branch2.0",
    "model.4.2.branch2.3",
    "model.4.3.branch2.0",
    "model.4.3.branch2.3",
    "model.4.4.branch2.0",
    "model.4.4.branch2.3",
    "model.4.5.branch2.0",
    "model.4.5.branch2.3",
    "model.4.6.branch2.0",
    "model.4.6.branch2.3",
    "model.5.branch1.0",
    "model.5.branch2.0",
    "model.5.branch2.3",
    "model.6.0.branch2.0",
    "model.6.0.branch2.3",
    "model.6.1.branch2.0",
    "model.6.1.branch2.3",
    "model.6.2.branch2.0",
    "model.6.2.branch2.3",
    "model.7.conv",
    "model.10.cv1.conv",
    "model.10.cv2.conv",
    "model.14.cv1.conv",
    "model.14.cv2.conv",
    "model.17.cv1.conv",
    "model.17.cv2.conv",
    "model.20.cv1.conv",
    "model.20.cv2.conv",
)

_DEPTHWISE_W8_LAYERS = (
    "model.1.branch1.0",
    "model.1.branch2.3",
    "model.2.0.branch2.3",
    "model.2.1.branch2.3",
    "model.2.2.branch2.3",
    "model.3.branch1.0",
    "model.3.branch2.3",
    "model.4.0.branch2.3",
    "model.4.1.branch2.3",
    "model.4.2.branch2.3",
    "model.4.3.branch2.3",
    "model.4.4.branch2.3",
    "model.4.5.branch2.3",
    "model.4.6.branch2.3",
    "model.5.branch1.0",
    "model.5.branch2.3",
    "model.6.0.branch2.3",
    "model.6.1.branch2.3",
    "model.6.2.branch2.3",
)

POLICY = Yolo5FaceNVTargetPolicy(
    workload=WORKLOAD,
    target=TARGET,
    profile_id="yolo5-face-nv-magnet-alpha-mixed-w4a4-v1",
    activation_qspec=_W4A4_ACTIVATION_QSPEC,
    output_activation_qspec=None,
    weight_qspec=_W4A4_WEIGHT_QSPEC,
    bias_qspec=None,
    w8_activation_qspec=_W8A8_ACTIVATION_QSPEC,
    w8_weight_qspec=_W8_WEIGHT_QSPEC,
    depthwise_w8_weight_qspec=_DWC_W8_WEIGHT_QSPEC,
    nongemm_qspec=_NONGEMM_QSPEC,
    w8_layers=_W8_LAYERS,
    depthwise_w8_layers=_DEPTHWISE_W8_LAYERS,
    input_channel_padding=_INPUT_CHANNEL_PADDING,
    output_channel_padding=_OUTPUT_CHANNEL_PADDING,
    first_layer_input_mode="padding",
    project_depthwise_scalar_bias=True,
    enable_target_recalibration=True,
    enable_quantization_region_qspec=True,
    default_imprecise_pt2e=True,
)


class _InputChannelPaddedConv2d(nn.Conv2d):
    """Conv2d with explicit zero padding on the input-channel dimension."""

    def __init__(self, conv: nn.Conv2d, padded_in_channels: int):
        if conv.groups != 1:
            raise ValueError("Input-channel padding requires groups=1")
        if padded_in_channels < conv.in_channels:
            raise ValueError("Padded input channels cannot shrink a Conv2d")

        super().__init__(
            in_channels=padded_in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.logical_in_channels = conv.in_channels
        self.input_channel_padding = padded_in_channels - conv.in_channels

        padded_weight = F.pad(
            conv.weight.detach(),
            (0, 0, 0, 0, 0, self.input_channel_padding),
        )
        self.weight = nn.Parameter(
            padded_weight,
            requires_grad=conv.weight.requires_grad,
        )
        if conv.bias is not None:
            self.bias = nn.Parameter(
                conv.bias.detach().clone(),
                requires_grad=conv.bias.requires_grad,
            )
        self.train(conv.training)

    def forward(self, inputs):
        padded_inputs = F.pad(
            inputs,
            (0, 0, 0, 0, 0, self.input_channel_padding),
        )
        return self._conv_forward(padded_inputs, self.weight, self.bias)


class _Im2ColConv2d(nn.Conv2d):
    """Pair unsupported external im2col with an Alpha-candidate 1x1 Conv2d."""

    def __init__(self, conv: nn.Conv2d):
        if conv.groups != 1:
            raise ValueError("First-layer im2col requires groups=1")
        if conv.padding_mode != "zeros":
            raise ValueError(
                "First-layer im2col requires zero-valued padding mode"
            )

        logical_in_channels = conv.in_channels
        kernel_elements = conv.kernel_size[0] * conv.kernel_size[1]
        flattened_in_channels = logical_in_channels * kernel_elements
        super().__init__(
            in_channels=flattened_in_channels,
            out_channels=conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=conv.bias is not None,
            padding_mode="zeros",
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.logical_in_channels = logical_in_channels
        self.im2col_kernel_size = conv.kernel_size
        self.im2col_stride = conv.stride
        self.im2col_padding = conv.padding
        self.im2col_dilation = conv.dilation

        flattened_weight = conv.weight.detach().reshape(
            conv.out_channels,
            flattened_in_channels,
            1,
            1,
        )
        self.weight = nn.Parameter(
            flattened_weight.clone(),
            requires_grad=conv.weight.requires_grad,
        )
        if conv.bias is not None:
            self.bias = nn.Parameter(
                conv.bias.detach().clone(),
                requires_grad=conv.bias.requires_grad,
            )
        self.train(conv.training)

    def forward(self, inputs):
        patches = F.unfold(
            inputs,
            kernel_size=self.im2col_kernel_size,
            dilation=self.im2col_dilation,
            padding=self.im2col_padding,
            stride=self.im2col_stride,
        )
        output_height = (
            inputs.shape[-2]
            + 2 * self.im2col_padding[0]
            - self.im2col_dilation[0] * (self.im2col_kernel_size[0] - 1)
            - 1
        ) // self.im2col_stride[0] + 1
        output_width = (
            inputs.shape[-1]
            + 2 * self.im2col_padding[1]
            - self.im2col_dilation[1] * (self.im2col_kernel_size[1] - 1)
            - 1
        ) // self.im2col_stride[1] + 1
        patches = patches.reshape(
            inputs.shape[0],
            self.in_channels,
            1,
            output_height * output_width,
        )
        outputs = self._conv_forward(patches, self.weight, self.bias)
        return outputs.reshape(
            inputs.shape[0],
            self.out_channels,
            output_height,
            output_width,
        )


def _rewrite_aov_input_channel_padding(model):
    """Match AoV's explicit C=32 padding for two early 1x1 convolutions."""
    for module_name, padded_channels in POLICY.input_channel_padding:
        conv = model.get_submodule(module_name)
        if isinstance(conv, _InputChannelPaddedConv2d):
            if conv.in_channels != padded_channels:
                raise ValueError(
                    f"{module_name} is already padded to "
                    f"{conv.in_channels}, expected {padded_channels}"
                )
            continue
        if not isinstance(conv, nn.Conv2d):
            raise TypeError(
                f"{module_name} must be Conv2d, got {type(conv).__name__}"
            )
        model.set_submodule(
            module_name,
            _InputChannelPaddedConv2d(conv, padded_channels),
        )


def _rewrite_first_layer_im2col(model):
    """Replace the first stem convolution with an equivalent im2col form."""
    conv = model.get_submodule(AOV_FIRST_STEM_CONV)
    if isinstance(conv, _Im2ColConv2d):
        return
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(
            f"{AOV_FIRST_STEM_CONV} must be Conv2d, got {type(conv).__name__}"
        )
    model.set_submodule(AOV_FIRST_STEM_CONV, _Im2ColConv2d(conv))


def _uses_first_layer_im2col(args):
    mode = (
        "im2col"
        if getattr(args, "conv2d_im2col", None)
        else POLICY.first_layer_input_mode
    )
    if mode not in ("padding", "im2col"):
        raise ValueError(f"Unsupported first-layer input mode: {mode}")
    return mode == "im2col"


def clear_yolo_imports():
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]


def _install_import_stubs():
    # Plotting/profiling libraries are imported by the AoV YOLO fork, but are
    # irrelevant to compiler export.
    if "seaborn" not in sys.modules:
        try:
            __import__("seaborn")
        except ImportError:
            sys.modules["seaborn"] = types.ModuleType("seaborn")

    if "thop" not in sys.modules:
        try:
            __import__("thop")
        except ImportError:
            thop_stub = types.ModuleType("thop")
            thop_stub.profile = lambda *args, **kwargs: (0, 0)
            thop_stub.clever_format = lambda values, *args, **kwargs: values
            sys.modules["thop"] = thop_stub


def _resolve_checkpoint() -> Path:
    configured = os.environ.get(YOLO_CHECKPOINT_ENVIRONMENT_VARIABLE)
    path = Path(configured) if configured else DEFAULT_CHECKPOINT
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "YOLO checkpoint not found. Pass --model_name_or_path, set "
            f"{YOLO_CHECKPOINT_ENVIRONMENT_VARIABLE}, or install it at "
            f"{DEFAULT_CHECKPOINT}."
        )
    return path


def _resolve_yolo_root() -> Path:
    root = Path(
        os.environ.get(
            YOLO_ROOT_ENVIRONMENT_VARIABLE,
            DEFAULT_YOLO_ROOT,
        )
    ).resolve()
    if not (root / "models" / "yolo.py").is_file():
        raise FileNotFoundError(
            "YOLO source tree not found. Set "
            f"{YOLO_ROOT_ENVIRONMENT_VARIABLE} or install it at "
            f"{DEFAULT_YOLO_ROOT}."
        )
    return root


def _load_qat_deployment_parameters(model, snapshot):
    """Validate and load deployment parameters for this exact pair."""
    apply_deployment_parameters(
        model,
        snapshot,
        expected_precision_policy=AOV_PRECISION_POLICY,
        expected_preprocessing_version=(
            AOV_QAT_DEPLOYMENT_PREPROCESSING_VERSION
        ),
        expected_image_size=AOV_SUPPORTED_QAT_IMAGE_SIZES,
    )


def _load_conventional_checkpoint(checkpoint, path):
    """Construct the YOLO model represented by one conventional checkpoint."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"].float().eval()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        from models.yolo import Model

        model = Model(checkpoint["model_cfg_yaml:"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.float().eval()
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint.float().eval()
    else:
        raise ValueError(f"Unsupported YOLO checkpoint format: {path}")
    return model


def _load_checkpoint(path, *, return_deployment=False):
    """Load one conventional or single-level deployment checkpoint."""
    model, deployment = load_optional_deployment_checkpoint(
        path,
        _load_conventional_checkpoint,
        expected_precision_policy=AOV_PRECISION_POLICY,
        expected_preprocessing_version=(
            AOV_QAT_DEPLOYMENT_PREPROCESSING_VERSION
        ),
        expected_image_size=AOV_SUPPORTED_QAT_IMAGE_SIZES,
    )
    if return_deployment:
        return model, deployment
    return model


def _replace_silu_with_relu(module):
    for child_name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, child_name, nn.ReLU())
        else:
            _replace_silu_with_relu(child)


def _rewrite_first_layer_grayscale(model):
    """Project an RGB first-layer kernel onto one grayscale input channel."""
    conv = model.get_submodule(AOV_FIRST_STEM_CONV)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(
            f"{AOV_FIRST_STEM_CONV} must be Conv2d, got {type(conv).__name__}"
        )
    if conv.in_channels == 1:
        return
    if conv.in_channels != 3 or conv.groups != 1:
        raise ValueError(
            "YOLO first-layer grayscale projection requires an ordinary "
            f"RGB convolution, got in_channels={conv.in_channels}, "
            f"groups={conv.groups}"
        )

    grayscale = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    grayscale.weight = nn.Parameter(
        conv.weight.detach().sum(dim=1, keepdim=True),
        requires_grad=conv.weight.requires_grad,
    )
    if conv.bias is not None:
        grayscale.bias = nn.Parameter(
            conv.bias.detach().clone(),
            requires_grad=conv.bias.requires_grad,
        )
    grayscale.train(conv.training)
    model.set_submodule(AOV_FIRST_STEM_CONV, grayscale)


def _ensure_detect_helpers(model):
    for module in model.modules():
        if module.__class__.__name__ != "Detect":
            continue

        # AoV checkpoints and their source tree carry optional diagnostic
        # state across revisions. The diagnostic performs data-dependent
        # scalar extraction and is not part of YOLO output semantics, so it
        # must be disabled before PT2E export.
        module.debug_print_max_conf_vector = False
        if not hasattr(module, "zero_last_entry"):
            module.zero_last_entry = False

        if not hasattr(module, "idents"):
            idents = []
            for i in range(getattr(module, "nl", 0)):
                ident = nn.Identity()
                setattr(module, f"ident_{i}", ident)
                idents.append(ident)
            module.idents = idents

        if not hasattr(module, "output_probes"):
            probes = []
            for i in range(getattr(module, "nl", 0)):
                probe = nn.Identity()
                setattr(module, f"output_probe_{i}", probe)
                probes.append(probe)
            module.output_probes = nn.ModuleList(probes)


def _set_raw_head_export(model):
    for module in model.modules():
        if hasattr(module, "export"):
            module.export = False
            module.train()
        if hasattr(module, "export_cat"):
            module.export_cat = False
            module.train()


def _fuse_conv_bn(model):
    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        torch.ao.quantization.fuse_modules(model, modules_to_fuse, inplace=True)


def _prepare_aov_depthwise_scalar_bias(model):
    """Project fused DWC bias to Alpha's one-scalar bias datapath.

    The supplied checkpoint uses ordinary per-channel BatchNorm. Folding it
    therefore produces a per-channel bias, while Alpha broadcasts one signed
    bias value across all DWC lanes. The least-squares scalar projection is
    the channel mean. Keep the projection here, before PT2E preparation, so
    calibration observes the graph that will actually be lowered. The
    returned residual makes this intentionally lossy model preparation
    visible instead of silently averaging in the collateral converter.
    """
    report = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if module.groups != module.in_channels:
                continue
            if module.out_channels != module.in_channels:
                continue
            if module.bias is None:
                continue

            original = module.bias.detach().clone()
            scalar = original.mean()
            residual = original - scalar
            module.bias.copy_(scalar.expand_as(module.bias))
            report.append(
                {
                    "module": name,
                    "channels": original.numel(),
                    "scalar": float(scalar),
                    "maximum_absolute_residual": float(residual.abs().max()),
                    "rmse": float(residual.square().mean().sqrt()),
                }
            )
    return report


def load_model(args, torch_dtype):
    yolo_root = str(_resolve_yolo_root())
    if yolo_root in sys.path:
        sys.path.remove(yolo_root)
    sys.path.insert(0, yolo_root)

    clear_yolo_imports()
    _install_import_stubs()

    if args.model_name_or_path is None:
        args.model_name_or_path = str(_resolve_checkpoint())
        print(f"Using yolo5-face-nv checkpoint: {args.model_name_or_path}")

    model, deployment = _load_checkpoint(
        args.model_name_or_path,
        return_deployment=True,
    )
    _rewrite_first_layer_grayscale(model)
    _replace_silu_with_relu(model)
    _fuse_conv_bn(model)
    if POLICY.project_depthwise_scalar_bias:
        bias_report = _prepare_aov_depthwise_scalar_bias(model)
        if bias_report:
            maximum = max(
                item["maximum_absolute_residual"] for item in bias_report
            )
            rmse = max(item["rmse"] for item in bias_report)
            calibration = getattr(args, "calibration_steps", 0)
            suffix = (
                " PT2E calibration observes the projected model, but does "
                "not optimize its bias."
                if calibration > 0
                else " Zero-calibration output is a compiler fallback, not "
                "AoV numerical parity."
            )
            warnings.warn(
                "Projected fused DWC biases onto Alpha's scalar-bias "
                f"datapath for {len(bias_report)} layers "
                f"(max residual={maximum:.6g}, max RMSE={rmse:.6g})."
                f"{suffix}",
                RuntimeWarning,
            )
    _rewrite_aov_input_channel_padding(model)
    if _uses_first_layer_im2col(args):
        _rewrite_first_layer_im2col(model)
    # Output-channel residency belongs to the compiler's dtype-aware matrix
    # padding pass. Keeping it out of model preprocessing lets the PPU
    # requantize the physical lanes before the logical output slice.
    _ensure_detect_helpers(model)
    if deployment is not None:
        _load_qat_deployment_parameters(
            model,
            deployment,
        )

    if args.bf16:
        model.bfloat16()

    _set_raw_head_export(model)

    first_conv = model.get_submodule(AOV_FIRST_STEM_CONV)
    first_weight = next(model.parameters())
    input_channels = getattr(
        first_conv,
        "logical_in_channels",
        first_weight.shape[1] if first_weight.ndim == 4 else 3,
    )
    if input_channels != 1:
        raise ValueError(
            "YOLO model loading must produce a single-channel first layer"
        )
    image_size = int(
        getattr(args, "aov_image_size", AOV_CALIBRATION_IMAGE_SIZE)
    )
    if image_size <= 0 or image_size % 32:
        raise ValueError("AoV image size must be a positive multiple of 32")
    example_args = (
        torch.randn(
            1,
            input_channels,
            image_size,
            image_size,
            dtype=torch_dtype,
        ),
    )
    return model, example_args


def uses_quantization(args):
    return args.pt2e or any(
        spec is not None
        for spec in (
            args.activation,
            args.output_activation,
            args.weight,
            args.bias,
        )
    )


def default_imprecise_pt2e(args):
    return bool(POLICY.default_imprecise_pt2e and uses_quantization(args))


def configure_quantizer(quantizer, args):
    if not uses_quantization(args):
        return quantizer

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    activation_qscheme = QuantizationSpec.from_str(args.activation).qscheme
    weight_qscheme = QuantizationSpec.from_str(args.weight).qscheme
    uses_block_quantization = (
        activation_qscheme == weight_qscheme
        and activation_qscheme in (QScheme.MICROSCALING, QScheme.VSQ)
    )
    if uses_block_quantization:
        int8_act_qspec = QuantizationSpec.from_str(POLICY.w8_activation_qspec)
        int8_weight_qspec = QuantizationSpec.from_str(POLICY.w8_weight_qspec)
        dwc_weight_qspec = QuantizationSpec.from_str(
            POLICY.depthwise_w8_weight_qspec
        )
        bias_qspec = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=None,
        )
        int8_qconfig = QuantizationConfig(
            int8_act_qspec,
            None,
            int8_weight_qspec,
            bias_qspec,
        )
        dwc_qconfig = QuantizationConfig(
            int8_act_qspec,
            None,
            dwc_weight_qspec,
            bias_qspec,
        )
        head_input_qspec = replace(
            QuantizationSpec.from_str(args.activation),
            ch_axis=1,
        )
        head_weight_qspec = replace(
            QuantizationSpec.from_str(args.weight),
            ch_axis=1,
        )
        if head_weight_qspec.coarse_qscheme == QScheme.PER_CHANNEL_SYMMETRIC:
            head_weight_qspec = replace(
                head_weight_qspec,
                coarse_ch_axis=0,
            )
        head_qconfig = QuantizationConfig(
            head_input_qspec,
            int8_act_qspec,
            head_weight_qspec,
            None,
        )
        dwc_layers = set(POLICY.depthwise_w8_layers)
        for layer in POLICY.w8_layers:
            if layer in dwc_layers:
                continue
            quantizer.set_module_name(
                f"^{re.escape(layer)}$",
                int8_qconfig,
            )
        for layer in POLICY.depthwise_w8_layers:
            quantizer.set_module_name(f"^{re.escape(layer)}$", dwc_qconfig)
        for layer, _ in POLICY.output_channel_padding:
            quantizer.set_module_name(
                f"^{re.escape(layer)}$",
                head_qconfig,
            )

    # AoV's MPQT rule keeps non-GEMM/elementwise ops at int8 per-tensor.
    nongemm_qspec = QuantizationSpec.from_str(POLICY.nongemm_qspec)
    nongemm_qconfig = QuantizationConfig(nongemm_qspec, None, None, None)
    quantizer.set_object_type(torch.ops.aten.add.Tensor, nongemm_qconfig)
    quantizer.set_object_type(torch.ops.aten.add_.Tensor, nongemm_qconfig)
    return quantizer


def retrieve_dataset(args):
    if args.calibration_steps <= 0 or not uses_quantization(args):
        return None
    return _AoVCalibrationDataset(args.calibration_steps)


def _load_camera_calibration(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AoV camera calibration not found: {path}")

    with open(path, "rb") as calibration_file:
        calibration = pickle.load(calibration_file)

    if not isinstance(calibration, dict):
        raise ValueError("AoV camera calibration must be a dictionary")
    if "mtx" not in calibration or "dist" not in calibration:
        raise ValueError("AoV camera calibration requires 'mtx' and 'dist'")

    camera_matrix = np.asarray(calibration["mtx"])
    distortion = np.asarray(calibration["dist"])
    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "AoV camera matrix must have shape (3, 3), got "
            f"{camera_matrix.shape}"
        )
    valid_distortion_sizes = (4, 5, 8, 12, 14)
    if (
        distortion.ndim not in (1, 2)
        or distortion.size not in valid_distortion_sizes
    ):
        raise ValueError(
            "AoV distortion coefficients must contain 4, 5, 8, 12, or "
            f"14 values, got shape {distortion.shape}"
        )

    return camera_matrix, distortion


def _resize_and_letterbox_grayscale(image):
    height, width = image.shape
    scale = AOV_CALIBRATION_IMAGE_SIZE / max(height, width)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=interpolation,
        )

    height, width = image.shape
    width_padding = AOV_CALIBRATION_IMAGE_SIZE - width
    height_padding = AOV_CALIBRATION_IMAGE_SIZE - height
    width_padding = int(np.mod(width_padding, 64))
    height_padding = int(np.mod(height_padding, 64))
    left = int(round(width_padding / 2 - 0.1))
    right = int(round(width_padding / 2 + 0.1))
    top = int(round(height_padding / 2 - 0.1))
    bottom = int(round(height_padding / 2 + 0.1))
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=AOV_CALIBRATION_PAD_VALUE,
    )


def _preprocess_calibration_input(
    frame,
    input_channels,
    torch_dtype,
    camera_matrix=None,
    distortion=None,
):
    if input_channels <= 0:
        raise ValueError(
            f"Input channels must be positive, got {input_channels}"
        )
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        shape = None if frame is None else frame.shape
        raise ValueError(f"AoV calibration frame must be BGR HWC, got {shape}")
    if (camera_matrix is None) != (distortion is None):
        raise ValueError(
            "Camera matrix and distortion coefficients must be provided "
            "together"
        )

    if camera_matrix is not None:
        height, width = frame.shape[:2]
        undistort_maps = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            np.eye(3),
            None,
            (width, height),
            cv2.CV_32FC1,
        )
        frame = cv2.remap(
            frame,
            undistort_maps[0],
            undistort_maps[1],
            cv2.INTER_LINEAR,
        )
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale = _resize_and_letterbox_grayscale(grayscale)
    if grayscale.shape != (
        AOV_CALIBRATION_IMAGE_SIZE,
        AOV_CALIBRATION_IMAGE_SIZE,
    ):
        raise ValueError(
            "AoV calibration preprocessing produced shape "
            f"{grayscale.shape}, expected "
            f"({AOV_CALIBRATION_IMAGE_SIZE}, "
            f"{AOV_CALIBRATION_IMAGE_SIZE})"
        )

    inputs = torch.from_numpy(np.ascontiguousarray(grayscale)).float()
    inputs = inputs.unsqueeze(0) / 255.0
    if input_channels != 1:
        inputs = inputs.repeat(input_channels, 1, 1)
    return inputs.unsqueeze(0).to(torch_dtype)


class _AoVCalibrationDataset:
    """Eagerly decode the requested AoV video frames for calibration."""

    def __init__(self, calibration_steps):
        if calibration_steps <= 0:
            raise ValueError(
                "AoV calibration requires a positive number of steps"
            )
        if not os.path.isfile(AOV_CALIBRATION_VIDEO):
            raise FileNotFoundError(
                f"AoV calibration video not found: {AOV_CALIBRATION_VIDEO}"
            )

        self.camera_matrix, self.distortion = _load_camera_calibration(
            AOV_CAMERA_CALIBRATION
        )
        capture = cv2.VideoCapture(AOV_CALIBRATION_VIDEO)
        if not capture.isOpened():
            raise ValueError(
                f"Could not open AoV calibration video: {AOV_CALIBRATION_VIDEO}"
            )

        self.frames = []
        try:
            for frame_index in range(calibration_steps):
                decoded, frame = capture.read()
                if not decoded:
                    raise ValueError(
                        "AoV calibration video ended after "
                        f"{frame_index} frames; {calibration_steps} requested"
                    )
                self.frames.append(frame)
        finally:
            capture.release()

    def __len__(self):
        return len(self.frames)

    def preprocess(self, index, input_channels, torch_dtype):
        return _preprocess_calibration_input(
            self.frames[index],
            input_channels,
            torch_dtype,
            self.camera_matrix,
            self.distortion,
        )


def _get_aov_calibration_profile(args):
    profile = getattr(
        args,
        "aov_calibration_profile",
        AOV_DEFAULT_CALIBRATION_PROFILE,
    )
    if profile not in AOV_CALIBRATION_PROFILES:
        raise ValueError(f"Unsupported AoV calibration profile: {profile}")
    return profile


def _released_status_amax(status, layer, role):
    key = f"{role}_accumulated_static_scales"
    entry = status.get(layer)
    if not isinstance(entry, dict) or key not in entry:
        raise KeyError(f"Missing released status {layer}.{key}")

    value = torch.as_tensor(entry[key]).detach().reshape(-1)
    if value.numel() != 1:
        raise ValueError(
            f"Expected scalar released status {layer}.{key}, "
            f"got {value.numel()} values"
        )
    amax = float(value[0])
    if not np.isfinite(amax) or amax <= 0.0:
        raise ValueError(
            f"Released status {layer}.{key} must be finite and positive"
        )
    return amax


def _aov_weight_attribute(value):
    queue = [value]
    visited = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, torch.fx.Node) or current in visited:
            continue
        visited.add(current)
        if current.op == "get_attr" and str(current.target).endswith(".weight"):
            return str(current.target)
        queue.extend(current.all_input_nodes)
    return None


def _aov_layer_for_conv(node):
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.conv2d.default
    ):
        return None
    weight = node.args[1] if len(node.args) > 1 else node.kwargs.get("weight")
    attribute = _aov_weight_attribute(weight)
    return attribute.removesuffix(".weight") if attribute is not None else None


def _aov_observer_nodes(model):
    modules = dict(model.named_modules(remove_duplicate=False))
    return {
        node: modules[str(node.target)]
        for node in model.graph.nodes
        if node.op == "call_module"
        and isinstance(
            modules.get(str(node.target)),
            FusedAmaxObsFakeQuantize,
        )
    }


def _aov_conv_input_anchors(model, status, observer_nodes):
    anchors = defaultdict(list)
    weight_targets = set()
    for conv in model.graph.nodes:
        layer = _aov_layer_for_conv(conv)
        if layer is None:
            continue

        input_node = conv.args[0] if conv.args else conv.kwargs.get("input")
        if input_node not in observer_nodes:
            raise ValueError(f"Conv input has no fused observer: {layer}")
        weight_node = (
            conv.args[1] if len(conv.args) > 1 else conv.kwargs.get("weight")
        )
        if weight_node not in observer_nodes:
            raise ValueError(f"Conv weight has no fused observer: {layer}")

        anchors[str(input_node.target)].append(
            {
                "layer": layer,
                "raw_amax": _released_status_amax(
                    status,
                    layer,
                    "input",
                ),
            }
        )
        weight_targets.add(str(weight_node.target))
    return anchors, weight_targets


def _aov_region_members(observer_nodes):
    members = defaultdict(set)
    for node in observer_nodes:
        region = node.meta.get(QUANTIZATION_REGION)
        if region is not None:
            members[region].add(str(node.target))
    return members


def _aov_is_conv_activation_input(observer, user):
    if (
        user.op != "call_function"
        or user.target != torch.ops.aten.conv2d.default
    ):
        return False
    input_value = user.args[0] if user.args else user.kwargs.get("input")
    return input_value is observer


def _aov_head_output_anchors(
    observer_nodes,
    status,
    activation_targets,
):
    anchors = {}
    for node in observer_nodes:
        target = str(node.target)
        if target not in activation_targets or not node.args:
            continue
        if any(
            _aov_is_conv_activation_input(node, user) for user in node.users
        ):
            continue

        weight = _aov_weight_attribute(node.args[0])
        if weight is None or not weight.startswith("model.21.m."):
            continue
        head_index = weight.removeprefix("model.21.m.").removesuffix(".weight")
        status_layer = f"model.21.output_probes.{head_index}"
        anchors[target] = {
            "layer": status_layer,
            "raw_amax": _released_status_amax(
                status,
                status_layer,
                "input",
            ),
        }
    return anchors


def _validate_released_mapping_counts(report):
    expected = AOV_RELEASED_COLLATERAL_EXPECTED_COUNTS
    mismatches = {
        name: {
            "expected": expected_value,
            "actual": report.get(name),
        }
        for name, expected_value in expected.items()
        if report.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(
            "Released AoV status does not match the prepared graph: "
            f"{mismatches}"
        )


def _set_released_scalar_amax(observer, value):
    history_len = observer.amax_history_len
    if history_len is None:
        history_len = 1
    observer.amax_history.resize_((history_len,)).fill_(value)
    observer.calculate_qparams()
    observer.disable_observer()


def _load_released_activation_status(model, status, xscale):
    """Load released activation maxima by graph semantics, not FX ordinals."""
    if not np.isfinite(xscale) or xscale <= 0.0:
        raise ValueError("Released AoV xscale must be finite and positive")

    observer_nodes = _aov_observer_nodes(model)
    modules = dict(model.named_modules(remove_duplicate=False))
    conv_anchors, weight_targets = _aov_conv_input_anchors(
        model,
        status,
        observer_nodes,
    )
    activation_targets = {
        str(node.target)
        for node in observer_nodes
        if str(node.target) not in weight_targets
    }
    assignments = {}

    for target, anchors in conv_anchors.items():
        raw_values = {item["raw_amax"] for item in anchors}
        if len(raw_values) != 1:
            layers = sorted(item["layer"] for item in anchors)
            raise ValueError(
                f"Shared observer {target} has conflicting AoV ranges "
                f"for {layers}"
            )
        assignments[target] = next(iter(raw_values))

    regions = _aov_region_members(observer_nodes)
    for region, targets in sorted(regions.items()):
        raw_values = {
            assignments[target] for target in targets if target in assignments
        }
        if len(raw_values) != 1:
            raise ValueError(
                f"Region {region} needs exactly one AoV activation domain"
            )
        raw_amax = next(iter(raw_values))
        for target in targets:
            assignments[target] = raw_amax

    head_anchors = _aov_head_output_anchors(
        observer_nodes,
        status,
        activation_targets,
    )
    for target, anchor in head_anchors.items():
        assignments[target] = anchor["raw_amax"]

    missing = sorted(activation_targets - set(assignments))
    extra = sorted(set(assignments) - activation_targets)
    if missing or extra:
        raise ValueError(
            "Incomplete released AoV activation mapping: "
            f"missing={missing}, extra={extra}"
        )

    uninitialized_weights = sorted(
        target
        for target in weight_targets
        if modules[target].amax_history.numel() == 0
    )
    if uninitialized_weights:
        raise ValueError(
            "Released AoV profile requires initialized weight observers: "
            f"{uninitialized_weights}"
        )

    report = {
        "activation_observer_count": len(assignments),
        "weight_observer_count": len(weight_targets),
        "conv_input_observer_count": len(conv_anchors),
        "conv_input_layer_count": sum(
            len(anchors) for anchors in conv_anchors.values()
        ),
        "head_output_observer_count": len(head_anchors),
        "quantization_region_count": len(regions),
    }
    _validate_released_mapping_counts(report)

    for target, raw_amax in assignments.items():
        _set_released_scalar_amax(
            modules[target],
            raw_amax * xscale,
        )
    return report


def _released_weight_parameter(model, weight_node):
    source = weight_node.args[0] if weight_node.args else None
    if not isinstance(source, torch.fx.Node) or source.op != "get_attr":
        raise ValueError("Released AoV weight observer has no parameter")
    try:
        value = model.get_parameter(str(source.target))
    except AttributeError:
        value = model.get_buffer(str(source.target))
    return value.detach()


def _load_released_w4_weight_encoding(model):
    """Attach released W4 payload and scales to semantic weight observers."""
    modules = dict(model.named_modules(remove_duplicate=False))
    loaded_layers = []
    for conv in model.graph.nodes:
        layer = _aov_layer_for_conv(conv)
        if layer is None:
            continue
        weight_node = (
            conv.args[1] if len(conv.args) > 1 else conv.kwargs.get("weight")
        )
        if (
            not isinstance(weight_node, torch.fx.Node)
            or weight_node.op != "call_module"
        ):
            continue
        observer = modules.get(str(weight_node.target))
        if (
            not isinstance(observer, FusedAmaxObsFakeQuantize)
            or observer.qscheme != QScheme.VSQ
        ):
            continue

        parameter = _released_weight_parameter(model, weight_node)
        if parameter.ndim != 4:
            raise ValueError(f"Released AoV W4 weight is not KCRS: {layer}")
        output_channels, input_channels, kernel_h, kernel_w = parameter.shape
        directory = os.path.join(
            AOV_RELEASED_COLLATERAL_ROOT,
            layer.replace(".", "_"),
        )
        prefix = os.path.join(directory, f"{layer}_weight")
        paths = {
            "payload": f"{prefix}.csv",
            "fine": f"{prefix}_fine_scale.csv",
            "coarse": f"{prefix}_coarse_scale.csv",
        }
        missing = [path for path in paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing released W4 weight collateral for {layer}: {missing}"
            )

        payload = torch.from_numpy(
            np.asarray(
                np.loadtxt(paths["payload"], delimiter=","),
            ).reshape(parameter.shape)
        )
        fine_shape = (
            output_channels,
            (input_channels + 31) // 32,
            kernel_h,
            kernel_w,
        )
        fine_mantissa = torch.from_numpy(
            np.asarray(
                np.loadtxt(paths["fine"], delimiter=","),
            ).reshape(fine_shape)
        )
        physical_coarse = torch.from_numpy(
            np.asarray(
                np.loadtxt(paths["coarse"], delimiter=","),
            ).reshape(-1)
        )
        if physical_coarse.numel() not in (1, output_channels):
            raise ValueError(
                f"Released W4 coarse scale count is invalid for {layer}"
            )
        physical_coarse = physical_coarse.reshape(
            physical_coarse.numel(),
            1,
            1,
            1,
        )

        fine_endpoint = AOV_RELEASED_W4_WEIGHT_FINE_ENDPOINT
        coarse_endpoint = AOV_RELEASED_W4_WEIGHT_COARSE_ENDPOINT
        mantissa_max = AOV_RELEASED_W4_WEIGHT_MANTISSA_MAX
        logical_fine = (
            torch.round(fine_mantissa)
            * coarse_endpoint
            / (fine_endpoint * mantissa_max)
        )
        logical_coarse = (
            physical_coarse * fine_endpoint * mantissa_max / coarse_endpoint
        )
        observer.set_static_vsq_components(
            torch.round(payload).to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            logical_fine.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            logical_coarse.to(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            reference=parameter,
        )
        loaded_layers.append(layer)

    if len(loaded_layers) != AOV_RELEASED_W4_WEIGHT_COUNT:
        raise ValueError(
            "Released AoV W4 collateral does not match the prepared graph: "
            f"{len(loaded_layers)} != {AOV_RELEASED_W4_WEIGHT_COUNT}"
        )
    return {
        "static_w4_weight_count": len(loaded_layers),
        "static_w4_weight_layers": sorted(loaded_layers),
        "static_w4_weight_source": AOV_RELEASED_COLLATERAL_ROOT,
    }


def _initialize_released_weight_observers(
    gm,
    example_args,
    calibration_data,
    args,
):
    if args.calibration_steps <= 0:
        raise ValueError(
            "Released AoV calibration profile requires positive steps"
        )
    if calibration_data is None:
        raise ValueError(
            "Released AoV calibration profile requires calibration data"
        )
    if len(calibration_data) != args.calibration_steps:
        raise ValueError(
            "AoV calibration data count does not match calibration_steps: "
            f"{len(calibration_data)} != {args.calibration_steps}"
        )

    input_channels = example_args[0].shape[1]
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    inputs = calibration_data.preprocess(
        0,
        input_channels,
        torch_dtype,
    )
    with torch.no_grad():
        gm(inputs)


def _validate_released_profile(hardware_config):
    if hardware_config is None or hardware_config.target != POLICY.target:
        raise ValueError(
            "Released AoV calibration status requires its registered target"
        )
    if not os.path.isfile(AOV_RELEASED_COLLATERAL_STATUS):
        raise FileNotFoundError(
            "Released AoV calibration status not found: "
            f"{AOV_RELEASED_COLLATERAL_STATUS}"
        )


def _get_quantization_region_qspec(args):
    if not POLICY.enable_quantization_region_qspec:
        return None

    region_qspec = QuantizationSpec.from_str(POLICY.nongemm_qspec)
    region_qspec.observer_or_fake_quant_ctr = (
        FusedAmaxObsFakeQuantize.with_args(
            force_scale_power_of_two=args.force_scale_power_of_two,
        )
    )
    return region_qspec


def _restore_released_profile(
    gm, example_args, calibration_data, args, hardware_config
):
    _validate_released_profile(hardware_config)
    _initialize_released_weight_observers(
        gm,
        example_args,
        calibration_data,
        args,
    )
    weight_report = _load_released_w4_weight_encoding(gm)
    finalize_pt2e_calibration(gm, hardware_config)
    status = torch.load(
        AOV_RELEASED_COLLATERAL_STATUS,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(status, dict):
        raise ValueError("Released AoV calibration status must be a dictionary")
    profile_report = _load_released_activation_status(
        gm,
        status,
        AOV_RELEASED_COLLATERAL_XSCALE,
    )
    profile_report.update(weight_report)
    profile_report.update(
        {
            "profile": _get_aov_calibration_profile(args),
            "activation_source": AOV_RELEASED_COLLATERAL_STATUS,
            "activation_xscale": AOV_RELEASED_COLLATERAL_XSCALE,
            "initialized_weight_frames": 1,
            "activation_scale_semantics": {
                "released_w8_denominator": (
                    AOV_RELEASED_W8_ACTIVATION_DENOMINATOR
                ),
                "voyager_w8_denominator": (
                    AOV_VOYAGER_W8_ACTIVATION_DENOMINATOR
                ),
                "released_vsq_physical_coarse_denominator": (
                    AOV_RELEASED_VSQ_PHYSICAL_COARSE_DENOMINATOR
                ),
                "native_endpoint_parity": True,
                "note": (
                    "Released maxima and xscale are loaded without "
                    "compensation. Voyager uses the native MPQT "
                    "activation scale endpoints."
                ),
            },
        }
    )
    return profile_report


def quantization_recipe(args, example_args, calibration_data, quantizer):
    from voyager_compiler.targets.magnet_alpha.quantization_flow import (
        AlphaQuantizationRecipe,
    )

    profile = _get_aov_calibration_profile(args)

    def samples():
        if (
            calibration_data is None
            or len(calibration_data) != args.calibration_steps
        ):
            raise ValueError(
                "AoV calibration data count does not match calibration_steps"
            )
        for index in range(args.calibration_steps):
            inputs = calibration_data.preprocess(
                index,
                example_args[0].shape[1],
                torch.bfloat16 if args.bf16 else torch.float32,
            )
            yield (inputs,), {}

    released = profile == AOV_RELEASED_COLLATERAL_CALIBRATION_PROFILE
    return AlphaQuantizationRecipe(
        quantizer=quantizer,
        samples=samples,
        steps=args.calibration_steps,
        bias=args.bias,
        region_qspec=_get_quantization_region_qspec(args),
        recalibrate=POLICY.enable_target_recalibration,
        restore_profile=(
            lambda graph, config: _restore_released_profile(
                graph, example_args, calibration_data, args, config
            )
        )
        if released
        else None,
        expected_exact_boundaries=AOV_RELEASED_COLLATERAL_EXACT_BOUNDARY_COUNT
        if released
        else None,
        profile=profile,
        activation_source=AOV_RELEASED_COLLATERAL_STATUS
        if released
        else "fresh-192-video-frames",
    )


def prepare_calibrated_model(
    model, example_args, quantizer, calibration_data, args, hardware_config
):
    """Compatibility helper for numerical probes; delegates to target PTQ."""
    from voyager_compiler.targets.magnet_alpha.quantization_flow import (
        calibrate_and_convert,
    )

    if not uses_quantization(args):
        raise ValueError("AoV calibrated model requires quantization")
    graph, _ = calibrate_and_convert(
        model,
        example_args,
        {},
        hardware_config,
        quantization_recipe(args, example_args, calibration_data, quantizer),
        quantizer,
    )
    return graph


def checkpoint_format(state: dict) -> str:
    """Return the checkpoint schema, recognizing legacy resume files."""
    checkpoint_type = state.get("format")
    if checkpoint_type in {
        RESUME_CHECKPOINT_FORMAT,
        DEPLOYMENT_CHECKPOINT_FORMAT,
    }:
        return checkpoint_type
    if (
        state.get("format_version") == 1
        and "qat_source" in state
        and "qat_phase" in state
    ):
        return RESUME_CHECKPOINT_FORMAT
    raise ValueError("Unsupported AoV QAT checkpoint schema")


def validate_deployment_checkpoint(
    state,
    *,
    expected_precision_policy,
    expected_preprocessing_version,
    expected_image_size,
) -> None:
    """Validate the shared compile-ready AoV deployment schema."""
    if not isinstance(state, Mapping):
        raise ValueError("AoV QAT deployment snapshot must be a mapping")
    if checkpoint_format(state) != DEPLOYMENT_CHECKPOINT_FORMAT:
        raise ValueError("Expected an AoV QAT deployment checkpoint")
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported AoV QAT deployment snapshot version")

    source_checkpoint = state.get("source_checkpoint")
    if not isinstance(source_checkpoint, (str, Path)):
        raise ValueError(
            "AoV QAT deployment snapshot requires source_checkpoint"
        )
    source_sha256 = state.get("source_checkpoint_sha256")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256) is None
    ):
        raise ValueError("AoV QAT deployment source checksum must be SHA-256")
    if not isinstance(state.get("parameters"), Mapping):
        raise ValueError(
            "AoV QAT deployment snapshot requires parameter tensors"
        )

    provenance = state.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("AoV QAT deployment snapshot requires provenance")
    policy = provenance.get("precision_policy")
    if policy != expected_precision_policy:
        raise ValueError(
            "AoV QAT deployment precision policy differs: "
            f"{policy!r} != {expected_precision_policy!r}"
        )
    if provenance.get("released_weight_sidecars") is not False:
        raise ValueError("AoV QAT deployment snapshot must be sidecar-free")
    if (
        provenance.get("preprocessing_version")
        != expected_preprocessing_version
    ):
        raise ValueError("AoV QAT deployment preprocessing version differs")
    if isinstance(expected_image_size, Real):
        expected_image_sizes = (int(expected_image_size),)
    else:
        expected_image_sizes = tuple(
            int(value) for value in expected_image_size
        )
    if provenance.get("image_size") not in expected_image_sizes:
        raise ValueError(
            f"AoV QAT deployment snapshot must come from "
            f"one of {expected_image_sizes}px QAT"
        )
    fitness = provenance.get("fitness")
    if (
        isinstance(fitness, bool)
        or not isinstance(fitness, Real)
        or not math.isfinite(float(fitness))
    ):
        raise ValueError("AoV QAT deployment snapshot requires finite fitness")


def deployment_source_path(state, checkpoint_path) -> Path:
    """Resolve and checksum the conventional checkpoint under a deployment."""
    source = Path(state["source_checkpoint"])
    checkpoint_path = Path(checkpoint_path).resolve()
    if not source.is_absolute():
        source = checkpoint_path.parent / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"AoV QAT deployment source not found: {source}"
        )

    actual = checkpoint_sha256(source)
    expected = state["source_checkpoint_sha256"].lower()
    if actual != expected:
        raise ValueError(
            "AoV QAT deployment source checksum differs: "
            f"{actual} != {expected}"
        )
    return source


def checkpoint_sha256(path) -> str:
    """Return the SHA-256 digest used by all AoV checkpoint provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        while chunk := checkpoint.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_deployment_parameters(model, state) -> None:
    """Overlay one validated deployment parameter set exactly."""
    expected = dict(model.named_parameters())
    saved = state["parameters"]
    missing = sorted(set(expected) - set(saved))
    unexpected = sorted(set(saved) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "AoV QAT deployment parameter keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    errors = []
    for name, parameter in expected.items():
        value = saved[name]
        if not isinstance(value, torch.Tensor):
            errors.append(f"{name}: not a tensor")
        elif parameter.shape != value.shape:
            errors.append(
                f"{name}: shape {tuple(value.shape)} != "
                f"{tuple(parameter.shape)}"
            )
        elif parameter.dtype != value.dtype:
            errors.append(f"{name}: dtype {value.dtype} != {parameter.dtype}")
        elif not torch.isfinite(value).all():
            errors.append(f"{name}: contains non-finite values")
    if errors:
        raise ValueError(
            "AoV QAT deployment parameters are invalid: " + "; ".join(errors)
        )

    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(saved[name].to(device=parameter.device))


def load_optional_deployment_checkpoint(
    checkpoint_path,
    load_conventional_checkpoint,
    *,
    expected_precision_policy,
    expected_preprocessing_version,
    expected_image_size,
):
    """Load a conventional model, optionally through one deployment snapshot.

    Deployment files wrap exactly one conventional model checkpoint. This
    exact-pair adapter owns schema validation, source checksum resolution,
    and the prohibition on deployment-to-deployment chains.
    """
    resolved = Path(checkpoint_path).resolve()
    checkpoint = torch.load(
        resolved,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not (
        isinstance(checkpoint, Mapping)
        and checkpoint.get("format") == DEPLOYMENT_CHECKPOINT_FORMAT
    ):
        return load_conventional_checkpoint(checkpoint, resolved), None

    validate_deployment_checkpoint(
        checkpoint,
        expected_precision_policy=expected_precision_policy,
        expected_preprocessing_version=expected_preprocessing_version,
        expected_image_size=expected_image_size,
    )
    source_path = deployment_source_path(checkpoint, resolved)
    source_checkpoint = torch.load(
        source_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if (
        isinstance(source_checkpoint, Mapping)
        and source_checkpoint.get("format") == DEPLOYMENT_CHECKPOINT_FORMAT
    ):
        raise ValueError(
            "AoV QAT deployment source must be a conventional checkpoint"
        )
    return (
        load_conventional_checkpoint(source_checkpoint, source_path),
        checkpoint,
    )


def apply_deployment_parameters(
    model,
    state,
    *,
    expected_precision_policy,
    expected_preprocessing_version,
    expected_image_size,
) -> None:
    """Validate and overlay one deployment snapshot on a prepared model."""
    validate_deployment_checkpoint(
        state,
        expected_precision_policy=expected_precision_policy,
        expected_preprocessing_version=expected_preprocessing_version,
        expected_image_size=expected_image_size,
    )
    load_deployment_parameters(model, state)


def recipe_options(options):
    values = dict(
        model_name_or_path=None,
        bf16=False,
        calibration_steps=0,
        conv2d_im2col=True,
        first_layer_input_mode=None,
        aov_image_size=192,
        activation=POLICY.activation_qspec,
        output_activation=POLICY.output_activation_qspec,
        weight=POLICY.weight_qspec,
        bias=POLICY.bias_qspec,
        force_scale_power_of_two=False,
        pt2e=True,
        target=POLICY.target,
        aov_calibration_profile=AOV_DEFAULT_CALIBRATION_PROFILE,
        aov_calibration_video=AOV_CALIBRATION_VIDEO,
        aov_camera_calibration=AOV_CAMERA_CALIBRATION,
    )
    values.update(options)
    return SimpleNamespace(**values)


def build(options):
    settings = recipe_options(options)
    path = (
        Path(settings.model_name_or_path)
        if settings.model_name_or_path
        else _resolve_checkpoint()
    )
    checksum = checkpoint_sha256(path)
    expected = options.get("checkpoint_sha256")
    if expected is not None and checksum != expected:
        raise ValueError(f"YOLO checkpoint checksum mismatch: {path}")
    torch.manual_seed(options.get("seed", 0))
    model, args = load_model(
        settings, torch.bfloat16 if settings.bf16 else torch.float32
    )
    model._voyager_examples = args
    model._voyager_assets = {str(path.resolve()): checksum}
    return model


def example_inputs(model, options):
    return model._voyager_examples, {}


def prepare_inference(model, args, kwargs, options):
    """Supply model/assets and a target recipe; target quantization runs later."""
    from voyager_compiler.ir.graph import PreparedWorkload

    settings = recipe_options(options)
    if settings.target != TARGET:
        raise ValueError(f"this numerical recipe requires {TARGET}")
    options.setdefault(
        "alpha_quantization",
        quantization_recipe(
            settings,
            args,
            options.get("calibration", []),
            configure_quantizer(None, settings),
        ),
    )
    with torch.no_grad():
        graph = export_model(model, args, kwargs)
        reference = graph(*args, **kwargs)
    return PreparedWorkload(
        graph,
        tuple(args),
        dict(kwargs),
        WORKLOAD,
        reference,
        dict(model._voyager_assets, seed=options.get("seed", 0)),
    )


WORKLOAD_SPEC = registry.register(
    WorkloadSpec(WORKLOAD, build, example_inputs, prepare=prepare_inference)
)
