# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Adapter for the exact ResNet-18 x Gemmini W-stage pairing."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.fx import GraphModule, Node
from torchvision.models import resnet18

from voyager_compiler import (
    compile,
    convert_pt2e,
    export_model,
    finalize_pt2e_calibration,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from voyager_compiler.codegen import (
    DEFAULT_EXECUTION_ORDER_MODE,
    GEMMINI_COMPILATION_TARGET,
    GEMMINI_HARDWARE_CONFIG,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    get_physical_dtype,
)
from voyager_compiler.codegen.target_backends.gemmini.lowering import (
    GemminiWorkloadRegion,
    gemmini_workload_add_regions,
    gemmini_workload_average_regions,
    gemmini_workload_linear_regions,
    gemmini_workload_regions,
)

from .utils import get_compile_args

WORKLOAD = "resnet18"
TARGET = GEMMINI_COMPILATION_TARGET
PROFILE_ID = "resnet18-gemmini-w-20260818-v1"
SEED = 20260818
INPUT_SHAPE = (1, 3, 224, 224)
CALIBRATION_STEPS = 2
COMPILER_THREADS = 8
QSPEC = "int8,qs=per_tensor_symmetric,ahl=1,qmin=-128,qmax=127,rnd=nearest"
BIAS_QSPEC = "int32"
CONV_REGION_COUNT = 20
ADD_REGION_COUNT = 8
AVERAGE_REGION_COUNT = 1
LINEAR_REGION_COUNT = 1
RELU_DOMAIN_ALIGNMENT_COUNT = 17
SOURCE_OPERATION_DENOMINATOR = 69
POST_FOLD_OPERATION_DENOMINATOR = 49
CAPTURE_PRODUCER = "voyager-resnet18-gemmini-w-v1"

_UNSUPPORTED_BOOLEAN_OPTIONS = (
    "compile_single_layer",
    "conv2d_im2col",
    "disable_reshape_fusion",
    "dump_dataset",
    "enable_mixed_precision",
    "evaluate",
    "quantize_attention_mask",
    "quantize_fc",
    "split_spmm",
    "transform_layout",
    "transpose_fc",
    "use_maxpool_2x2",
)


@dataclass(frozen=True)
class ResNet18GemminiRunResult:
    """Typed result returned to the shared test-codegen dispatcher."""

    workload: str
    target: str
    profile_id: str
    graph_module: Any
    reference_output: Any
    transformed_output: Any
    preprocess_fn: Any
    imprecise_pt2e: bool


@dataclass(frozen=True)
class _CaptureSpec:
    """Final-FX authority for one live Conv-region capture."""

    module_name: str
    submodule: GraphModule
    region_name: str
    input_logical_dtype: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dtype_name(value: object) -> str:
    name = str(value)
    return name.removeprefix("torch.")


def _node_shape(node: Node, label: str) -> tuple[int, ...]:
    candidates = (
        getattr(node, "shape", None),
        getattr(getattr(node, "value", None), "shape", None),
        getattr(node.meta.get("val"), "shape", None),
        getattr(node.meta.get("tensor_meta"), "shape", None),
    )
    for shape in candidates:
        if shape is not None:
            return tuple(int(dimension) for dimension in shape)
    raise RuntimeError(f"{label} has no final-FX tensor shape")


def _node_public_dtype(node: Node, label: str) -> str:
    candidates = (
        getattr(getattr(node, "value", None), "dtype", None),
        getattr(node.meta.get("val"), "dtype", None),
        getattr(node.meta.get("tensor_meta"), "dtype", None),
        node.meta.get("dtype"),
    )
    for dtype in candidates:
        if dtype is not None:
            return _dtype_name(dtype)
    raise RuntimeError(f"{label} has no final-FX public dtype")


def _capture_spec(region: GemminiWorkloadRegion) -> _CaptureSpec:
    """Derive the runtime logical domain from validated region semantics."""

    if not region.outer.args or not isinstance(region.outer.args[0], Node):
        raise RuntimeError(f"{region.outer.name} has no tensor ingress node")
    input_node = region.outer.args[0]
    input_logical_dtype = (
        "float32" if region.input_quantize is not None else "int8"
    )
    actual_input_dtype = (
        _node_public_dtype(input_node, f"{region.outer.name} input")
        if input_logical_dtype == "float32"
        else _dtype_name(get_physical_dtype(input_node))
    )
    if actual_input_dtype != input_logical_dtype:
        raise RuntimeError(
            f"{region.outer.name} final-FX input domain is "
            f"{actual_input_dtype!r}, expected {input_logical_dtype!r}"
        )
    actual_output_dtype = _dtype_name(get_physical_dtype(region.outer))
    if actual_output_dtype != "int8":
        raise RuntimeError(
            f"{region.outer.name} final-FX output domain is "
            f"{actual_output_dtype!r}, expected 'int8'"
        )
    return _CaptureSpec(
        module_name=str(region.outer.target),
        submodule=region.submodule,
        region_name=region.outer.name,
        input_logical_dtype=input_logical_dtype,
        input_shape=_node_shape(input_node, f"{region.outer.name} input"),
        output_shape=_node_shape(region.outer, f"{region.outer.name} output"),
    )


def _validate_capture_tensor(
    value: torch.Tensor,
    *,
    logical_dtype: str,
    shape: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"{label} shape {tuple(value.shape)!r} differs from {shape!r}"
        )
    numeric = value.detach().cpu().contiguous().to(torch.float32)
    if not torch.isfinite(numeric).all():
        raise RuntimeError(f"{label} contains non-finite values")
    if not torch.count_nonzero(numeric):
        raise RuntimeError(f"{label} is degenerate")
    if logical_dtype == "int8":
        if not torch.equal(numeric, torch.round(numeric)):
            raise RuntimeError(f"{label} is not integral INT8 code data")
        if int(numeric.min()) < -128 or int(numeric.max()) > 127:
            raise RuntimeError(f"{label} is outside signed INT8 range")
    elif logical_dtype != "float32":
        raise RuntimeError(f"{label} has unsupported logical dtype")
    return numeric


def _write_capture_tensor(
    root: Path,
    relative: str,
    value: torch.Tensor,
    *,
    logical_dtype: str,
) -> dict[str, Any]:
    array = value.numpy()
    path = root / relative
    path.write_bytes(array.astype("<f4", copy=False).tobytes(order="C"))
    return {
        "path": relative,
        "sha256": _sha256(path),
        "container_dtype": "float32",
        "logical_dtype": logical_dtype,
        "shape": list(array.shape),
    }


def _effective_args(args):
    effective = copy.copy(args)
    if effective.model != WORKLOAD or effective.target != TARGET:
        raise ValueError(
            "Adapter identity does not match the requested exact pair: "
            f"{effective.model!r} x {effective.target!r}"
        )
    if effective.model_name_or_path not in (None, "NONE"):
        raise ValueError(
            "ResNet-18 x Gemmini uses deterministic weights=None; "
            "--model_name_or_path must be omitted or NONE"
        )
    if effective.bf16:
        raise ValueError("ResNet-18 x Gemmini requires float32 export")
    if effective.bufferize:
        raise ValueError("ResNet-18 x Gemmini uses operator-graph compilation")
    enabled_unsupported = tuple(
        name
        for name in _UNSUPPORTED_BOOLEAN_OPTIONS
        if bool(getattr(effective, name, False))
    )
    if enabled_unsupported:
        formatted = ", ".join(f"--{name}" for name in enabled_unsupported)
        raise ValueError(
            "ResNet-18 x Gemmini does not consume these legacy CLI "
            f"options: {formatted}"
        )
    if effective.scratchpad_mapping != PER_OPERATION_SCRATCHPAD_MAPPING:
        raise ValueError(
            "ResNet-18 x Gemmini requires per_operation scratchpads"
        )
    if effective.execution_order != DEFAULT_EXECUTION_ORDER_MODE:
        raise ValueError("ResNet-18 x Gemmini requires FX execution order")
    if effective.force_scale_power_of_two:
        raise ValueError("ResNet-18 x Gemmini preserves calibrated scales")
    if effective.residual is not None:
        raise ValueError("ResNet-18 x Gemmini does not override residual Q/DQ")
    if effective.calibration_steps not in (0, CALIBRATION_STEPS):
        raise ValueError(
            "ResNet-18 x Gemmini owns an exact two-execution calibration"
        )
    if effective.local_scheduler is not None:
        raise ValueError(
            "ResNet-18 x Gemmini delegates local control to stock helpers"
        )
    if effective.total_memory is not None:
        raise ValueError(
            "ResNet-18 x Gemmini owns its canonical DRAM allocation; "
            "do not pass --total_memory"
        )
    if effective.imprecise_pt2e is True:
        raise ValueError(
            "ResNet-18 x Gemmini requires its exact semantic-lowering gate"
        )

    qspecs = {
        "activation": QSPEC,
        "output_activation": QSPEC,
        "weight": QSPEC,
        "bias": BIAS_QSPEC,
    }
    for name, value in qspecs.items():
        supplied = getattr(effective, name)
        if supplied is not None and supplied != value:
            raise ValueError(
                f"CLI {name} qspec conflicts with {PROFILE_ID!r}: "
                f"{supplied!r} != {value!r}"
            )
        setattr(effective, name, value)

    effective.pt2e = True
    effective.calibration_steps = CALIBRATION_STEPS
    effective.dump_tensors = True
    effective.model_name_or_path = "NONE"
    effective.transform_layout = False
    effective.transpose_fc = False
    return effective


def _require_fresh_output(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path must be a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be fresh: {path}")


def _workload_capture_specs(model: GraphModule) -> tuple[_CaptureSpec, ...]:
    conv_regions = gemmini_workload_regions(model)
    counts = {
        "conv": len(conv_regions),
        "add": len(gemmini_workload_add_regions(model)),
        "average": len(gemmini_workload_average_regions(model)),
        "linear": len(gemmini_workload_linear_regions(model)),
    }
    expected = {
        "conv": CONV_REGION_COUNT,
        "add": ADD_REGION_COUNT,
        "average": AVERAGE_REGION_COUNT,
        "linear": LINEAR_REGION_COUNT,
    }
    if counts != expected:
        raise RuntimeError(
            f"Gemmini workload region partition {counts!r} differs from "
            f"{expected!r}"
        )
    specs = tuple(_capture_spec(region) for region in conv_regions)
    if len({spec.module_name for spec in specs}) != len(specs):
        raise RuntimeError("Gemmini Conv capture module names are not unique")
    if len({spec.region_name for spec in specs}) != len(specs):
        raise RuntimeError("Gemmini Conv capture region names are not unique")
    histogram = {
        dtype: sum(spec.input_logical_dtype == dtype for spec in specs)
        for dtype in ("float32", "int8")
    }
    if histogram != {"float32": 1, "int8": 19}:
        raise RuntimeError(
            f"Gemmini Conv ingress-domain partition differs: {histogram!r}"
        )
    return specs


def _capture_regions(
    model: GraphModule,
    example: torch.Tensor,
    specs: tuple[_CaptureSpec, ...],
) -> tuple[Any, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    by_module = {spec.module_name: spec for spec in specs}
    captures: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    modules = dict(model.named_modules(remove_duplicate=False))
    missing = sorted(set(by_module) - set(modules))
    if missing:
        raise RuntimeError(
            f"Gemmini Conv capture modules are missing: {missing}"
        )

    for module_name, spec in by_module.items():
        module = modules[module_name]
        if module is not spec.submodule:
            raise RuntimeError(
                f"{spec.region_name} capture module identity differs"
            )

        def capture(
            _module,
            inputs,
            kwargs,
            result,
            *,
            capture_spec=spec,
        ):
            if kwargs or len(inputs) == 0:
                raise RuntimeError(
                    f"{capture_spec.region_name} has no positional ingress"
                )
            if capture_spec.region_name in captures:
                raise RuntimeError(
                    f"{capture_spec.region_name} executed more than once"
                )
            if not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(
                    f"{capture_spec.region_name} ingress is not a tensor"
                )
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(
                    f"{capture_spec.region_name} output is not a tensor"
                )
            input_value = _validate_capture_tensor(
                inputs[0],
                logical_dtype=capture_spec.input_logical_dtype,
                shape=capture_spec.input_shape,
                label=f"{capture_spec.region_name} input",
            )
            output_value = _validate_capture_tensor(
                result,
                logical_dtype="int8",
                shape=capture_spec.output_shape,
                label=f"{capture_spec.region_name} output",
            )
            captures[capture_spec.region_name] = (
                input_value.clone(),
                output_value.clone(),
            )

        handles.append(module.register_forward_hook(capture, with_kwargs=True))
    try:
        with torch.inference_mode():
            output = model(example)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != {spec.region_name for spec in specs}:
        raise RuntimeError("Gemmini Conv capture set is incomplete")
    return output, captures


def _write_capture(
    output_dir: Path,
    captures: dict[str, tuple[torch.Tensor, torch.Tensor]],
    specs: tuple[_CaptureSpec, ...],
) -> dict[str, Any]:
    capture_dir = output_dir / "runtime-capture"
    capture_dir.mkdir()
    by_region = {spec.region_name: spec for spec in specs}
    regions = {}
    for region_name, (input_value, output_value) in sorted(captures.items()):
        spec = by_region[region_name]
        regions[region_name] = {
            "input": _write_capture_tensor(
                capture_dir,
                f"{region_name}.input.bin",
                input_value,
                logical_dtype=spec.input_logical_dtype,
            ),
            "output_codes": _write_capture_tensor(
                capture_dir,
                f"{region_name}.output-codes.bin",
                output_value,
                logical_dtype="int8",
            ),
        }
    manifest = {
        "schema_version": 1,
        "producer": CAPTURE_PRODUCER,
        "canonical_model": {
            "path": "model.txt",
            "sha256": _sha256(output_dir / "model.txt"),
        },
        "regions": regions,
    }
    manifest_path = capture_dir / "runtime-capture.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_profile(
    output_dir: Path,
    manifest: dict[str, Any],
    model: GraphModule,
    semantic_comparison: dict[str, Any],
) -> None:
    graph_path = output_dir / "final_fx.txt"
    graphs = []
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, GraphModule):
            graphs.append(f"## {name or '<root>'}\n{module.graph}")
    graph_path.write_text("\n\n".join(graphs) + "\n", encoding="utf-8")
    capture_path = output_dir / "runtime-capture" / "runtime-capture.json"
    profile_path = output_dir / "gemmini-resnet18-profile.json"
    inventory_path = output_dir / "artifact-inventory.json"
    inventory_entries = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path in {profile_path, inventory_path}:
            continue
        if path.is_symlink():
            raise RuntimeError(
                f"Gemmini collateral cannot be a symlink: {path}"
            )
        relative = str(path.relative_to(output_dir))
        inventory_entries[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    required = {
        "final_fx.txt",
        "hardware_config.json",
        "layers.txt",
        "model.txt",
        "resnet18.svg",
        "runtime-capture/runtime-capture.json",
        "storage_plan.json",
    }
    missing = sorted(required - inventory_entries.keys())
    if missing:
        raise RuntimeError(f"Gemmini collateral inventory is missing {missing}")
    tensor_file_count = sum(
        relative.startswith("tensor_files/") for relative in inventory_entries
    )
    capture_payload_count = sum(
        relative.startswith("runtime-capture/")
        and relative != "runtime-capture/runtime-capture.json"
        for relative in inventory_entries
    )
    if tensor_file_count == 0 or capture_payload_count != 40:
        raise RuntimeError(
            "Gemmini collateral inventory has an invalid tensor/capture count"
        )
    inventory = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "file_count": len(inventory_entries),
        "tensor_file_count": tensor_file_count,
        "capture_payload_count": capture_payload_count,
        "files": inventory_entries,
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profile = {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "workload": WORKLOAD,
        "target": TARGET,
        "generator_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
        },
        "seed": SEED,
        "input_shape": list(INPUT_SHAPE),
        "input_distribution": "uniform[-1,1]",
        "calibration_executions": CALIBRATION_STEPS,
        "compiler_threads": COMPILER_THREADS,
        "qspec": QSPEC,
        "bias_qspec": BIAS_QSPEC,
        "fold_conv_bn": True,
        "transform_layout": False,
        "source_operation_denominator": SOURCE_OPERATION_DENOMINATOR,
        "post_fold_operation_denominator": POST_FOLD_OPERATION_DENOMINATOR,
        "helper_region_partition": {
            "conv": CONV_REGION_COUNT,
            "add": ADD_REGION_COUNT,
            "average": AVERAGE_REGION_COUNT,
            "linear": LINEAR_REGION_COUNT,
        },
        "capture_input_logical_dtype_counts": {
            dtype: sum(
                binding["input"]["logical_dtype"] == dtype
                for binding in manifest["regions"].values()
            )
            for dtype in ("float32", "int8")
        },
        "capture_output_logical_dtype_counts": {
            "int8": len(manifest["regions"])
        },
        "semantic_transform_comparison": semantic_comparison,
        "artifacts": {
            "model_txt_sha256": manifest["canonical_model"]["sha256"],
            "runtime_capture_sha256": _sha256(capture_path),
            "final_fx_sha256": _sha256(graph_path),
            "collateral_inventory_path": inventory_path.name,
            "collateral_inventory_sha256": _sha256(inventory_path),
            "collateral_inventory_file_count": len(inventory_entries),
        },
    }
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exact_semantic_comparison(
    reference: object,
    transformed: object,
) -> dict[str, Any]:
    if not isinstance(reference, torch.Tensor) or not isinstance(
        transformed,
        torch.Tensor,
    ):
        raise RuntimeError("Gemmini exact pair must return one tensor")
    if (
        reference.dtype != transformed.dtype
        or reference.shape != transformed.shape
    ):
        raise RuntimeError("Gemmini transformed output contract differs")
    difference = torch.abs(reference - transformed)
    mismatch_count = int(torch.count_nonzero(difference))
    maximum = float(torch.amax(difference).item())
    if mismatch_count != 0:
        raise RuntimeError(
            "Gemmini transform changed the frozen workload output: "
            f"{mismatch_count}/{reference.numel()} values, max {maximum}"
        )
    return {
        "element_count": reference.numel(),
        "exact_mismatch_count": mismatch_count,
        "maximum_absolute_error": maximum,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "status": "passed",
    }


def run(
    args,
    *,
    vector_stages,
    hardware_config,
    scheduler_backend,
):
    """Compile and capture deterministic ResNet-18 Gemmini collateral."""

    del vector_stages
    effective = _effective_args(args)
    if hardware_config is not GEMMINI_HARDWARE_CONFIG:
        raise ValueError(
            "ResNet-18 x Gemmini requires the registered hardware profile"
        )
    if scheduler_backend is not None:
        raise ValueError("ResNet-18 x Gemmini does not use a local scheduler")

    requested_output = Path(effective.model_output_dir)
    _require_fresh_output(requested_output)
    output_dir = requested_output.resolve()

    torch.set_num_threads(COMPILER_THREADS)
    torch.manual_seed(SEED)
    source = resnet18(weights=None).eval()
    example = torch.empty(INPUT_SHAPE).uniform_(-1.0, 1.0)
    graph = export_model(source, (example,), strict=False)
    quantizer = get_default_quantizer(
        input_activation=QSPEC,
        output_activation=QSPEC,
        weight=QSPEC,
        bias=BIAS_QSPEC,
    )
    graph = prepare_pt2e(
        graph,
        quantizer,
        hardware_config=hardware_config,
        fold_conv_bn=True,
    )
    with torch.inference_mode():
        for _ in range(CALIBRATION_STEPS):
            graph(example)
    finalize_pt2e_calibration(graph, hardware_config=hardware_config)
    alignment_count = graph.meta.get("gemmini_relu_domain_alignment_count")
    if alignment_count != RELU_DOMAIN_ALIGNMENT_COUNT:
        raise RuntimeError(
            "Gemmini ReLU-domain calibration covered "
            f"{alignment_count!r} sites, expected "
            f"{RELU_DOMAIN_ALIGNMENT_COUNT}"
        )
    graph = convert_pt2e(
        graph,
        calibration_steps=CALIBRATION_STEPS,
        hardware_config=hardware_config,
    )
    with torch.inference_mode():
        reference_output = graph(example)

    graph = transform(
        graph,
        (example,),
        hardware_config=hardware_config,
        transform_layout=False,
    )
    specs = _workload_capture_specs(graph)
    transformed_output, captures = _capture_regions(graph, example, specs)
    semantic_comparison = _exact_semantic_comparison(
        reference_output,
        transformed_output,
    )

    compile_args = get_compile_args(
        effective,
        hardware_config=hardware_config,
        scheduler_backend=scheduler_backend,
    )
    compile(graph, (example,), **compile_args)
    model_txt = output_dir / "model.txt"
    tensor_files = output_dir / "tensor_files"
    if not model_txt.is_file() or not tensor_files.is_dir():
        raise RuntimeError(
            "Gemmini compilation did not emit canonical collateral"
        )
    manifest = _write_capture(output_dir, captures, specs)
    _write_profile(
        output_dir,
        manifest,
        graph,
        semantic_comparison,
    )

    return ResNet18GemminiRunResult(
        workload=WORKLOAD,
        target=TARGET,
        profile_id=PROFILE_ID,
        graph_module=graph,
        reference_output=reference_output,
        transformed_output=transformed_output,
        preprocess_fn=None,
        imprecise_pt2e=False,
    )


__all__ = [
    "PROFILE_ID",
    "ResNet18GemminiRunResult",
    "TARGET",
    "WORKLOAD",
    "run",
]
