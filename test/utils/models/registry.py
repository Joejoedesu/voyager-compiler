# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Exact workload-target adapter registrations for the test CLI."""

from types import MappingProxyType

from voyager_compiler.codegen import DEFAULT_COMPILATION_TARGET

from . import (
    resnet18_gemmini,
    resnet18_magnet_alpha,
    yolo5_face_nv_magnet_alpha,
)

WORKLOAD_TARGET_REGISTRY = MappingProxyType(
    {
        (
            resnet18_gemmini.WORKLOAD,
            resnet18_gemmini.TARGET,
        ): resnet18_gemmini,
        (
            yolo5_face_nv_magnet_alpha.WORKLOAD,
            yolo5_face_nv_magnet_alpha.TARGET,
        ): yolo5_face_nv_magnet_alpha,
        (
            resnet18_magnet_alpha.WORKLOAD,
            resnet18_magnet_alpha.TARGET,
        ): resnet18_magnet_alpha,
    }
)

_REGISTERED_WORKLOADS = frozenset(
    workload for workload, _ in WORKLOAD_TARGET_REGISTRY
)
_LEGACY_FALLBACK_PAIRS = frozenset(
    {
        (
            resnet18_magnet_alpha.WORKLOAD,
            DEFAULT_COMPILATION_TARGET,
        ),
    }
)


def resolve_workload_target(model: str, target: str):
    """Resolve one exact adapter without inventing target fallback."""

    adapter = WORKLOAD_TARGET_REGISTRY.get((model, target))
    if adapter is not None:
        return adapter
    if (model, target) in _LEGACY_FALLBACK_PAIRS:
        return None
    if model not in _REGISTERED_WORKLOADS:
        return None
    raise ValueError(
        f"No registered workload-target adapter for {model!r} x " f"{target!r}"
    )


__all__ = [
    "WORKLOAD_TARGET_REGISTRY",
    "resolve_workload_target",
]
