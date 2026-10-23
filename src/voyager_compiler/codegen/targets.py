# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compiler-owned compilation-target identity and model binding."""

DEFAULT_COMPILATION_TARGET = "generic"
GEMMINI_COMPILATION_TARGET = "gemmini"
MAGNET_ALPHA_COMPILATION_TARGET = "magnet-alpha"
COMPILATION_TARGETS = (
    DEFAULT_COMPILATION_TARGET,
    GEMMINI_COMPILATION_TARGET,
    MAGNET_ALPHA_COMPILATION_TARGET,
)
COMPILATION_TARGET_META = "voyager_compilation_target"


def resolve_compilation_target(hardware_config=None):
    """Resolve the target solely from the typed hardware configuration."""
    target = (
        DEFAULT_COMPILATION_TARGET
        if hardware_config is None
        else hardware_config.target
    )

    if target not in COMPILATION_TARGETS:
        raise ValueError(f"Unsupported compilation target: {target}")
    return target


def bind_compilation_target(
    model,
    hardware_config=None,
    *,
    stage: str,
):
    """Bind one model to its compilation target across public stages."""
    target = resolve_compilation_target(hardware_config)
    metadata = getattr(model, "meta", None)
    if not isinstance(metadata, dict):
        raise TypeError(
            f"{stage}: compilation requires a GraphModule with metadata"
        )

    missing = object()
    bound_target = metadata.get(COMPILATION_TARGET_META, missing)
    if bound_target is missing:
        metadata[COMPILATION_TARGET_META] = target
    elif bound_target != target:
        raise ValueError(
            f"{stage}: model is bound to compilation target "
            f"{bound_target!r}, not {target!r}"
        )
    return target


__all__ = [
    "COMPILATION_TARGET_META",
    "COMPILATION_TARGETS",
    "DEFAULT_COMPILATION_TARGET",
    "GEMMINI_COMPILATION_TARGET",
    "MAGNET_ALPHA_COMPILATION_TARGET",
    "bind_compilation_target",
    "resolve_compilation_target",
]
