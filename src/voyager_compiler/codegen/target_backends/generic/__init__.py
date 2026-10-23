# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generic target-owned declarations."""

from voyager_compiler.codegen.target_backends.generic.hardware_pipeline import (
    GENERIC_HARDWARE_PIPELINE,
    GENERIC_MATRIX_OPERATIONS,
    GENERIC_QUANTIZE_OPERATIONS,
)

__all__ = [
    "GENERIC_HARDWARE_PIPELINE",
    "GENERIC_MATRIX_OPERATIONS",
    "GENERIC_QUANTIZE_OPERATIONS",
]
