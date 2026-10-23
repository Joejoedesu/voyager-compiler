# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voyager compiler test support package."""

import sys

from . import verify_help

# Model helpers also support direct execution from ``test/test_codegen.py``,
# where this module historically appears as a top-level import.
sys.modules.setdefault("verify_help", verify_help)
