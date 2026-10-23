# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pin the generic target's source-owned hardware-pipeline matcher."""

import hashlib
import json
import unittest

from voyager_compiler.codegen.target_backends import (
    get_target_hardware_pipeline,
)
from voyager_compiler.codegen.target_backends.generic import (
    GENERIC_HARDWARE_PIPELINE,
)

# Filled from the exact historical test_codegen matcher after moving it into
# the generic target. Changing this fingerprint means changing the generic
# target's fusion behavior and requires deliberate collateral revalidation.
GENERIC_HARDWARE_PIPELINE_FINGERPRINT = "b52bac0a4f736889"


def _matcher_inventory(pipeline):
    return [
        [
            {
                "targets": sorted(str(target) for target in stage.targets),
                "predicate": getattr(stage.predicate, "__name__", None),
            }
            for stage in pattern
        ]
        for pattern in pipeline
    ]


def _fingerprint(pipeline):
    payload = json.dumps(
        _matcher_inventory(pipeline),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class GenericHardwarePipelineTest(unittest.TestCase):
    def test_registry_returns_named_generic_pipeline(self):
        self.assertIs(
            get_target_hardware_pipeline("generic"),
            GENERIC_HARDWARE_PIPELINE,
        )

    def test_historical_pipeline_shape_is_preserved(self):
        self.assertEqual(
            tuple(len(pattern) for pattern in GENERIC_HARDWARE_PIPELINE),
            (6, 4, 5, 2),
        )

    def test_historical_pipeline_fingerprint_is_pinned(self):
        self.assertEqual(
            _fingerprint(GENERIC_HARDWARE_PIPELINE),
            GENERIC_HARDWARE_PIPELINE_FINGERPRINT,
        )


if __name__ == "__main__":
    unittest.main()
