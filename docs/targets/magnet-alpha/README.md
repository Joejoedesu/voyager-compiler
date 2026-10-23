<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Magnet Alpha target backend

This directory documents Magnet Alpha-specific compiler behavior. The shared
[compiler architecture](../../architecture.md) defines when Voyager invokes a
target backend; this document maps those extension points to their owners.

## Implementation binding

| Target responsibility | Magnet Alpha owner |
| --- | --- |
| Hardware profile | `target_backends/magnet_alpha/hardware.py` |
| Hardware pipeline and operator traits | `target_backends/magnet_alpha/hardware_pipeline.py`, `traits.py` |
| PTQ lifecycle and target lowering | `target_backends/magnet_alpha/quantization.py` |
| Fixed-point arithmetic and target operators | `target_backends/magnet_alpha/fixed_point.py`, `operators.py` |
| Graph rewrites and physical movers | `target_backends/magnet_alpha/graph.py`, `lowering.py` |
| Memory planning and allocation | `target_backends/magnet_alpha/memory_plan.py`, `memory.py` |
| Serialization validation | `target_backends/magnet_alpha/serialization.py` |
| Rule-based scheduling | `scheduler/magnet_alpha_scheduler/` |
| Alpha collateral conversion | `conversion/magnet-alpha/` |

Magnet Alpha uses the shared compiler lifecycle:

1. PT2E preparation installs target-owned calibration and fake-forward state.
2. Conversion leaves executable reference quantized operators.
3. Target lowering creates explicit Alpha operators and operands.
4. Memory rewriting inserts physical movers before execution ordering,
   storage planning, and scheduling.
5. Canonical serialization emits final FX semantics in `model.txt`.
6. The standalone converter translates supported serialized operations into
   Alpha collateral without choosing schedules or changing numerical meaning.

Compilation and collateral conversion are the maintained open-source
boundaries. They do not establish external target execution, latency, energy,
accuracy, or accelerator call counts.

## Workload-specific policy

The YOLO precision table, preprocessing, calibration data, accuracy metric,
and optional QAT recipe remain outside the target backend. See the
[YOLO compiler and conversion example](../../../examples/yolo5_face_nv_magnet_alpha/README.md).

## Integration contracts

- [Regression](regression.md) defines the compiler and converter checks.
- [Operation inventory](operations.yaml) records compiler and converter
  support without claiming external execution.
- [Ownership constraints](ownership.md) identify legacy files that combine
  several adoption responsibilities.

These contracts are Magnet Alpha-specific and are not defaults for another
backend.
