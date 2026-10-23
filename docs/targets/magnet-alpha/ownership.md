<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Magnet Alpha adoption ownership constraints

Read this file only when an adoption or refactor changes the existing Magnet
Alpha target. The package predates the generic role split and is not safe for
parallel conceptual-role edits:

- `src/voyager_compiler/codegen/target_backends/magnet_alpha/quantization.py`
  owns PTQ/QAT preparation, calibration lifecycle, cleanup, and the ordered
  target-lowering pipeline, but not feature-specific converted-graph
  rewrites;
- `pe_lowering.py`, `nmp_lowering.py`, `movement_lowering.py`,
  `residual_lowering.py`, `boundary_lowering.py`, and `pool_lowering.py` own
  their respective converted-graph rewrites, while `lowering_utils.py`
  contains graph facts and construction helpers shared by those rewrites;
- `pe_lowering.py` also owns physical VSQ operand finalization and proved PPU
  ReLU folding; `movement_lowering.py` is restricted to executable physical
  movement rewrites such as signed-INT8 nearest-neighbor replication;
- `pe_fake_epilogue.py` and `nmp_fake_epilogue.py` own temporary mutable
  calibration simulation; `operators.py` owns executable target-IR operator
  semantics; and `fixed_point.py` owns their shared stateless instruction
  encoding, range proofs, and bit-accurate arithmetic;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/graph.py` combines
  structural recognition, channel-shuffle graph rewrites, and late
  storage-mover insertion;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/lowering.py`
  binds the existing operation classification after the late
  graph rewrite; final FX remains the semantic instruction carrier;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/hardware_pipeline.py`
  owns Alpha's `OpMatcher` semantic hardware paths and the inventory from
  which required composite execution patterns are derived;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/traits.py`
  combines quantization transfer, padding/tiling traits, operand roles, and
  mapper policy;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/memory.py` owns
  storage relations, physical allocation, addresses, and validation, while
  `memory_plan.py` owns the typed graph-analysis plan consumed by allocation;
- `src/voyager_compiler/codegen/target_backends/magnet_alpha/serialization.py`
  serializes and validates canonical Magnet Alpha compiler facts, including
  the scheduler-owned GEMM named factors, while `conversion/magnet-alpha/`
  owns target route, disposition, Q/DQ absorption, descriptor, epilogue, skip,
  encoding policy, and independent handoff checks.

Workload routing now has an explicit staged boundary:

- `test/utils/models/yolo5_face_nv_magnet_alpha.py` is the single combined
  adapter for the exact `yolo5-face-nv` × `magnet-alpha` tuple. It owns the
  AoV model, data, preprocessing, reference, PTQ, calibration, and compile
  orchestration plus every pair-specific qspec, projection, recalibration, and
  comparison rule;
- `test/utils/models/utils.py::get_hardware_config` contains shared test-side
  target selection; and
- `test/utils/models/registry.py` owns validated mapping construction and exact
  tuple-to-adapter registration while `test/test_codegen.py` is the neutral
  parser/resolver/caller. Unregistered branches remain legacy compatibility
  paths and must not acquire new Magnet Alpha policy.

Assign the combined AoV × Magnet Alpha file to `workload-target-overlay` as
one whole-file mutation window. Do not require a separate neutral adapter and
policy-only overlay. Extract a target-independent helper only after another
pair actually shares the same code. Integration owns tuple selection, registry
construction, and `WORKLOAD_TARGET_REGISTRY` entries.
Serialize shared-registry edits.

This existing pair genuinely requires a combined module; that does not impose
one Python file on every future exact pair. Exact tuples remain routing and
evidence units, while a tuple with no pair-specific behavior may select a
reusable adapter.

The maintained open-source boundary ends at compiler serialization and
fail-closed collateral conversion. These checks do not establish external
execution, a W or X workload result, latency, energy, or accuracy.

The exact-pair compile lane provides behavioral dispatch coverage. A dedicated
test that only repeats registry entries is not an ownership or regression
requirement, and compile dispatch does not count as operation-execution
evidence.

Physical shuffle copies remain late by design: the existing memory-rewrite
callback inserts explicit `magnet_alpha::channel_shuffle_copy` nodes at the
start of memory mapping, before execution order, storage planning, and
scheduling. This target-local separation adds no shared hook, IR, or protobuf
field.

When lowering one operation into several accelerator calls, use
`multi-call-tiling` ownership. Represent the calls and result construction in
final FX and `model.txt`; record split-specific details only when required for
correctness. Do not assign local loop factors, Alpha banks, or addresses to
that role. Because `graph.py`, `hardware_pipeline.py`, and `traits.py` jointly
define graph recognition, composite admission, and target traits, serialize
their mutation with composite and storage owners.

Assign an entire affected file to one worker and serialize later roles when
legacy responsibilities still overlap. Do not create parallel write orders
from the generic role names. Any further extraction must preserve current
behavior through the Magnet Alpha regression contract and must not move target
semantics into shared compiler code.
