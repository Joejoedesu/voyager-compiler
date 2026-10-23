<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Workload × target runner hierarchy

This document records the migration that keeps `test/test_codegen.py` as a
small parser/resolver/caller and makes the exact `(workload, target)` tuple the
unit of routing, policy, measurement, and evidence. A tuple has one explicit
executable adapter, but it needs a new pair-named Python module only when its
behavior is genuinely pair-specific. Run-specific workload and target
instructions belong in the invoking guide, not this base contract.

## Current base-repository slice

Three exact cells are registered: Gemmini x ResNet-18, Magnet Alpha x YOLO,
and Magnet Alpha x ResNet-18.

```text
test/test_codegen.py
  parse common CLI
  resolve the exact workload × target adapter
  build common hardware and scheduler context
  call adapter.run(...)
  retain unchanged legacy branches for unregistered cells

test/utils/models/registry.py
  WORKLOAD_TARGET_REGISTRY

test/utils/models/yolo5_face_nv_magnet_alpha.py
  model and checkpoint loading
  data, preprocessing, reference, PTQ, calibration, transform, and compile
  target qspec/profile/projection/comparison behavior
  typed result with workload, target, and profile identity

test/utils/models/resnet18_{gemmini,magnet_alpha}.py
  exact-pair workload policy and target-specific compile orchestration

```

The canonical registration maps the exact tuple directly to that adapter:

```python
WORKLOAD_TARGET_REGISTRY = {
    ("resnet18", "gemmini"): resnet18_gemmini,
    ("yolo5-face-nv", "magnet-alpha"): yolo5_face_nv_magnet_alpha,
    ("resnet18", "magnet-alpha"): resnet18_magnet_alpha,
}
```

`yolo5_face_nv_magnet_alpha.py` directly owns the established pair-specific
flow. It is not split into a mandatory neutral YOLO adapter plus a policy-only
overlay: model preparation, calibration, and target policy remain together.

The registry resolves exact cells; it does not infer a target from a workload
or use a target name as proof of support. Two cells may deliberately reference
the same reusable adapter while retaining separate policy selection and
evidence. An unknown named tuple fails instead of inheriting another cell's
behavior or implying CPU support, with one narrow exception: a pair listed in
`_LEGACY_FALLBACK_PAIRS` returns `None` and takes the legacy route even though
its workload is registered. That list is a staged-migration escape hatch, not a
resolution rule; every entry needs a removal condition. Unregistered workloads retain their legacy
route during staged migration; that compatibility path is not support evidence
for any tuple.

The deliberate `("resnet18", "generic")` fallback remains a compiler-native
legacy path. It has no concrete target injection point and therefore carries
no R/W/X adoption claim.

Repository modules use the stable `test.utils...` identity. The external YOLO
fork imports its own top-level `utils` package and the loader intentionally
evicts `utils.*`; using `test.utils` prevents registry duplication or eviction.
Direct CLI execution adds the repository root before those imports.

## Adapter boundary

A pair-specific workload × target adapter may own:

- reproducible model, checkpoint, data, and preprocessing behavior;
- reference, PTQ, calibration, evaluation, transform, and compile orchestration;
- the pair's qspec, mixed-precision, projection, and comparison policy; and
- a typed result that reports the resolved workload, target, and profile.

It must not implement target operators, lowering, scheduling, mapping,
serialization, collateral conversion, or runtime launch. Those remain with
the target/compiler owners. It must not construct or silently modify a
`HardwareConfig`; R establishes the default-derived profile under the
hardware-profile role, and the workload adapter validates and consumes it.

Here “workload adapter” is the model/PTQ runner above. A target adapter is the
general Voyager-to-target integration boundary, while a converter is the
specific component that turns canonical Voyager output into injection
collateral.

Selecting `magnet-alpha` alone does not enable PTQ. The exact registered
adapter applies its defaults only after PT2E or an explicit qspec requests
quantization, and rejects a conflicting CLI qspec instead of silently changing
the named profile.

A target-independent workload helper or adapter is optional. Extract one only
after two or more exact tuples genuinely share executable model/data/PTQ
behavior. A helper may remain an implementation detail imported by pair
adapters. A complete reusable adapter may be registered directly for multiple
exact tuples when all target differences are already represented by explicit
resolved policy; neither form is a required base layer.

## CLI boundary

The CLI owns argument definitions, common validation, exact-tuple resolution,
common hardware/scheduler context, `adapter.run(...)`, and final comparison
reporting. It does not import individual pair modules or contain workload- or
target-specific branches for migrated cells.

Registered adapters return a typed result with graph module, reference output,
transformed output, preprocessor, comparison mode, and the resolved
workload/target/profile identity. The CLI reports that identity before
comparison, and regression evidence obtains the same profile identity from the
registered adapter rather than duplicating it. Adapter reuse never merges
evidence: calibration, assignment, execution, coverage, and performance remain
reported for the exact tuple that ran.

Unregistered torchvision, BERT, LLM, ViT, YOLO5, MobileViT, and Mamba branches
remain compatibility fallbacks during staged migration. Moving them is
separate work; a cell migration must not hide a large unrelated relocation.

## Future workload slices

Each migrated cell gets one explicit registry entry. Add a combined entry
module only when it owns real pair-specific behavior:

```text
test/utils/models/<workload>_<target>.py
  load model and frozen data
  reference/PTQ flow and evaluation
  exact target qspec/profile/projection/comparison behavior
```

Four ResNet-18 cells therefore do not automatically require four Python files.
If all four use the same frozen model/data/reference flow, the exact tuples may
route to one reusable ResNet adapter and explicit target policies. A pair module
is added only for an inseparable difference such as target-specific model
projection, calibration, mixed precision, or comparison behavior:

```text
test/utils/models/resnet18.py                    # reusable, if complete
test/utils/models/resnet18_magnet_alpha.py       # exists: Alpha-specific
```

`resnet18_magnet_alpha.py` was registered as a combined pair adapter. Record
which inseparable difference justified that, so a later reusable
`resnet18.py` can be split out against a stated reason rather than by guess.

Do not require four policy-only files, and do not require a premature common
adapter when behavior is not actually shared. The exact tuple remains the
resolution and evidence key in both designs.

Regional compilation remains compile/capture evidence, not target execution.
Candidate assignment is not final binding; final binding requires successful
translation of final FX and `model.txt` through the selected backend. Operations
assumed to execute on CPU are accelerator-module `unsupported`; do not build or
validate a CPU executor for them, and exclude them from accelerator coverage.
Connected target operation execution and performance are separate gates defined
by the backend-adoption contracts.

For W adoption, the exact-pair adapter exposes the workload inventory but does
not choose target decomposition. Target work first attempts every natural
operation kind under relevant declared qschemes and every real metadata
signature directly; target lowering introduces composites or multi-call forms
only after a documented direct-legality gap.

## Acceptance

Each migration slice requires:

1. deterministic exact-tuple resolution and workload/target/profile agreement,
   including deliberate reusable-adapter selection;
2. no adapter mutation of CLI state or hardware configuration;
3. explicit rejection of unknown named tuples;
4. behavioral dispatch evidence from the affected exact-pair compile lanes;
5. workload/reference/PTQ checks appropriate to the selected adapter;
6. no test that merely restates static registry dictionary contents; and
7. separate exact-tuple assignment, connected-execution, and performance
   evidence when those claims are made; and
8. shared regressions and all affected target lanes, including an explicit
   im2col lane when that target contract changes.

Do not create a target-specific unit or `test/integration/` directory for this
routing. The selected target's regression runner and exact-pair compile lanes
are the durable acceptance surface; target-specific imports do not belong
under shared `test/unit_tests/`.
