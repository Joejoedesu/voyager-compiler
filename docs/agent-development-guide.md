<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Agent development guide

Use this guide for implementation and review work in Voyager. It complements
the current ownership map in [architecture.md](architecture.md).

## Start with evidence

Run these read-only checks before editing:

```bash
git status --short
rg --files src/voyager_compiler src/scheduler test/unit_tests
rg -n "symbol_or_concept" src test
```

Read the complete files that own the relevant stage. For a shared component,
also read its target-neutral unit tests. For a backend component, read the
selected target's end-to-end regression contract. The worktree may contain
unrelated user work. Preserve it, avoid broad formatting, and never use a
clean checkout as a substitute for understanding the current state.

If the task touches target-specific regression or an external runtime, select
the relevant contract from [targets/](targets/README.md). Follow that target's
collateral ownership and execution rules; external reference artifacts remain
read-only unless regeneration is explicit.

## Route the change before coding

Write down the smallest contract that the change affects:

- logical inputs and outputs;
- explicit payloads, scales, axes, blocks, dtype, rounding, and saturation;
- graph stage at which the representation becomes available;
- generic reference behavior and target-specific behavior;
- persistent artifact fields, if any; and
- equivalence level the tests will prove.

Then assign one owner. Common routes are:

| Change | First owner to inspect | Durable validation owner |
| --- | --- | --- |
| Quantization spec or parser | `quantizer/quantizer.py` | `test_vsq_quantization.py` |
| Observer, fake quant, or STE | `fake_quantize.py` | `test_vsq_quantization.py` |
| PT2E prepare or conversion | `quantize_pt2e.py` | quantization and shuffle tests |
| Shared prepared-region construction or finalization | `quantization_regions.py` | prepared-domain and quantization tests |
| Generic PyTorch export or FX helper | `pt2e_utils.py` | caller-focused graph tests |
| Generic graph rewrite | `codegen/passes/` | matching tiling/lowering test |
| Target graph or quantization rewrite | `codegen/target_backends/<target>/` | target end-to-end contract |
| Target final-FX classification binding | target backend lowering/operator owner | target end-to-end contract |
| Shared hardware schema | `codegen/hardware.py` | target-neutral schema tests |
| Target hardware profile | target backend profile | target end-to-end contract |
| Shared local-schedule contract | `scheduler/` types and generic provider | target-neutral scheduler test |
| X-oriented bit-accurate target lowering | target backend arithmetic/quantization owner | complete exact target-output comparison against the frozen workload-semantic reference, with provenance for any packaged expected stream |
| Full-control target schedule | target backend spec plus the selected engine's `providers.py`; generic dispatch consumes `SchedulerProviderSet` | target end-to-end contract proving endpoint consumption; causal variation when that stronger claim is needed |
| Logical/physical tensor relation or liveness | `codegen/storage/` | storage and memory-planning tests |
| Generic physical address placement | mapping plus generic allocator | target-neutral memory-planning tests |
| Voyager-visible target allocation | selected target allocator | target end-to-end contract |
| Full-control accelerator-local placement/address work | selected target allocator | target end-to-end contract proving endpoint consumption; causal variation when that stronger claim is needed |
| Serialized field | `param.proto` and mapping | serialization-focused test |
| External collateral route or encoding | `conversion/<backend>/` under `collateral-conversion` | converter contract |
| Required target packaging, injection, launch, or readback | `conversion/<backend>/**` under `collateral-conversion` | one-operation target execution lane |
| Bufferized behavior | `codegen/lowering/` | bufferized tests and compile |

Do not create a shared abstraction until at least two real consumers need the
same contract. Do not keep two defaults, registries, or target selectors in
sync manually.

## Preserve the boundaries

Use this checklist during design and review:

- Is model-specific policy confined to model configuration or helpers?
- Is the generic representation executable without the target backend?
- Does generic code call a target hook instead of naming Magnet Alpha or Alpha?
- Does final FX remain the semantic instruction carrier after late target
  rewrites, without adding a shared execution IR?
- Does the hardware profile remain the target and dimension source of truth?
- Does scheduler dispatch remain policy-free?
- Does a scheduler return canonical data instead of mutating the graph or
  allocating storage?
- Does `StoragePlan` own aliases, copies, components, and liveness while
  mapping and the selected allocator own placement?
- Does every multi-output tensor use `ValueRef(node, path)` instead of an
  ad hoc tuple key or the producer node alone?
- Does allocator lookup require the finalized plan instead of falling back to
  an FX node key in production, and is any raw test-only allocation or node
  compatibility lookup limited to a unique root tensor output?
- Does an executable slice/transpose/permute materialize unless a real
  address-only lowering can encode its offset, extent, and access pattern?
- Are cross-plane loads represented explicitly instead of treating equal
  logical values in DRAM and scratchpad as the same physical identity?
- Does serialization emit canonical compiler decisions without external
  runtime taxonomy or encoding policy?
- Does external conversion project canonical facts without repairing compiler
  computation, schedules, storage, addresses, or qparams?
- Does optional endpoint integration consume declared encodings and
  layouts instead of guessing them, and compare every produced output?
- For adoption work, do the accelerator-coverage, semantic-support, injection-
  control, and X rules in `docs/backend-adoption/adoption-levels.md` hold? They
  are owned there and are not restated in this guide.
- Are values needed downstream explicit or persisted, rather than held only
  in temporary metadata?
- Are real and fake custom-operator implementations validated against the same
  shape and error contract?
- Is duplicated logic being shared at a neutral layer rather than copied
  between generic and target implementations?
- Are old helpers, registrations, metadata fields, and imports removed when
  their last consumer disappears?

When target policy must run at several compiler stages, keep the stage
implementations together under the target backend and register their order in
the central backend registry. This keeps one target boundary while making pass
order explicit.

## Make a coherent patch

1. Trace the current end-to-end path and tests.
2. State invariants and unsupported cases.
3. Change the smallest complete vertical slice.
4. Remove superseded code only after verifying all call sites with `rg`.
5. Update architecture docs when ownership or representation changes.
6. Run syntax, shared unit tests when applicable, and the appropriate
   end-to-end compilation lanes.
7. Inspect `git diff --check` and the final scoped diff.

Avoid drive-by cleanup. A large file is not, by itself, permission to rewrite
it; extract code only when the new boundary has a clear name, contract, and
test owner.

## Validation ladder

Choose checks in proportion to the affected boundary.

### Fast structural checks

```bash
python -m compileall -q src test/unit_tests
git diff --check
```

Also verify import order when changing package boundaries or deferred imports.
At minimum, import the affected provider before and after
`voyager_compiler` in fresh Python processes.

### Permanent shared unit tests

Tracked tests under `test/unit_tests/` validate target-neutral compiler and
scheduler contracts. A permanent shared unit test may use a small synthetic
backend fixture to exercise a registration or dispatch interface, but it must
not depend on a production backend's operator names, physical address format,
packing rules, scheduler factors, workload checkpoint, or external runtime.

Add a permanent unit test when shared behavior changes. Examples include
generic quantization contracts, storage identity, scheduler result types,
protobuf emission, work-order validation, and target-registry dispatch. Name
the shared contract being tested; do not use a production backend as an
implicit fixture for general behavior.

```bash
python -m unittest discover \
  -s test/unit_tests \
  -p 'test_vsq_quantization.py'
python -m unittest discover \
  -s test/unit_tests \
  -p 'test_matrix_tiling.py'
python -m unittest discover \
  -s test/unit_tests \
  -p 'test_global_memory_planning.py'
```

The shared-test dependency boundary is checked by
`test_shared_test_dependency_boundary.py`. It rejects direct imports of
production backend subpackages, production model adapters, converter packages,
and registered production target names anywhere under `test/unit_tests/`.
This is a static dependency guard: it neither executes the shared compilation
path nor enforces work-order ownership. Preserve shared behavior with affected
target-neutral tests, the full shared suite when required, and affected target
end-to-end lanes. Put target behavior in the selected target's end-to-end
regression harness. Removing a former target unit test does not prove
equivalent end-to-end coverage; report a missing runner assertion as an
explicit verification gap until that assertion exists. QAT/STE training is
outside the PTQ-only backend-adoption flow and requires a separately owned
training compatibility contract when it needs durable target-specific
coverage.

Run only the relevant shared subset while iterating, then the full shared unit
suite when a shared pipeline, representation, target registry, scheduler
contract, or mapping contract changes:

```bash
python -m unittest discover -s test/unit_tests -p 'test_*.py'
```

### Target end-to-end validation

Backend implementation is validated through the public Voyager pipeline, not
by adding `test_<backend>_*.py` files under `test/unit_tests/`. Each target
contract must name end-to-end runner lanes that cross every affected boundary:

```text
export -> PTQ/calibration -> reference conversion -> target lowering
       -> late target movers/final classification -> scheduling
       -> storage/mapping -> serialization
serialization -> collateral conversion
collateral conversion -> optional endpoint integration and execution
target execution evidence -> external system workload runtime and accuracy
```

A small target-specific model may isolate an operation, edge case, or memory
rule, but its lane must still enter through public prepare, transform, and
compile APIs and inspect produced artifacts rather than call one backend
helper as a unit. Include a representative workload lane in addition to
micro-model lanes needed for coverage.

An external simulator, emulator, or runtime is optional only as a repository
capability. If it is out of scope, record `not_in_scope`; if it is required
but a concrete dependency is unavailable, record `blocked`. Insufficient
evidence without a known blocker is `unverified`. Compile success does not
prove target numerical execution. When available and in scope, use the
target-named runtime contract.

For a newly adopted target, close one operation anchor before growing
the workload matrix: run a stock target-native operation through the chosen
endpoint to demonstrate that boundary, freeze the collateral injection
contract in the target regression
contract or a target-local document it links, following the normative
[injection-contract fields](backend-adoption/contracts/execution.md#injection-contract),
and apply any required converter-owned byte/layout encoding. When the contract
requires integrated execution, package, inject, execute, read back, and fully
compare one Voyager-generated operation or composite against the registered R
reference at its declared tolerance. Report emitted-call and
runtime-observed-call evidence distinctly. This gate is operation-local; it
requires neither
model-wide residency nor an external hybrid runtime. Do not infer CPU-runtime
implementation as the next backend-adoption action.

A failure is terminal only when its blocker locus supports that decision.
Compiler, converter, or required runtime-integration gaps need a successor.
Hardware non-support terminates the affected operation form, not the lane.
Runtime numerical failure remains a failed, actionable lane. An external
resource or authority gap may terminate a lane only after an exact attempt,
exhausted in-scope alternatives, and a concrete unblock condition. An unknown
locus is `unverified` and nonterminal.

### Temporary diagnostic tests

After a target end-to-end failure, an agent may create a narrow unit-style
diagnostic under `/tmp/voyager-debug/<task-id>/` or another explicitly ignored
scratch directory. It must not be added to `test/unit_tests/`, used as
acceptance evidence, or committed as target-specific permanent coverage.

- If it exposes a shared bug, reproduce the behavior without a production
  backend and add that target-neutral regression to `test/unit_tests/`.
- If it exposes a backend bug, retain or add the failing end-to-end
  lane in the target regression harness, fix the backend, and remove the
  temporary diagnostic.
- Record the diagnostic command and finding in the handoff, but use the
  passing durable unit or end-to-end lane as check-in evidence.

### Target and workload regression contracts

Before declaring a compiler change ready for submission, identify every
affected backend and workload, then load its concrete regression contract from
[targets/](targets/README.md). Here, “submission” means code-review readiness;
it is unrelated to submitting an external batch job.

- A target-local change runs that target's applicable contract.
- A shared compiler change runs the contracts for every supported target whose
  path consumes the changed behavior.
- A workload or workload-target change runs every affected matrix cell named
  by its adoption/status contract.
- A new target cannot claim submission readiness until it satisfies the R
  registration requirements in `docs/backend-adoption/adoption-levels.md`.

Keep shared assertions in target-neutral unit tests and target assertions in
the target's end-to-end lanes. Run conversion, an external runtime,
or workload accuracy only when the changed contract requires that stronger
claim. Successful compilation and internal numerical checks do not prove
schedule parity, legal external execution, or task accuracy.

If an environment, dependency, or input artifact prevents a required lane
from running, record that lane as `blocked` with its concrete blocker. Do
not replace it with an uncalibrated run, an implicit scheduler, a different
quantization policy, or another weaker smoke test.

### Packaging and submodules

When moving packages or scheduler modules, also verify:

```bash
git submodule status
python -m pip wheel --no-deps --no-build-isolation .
```

Confirm the wheel contains only the intended Python packages and resources.

## Handoff

Report:

- the behavior and boundaries changed;
- the source of truth chosen;
- files changed by responsibility;
- exact checks run and their results;
- unsupported or unverified cases; and
- the first remaining risk, if one exists.

Use the strongest conclusion supported by the evidence. For example,
"all required target/workload lanes compile" is narrower and more useful than
an unsupported claim of end-to-end equivalence.
