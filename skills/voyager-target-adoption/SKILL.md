---
name: voyager-target-adoption
description: Bind a new accelerator backend to Voyager's shared compilation flow, or diagnose a target whose search selects poor schedules. Build evidence-backed hardware and execution models, target constraints, buffer plans and ISA realization; validate schedule ranking, utilization, correctness and default regressions.
---

# Voyager target adoption

Make the shared compiler search for useful schedules that the target can actually
execute. A legal mapping or a successful simulator run alone does not establish
that the target's search model is reliable. An approximate model with sound
resource accounting and useful ranking is the goal; cycle accuracy is not.

## Establish the implementation boundary

Honor the selected checkout and existing work. In this workspace,
`AGEN-voyager/voyager-extensible` contains the shared extension interfaces;
`voyager-base` and the Gemmini siblings are comparators, not interchangeable
working directories. Inspect local instructions, Git state, `docs/compilation.md`,
`docs/hardware-ir.md` and the current source before using an interface name below.
The companion `voyager-compiler-architecture` skill provides general stage and
regression guidance; this skill focuses on adoption and model quality.

Identify the pinned device/configuration, command interface, precisions, kernels,
objective and measurement scope. Preserve prior no-commit and runtime/RTL editing
constraints. Treat pinned hardware/runtime sources as evidence, not files to
change merely to improve compiler results. Reuse existing records before running
expensive experiments, authenticating their provenance. Ask only for missing
information that prevents choosing a valid target or comparison.

## Separate facts, choices and results

- **Hardware IR:** physical memories, replication, banks/ports, operand and
  accumulator widths, compute modes/throughput, transfer geometry, bandwidth,
  latency, startup/issue costs and queue capacities. Record units and evidence:
  documented, measured, derived or assumed. Unknown timing is not zero.
- **Target policy:** selected dataflow, search restrictions, bank placement
  preferences, buffering candidates, conservative temporary-storage allowances
  and instruction submission bursts. Keep tunable choices in explicit hooks;
  do not disguise them as physical capacities.
- **Selected plan:** per-kernel mapping, loop order, buffer generations, physical
  footprints and estimated resource demand. Search returns this plan explicitly;
  downstream stages must not recover it from the cost model's last evaluation.

Derive constraints from physical facts, supported lowering or explicit policy.
Distinguish those three reasons when a candidate is rejected. Do not invent a
smaller SRAM, a nonexistent engine/connection, or benchmark-specific tile presets
to make an existing Voyager model apply. Architecture alone does not determine
an optimal schedule. A heuristic is acceptable when identified, configurable and
consistently interpreted by search and realization.

## Use the shared flow and its extension points

The current interfaces are in `targets.py`, `compilation.py` and
`codegen/transform/tiling/`:

| Boundary | Integration |
| --- | --- |
| Legalization and fusion | Backend validation, graph preparation, padding and fusion legality; quantization-family policy remains separate |
| Mapping search | `interstellar_memory`, `schedule`, `prepare_matrix`; reuse candidate enumeration instead of a parallel production scheduler |
| Resource and execution estimates | `partition`, `evaluate`, `CandidateEvaluation`, `StorageRequirement`; model the execution that lowering can realize |
| Nonmatrix search | `vector_limits`, `nonmatrix_slot_size`, `nonmatrix_footprint`, `nonmatrix_cost`; footprint and bank groups must agree |
| Buffering and placement | `KernelBufferPlan`, `SelectedMapping`, `place_local_buffers`; retain shared lifetime, alias and semaphore analysis |
| Instruction realization | `realize`, `options`, `restore_mapping_policy`; consume the existing collaterals and their dependencies |

Prefer the smallest extension to a missing shared contract over target-name
branches throughout the compiler. Do not force unsupported hierarchies or loop
orders into the current four-level matrix-builder contract. Implement the missing
capability or reject it explicitly. Treat resident/per-kernel support separately;
a backend need not claim both.

Resolve hardware, policy and objective once through `CompilerContext`. Persist
reconstructible policy in existing run metadata. Conversion must restore it or
check an explicit context against it. A buffer plan chooses copies/reuse; the
storage allocator chooses addresses and checks physical placement. Neither is
a replacement for the other.

The converter realizes the selected software schedule: ISA expansion, supported
fusion and dependency-preserving reordering are legitimate. It must not silently
search for different software tiles or invent transfers absent from the shared
program. Internal estimation descriptions are not a new external collateral
format. Share geometry, storage sizing and traversal interpretation where practical
without expanding every search candidate into a complete instruction stream.

## Establish model quality before trusting the search

Read [model-validation.md](references/model-validation.md) when implementing or
changing resource/cycle estimates, explaining poor utilization, or comparing
against another scheduler. It covers transfer/compute overlap, finite submission
windows, calibration, candidate ranking and trustworthy performance evidence.

Start with representative legal candidates, including an available good schedule
from the prior compiler, Exo or another baseline. Separate representability,
legality, scoring and realization failures before widening search or adding a
cost term. Diagnostic fixed-tile probes are useful evidence; keep them outside
production selection. Use the requested objective; preserve default Voyager's
objective/tie behavior while binding a speed-only target.

Validate numerical correctness, schedule quality and default compatibility
separately. Use the selected regression suite against a fixed pre-change baseline,
including numerical-warning signatures and resident coverage when affected.
For a simulator-backed target, compare predicted and measured cycles/utilization
on the agreed kernels and validate the actual emitted instructions and outputs.
Do not substitute reference buffers for hardware results in a network chain.

## Finish with reproducible evidence

Record the hardware/policy configuration and its sources; supported and rejected
execution forms; search restrictions and their reasons; predicted versus measured
results and remaining gaps; default regression results; and commands/artifact
locations. Explain whether simulation is fresh or authenticated reuse, and what
timing excludes. Do not promise a universal utilization percentage or report
unrun cases as validated. Follow the user's repository and commit/push scope.
