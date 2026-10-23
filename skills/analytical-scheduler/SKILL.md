---
name: analytical-scheduler
description: Analyze and reconcile analytical schedules across Voyager graph tiling, memory mapping, Interstellar loop search, and Magnet Alpha hardware constraints. Use for explaining why a schedule won, comparing candidate loop nests, validating buffer and address-generator legality, designing target-specific runtime or energy models, tracing schedule fields, and locating trustworthy performance or energy evidence.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Analytical Scheduler

Trace a scheduling decision from logical layer dimensions through candidate
generation, capacity checks, objective evaluation, target legalization, and
serialization. Distinguish modeled behavior from measured hardware behavior.

Read [references/scheduler-map.md](references/scheduler-map.md) before assigning
ownership, comparing hierarchy levels, or quoting energy numbers.

## Establish the evidence

1. Read repository instructions and inspect the dirty worktree before edits.
2. Record the exact Voyager output, AoV schedule, hardware profile, precision,
   and modification times used in the comparison.
3. Keep AoV and Alpha collateral read-only unless regeneration is explicit.
4. Identify whether each value is a logical extent, blocking factor, spatial
   partition, tile count, address-generator field, or physical capacity.

Never infer schedule equality from matching names or a successful compile.

## Reconstruct the decision

1. Recover the full `IC/OC/OX/OY/FX/FY/ON` layer and dtype widths.
2. Recover Interstellar resource capacities, partitions, parallel counts,
   schedule hints, runtime callback, and optimization ordering.
3. Recover transform-time Voyager tiling and compile-time scratchpad refinement.
4. Decode the winning mapping into local factors and outer tile counts.
5. Re-evaluate plausible alternatives with
   `scripts/compare_interstellar_candidates.py`.
6. Report runtime terms, access counts, energy score, and candidate legality.
7. Apply Alpha constraints only after the generic candidate is understood.

When a candidate loses narrowly, inspect the objective equation term by term.
Do not describe the result as a hardware preference until the relevant
physical event is represented in the model.

## Gate target-specific scheduler work

Freeze the execution endpoint's injection contract before adding a scheduler.
Classify control by the performance-relevant decisions that remain after
Voyager emits a call:

- A point is partial-control when a helper or runtime privately chooses a
  performance-relevant mapping. Compiler factors are advisory there.
- A point is full-control when Voyager's scheduler and mapper determine those
  variable decisions and the endpoint only consumes or deterministically
  projects them. Explicit schedule fields are evidence, not the criterion.
- Reuse a shared provider when it enforces every endpoint-consumed decision and
  target legality fact.
- If an engine is adequate but target projection or provider wiring is
  missing, bind that engine through the target scheduler interface.
- Implement a target scheduler only when eliminating an otherwise hidden
  mapping decision requires target-owned selection and no existing engine
  enforces the required target facts.

Unknown authority defers scheduler work; it does not justify speculative
implementation. When a fixed helper ignores compiler factors, either keep the
schedule advisory or constrain and validate scheduler output against the
helper's representable normal form. Never preserve authoritative schedule
metadata that the selected execution path discards.

## Model Magnet Alpha cleanly

Keep generic enumeration and target evaluation separate:

```text
Interstellar loop-factor candidate
  -> Magnet Alpha legality and factor normalization
  -> Magnet Alpha event counts
  -> Magnet Alpha runtime and energy evaluation
  -> target-owned schedule descriptor
```

Prefer optional evaluator and validator callbacks over embedding Magnet Alpha names
in Interstellar's generic cost model. Keep graph splitting in Voyager tiling,
physical placement and copies in mapping, and serialization policy-free.

Keep operation scheduling behind a target-owned interface:

```text
generic mapping context
  -> resolve the FX operation and current tensor shapes
  -> generic target-scheduler dispatch
  -> magnet_alpha_scheduler recognizes and validates the operation
  -> shared MappingPoint/LocalScheduleResult, or None when unsupported
```

`magnet_alpha_scheduler` owns the supported NMP operation list, operation-mode
recognition, Alpha legality, and `P1/Q1/C1/K1` derivation. Generic code may
select a registered target scheduler and attach its shared result, but it must
not construct Magnet Alpha workloads, name NMP modes, reproduce factor formulas,
or cache by target-specific fields. An unsupported Magnet Alpha operation returns
`None` and receives no tiling; it must not be forced through GEMM scheduling.
Keep this target-operation path independent of the selected GEMM optimizer so
Interstellar and rule-based GEMM scheduling can reuse it.

Count physical events rather than only scalar tensor elements. At minimum,
separate vector MACs, input-buffer words, weight-buffer words, collector word
reads and writes, accumulation-buffer traffic, global-buffer traffic, address
generation, collector drain/fill, and PPU work. Keep read and write energy
separate where evidence supports it.

## Qualify energy claims

Label every number as one of:

- post-layout or gate-level measurement;
- synthesis estimate;
- calibrated analytical estimate;
- analytical estimate for an abstract architecture;
- historical proxy from another technology or configuration;
- placeholder relative cost.

Do not transfer AoV PAT costs or old spreadsheet values into the Magnet Alpha profile
without matching technology, voltage, frequency, word width, depth, banking,
ports, and access definition. If no matching source exists, report the gap and
propose the measurement needed to fill it.

## Validate proposed changes

Use a small candidate set first. Require:

- total loop products cover the logical layer;
- local factors satisfy collector, buffer, and address-generator constraints;
- predicted cycles explain the selected candidate;
- event counts agree with serialized target counters or schedule-derived counts;
- several AoV schedules are reproduced before claiming generality;
- generic-target behavior remains unchanged.

Treat the bufferized backend as a separate consumer.
