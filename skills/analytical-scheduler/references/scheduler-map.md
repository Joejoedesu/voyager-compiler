<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Scheduler map and evidence guide

## Contents

1. [Source ownership](#source-ownership)
2. [Representations](#representations)
3. [Current Interstellar model](#current-interstellar-model)
4. [Alpha and Magnet constraints](#alpha-and-magnet-constraints)
5. [Energy evidence](#energy-evidence)
6. [Required reporting](#required-reporting)

## Source ownership

| Responsibility | Primary source |
| --- | --- |
| Voyager pass order | `voyager-compiler/src/voyager_compiler/__init__.py` |
| Graph-level matrix tiling | `codegen/passes/tiling.py` |
| Interstellar resource and runtime adapter | `codegen/tiler.py` |
| Shared schedule/result types | `src/scheduler/types.py` |
| Generic local and target-operation dispatch | `src/scheduler/dispatch.py` plus `codegen/mapping.py` |
| Magnet Alpha GEMM and NMP scheduling policy | `src/scheduler/magnet_alpha_scheduler/` |
| Scratchpad refinement, schedule attachment, and target memory rewrites | `codegen/mapping.py` |
| Interstellar candidate enumeration | `src/interstellar/mapping_point_generator.py` |
| Generic capacity and access model | `src/interstellar/cost_model.py` |
| Generic hierarchy description | `src/interstellar/resource.py` |
| Alpha schedule serialization | `conversion/magnet-alpha/convert.py` |
| Alpha factor legalization | `src/scheduler/magnet_alpha_scheduler/scheduler.py` |
| Alpha physical buffer model | `src/voyager_compiler/codegen/target_backends/magnet_alpha/hardware.py` |

Verify symbols and line numbers with `rg`; do not rely on this table as a
versioned API.

### Target-operation scheduling boundary

The common compiler passes an FX operation, resolved semantic operand shapes,
the output shape, and hardware configuration to a registered target scheduler.
The target package either returns a shared mapping plus target factors or
returns `None` when that operation is not implemented.

For Magnet Alpha, keep all of the following in `magnet_alpha_scheduler`:

- supported NMP target/operator lists;
- depthwise, pooling, shuffle, upsample, normalization, and Softmax checks;
- target legality and unsupported-case errors; and
- `P1/Q1/C1/K1` or later Alpha-specific factor derivation.

`scheduler/dispatch.py` owns the target-to-provider gate and shared result
contract. `codegen/mapping.py` resolves current shapes, invokes that neutral
interface, and attaches the result. It must not construct Magnet Alpha
workloads, branch on NMP modes, or derive/cache target factors. Target memory
rewrites may remain in mapping because placement and physical copies are a
different responsibility from schedule derivation.

Invoke target-operation scheduling before generic GEMM scheduling. This lets
Magnet Alpha claim depthwise convolution as an NMP operation even though its FX
target is also classified as convolution. An unclaimed operation may continue
to an applicable generic path; an unimplemented non-GEMM operation gets no
tiling metadata.

### Scheduler test ownership

Keep scheduler tests aligned with the same production boundary:

| Test responsibility | Owner |
| --- | --- |
| Graph tile shapes, padding, and scheduler admission | `test_matrix_tiling.py` |
| Magnet Alpha namespace and scheduler regression | `test/run_magnet_alpha_scheduler_regression.py` |
| Physical addresses, banks, copies, aliases, and liveness | `test_global_memory_planning.py` |

Do not test target runtime or factor formulas from matrix-tiling tests. Do not
test physical shuffle copies from scheduler tests. Keep collateral-dependent
parity checks separate from deterministic parser unit tests so missing local
AoV data cannot silently reduce normal coverage. Keep any custom-test audit in
the review evidence supplied with the change.

## Representations

Keep these representations distinct:

```text
logical layer extents
  -> Voyager graph tiles
  -> Interstellar blocking and partitioning
  -> target-local factors and outer traversal
  -> address-generator maxima and strides
  -> emitted collateral
```

AoV `P0/Q0/P1/Q1/P2/Q2` suffixes are not automatically equivalent to
Interstellar memory levels. Alpha factors encode collector and address behavior
as well as loop blocking. If Voyager materializes outer tiles as graph nodes or
explicit loops, it may not need to serialize `P2/Q2`, but it must preserve the
same coverage and offsets elsewhere.

## Current Interstellar model

Voyager normally builds three levels:

```text
L0: parallel IC0*OC0 processing/register abstraction
L1: separate input, output/psum, and weight capacities
L2: unified cache capacity
```

The current L1 capacities are expressed as scalar-element counts:

```text
IF = input_buffer_size * IC0
OF = accum_buffer_size * OC0
FL = weight_buffer_size * OC0
```

The generic OF working set is a product of blocked and partitioned
`OX/OY/OC/ON` extents. Output accesses depend on loop order: reduction loops
`IC/FX/FY` outside output loops cause additional output traffic; inner
reduction loops assume output residency.

Important limitations:

- the resource documentation says bytes, while Voyager currently supplies
  normalized element capacities;
- `psum_dtype_bits` is recorded but does not scale capacity checks;
- physical word grouping and read/write direction are not represented;
- the generic runtime callback approximates overlap and does not model Alpha
  collector drain, address configuration, or PPU handoff;
- optimization is lexicographic: runtime first, energy only on exact runtime
  ties.

## Magnet Alpha constraints

For the common output-stationary plus local-weight-stationary path:

- `IC0` is vector size and `OC0` is vector lanes;
- the output collector contains `OC_DEPTH` words;
- each collector word contains `OC0` accumulator lanes;
- ordinary convolution uses `P0=1` and addresses collector words with `Q0`;
- supported `P0>1` paths flatten `P0*Q0` into collector addresses;
- require the resident local spatial positions to fit `OC_DEPTH`;
- `Q0` must divide `Q1`, and current compatibility code strongly favors
  `Q0=Q1`;
- non-unit `P0` has restrictions on kernel, stride, folding, scaling, and
  outer factors;
- output address generation has additional supported-shape cases and fixups.

Do not reduce these rules to a single capacity inequality. A candidate can fit
all scalar capacities and still be impossible or expensive to address.

## Energy evidence

Use this evidence order:

1. post-layout or gate-level power using activity from the matching Alpha
   configuration;
2. synthesis power for the matching module, precision, voltage, and clock;
3. a characterized SRAM/macro table matching word width, depth, banks, ports,
   and technology;
4. serialized target event counts multiplied by characterized event energy;
5. external analytical estimates for structural comparison only;
6. historical spreadsheet values as methodology or sensitivity proxies only;
7. Interstellar's current `[1, 10, 100]` access costs as relative placeholders.

Useful searches:

```text
rg -n -i "Vector access energy|Energy \(per-scalar-access\)" <collateral-root>
rg -n -i "pJ/access|fJ/MAC|dynamic power" <accelerator-root>
find <accelerator-root> -iname '*power*' -o -iname '*.rpt'
```

## Required reporting

Report:

- exact layer, precision, resource profile, and artifacts;
- winning and alternative factors;
- legality result and violated rule, if any;
- decomposed runtime terms;
- per-level, per-operand access counts;
- energy values with units and provenance;
- whether the conclusion is generic, AoV-specific, or Magnet-Alpha-specific;
- the smallest clean ownership point for any proposed change.
