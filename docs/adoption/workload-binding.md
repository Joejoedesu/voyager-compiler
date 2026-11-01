# Workload × target binding and refinement

Start with [current source and applicable evidence](../adoption.md#start-from-current-state).
A new pair can reuse both a workload and a registered target. Follow only the
steps relevant to the requested gap; load the needed [roles](../roles.md).

1. **Identify the workload and reference.** Reuse its `WorkloadSpec`, weights,
   inputs/state, preprocessing and calibration samples/profile assets. Select the
   target quantization policy through the binding; the workload supplies data,
   while the target owns numerical conversion. For a new workload,
   start with one module and only the necessary hooks. State the numerical
   comparison boundary and asset identities relevant to the change.
2. **Inspect the real graph and target support.** For whole-workload binding,
   inventory every distinct operation/signature and connected region in the
   declared workload/input scope, including non-matrix ops, views, transfers and
   state updates. Compare shape/dtype/layout/state with current recipes and
   evidence. For a scoped fix, inspect the affected regions and boundaries.
   Classify known support, gaps and unverified behavior. A single matmul example does not prove
   every matmul signature. Use the [padding decision guide](../compilation/padding.md) for
   alignment requirements before placing hardware padding in a workload adapter.
3. **Explore missing or inefficient mappings.** First review the exposed ISA/API,
   including load/store features, fused modes and hardware macros; an absent
   compiler recipe is not a hardware limitation. Then inspect
   [shared rewrites](../compilation/rewrites.md), compositions/decompositions,
   layouts and neighboring-operation fusion that could make the work executable.
   Use [input preprocessing](../roles/input-preprocessing.md) when a changed input
   representation helps. A missing one-to-one instruction does not prove the
   computation unsupported.
   Reuse existing builders and expose useful alternatives as recipes. Preserve
   the agreed accuracy and resource limits; compare useful utilization and total
   execution cost, including preparation, movement and added computation. A legal
   but slow mapping remains supported work with a performance gap. Mark untested
   ideas as unverified; support an infeasibility claim with a specific constraint
   or test. Use [target registration](registration.md) for missing capabilities.
4. **Connect and check the affected regions.** Use the common pipeline. Check
   state, region boundaries and combined outputs; when splitting a logical op,
   validate the calls and reconstruction. Reuse existing end-to-end evidence
   where applicable; isolated kernel checks cannot establish whole-model behavior.
5. **Explain coverage.** Identify mapped, unsupported, unverified and host work.
   Distinguish compile-time eliminated work from unsupported work. Graph
   diagnostics or a brief result summary suffice; no mandatory ledger or percent
   report. If comparing percentages, use the fixed semantic-work denominator in
   the [compiler objectives](../compilation.md#guide-scope-and-implementation-objectives).
6. **Refine performance when requested.** Explore legal recipes, tiles, placement
   and reuse for the current mapped regions while coverage grows. Include relevant
   movement, materialization and launch/boundary costs. Follow the
   [performance role](../roles/performance.md) and
   [model/converter guidance](../compilation.md#execution-model-and-converter-consistency)
   whenever execution assumptions change. Models can include stable native-owned
   decisions; full compiler control is not a prerequisite for empirical tuning.
7. **Validate the final changed behavior.** Recheck affected numerical/state and
   endpoint claims after mapping changes; run required pinned gates for affected
   shared behavior. Return source/configuration, reused/new evidence and remaining
   limitations. Claims must match the boundary actually checked.

The steps iterate. Coverage is an adoption priority, not a 100% barrier before
performance work. Exact arithmetic or explicit execution-control work uses the
relevant role/contract when needed; it is not a separate adoption stage.

## Input preprocessing before quantization

For poor utilization caused by input representation, consider
[input preprocessing](../roles/input-preprocessing.md). Compare preprocessing and
quantization choices with fixed references, accuracy and timing boundaries.

## Quantization is part of the binding search

For requested performance work, minimize complete execution time among assignments
meeting the fixed numerical and coverage requirements. A passing assignment is an
incumbent. If its lowering causes narrow matrix work, extra requantization or
materialization, compare an alternative that changes that cause before limiting
tuning to tiles. Use the [adoption work record](harness.md) to retain promising
alternatives independently of correctness checks.

Compare equivalent lowerings under the same integer contract. Compare changed
quantization policies under the same model, inputs and accuracy limits, with the
changed arithmetic explicit. Include their full executable costs, not merely
quantizer error or matrix MAC count.

The target's default is a starting candidate. Compare supported formats, scale
granularity, observer placement and residual-domain sharing when accuracy or
execution cost calls for it. These choices can change rounding, fusion and
requantization traffic even with the same INT8 dtype. Use target policy settings
and the shared quantizer's module/operator controls before adding another flow.

Keep calibration data, pre-quantization references and accuracy thresholds fixed
for the comparison. Reject schemes outside hardware or accuracy constraints;
compare execution cost among the survivors. Check the first changed boundary
when one fails, including unintended sharing across branches. Record the chosen
scheme and measured tradeoff in the binding evidence. Do not silently replace the
target-wide default because one workload prefers a different scheme, or mistake
agreement with the converted graph for accuracy against the original model.

## Scope of operation exploration

Aim for a complete inventory of the requested workload's signatures and systematic
exploration of useful realizations. This is not exhaustive enumeration of every
ISA instruction, graph partition, algorithm and tile combination. Use legality,
representative boundary cases, cost estimates and the task budget to focus search;
retain promising alternatives and explain material pruning or unexplored options.
Mark an uninvestigated signature as unverified, rather than declaring it impossible.
A shape-specialized export does not establish coverage outside its input scope.

Coverage completeness and performance-search completeness are different claims.
Prioritize useful coverage and improve mapped regions together. A passing matrix
kernel is not sufficient acceptance for a requested whole-workload binding.

When performance is part of the task, use the
[required utilization diagnosis](../compilation/performance-models.md#required-diagnosis-when-utilization-or-selected-performance-is-poor)
before concluding a gap is unavoidable. Check permitted initial-input rewrites,
spatial/channel search coverage and retention, then candidate ranking and emitted
execution. The [input-boundary contract](../compilation/rewrites.md#input-boundary-rewrites)
explains when packing belongs in the workload binding versus target recipes.
