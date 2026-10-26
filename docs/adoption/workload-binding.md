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
3. **Address actual gaps.** For each unsupported or costly signature, investigate
   shared rewrites/builders, ISA primitives or macros, compositions/decompositions,
   fusion opportunities and alternative layouts/algorithms where relevant. A
   missing one-to-one instruction does not prove the computation unsupported.
   Expose competing legal realizations as recipes instead of stopping at the
   first working lowering. Try existing semantics, recipes and bufferized
   builders first. Isolate new behavior, boundary classes or a failing case when
   useful; do not rerun every signature as mandatory prework for an unchanged
   mapping. Add arithmetic only for a semantic gap. Use the
   [registration procedure](registration.md) for a missing target capability.
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
