# Target registration and extension

Start with [current source and applicable evidence](../adoption.md#start-from-current-state).
When the target is absent, use the fresh-target route below; there is no target
guide or prior implementation to recover.
Use the steps relevant to the missing capability; an existing target does not
need to be registered again. Load only the [roles](../roles.md) needed for it.

1. **Inspect available bindings and authoritative capabilities.** Read the ISA/API,
   hardware configuration and runtime contract, plus existing bindings/tests when
   present. Derive missing hardware facts from those sources; reuse an existing
   description only when it applies.
   Keep capacities separate from software policy and empirical corrections.
2. **Identify the missing scope.** Specify affected operations, shapes, dtypes,
   numerical behavior, layouts and resources. Reuse shared semantics and builders
   where possible. Identify an endpoint when the requested work depends on one.
3. **Pair alternative implementations.** Discover equivalent ISA/algorithm forms,
   such as explicit instructions and a hardware-loop macro for the same operation.
   For missing forms, pair recipe legality, candidates, model, builder and
   emission. Check existing alternatives before adding duplicates; first-match
   dispatch does not explore them.
   For a fresh target, use the [model creation instructions](../compilation/performance-models.md#create-and-register-the-model)
   to choose shared search machinery and target-owned fit/cost components.
4. **Bind the missing extension.** Use the relevant `TargetSpec` hooks and shared
   pipeline. For new operations, define executable behavior and schema effects;
   register `OperatorSemantics` classification as needed.
   Reuse existing allocation and compilation decisions. A new endpoint does not
   justify a private compilation pipeline.
5. **Check affected behavior.** Use representative cases for changed legality,
   arithmetic, execution or resource ownership; force alternative recipes when
   needed to verify them. Reuse unaffected evidence. Compile/emit through Voyager
   and compare host results under the declared contract. Submit and compare at
   the endpoint when claiming endpoint execution; a stock native example alone
   does not validate the Voyager path.
6. **Return the scoped result.** Link bindings, changes, restrictions and evidence.
   Distinguish compiler checks from endpoint execution and report unavailable
   evidence honestly. Compiler-only work may finish without device access.

Analytical models suffice for registration/search. Reuse applicable measurements
with provenance; characterize only when the requested tuning or performance claim
needs it. Endpoint ownership and numerical comparison rules live in the
[adoption evidence guidance](../adoption.md#evidence-follows-the-claim).

## When no target implementation exists

These are decision routes within registration, not extra adoption stages. Follow
the requested workload/endpoint scope and revisit decisions as evidence changes.

| Question | Action and focused owner |
| --- | --- |
| What facts can the agent use? | Start from the hardware/ISA/API revision, configuration, available runtime/simulator and workload/reference. Identify missing facts; the [target role](../roles/target.md) derives the physical and endpoint contract. A previous compiler port is not required. |
| What can implement the workload? | Export representative graphs and inspect actual signatures, including non-matrix ops and state. Inspect ISA/API alternatives rather than stopping at the first GEMM example. Use [implementation discovery](../roles/implementations.md#implementation-decisions). |
| Which numerical and graph work is needed? | Reuse shared [quantization](../roles/quantization.md) and [normalization](../roles/normalization.md). Add [operation definitions](../roles/semantics.md) or [custom-operation lowering](../roles/lowering.md) only for a real mismatch. |
| How does this target execute? | Pair [buffer construction](../roles/bufferization.md) with [model construction](../compilation/performance-models.md). Explicit commands, native macros and heterogeneous engine pipelines can coexist; they are not separate compiler architectures. |
| What establishes a working path? | Compile one representative case through selected recipe, allocation, [emission](../roles/emission.md) and [runtime](../roles/runtime.md) when endpoint access exists. Check logical results and realized control/movement. Extend across signatures and connected regions from there. |
| How do coverage and speed improve? | Use [workload binding](workload-binding.md) to expand supported work; compare forced legal alternatives and inspect bottlenecks. Use [characterization](../roles/characterization.md) for empirical tuning when needed. These loops can proceed together. |

A command/macro-oriented target often needs careful arithmetic, private storage,
ISA alternatives and command placement. A target with several compute engines
and local memories often needs layout/transfer choices, scratch accounting,
instruction fusion and empirical correction of analytical costs. An agent should
derive these needs from the target specification, not select a preset template
from a target name. Reuse shared mechanisms in both cases.

No default target source is required to understand the contracts. If it is
available, treat it as an optional example and check its assumptions. Do not
require a target guide that does not exist yet; write its supported bindings and
restrictions after establishing them. Missing specs or endpoint access limit the
claim, rather than justify invented facts. A compiler-only result does not fulfill
a request for comparable measured performance.
