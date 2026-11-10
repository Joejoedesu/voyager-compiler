# Hardware–compiler co-design and DSE

Use this procedure when the task includes choosing hardware configurations,
implementing new hardware capabilities, or developing an accelerator and its
compiler together. Design-space exploration (DSE) compares hardware choices
with the workload mappings they enable under the task's constraints. Mapping
search within one fixed hardware candidate is part of that exploration.

The task defines the workloads, numerical requirements, resource limits,
permitted hardware changes, objective and final validation. Documentation routes
implementation work; it does not reduce those requirements. When hardware
implementation is in scope, missing hardware, ISA, compiler or runtime support
is work to implement or address by revising the design. An unimplemented feature
is not evidence that the task is infeasible.

## Decide what needs to change

Inspect complete workload graphs, available hardware primitives and actual
execution before choosing a feature. Include non-matrix operations, numerical
boundaries, data movement and control. Distinguish a missing compiler mapping
from a missing hardware capability and from a slow existing implementation.

| Finding | Next action |
| --- | --- |
| Available hardware primitives can implement the work | Extend [compiler support](registration.md); compare compositions and [algorithm alternatives](../compilation/algorithm-search.md) before adding hardware. |
| The hardware supports the work but the selected execution is poor | Diagnose [mapping, model and realization](../compilation/performance-models.md#required-diagnosis-when-utilization-or-selected-performance-is-poor); improve workload mapping or compiler search where needed. |
| Supported generator parameters can address the limitation | Compare configurations with mappings regenerated for each candidate, including capacity, bandwidth, banking and useful utilization. |
| Required behavior or performance needs a hardware change | Propose and implement datapath, storage, control or ISA alternatives; bind their semantics, mappings and costs into the compiler and validate the resulting execution. |

A feature proposal should explain which workload gap or bottleneck it addresses,
why existing realizations are insufficient, and its expected benefit and resource
cost. Consider effects across the requested workloads, including short reductions,
tails, movement and synchronization. Avoid selecting a feature solely because it
is absent or because a larger unit has higher peak throughput.

## Iterate hardware and compilation together

1. **Establish the task contract and candidate.** Separate fixed requirements
   from design variables. Identify the hardware/generator revision, configuration,
   ISA, numerical behavior and compiler bindings for each candidate. A fixed
   target adoption task does not authorize hardware changes; a co-design task
   may authorize them within its declared design space.
2. **Find a useful next experiment.** Inventory required support and investigate
   performance bottlenecks. Compare mapping changes, supported parameter changes
   and new hardware features. Use analytical estimates and representative cases
   to prioritize implementation. Use estimates to compare proposed hardware
   features before implementing them. Identify which results are estimates and
   which come from the implemented design. Establish a complete executable
   baseline early where practical.
3. **Implement the candidate and its support.** Update hardware/generator, ISA,
   arithmetic, compiler recipes, models, builders, emission and runtime wherever
   the change requires it. Use [target registration](registration.md) and the
   [extension catalog](../extensions.md) for compiler ownership. Updating a
   resource description alone does not implement a physical resource. Ensure
   selected mappings and layouts are realized by the generated execution.
4. **Validate and integrate.** Use modular checks during development, then satisfy
   the task's connected-workload and numerical checks. Compare hardware-aligned
   reference behavior with actual execution, including intermediate boundaries
   and required stores. Freeze each tested candidate's hardware and compiler
   identity; a separate candidate may continue evolving. Revalidate affected
   behavior after changes rather than combining incompatible evidence.
5. **Compare and revise.** Evaluate feasible candidates under the same workload,
   numerical, measurement and resource contract. Include movement and control
   costs; distinguish useful work from padding or idle capacity. Remap workloads
   when hardware changes. Investigate discrepancies between modeled and realized
   execution before using estimates to justify further design choices.
6. **Select and deliver the complete design.** Satisfy the requested features,
   workload mappings and validation, and explore the design space sufficiently
   to justify the final point under the task's objective. Explain representative
   alternatives and material pruning; exhaustive enumeration is not implied.
   Missing required support returns to implementation or candidate revision.
   Completing a registration step or documenting a partial result does not
   complete the enclosing co-design task.

An optional [work record](harness.md) can retain candidate identities, required
unfinished work, checks and optimization opportunities. A partial prototype is
development evidence. A feasible design meets the task's hard requirements;
selection among feasible designs additionally needs the requested DSE evidence.
Optional improvements may remain after a justified final selection, but required
implementation and validation cannot be deferred as optional opportunities.

If an external dependency blocks progress, identify it and request the needed
input while continuing independent work. Report evidence accurately throughout
development; narrowing a claim does not narrow the requested deliverable.
