# Documentation entry point

Choose a route for the current task. Keep detailed contracts available on demand;
there is no requirement to preload the entire documentation set. For a complete
human-readable tour, use the [compiler walkthrough](../../reports/voyager-compiler-walkthrough.html).

| Task | Read first | Then load as needed |
| --- | --- | --- |
| Install or edit skills for an isolated checkout | [Editable skill packages](skills/README.md) | Local installer; relative checkout links |
| Locate an extension point | [Extension catalog](extensions.md) | Actual target registration and its guide |
| Understand the compiler | [Compilation flow](compilation.md#the-compilation-flow) | [IR decisions](compilation.md#ir-ownership-and-why-these-records-differ), relevant source |
| Distinguish graphs and plans | [Tensor graph, execution graph and decisions](compilation.md#tensor-graph-execution-graph-and-compilation-decisions) | [Async and native completion](compilation.md#asynchronous-buffer-operations-and-native-completion) |
| Choose preprocessing before quantization | [Input preprocessing](roles/input-preprocessing.md) | [Hook contract](extensions.md#input-preprocessing), input rewrite helpers |
| Reuse an existing compiler transformation | [Shared transformations](compilation/rewrites.md) | Focused feature reference, current pass and target binding |
| Compare algorithms for one operation/region | [Algorithm comparison](compilation/algorithm-search.md) | Implementation recipes, component builders and whole-region costs |
| Adopt a target or workload | [Adoption overview](adoption.md) | [Target registration](adoption/registration.md) or [workload × target binding](adoption/workload-binding.md), plus relevant roles |
| Implement a specific change | [Role index](roles.md) | Selected role, linked contract section, current callable source |
| Build a model without an existing target | [Performance role](roles/performance.md) | [Create and register the model](compilation/performance-models.md#create-and-register-the-model), builder and ISA/runtime contract |
| Diagnose poor selection/performance | [Performance role](roles/performance.md) | [Model/converter guidance](compilation.md#execution-model-and-converter-consistency), target implementation |
| Resume adoption experiments or organize pending checks | [Adoption work record](adoption/harness.md) | Task-local artifacts, relevant procedure and role |
| Check a finished change | [Integration role](roles/integration.md) | [Pinned validation](compilation.md#reference-gates-and-reproducible-commands) |

The compilation guide owns shared compiler contracts and links focused references
for cross-cutting details. Adoption pages own task routes and evidence guidance.
Role readers are grouped by registered extension point, with shared-framework
work listed separately. They describe decisions, inputs, source owners and handoffs; they link to contracts instead of copying
them. Shared docs and skills remain target agnostic; the default Voyager target may
serve as an example. [Target-owned guides](../src/voyager_compiler/targets/README.md)
describe actual extension bindings, numerical/ABI constraints, supported forms,
padding/storage decisions and validation. Keep other target examples, status and
endpoint commands in those guides, not in shared role readers. Skills route the task;
HTML reports explain design and dated evidence.

These are documentation boundaries, not required source modules. A workload can
still live in one Python file, and one person/agent can fill multiple roles.
Historical APIs and snapshots are reference evidence, not current interfaces.
If a guide and source disagree, resolve or record the discrepancy rather than
silently weakening the required contract.

## When to create or split a document

Create documentation for a recurring reader decision, not for every feature,
source file or exploration session. First extend the existing owner. Split out a
focused reference when the decision has substantial reusable detail, spans several
roles or stages, and can be loaded independently. A small detail stays in its
owning section. A new extension point does not automatically need a new role.

| Reader question | Canonical home | Example |
| --- | --- | --- |
| What runs, in what order, with which invariants? | Compilation guide and linked technical references | Graph/plan meaning; transformation catalog with focused feature references |
| How do I carry out this task? | Adoption procedure | Bind a workload using existing target support |
| What do I own, and where must I hand off? | Role reader | Emitter consumes selected tiles; search chooses them |
| What differs for this target? | Target README and callable bindings | Bound hooks, shared defaults, supported shapes and endpoint evidence |
| What happened in a particular run or review? | Dated report/evidence | Source identity, commands, results and limitations |

The technical hierarchy follows the compilation flow:

```text
compilation.md                 stage order, IR and shared contracts
compilation/
  rewrites.md                  catalog of shared transformations and activation
  padding.md                   focused detail: logical/physical meaning and placement
  performance-models.md        optional model-construction and search-reuse details
adoption.md + adoption/        how to carry out integration tasks
roles.md + roles/              who owns decisions and handoffs
skills/                       editable, locally installable skill packages
```

Padding is a feature under the compiler reference, not a sibling documentation
category to adoption or roles. The [transformation catalog](compilation/rewrites.md)
lets an agent discover existing behavior before implementing it. Keep short
features in the catalog; link substantial details, such as padding, only when
needed. Do not consolidate all implementation detail into one mandatory long read.

A maintained page should make its use case, ownership boundary, relevant source
and checks discoverable. A role reader includes a short **Implementation decisions** table for likely
problems, explicit **Do** and **Do not** sections, and a concrete handoff/check; a technical reference explains alternatives and when each applies.
Link a new page from the existing router/owner. Do not introduce a parallel
summary that must be manually kept consistent with the same contract.

## Preventing stale guidance from becoming implementation work

Read current registrations, implementations and relevant tests before interpreting
a procedure step or old gap as missing work. Use the adoption
[start-from-current-state rule](adoption.md#start-from-current-state) for evidence
reuse. Source establishes implementation; tests establish only their exercised
scope; contracts establish intended behavior. A disagreement calls for a scoped
diagnosis, not automatic reimplementation or automatic dismissal of the contract.

Keep durable docs about decisions and responsibilities. Put run status and open
observations in dated evidence with source/configuration identities; they are not
standing work orders. Do not create stage-completion status trackers as an adoption
requirement. If a live coordination note is actually needed, keep it task-scoped
and retire it when the task ends. Update affected canonical docs in the same
change as an interface or ownership change; keep historical evidence historical.

## Keep implementation guidance small and useful

A role reader must help choose the next implementation action, including when its
target code does not yet exist. Use a few representative problems and decisions,
not an exhaustive troubleshooting list or a restatement of every API field.
Inputs may be authoritative specifications and a failing case; existing target
source is optional. Link long algorithms, signatures and technical constraints to
their canonical owner. The performance-model construction reference is an example.

Aim for roughly 300–600 words per role as an editorial guide, not a correctness
gate. A task normally loads the router, one or two applicable roles and the linked
contract sections it needs. Load more only when a concrete dependency warrants
it. One agent may own several roles; role separation does not require delegation.

Check a guide by asking: can a reader identify a starting point without a donor,
choose reuse versus new implementation, recognize an unsupported contract, and
produce a checkable handoff? Link validity and concise text are necessary but do
not prove agents can reproduce target quality or measured performance. That needs
an independent reconstruction experiment with declared inputs and acceptance.
