---
name: voyager-target-adoption
description: Bind an accelerator to Voyager's shared bufferized compiler or diagnose poor schedule selection. Establish supported workload coverage, pair ISA alternatives with models and builders, validate converter/runtime agreement, and calibrate performance with attributable evidence.
---

# Voyager target adoption

## Locate the selected checkout

Use the user-selected checkout and inspect current source and local instructions.
This editable package lives under `docs/skills/`; install it under that checkout's
`.agents/skills/` so the relative links continue to resolve locally. Preserve
unrelated work and the experiment's declared input/evidence boundary.

Start with the [documentation router](../../../docs/README.md) and relevant
[role reader](../../../docs/roles.md). Follow
[target registration](../../../docs/adoption/registration.md) for a missing
capability, then [workload × target binding](../../../docs/adoption/workload-binding.md)
for coverage and performance. Binding can return to registration. No completed
target guide or donor is required: use the fresh-target route and authoritative
ISA/API/configuration when the target is absent.

### Create a performance model/search binding

Read the [creation instructions](../../../docs/compilation/performance-models.md#create-and-register-the-model)
when implementing a model; use the [performance role](../../../docs/roles/performance.md)
for ownership. Pair execution with capacity/placement and cost. Choose shared
matrix/grid/attention search where representable, otherwise recipe candidates
and a whole-region estimate. Always bind `TargetSpec.model`; recipe model
callbacks can override it. Check feasible/rejected candidates and realization
before fitting. Existing target formulas are optional evidence, not prerequisites.

Keep target numerical/ABI rules and scoped support in its own guide once they
are established. Shared skills describe decisions and contracts, not target
answers. Do not require one source file per hook or every optional hook.

## Start from existing behavior

Inspect callable source and relevant tests/artifacts before treating an old gap
as missing work. Reuse applicable implementations and evidence. A historical
report, TODO or status tracker does not authorize a new implementation campaign.
Resolve source/contract discrepancies as stale wording, a real violation or a new
requirement. R/W/X labels and the separate X stage are retired; exact arithmetic
and execution-control work belong to relevant roles when requested.

State the requested workload/signature, numerical boundary, hardware, endpoint and
performance claim. Target registration can be compiler-only; claiming execution
requires endpoint evidence. Full compiler control and bit-exact arithmetic are
independent properties, not universal prerequisites for coverage or model fitting.

## Bind through the shared pipeline

`WorkloadSpec` supplies model/inputs and optional assets, calibration, state or
reference behavior. `TargetSpec` supplies hardware and recipes plus applicable
input preprocessing, quantization policy, legalization, normalization, custom-operation lowering,
refinement, encoding/emission and runtime
hooks. Use the smallest existing hook; do not add target-name switches or invoke
a donor pipeline inside a hook.

For input-representation utilization problems, see
[input preprocessing](../../../docs/roles/input-preprocessing.md), before quantization.

Pair each equivalent ISA/algorithm form with semantic matching, legal candidates,
resource requirements, a model/estimate and a builder. Competing recipes need
finite comparable costs and must survive selection into emission. First-match
dispatch is not a search. Reuse the shared mapping/tile engines when they express
the space; a missing engine protocol requires a scoped shared-contract change.

Builders establish loops, buffers, copies and native calls. Final shared allocation
checks expanded capacity, banks and lifetimes. Declare packed components, physical
padding and native reservations rather than allocating again in the converter.
Allocation and scheduling ownership are independent. Native calls complete
synchronously; native async completion needs explicit tokens. Emitters consume
selected tiling/ISA/addresses and must not reinvent them.

## Coverage, utilization and model agreement

Grow correctly supported workload coverage and tune already mapped work as it
grows. A coverage percentage is optional; if useful, fix its semantic-work
denominator and distinguish unsupported work from compile-time elimination.
Introduced copies, fused call counts and padded activity do not increase coverage
or useful utilization. Isolated kernel support does not establish connected-model
execution.

The search engine enumerates/selects, the model evaluates, the builder constructs
execution and the converter/runtime realizes it. Cost is target-owned, not a
universal latency formula. A native macro can have an analytical/empirical model
without compiler expansion of its private schedule.

When execution assumptions change, check relevant reuse, transfer placement,
reduction lifetime, scratch and submission visibility. Equal outputs or aggregate
command counts do not establish overlap. Choose target-specific checks/traces; no
universal checker or fixed characterization campaign is required.

For performance diagnosis or fitting, read
[model-validation.md](references/model-validation.md). Reuse attributable evidence
when hardware, execution and observation scope apply. Collect new measurements
when changed behavior or the claim requires them. Do not fit away a known
implementation mismatch. `performance/` supports measurement adapters, fitting and
held-out evaluation; assess candidate ranking as well as timing error. Hardware
facts, policy choices and empirical corrections remain distinct.

## Validation and handoff

Across experiments or resumptions, use the checkout's
[adoption harness](../../../docs/adoption/harness.md) to retain the best validated
artifact, opportunities and pending checks. Follow registration for missing target
capabilities and workload binding for accuracy-constrained execution choices.
Compare equivalent lowerings under the integer contract; compare changed numerical
policies under fixed model/data/accuracy limits. Use the endpoint's timing unit and
scope, keeping estimates separate from measurements.

Run affected shared contracts and agreed gates, using a target guide when present
or the declared workload/endpoint contract for a fresh target. Preserve
pinned comparators, approved replacements and retained warnings. Use existing
environments and fresh outputs; freeze the tested source/artifact snapshot while
a gate runs. A separate candidate can develop independently. Host reference,
artifact compatibility, endpoint correctness and performance are separate claims.
Static/calibration inputs are not substitute hardware outputs.

Report changed/reused behavior, bound extension points, supported/rejected forms,
model/emission findings, measured versus estimated results and reproducible
artifacts. Keep general docs target agnostic; target guides own local constraints
and dated reports own run status. Follow the user's commit/push scope.

For requested performance work, investigate material utilization gaps: permitted
initial-input rewrites, candidate/adapter/builder expressiveness, ranking, and
model–conversion–runtime agreement. Follow the checkout's
[model construction and required diagnosis](../../../docs/compilation/performance-models.md).
Default selection is strict modeled latency. Reuse Interstellar through
`search_matrix` with target-owned constraints; do not inherit the default target's
schedule or call a narrow candidate list a hardware limit.
