# Performance characterization

Use for: Measure compiled cases and supply attributable evidence for model fitting.

## Extension and inputs

Binding: `TargetSpec.characterize`.

Compiled result, case, resolved config, selected execution identity and measurement endpoint.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#performance-characterization-and-model-fitting).
Source: [performance/characterize.py](../../src/voyager_compiler/performance/characterize.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Analytical model ranks candidates poorly | Force comparable legal alternatives on the same operation/input boundary and measure realized work. First exclude missing candidates or model/emitter mismatches. |
| Only a subset of workloads is covered | Sample representative supported families and resource boundaries while coverage grows. Record unsupported/host work; fitting only easy kernels does not prove whole-workload speed. |
| Need an empirical correction | Use disjoint fitting/held-out cases and stable endpoint/configuration. Fit only after structural agreement; inspect ranking and selected schedules, not only average timing error. |

## Use the existing measurement path

`performance.characterize` compiles cases and calls a supplied adapter (or the
registered hook). Return `Measurement` with matching case/family/target/hardware
identity, cycles/seconds, clock and provenance. The current fitter learns a
nonnegative affine correction per family; each fitted family needs at least two
distinct modeled training timings and held-out coverage. Apply the returned
profile through the compile API's `performance_profile` argument.

This is not automatic candidate exploration. Arrange forced alternatives through
the target's candidate controls or a scoped test registration, and retain the
selected recipe/parameters, artifact identity and timing boundary in provenance.
When deriving a test target, retain noncompeting recipes needed by the case and
pass it directly without re-registering its name. Use a family key matching the
recipe name or `OpModel` class being corrected, and measure the same work its
estimate covers. The current measurement record has no dedicated candidate-identity field, and
held-out output reports timing error, not candidate-ranking quality. Compare
rankings separately when the claim depends on selection. A richer request/record
is framework work only when the concrete experiment needs it.

## Do

- Use `characterize(result, case, config) -> Measurement`, or supply a measurement adapter to the shared characterization runner.
- Keep training and held-out cases distinct and preserve case/family/target/hardware identity.
- Reuse applicable observations; fix execution/model mismatches before fitting and assess ranking as well as error.

## Do not

- Do not use host wall time as accelerator timing, or fit away missing computation/incorrect transfers.
- Do not make a fresh measurement campaign mandatory for ordinary registration or compiler-only changes.

## Handoff and validation

Return measurements, provenance and fitting/held-out results to the performance-model owner. This is an offline workflow, not the runtime scheduler or a required compile stage.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)
