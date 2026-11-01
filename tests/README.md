# Compiler contract tests

Run from the repository root with an existing compatible environment:

```bash
PYTHONPATH=src:test python -m unittest discover -s tests -v
```

These tests use synthetic inputs and temporary output directories. They do not
download models or launch hardware simulation. Add cases to the existing owner
below; create a module when a distinct contract needs one, rather than for each
implementation change. Use subtests for the same behavior across shapes or modes.

| Responsibility | Test modules |
| --- | --- |
| Registration, selection and frozen artifacts | `test_extensions`, `test_pipeline` (also relocated, fresh-process emission; complete concat stores and replay images) |
| Input preparation and layout | `test_input_preprocessing` (also im2col), `test_constant_folding`, `test_depthwise_dispatch` |
| Allocation and residency | `test_memory` (async lifetimes, bank conflicts, packed backing, resident views) |
| Quantization | `test_quantization`, `test_quantization_flow` (target stage overrides) |
| Search and performance | `test_matrix_search`, `test_region_algorithms`, `test_performance` |
| Gemmini | `test_gemmini` (raw ISA alternatives), `test_gemmini_deployment` (native command realization, depthwise, replay), `test_gemmini_quantization` (domains, folding, ReLU6) |
| Trainium alternatives | `test_trainium_alternatives` |
| Adoption work record | `test_adoption_harness` |

Passing host replay and command-encoding checks is not a device correctness or
performance claim. Target guides under `src/voyager_compiler/targets/` identify
applicable pinned gates and endpoint checks; runners live in `validation/`.
Keep generated outputs in ignored `results/` or external artifact storage.
