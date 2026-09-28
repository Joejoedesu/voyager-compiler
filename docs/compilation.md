# Model × hardware compilation

The default target is `voyager` (there is no separate `default` profile).
Existing commands without a target or recipe retain their CLI defaults.

```sh
python test/test_codegen.py resnet18 --target_hardware voyager \
  --quantization_recipe E4M3 --pe_array_size 16,16 \
  --model_output_dir /tmp/resnet-voyager --debug
python test/run_ci.py /tmp/ci --baseline /path/to/fixed/timestamp \
  --suite test/regression_suite.txt --jobs 2
```

## Model × hardware instances

Previously, `test/test_codegen.py` selected a model, `run_ci.py` supplied shared
quantization flag strings, and model adapters embedded Voyager quantizer rules
and repeated the transform/compile flow. The refactor preserves the entry point
while separating those choices:

```text
CI case: model/stage + target instance + recipe + hardware overrides
    → test_codegen.py: parse CLI and dispatch the model adapter
    → compilation/context.py: resolve target, family, recipe, hardware
    → model adapter: load/export/calibrate using the family's quantizer rules
    → compilation/pipeline.py: transform, compile, verify
    → target's backend: hardware-specific lowering and emission
```

For example, `resnet18 × voyager` with recipe `E4M3` and PE size `16,16`
selects the registered `voyager` target, its `voyager` family and backend,
applies that family's E4M3 defaults, and constructs one hardware configuration
with the requested geometry. The same model adapter can be paired with another
registered target without embedding its quantization rules in the adapter.

A target instance has a name, family, backend, hardware factory and optional
recipe overrides. Multiple instances can share one family's policies while
providing different hardware factories or overriding individual recipes.
Currently only the `voyager` instance/backend is registered; its CLI geometry
variants remain separate test cases. Gemmini and Trainium implementations are
future work, and their names are not accepted until registered.

Quantization resolution has two parts: the named `--quantization_recipe`
selects instance-specific defaults, falling back to the family, with optional
model/stage overrides and explicit CLI values taking precedence. The family's
`configure_model` then applies per-model/per-operator rules; `--qconfig`, when
used, selects a named per-operand table from that family. Selecting hardware
alone does not automatically choose a precision: each CI case explicitly
selects a recipe, while old raw-flag commands remain supported.

To add a supported model/target combination, register its backend and family
policy, register a `Target` with its hardware factory, then add an explicit CI
case containing the target and recipe. The framework supplies the selection
and dispatch points; backend lowering and quantization support still need to
be implemented and validated for that hardware.

## Resolution and ownership

`test/compilation/context.py` resolves the target and its family policy before
loading model or dataset assets. `CompilationContext` shares one hardware
instance across model preparation, fusion, transform, compile and reporting.
Each case writes `compilation.json` with resolved CLI options and the hardware
IR, without changing `model.txt`.

`src/voyager_compiler/targets.py` registers `Target` instances and backends.
An instance selects a family, a backend and a hardware factory. The hardware
factory constructs the target's own graph; it must not adopt Voyager topology
or parser defaults accidentally. A backend provides validation, fusion policy,
transformation and compilation. Only the Voyager backend is implemented.
Unknown targets, missing backends and unsupported Voyager topologies fail.

`quantization/recipes.py` defines the generic `Recipe` and `FamilyPolicy` types
and the family registration/lookup functions. It contains no hardware-specific
presets. `quantization/voyager.py` owns Voyager's presets, idempotent family
registration, and model-specific quantizer rules. A family supplies named
recipes, per-operand qconfig tables, and those model rules.
Recipes keep quantization settings separate from compilation defaults, with
optional model/stage overrides. Resolution order is:

1. Select the instance's named recipe, or fall back to its family's recipe.
2. Overlay that recipe's model/stage settings on its common settings.
3. Apply explicitly supplied CLI arguments.
4. Apply model quantizer rules, including the selected per-operand `--qconfig`.

Omitting `--quantization_recipe` preserves the legacy raw-flag workflow;
selecting a target alone does not silently enable a new precision. Voyager
recipes are E4M3, P8_1, INT8, MXINT8 and MXNF4. The store beat is derived from
the effective activation bit width and PE columns unless `--bank_width` is
supplied. Boolean recipe defaults can be overridden with `--no-bf16`,
`--no-force_scale_power_of_two`, `--no-quantize_fc`, etc.

Voyager's LLM tables and annotation helpers live in
`quantization/voyager_llm_configs.py`. The language-modeling example re-exports
them for compatibility. These configurations belong to the Voyager policy;
support for another hardware family must be implemented and validated separately.

## Stage map

Start with `test/test_codegen.py`, which owns the CLI and model dispatch.

| Concern | Owner |
| --- | --- |
| Target name, hardware factory, backend registration | `src/voyager_compiler/targets.py` |
| Immutable hardware graph and Voyager factory | `src/voyager_compiler/hardware_config.py` |
| Voyager topology and ISA-fusion adapter | `src/voyager_compiler/voyager_adapter.py` |
| Generic recipe/family types and lookup | `src/voyager_compiler/quantization/recipes.py` |
| Voyager presets, registration, and model rules | `src/voyager_compiler/quantization/voyager.py` |
| Voyager LLM tables and annotation helpers | `src/voyager_compiler/quantization/voyager_llm_configs.py` |
| Resolve CLI options once and write run metadata | `test/compilation/context.py` |
| Model loading, export, calibration, reference state | `test/utils/models/` |
| Shared transform/compile/verification orchestration | `test/compilation/pipeline.py` |
| Public backend dispatch and Voyager stage entry points | `src/voyager_compiler/__init__.py` |
| Tiling and cost search | `src/voyager_compiler/codegen/transform/tiling/` |
| Explicit buffers, tile loops, copies, waits, views | `src/voyager_compiler/codegen/transform/bufferize/` |
| Schedule estimation and calibration reports | `src/voyager_compiler/codegen/reporting/` |

Non-obvious invariants:

- All stages must see the same resolved hardware instance. CLI defaults must not
  override an explicit recipe; family policies must not leak across targets.
- Compute connections do not authorize fusion by themselves. Voyager fusion
  comes from explicit ISA pipelines and their legality predicates.
- `transform` and `compile` mutate the FX graph. Preserve vision's preprocessing
  extraction before final fusion. Restore decode KV-cache state before replay;
  replay is not necessarily idempotent at a chunk boundary.
- Shape propagation uses fake tensors. Running it is not numerical verification.
- Bufferization uses destination-passing computation with explicit memory and
  synchronization. Preserve allocation/view aliasing and copy/wait balance;
  a slot subview identifies storage, not a model slicing operation.
- The bufferizer and tiler must agree on SRAM occupancy and buffering depth.
  Memory units (bytes/elements/rows) and replicated capacity are explicit in IR.
- `compile` is backend-dispatched. Voyager's bufferized stages are reusable but
  are not a requirement for a future target. Do not restore the retired per-node
  path or unify paths unless that work is requested.

## Compilation stages

Model adapters in `test/utils/models` return `PreparedModel`: the exported and
quantized graph, example inputs, reference output, optional preprocessing and
state-restoration hooks. `test/compilation/pipeline.py` owns subsequent execution.
Vision preprocessing stays between transformation and final operator fusion.
The existing adapter `quantize_and_dump_model` functions remain wrappers.

The package's `transform()` and `compile()` dispatch by `config.backend`.
Voyager uses transformation → `lower_to_buffers()` → `emit_program()`.
`lower_to_buffers()` builds the tiler, bufferizes, and plans memory;
`emit_program()` writes the existing protobuf and tensor artifacts. Both
retain in-place graph semantics. `BufferizationOptions` holds algorithm choices
(single-buffer reduction tails, FA3 attention, boolean masks); buffer capacity
and slot count remain hardware properties. Other backends need not use these
stages or Voyager IR. This refactor does not restore or unify the retired
per-node emitter.

`--debug` executes the final bufferized graph for every model. Previously,
vision/BERT/MobileBERT verified the transformed graph before compilation.
A newly visible numerical warning is therefore not automatically a regression:
compare the generated program and reproduce the same execution stage against
the baseline. Do not loosen tolerances to hide a discrepancy. Stateful adapters
restore reference state before lowering. The optional `before_emit` observer
captures the lowered representation (including quantized caches and scale
tables) before tensor emission can execute it. Verification restores this state
and uses copies of input tensors and reference outputs.

## Regression comparisons

Case paths are `<target>/<network>/<scheme>/<geometry>`. The existing 26
cases all explicitly select Voyager. Targets without a PE-array override use
`default` for geometry. The matrix lists supported combinations, not a blind
Cartesian product. Legacy three-component baseline paths and Voyager selectors
remain recognized. `--suite` selects exact labels and rejects unknown ones;
`--only` retains prefix matching. `--baseline` pins a comparison so a failed
attempt cannot silently become the next reference.

`--jobs N` runs up to N model processes concurrently (default: 1). Each case
retains its own artifact directory and log; the final report stays in matrix
order. Parallel runs divide the available CPU affinity across workers, capped
at 32 PyTorch threads per process. `--threads-per-job N` overrides this budget.
Serial execution keeps the previous 32-thread default. Start with two workers:
large models require separate memory, and excessive concurrency can slow the
suite. Interrupting a run cancels queued cases and terminates active processes.

Use the checkout's active Python environment (in this workspace,
`/home/zhouhua/Research/ML/ml-env/bin/python`).

1. Inspect `test/run_ci.py`, the marked cases in `test/regression_suite.txt`,
   and the fixed reference report. Use a pre-change baseline or capture the
   selected cases from an immutable copy before editing their source.
2. Use `run_ci.py --list` to inspect case commands. Run the marked suite with
   `run_ci.py OUT --baseline FIXED_RUN --suite test/regression_suite.txt`.
   Unmarked cases are not required unless broader coverage is requested.
3. Run `test/test_hardware_config.py` and `test/test_compilation.py` for the
   configuration and shared pipeline contracts.
4. Inspect every selected comparison, not just the process exit code. Compare
   `model.txt` contents and failure signatures, and retain the report and
   resolved `compilation.json` data. NEW, MISSING or unverified selected cases
   do not establish equivalence. Baseline-only cases intentionally excluded
   by the suite can appear as MISSING in the report.

Numerical warnings remain visible and non-gating; compile failures and artifact
differences fail CI. Identify existing failures from the chosen reference rather
than hardcoding exceptions. Compare the same execution stage and initial state
when assessing warnings; reproduce against old source instead of loosening
tolerances. Record environmental failures separately from compiler failures.
Do not claim success from an import-only, skipped, or incomplete selected suite.
