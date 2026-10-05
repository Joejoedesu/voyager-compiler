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
This checkout registers `voyager`, `trainium-v2`, and `trainium-v3`. Voyager's
CLI geometry variants remain separate test cases. Trainium reuses the shared
bufferized pipeline and converts its emitted collaterals to NKI; see
[Trainium support and limits](trainium.md). Gemmini is not registered here.

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
transformation and compilation. Voyager and Trainium backends are implemented
here. Unknown targets, missing backends and unsupported topologies fail.

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
| Observer-time operation rules and shared calibration groups | `src/voyager_compiler/quantization/rules.py` |
| Resolve CLI options once and write run metadata | `test/compilation/context.py` |
| Model loading, export, calibration, reference state | `test/utils/models/` |
| Shared transform/compile/verification orchestration | `test/compilation/pipeline.py` |
| Public backend dispatch and Voyager stage entry points | `src/voyager_compiler/__init__.py` |
| Tiling and cost search | `src/voyager_compiler/codegen/transform/tiling/` |
| Explicit buffers, tile loops, copies, waits, views | `src/voyager_compiler/codegen/transform/bufferize/` |
| SRAM-region eligibility, placement trials, and boundary transfers | `src/voyager_compiler/codegen/transform/bufferize/residency.py` |
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

## Observer-time quantization rules

After normal operator/model annotations and before PT2E injects observers,
`XNNPACKQuantizer` runs its selected `QuantizationRule`s. A rule supplies a name,
operation targets, a graph collector returning edges/outputs that must share
parameters, and an applicability predicate over their resolved specs and target
context. The scanner merges overlapping groups and validates their specs before
writing `SharedQuantizationSpec` annotations. Graph folding is independent and
is not needed to establish these observer groups.

`FamilyPolicy.quantization_rules(context)` selects rules. The context contains
the target, hardware instance, model, model kind, and resolved CLI options, so a
family can vary its selection by hardware instance and model. A family without
that callback has no graph constraints. Direct library callers can use
`quantizer.set_quantization_rules(rules, context)`.

Voyager selects `CONCAT_INT8`: static per-tensor INT8 concat chains, including
supported reshape/transpose/alias views, share a calibrated scale. Applicability
uses resolved per-operation specs, including mixed-precision overrides. The
scanner constrains participating edges and concat/view outputs; a source's other
consumers remain separate. Conflicting precision/range/observer settings within
the required group fail explicitly. Implicit PT2E observer sharing is disabled
on affected fan-out consumers to prevent a same-dtype sibling from inadvertently
joining the group.

`SharedAmaxObsFakeQuantize` collects the cumulative maximum across every branch
and calibration call, including the current call. It passes values through
while observing, so changing branch order cannot change downstream statistics
through a moving fake-quant scale. Disabling observers enables fake quantization
with the common frozen scale; conversion uses that scale directly. Per-call
finite histories are intentionally not used for these groups. The first rule
supports symmetric per-tensor amax PTQ; QAT with graph rules is rejected, and
per-channel, microscaling, and group-wise affine schemes are not automatically
merged. `model.meta['quantization_rule_groups']` records rule names and members.

## Resident bufferization

`BufferizationOptions.flow` / `--bufferized_flow` selects `per_kernel` (the
original default) or `resident`. The new flow analyzes existing fused ISA
nodes before DRAM load/store construction. A region shares on-chip storage
between its kernels; its members remain separate ISA operations.

```sh
python test/test_codegen.py resnet18 --target_hardware voyager \
  --quantization_recipe INT8 --pe_array_size 16,16 \
  --bufferized_flow resident --parameter_loading on_demand \
  --model_output_dir /tmp/resnet-sram --debug
```

Residency is proposed before consulting the DRAM-streaming scheduler. Each
proposal reserves whole activation tensors and checks their lifetimes with the
actual allocator. Matrix kernels are then scheduled with those operands pinned
in SRAM: the search charges the simultaneously live, bank-rounded allocations
plus reduction workspace, without pricing DRAM transfers on internal edges.
Eligible boundary input loads and output stores are priced with their first-use
traffic and pipeline overlap. This lets boundary kernels select multiple compute
tiles even when the full tensors fit in SRAM. The search score averages reuse
and reduction phases; the emitted loop is walked for the final timing report.
Its cache key includes the live allocation budget, weight placement, and boundary
roles, so an internal-only or streaming schedule cannot answer a boundary query. Pointwise and
pooling kernels at a region boundary use their existing tiling builders;
internal pointwise kernels retain whole SRAM operands. Region
growth preserves graph order and connected dependencies. Retained branches and
multiple outputs extend lifetimes until their final access or boundary store.
Compatible contiguous views remain aliases; views needing an offset/layout
change or exposing a later external mutation retain the original path.
Compute destinations are allocated
before their instructions, so they cannot reuse banks that the same instruction
is still reading.

The initial bank policy is conservative: distinct simultaneously-live buffers
occupy disjoint whole banks. This satisfies all member kernels' concurrent
accesses without assuming that kernel-local bank groups can be merged. It can
reject a region that a more elaborate bank-sharing schedule could execute.
Address reuse after last access uses the existing allocator. The ordinary
convolution/GEMM builders retain their compute grid, reduction accumulation,
and fused-tail completion rules, but access resident operands using SRAM
subviews instead of DMA slots. A completed output tile is written into its
resident tensor; multiple compute tiles are not a residency rejection reason.

Padded dense and supported 3x3 depthwise convolutions use Voyager's native
boundary-zero generation (Matrix InputController / DwCUnit). Their SRAM inputs
remain unpadded and their compute instructions retain the convolution padding;
there is no padded DMA or SRAM-to-SRAM staging copy. The outer spatial grid
keeps the entire image visible to each invocation, while IC/OC tiling, split
reductions, and hardware-internal L1 spatial tiling remain available. This
prevents internal tile edges from being mistaken for image boundaries. The
search constraint and its cache identity enforce this rule; explicit spatial
tilings that violate it fall back. Native padding currently requires symmetric
padding of at most three, equal spatial strides, dilation one, and no input
codebook. Depthwise additionally requires a 3x3, multiplier-one kernel and the
backend's supported stride geometry. Other cases retain the original path.
Boundary ingress can pipeline unpadded channel windows; its cost uses the actual
input dimensions, including for stride-two convolutions.

The scheduled proposal is allocated again with its actual scratch/staging
buffers. Provisional addresses are cleared before every allocation. Both
byte-capacity and bank-granularity failures are recorded, and a rejected
extension closes the last feasible region. A failed search is reported as
no resident schedule found, not proof that residency is physically impossible.
An infeasible singleton uses `per_kernel`. A failure in
final whole-graph allocation rebuilds the original graph through `per_kernel`;
unexpected compiler errors are not swallowed.

The two `--parameter_loading` strategies are:

- `preload`: load a region's weights, biases, and stored quantization parameters
  before its first kernel. They consume capacity from region entry until last use.
- `on_demand`: first try loading each parameter at first use and retaining it
  until last use within the region. If placement/scheduling fails, retry with
  ordinary dense matrix weights streamed through a single tile buffer while
  activations stay pinned. Consecutive uses of the same weight tile reuse it.
  This bounded retry currently excludes GEMV, MX weight-scale pairs, and sparse
  formats. Each element of a retained parameter transfers once per region;
  that transfer may be tiled. Streamed weights may reload according to the
  compute grid. This is the default strategy.

Neither strategy promises persistence across model invocations. Region records
include `streamed_parameters` to distinguish retained and streamed weights.

Instruction-immediate scalars and lookup tables retain their existing treatment;
scalar-producing kernels use the original scalar path rather than region DMA.
Activation ingress/egress uses the existing DMA and wait primitives, pipelined
with the boundary kernel's compute tiles. The first input tile is primed before
the loop; each next first-use tile is prefetched directly into its window of the
full SRAM allocation before waiting for the current tile. Grid revisits and
later consumers reuse those windows without another load. This adds no duplicate
activation staging buffers. Input windows must be non-overlapping and cover the
whole tensor; halo or transposed input windows retain the whole-load path.
Preloaded parameters still load before the region's first kernel.

An externally visible output tile stores after its reduction and fused tail
finish, while the next tile computes. The final store is drained before leaving
the kernel. The SRAM output remains available to other region members, including
when that intermediate is also an external output. Internal edges use SRAM
buffers directly, without SRAM-to-SRAM DMA. A one-tile pointwise operation keeps
the simpler whole-operation path, since there is no next tile to overlap.
Execution is sequential between kernels; compute dependencies are preserved
without an additional asynchronous cross-kernel schedule. Pooling halo transfers, external
mutations, sparse kernels, explicit reduction workspaces, dynamic operands, and
unsupported layouts use the original builders. An in-place ISA tail operating
on a fresh internal intermediate is allowed.

For `resident`, the compiled SVG is rendered after the final placement decision,
using the original kernel graph. Graphviz boxes identify accepted regions, their
parameter strategy, and bank-rounded SRAM footprint. `sram_regions.json` records
the same accepted regions and fallback reasons. Existing schedule reports count
the emitted boundary transfers, so activation/weight DRAM bytes reflect the
chosen flow. A fully resident single-input/output graph needs only activation
ingress and egress, plus its parameter transfers. Each accepted region occupies
one contiguous layer range in `layers.txt`, including all of its boundary stores.

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

The CI driver also accepts `--bufferized-flow resident` and
`--parameter-loading preload|on_demand`, forwarding them to every selected case.
Use a separate output directory for each flow/strategy. Compare the default
flow to the fixed reference; Residency planning intentionally changes emitted programs,
so validate its numerical results and transfer counts separately. `NEW` means
a new artifact was produced, not that it matched the default-flow reference.

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
