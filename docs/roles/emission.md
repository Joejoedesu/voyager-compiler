# Encoding and emission

Use for: Canonical encoding and target collateral generation from frozen decisions.

## Binding and ownership

`TargetSpec.encode / TargetSpec.emit`. Project frozen decisions into the canonical artifact and target collateral. Runtime submission is a separate extension.

Encoding always saves a bundle. `encode(ir, program, config)` returns `EncodedIR`:
common protobuf `model`, JSON-compatible `target_data`, and optional NumPy-array
`payloads`. Capture every selected mode, geometry, address, constant and target
field that emission needs; version the target records. The shared writer saves
`model.txt`, `target_ir.json`, payloads, plan/config/bindings and their checksums.
A target without an encode hook uses the common projection.

`emit(bundle, output_directory)` reads a verified `SavedBundle`, with helpers for
JSON, protobuf and payload arrays. It has no live graph or hardware object.
Translate saved operations into commands or code without making new mapping
choices. A target consuming Voyager IR directly needs no emit hook.

```text
frozen program → encode → saved bundle → emit → target files
                           └ usable in a separate process ┘
```

Compilation calls emission by default. Use `emit_target=False` for encode-only,
then run from the repository root:

```bash
PYTHONPATH=src python -m voyager_compiler.compiler.emit BUNDLE OUTPUT
```

The saved bundle is schema 2; older bundles lack the complete emission contract
and must be recompiled. Do not pickle FX graphs or recreate search during loading.

## Load for this task

Frozen program/plan/bindings, target endpoint/ABI and selected recipe/policy, reference boundary and runtime revisions.

Read the relevant [shared contract](../compilation.md#execution-model-and-converter-consistency) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Frozen emission](../../src/voyager_compiler/compiler/emit.py)
- [Bundle/checksums](../../src/voyager_compiler/artifacts.py)
- [Runtime entry](../../src/voyager_compiler/runtime.py)
- [Target encoders, emitters and runtime](../../src/voyager_compiler/targets/)

## Generated output location

Use a fresh `results/<target>/<run>/` directory for local compilation and replay
collateral; the repository ignores `/results/`. `output_dir` remains caller-selected,
so external artifact storage is also supported. Keep each compilation bundle in
its own directory: `save_bundle` hashes every existing file below that directory.
The bundle is sealed before emission. `emission.json` separately records the
source bundle digest and generated-file checksums. Independent emission should
use a fresh sibling output directory; convenience compilation writes target
subdirectories beside the IR. Replay packaging and measurements also belong in
separate sibling directories and are not covered by the compilation manifest.

Keep source, small test fixtures and reference checksum specifications in Git.
Generated bundles, tensor binaries and per-run reports belong in artifact storage;
share their location and identity when retaining a reference. Do not globally ignore
JSON files: configuration, schemas and reference pins are maintained source inputs.

## Implementation decisions

| Problem | Next action |
| --- | --- |
| First target emitter is missing | Start with one selected operation and an endpoint-accepted artifact; map frozen operands, formats, dimensions, addresses and waits to its ABI. Include required host/boundary work. |
| Numerics pass but timing violates modeled reuse | Trace one load/compute/store sequence across tile iterations. Repair transfer scope or missing dependencies with model/builder/runtime owners; do not retile in translation. |
| Destination compiler rejects a form the simulator accepts | Check instruction operand/layout/resource restrictions and versioned API rules. Return new legality constraints to the candidate or operation owner. |

## Do

1. Consume finalized arithmetic, mapping and addresses. Projection may validate/encode/pack; it cannot repair missing semantics, retile or choose another ISA mode.

2. Include transfers, views, host conversions and state operations in the complete driver where needed. Native layer files alone may omit observable program work.

3. Hand the command/submission contract to the [runtime role](runtime.md). Review together when inserted loads/stores, fences or launch order may change reuse/overlap.

4. Record which downstream decisions remain native-owned. Preserve reference provenance; an endpoint-generated expected stream is not an independent semantic reference.

## Do not

- Reinvent tiling, choose a different ISA recipe, assign new addresses or repair missing arithmetic in the converter. Return those changes to search, allocation or semantics.
- Insert or move transfers/fences in a way that changes assumed reuse/overlap without coordinating the builder/model contract and validating the affected behavior.

## Optional replay reuse

For a flat native program, [shared replay images](../../src/voyager_compiler/replay.py)
provide finalized-allocation bounds, poison, input staging and a separate expected
image. Supply the target tensor encoder; keep endpoint command/manifest formats
in this extension. Nested/asynchronous programs reject explicitly and need their
own replay traversal. Never initialize intermediate accelerator inputs from
expected outputs. Reuse these checks rather than adding parallel image validators.

## Handoff and validation

Test emission from a relocated bundle in a new process and compare output bytes.
Include target constants/payloads and reject missing or changed inputs before
emission. Return encoded artifacts, the complete driver description and emission checks. Hand execution to the runtime owner and report transfer/control differences to model/builder owners.

Check actual operand/address/mode projection plus the endpoint when available. Host replay and syntactically valid code/commands are separate from device correctness and timing.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)
