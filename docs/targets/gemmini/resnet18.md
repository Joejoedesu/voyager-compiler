<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ResNet-18 workload binding

The `W-resnet18-gemmini-20260818` binding is a qualified best-effort mapping,
not connected model execution, classifier accuracy, or target-performance
evidence. The frozen workload is
`torchvision.models.resnet18(weights=None).eval()` with batch one, seed
`20260818`, and one `[1, 3, 224, 224]` FLOAT32 input.

The reusable implementation is the public
[Gemmini workload adapter](../../../conversion/gemmini/workload.py). It
dispatches ordinary canonical regions by semantic operation; helper
projections are not ResNet-specific. This report is model-specific because it
owns the frozen workload, coverage denominator, and retained result history.

## Current assignment

The intended current compiler package contains 30 target-owned shared-semantic
regions:

- 20 Conv regions projected to `tiled_conv_auto`;
- eight residual Add/ReLU regions projected to `tiled_resadd_auto`;
- one global average region projected to `tiled_global_average_auto`; and
- one final Linear region projected to `tiled_matmul_auto`.

Inference BatchNorm is folded into Conv parameters because the qualified
Gemmini helper library has no BatchNorm helper. ReLU consumes signed-INT8
qcodes within the admitted Conv or residual region. The stem MaxPool is a Conv
helper epilogue. Flatten is a layout-only boundary and does not emit a helper.

The coverage denominator is 69 source semantic occurrences and 49 occurrences
after BatchNorm folding. A complete mapped campaign must account for both
denominators even though it emits only 30 helper calls.

## Runtime-capture dtype boundary

Runtime capture separates storage representation from semantic type. Every
payload is a contiguous little-endian FLOAT32 container because that is the
canonical tensor-dump format. The capture manifest's `logical_dtype` instead
records the final-FX domain consumed by the helper boundary.

The stem Conv consumes logical FLOAT32 data. The other 19 Conv regions consume
integral signed-INT8 codes held in FLOAT32 containers, and all 20 captured
outputs are logical INT8. The producer derives those labels from the validated
final-FX regions and checks integrality and range before writing. The converter
validates the declarations and never relabels or repairs payloads.

## Recorded W evidence

The W endpoint candidate compiled all 30 mapped regions from one canonical
package. The compiler semantic gate compared all 1,000 final outputs exactly
before serialization. Conversion passed 30/30, and 30 independent Spike runs
passed complete readback for 20 Conv, eight Add, one global-average, and one
Linear region.

Across all 2,635,752 signed-INT8 output codes, 2,367,030 were exact and 268,722
differed by one code. No value exceeded `rtol=0`, `atol=1`, and the maximum
absolute error was one. This completes W for the fixed pair. These exact and
one-code counts are W accounting, not an X evaluation: Gemmini X has not
started, the `_auto` helper point is partial-control, and the same frozen W
binding has not been exercised through a full-control point.

The sole preserved Voyager compiler-collateral identity and the 30/30
conversion status are recorded in [operations.yaml](operations.yaml). The
recorded counts are result history, not a packaged golden; promotion requires
a fresh run of every gate below.

## Golden ownership and promotion

The exact generator and run recipe belong in this repository. A promoted
descriptor should live under
`conversion/gemmini/goldens/resnet18-w/<version>/` and bind the complete
compiler inventory, W aggregate, contracts, stack lock, and source revision.
The 86 MiB compiler payload bundle should be stored immutably by inventory hash
under a configured `GEMMINI_ARTIFACT_ROOT`. If clone-local payloads are
required, first adopt Git LFS rather than adding ordinary large blobs.

Generated collateral is only a candidate until the exact compiler semantic
gate, all 30 conversions, all 30 endpoint executions, complete-output
comparison, and the single Voyager-collateral inventory check pass against that
same package. Converted artifacts are regenerated and judged by pass count,
not frozen by content hash.
Promotion copies the package unchanged into a new immutable version. Never
relabel or regenerate a golden in place; any content change requires a new
version and the complete promotion gate.

For ordinary converter maintenance, a five-region sanity set may cover the
stem FLOAT32-ingress Conv, one downstream INT8-ingress Conv, one residual Add,
global average, and Linear. That subset cannot authorize golden replacement or
a new W claim; promotion always requires all 30 regions.

## Claim limits

Each helper region is executed independently from canonical or captured input.
Passing all rows does not prove connected ResNet-18 execution or accelerator-
resident intermediate transfer. Functional Spike does not provide target
cycles, frequency, traffic, utilization, energy, latency, speedup, or RTL
evidence. Dataset accuracy is not measured.
