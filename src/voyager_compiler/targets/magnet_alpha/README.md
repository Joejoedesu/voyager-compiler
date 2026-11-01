# Alpha target

This guide specializes the shared [compilation contract](../../../../docs/compilation.md).
Registration is [`target.py`](target.py); start from these callable bindings, not
a historical completion tracker.

## Extension points used

| Extension | Binding and responsibility |
| --- | --- |
| `TargetSpec.hardware` | `hardware.py:MagnetAlphaHardwareConfig`: engines, bank geometry and physical capacities |
| Workload preparation | `workloads/yolo5_face_nv.py`: external model/checkpoint, frozen state and pair-specific input preparation |
| `quantization` | `quantization_flow.py:POLICY`: target format conversion and optional exact-arithmetic recalibration with the caller's resolved hardware configuration |
| `operators` and `legalize` | `operators.py`, `quant_ops.py`, `target.py:legalize`: exact integer primitives, graph/quantization rewrites, dtype-aware channel padding, input preprocessing and region formation |
| `implementations` / `model` | `magnet-alpha.whole`, `build_alpha`, `AlphaModel`: shared whole-tensor construction with instruction constants; no characterized candidate timing model |
| `refine` | `target.py:refine`: memory-instance binding, shuffles, shared residency refinement and `schedule.py:select_schedules` before freeze |
| Storage declarations | `bank_layouts`, `layout.py`: packed scale components, physical shapes, clustered banks and port conflicts; shared allocator assigns final addresses |
| `encode` | `codegen.py`: resolved schedules, allocated addresses, explicit driver and binary payload arrays in the saved target IR |
| `emit` | `codegen.py`: read the saved target IR/payloads and write CMOD files and driver; no live graph required |

There is no `runtime` or `characterize` hook bound
here; numerical legalization is in `legalize`, and host replay is not a C-model
endpoint. Default shared stages still own allocation, freezing and bundling.

Depthwise dispatch is target-owned: legalization forms an accumulator plus NMP
requantization region, while dense convolution uses PE/PPU. Region formation in
[implementations.py](implementations.py) selects those instruction families;
[schedule.py](schedule.py) validates depthwise groups and selects NMP fields.
Alpha does not rely on the shared dense-convolution tiler or Voyager's fallback.

## Target quantization

`QuantizationPolicy.run` consumes `options["alpha_quantization"]`, an
`AlphaQuantizationRecipe`. It supplies a quantizer, optional region qspec, replayable
`samples()` returning `(args, kwargs)`, a step count and optional profile loader.
An explicit caller quantizer wins; `quantizer=None` skips this stage for an already
quantized graph. `bias` can also be overridden. The target runs preparation,
initial calibration, optional exact PE/NMP arithmetic setup and recalibration,
then Alpha's VSQ/symmetric converter. There is no second shared conversion.

The YOLO binding owns model/checkpoint loading, sample preprocessing and released
profile interpretation. Its prepare hook supplies the recipe, not a quantized
model or another default hardware config. `quantization_flow.py` imports no
workload module and can be reused with another model. A saved-profile loader
installs its state and returns provenance; the target subsequently configures
exact arithmetic without resetting those observers. Fresh multi-pass calibration
rejects exhausted/incorrect sample counts. Pure payload packing does not trigger
recalibration. The returned host reference models numerical behavior; native bank
encoding and endpoint validation remain separate.

## Numerics, padding and execution

Mixed W4A4/W8A8 inference/PTQ, domain propagation, PE/PPU/NMP/PVS operations and
embedded scales are retained. The donor-specific QAT trainer/optimizer/phase
orchestration is not ported. The workload imports external YOLO model source and
a verified checkpoint, not the donor compiler pipeline.

`padding.py` runs before region formation. Dtype-dependent channel lanes and
epilogues preserve logical extents in `hardware_logical_shape`, keep required
physical lanes through the epilogue, then slice logical outputs. Later bank layout
may enlarge `physical_shape` solely to reserve backing storage. These are distinct
from the workload's input-channel preparation for selected early convolutions.
Do not replace them with one fixed array-width padding rule.

The whole-region recipe uses deterministic `schedule.py:local_factors/select_schedules`.
It does not expose the donor's rule-based-versus-Interstellar provider choice and
has no calibrated timing model. Schedules remain selected before freeze. Hardware
scheduling uses compiler-selected global bank addresses; native scheduling does
not imply native allocation ownership.

Alpha schedules use explicit transfers for materialized split/concat views and
new addresses, so schedules are not byte-identical to the old resident program.
All arithmetic fields and payloads are checked against the reference, allowing
only documented storage-descriptor/default-field differences.

For Alpha, `alpha-cmod/layers.txt` alone is not a complete buffered execution
program. Use `alpha-cmod/program.json` together with the layer files and
canonical `model.txt`: it records transfers, views and six host output conversion
nodes. Host tensors model logical values; native bank packing is an encoding
contract, not a physical-bank simulator. No Alpha C-model or device run is claimed.
The pinned synthetic zero-calibration recipe also retains the documented
19-layer depthwise scalar-bias projection; matching it is not an accuracy result.

## Pinned workload check

Alpha acceptance verifies sealed reference checksums and checkpoint, exact
outputs, every intermediate payload, layer order/kinds, schedule arithmetic,
405 selected addresses, native bank-port disjointness, complete buffer-program
accounting and bundle checksums. Its `cmod_passed` field means collateral checks,
not C-model execution.

```bash
PYTHONPATH=src:../.validation-packages \
VOYAGER_YOLO_ROOT=../voyager-nv-integration/libraries/yolov5-face \
/home/zhouhua/Research/ML/ml-env/bin/python validation/run_alpha_reference.py \
  ../reports/alpha-RUN \
  --checkpoint ../voyager-nv-integration/libraries/weights/yolov5n-0.5.pt
```

Run from the repository root into a fresh output directory. The reference under
`reference-point/alpha-yolov5n05-2026-10-23` pins checkpoint, seed-0 synthetic input
and zero-calibration recipe. The checked scope includes three exact outputs,
631 intermediate payloads, 144 native calls, 174 views, 1,275 buffer operations
and six host output conversions. Lowered counts are not source-op coverage.
The synthetic recipe retains a 19-layer depthwise scalar-bias projection; matching
it does not establish model accuracy. See the
[dated acceptance](../../../../../reports/archive/voyager-nhwc-preparation-2026-10-24/acceptance.json).
Donor: `497bfbd40d08690345c769ec8983e4f9ad878c0e`; preserve its source notices.
