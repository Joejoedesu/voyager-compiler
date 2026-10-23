<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager Compiler

The Voyager Compiler is a hardware–software co-designed machine learning (ML) compiler that efficiently maps PyTorch models onto Voyager-generated deep neural network (DNN) accelerators.

The compiler ingests PyTorch models, extracts a static compute graph using [PyTorch 2 Export (PT2E)](https://docs.pytorch.org/docs/stable/export.html), and applies [PT2E-based quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html). It incorporates a custom quantization framework that provides fine-grained control over data types—including low-bitwidth integers, floating point, posit, and NormalFloat—as well as advanced techniques such as mixed-precision and codebook quantization.

After quantization, the compiler lowers models through hardware-aware operator fusion, architecture-specific optimizations, scheduling, and accelerator instruction generation. Voyager produces a hardware-oriented intermediate representation (IR) built on the PyTorch FX graph, which is serialized via [Protocol Buffers](https://github.com/protocolbuffers/protobuf) and consumed by a C-based backend to generate the final accelerator instruction bitstream.

## Getting Started

End-to-end examples of using the Voyager Compiler can be found in test/test_codegen.py, which demonstrates model ingestion, quantization, compilation, and instruction generation.

Target-neutral shared unit tests live in `test/unit_tests` and can be run with:

```bash
python -m unittest discover -s test/unit_tests -p 'test_*.py'
```

Backend-specific behavior is validated through the selected target's
end-to-end compilation contract. A target failure may use a temporary
unit-style diagnostic, but production target dependencies do not belong in
the shared unit-test directory.

### DRAM and scratchpad memory planning

The default ``per_operation`` scratchpad mode preserves Voyager's historical
two-level behavior. Full tensors receive liveness-based DRAM addresses in the
protobuf ``memory`` field, while each operation receives an independent tile
placement in the ``scratchpad`` field. Pass ``--total_memory BYTES`` to bound
the DRAM allocator.

Pass ``--scratchpad_mapping end_to_end`` with a hardware profile to keep the
compiled graph in accelerator-global memory. Inputs and parameters retain DRAM
backing addresses for initial loading, but computed intermediates have only a
model-wide ``scratchpad`` address. Parameters remain persistent; activation
storage is returned to a first-fit free list after its final graph use.

The hardware profile's global memory instances are the scratchpad capacity,
partition, and alignment source of truth. Weight and bias tensors are routed to
global weight memory; activations, outputs, and writable scales are routed to
global activation memory. Allocation fails instead of silently spilling when
a configured instance is too small or cannot be accessed by every declared
consumer compute unit. PE-local input, weight, and accumulation
buffers remain scheduler-owned and are not assigned model-wide tensor
addresses. Local scheduler selection is independent of the scratchpad mapping
mode.

For example, compile Magnet Alpha YOLO into model-wide global-memory addresses with:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export VOYAGER_YOLO_ROOT=/path/to/yolov5-face

python test/test_codegen.py yolo5-face-nv \
  --pt2e \
  --target magnet-alpha \
  --conv2d_im2col \
  --scratchpad_mapping end_to_end \
  --execution_order dp \
  --calibration_steps 0 \
  --local_scheduler magnet_alpha_rule_based \
  --model_name_or_path /path/to/yolov5n-0.5.pt \
  --model_output_dir models/yolo5-face-nv-scratchpad
```

The supported Magnet Alpha YOLO collateral contract uses AoV's fixed mixed
W4A4/W8A8 policy. Default-W8/W8A8 dumps are historical experiments and are
not accepted as closure or regression artifacts.

Magnet Alpha YOLO uses the native 3x3 first-layer contract with Alpha channel
padding through the combined exact-pair adapter in
`test/utils/models/yolo5_face_nv_magnet_alpha.py`; it is a direct
`yolo5-face-nv` × `magnet-alpha` route, not a generic `test_codegen.py`
default or a policy-only overlay.

For Magnet Alpha, end-to-end placement also resolves Alpha's implicit channel
movement. Eligible concat producers share one backing address and carry
``is_to_concat`` plus ``concat_*`` write-mode kwargs. A channel shuffle keeps
one logical FX output while serializing its ordered physical destinations in
``output_addresses`` and the corresponding
``output0_memory_instance``/``output1_memory_instance`` fields. Its explicit
``cpyN`` records contain exact physical source and destination scratchpad
views. Concat and split/view nodes are
serialized as ``nop`` only after the target allocator proves the implicit
addressing legal.

## Compiler model coverage

Compiler examples and historical checks cover the following model families:
- CNNs (ResNet, MobileNet, EfficientNet)
- Transformers (BERT, ViT, MobileBERT)
- LLMs (GPT-2, LLaMA)
- Detection and sequence models (YOLO, Mamba)

This list is not target-adoption evidence. An R/W/X claim must name a target,
workload scope where applicable, injection point, comparison rule, and linked
regression result under `docs/targets/`.

## Citation

If you use Voyager Compiler in your research, please cite:

```bibtex
@misc{prabhu2025voyagerendtoendframeworkdesignspace,
  title={Voyager: An End-to-End Framework for Design-Space Exploration and Generation of DNN Accelerators},
  author={Kartik Prabhu and Jeffrey Yu and Xinyuan Allen Pan and Zhouhua Xie and Abigail Aleshire and Zihan Chen and Ammar Ali Ratnani and Priyanka Raina},
  year={2025},
  eprint={2509.15205},
  archivePrefix={arXiv},
  primaryClass={cs.AR}
}
```
