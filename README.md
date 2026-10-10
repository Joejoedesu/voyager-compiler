# Voyager Compiler

The Voyager Compiler is a hardware–software co-designed machine learning (ML) compiler that efficiently maps PyTorch models onto Voyager-generated deep neural network (DNN) accelerators.

The compiler ingests PyTorch models, extracts a static compute graph using [PyTorch 2 Export (PT2E)](https://docs.pytorch.org/docs/stable/export.html), and applies [PT2E-based quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html). It incorporates a custom quantization framework that provides fine-grained control over data types—including low-bitwidth integers, floating point, posit, and NormalFloat—as well as advanced techniques such as mixed-precision and codebook quantization.

After quantization, the compiler lowers models through hardware-aware operator fusion, architecture-specific optimizations, scheduling, and accelerator instruction generation. Voyager produces a hardware-oriented intermediate representation (IR) built on the PyTorch FX graph, which is serialized via [Protocol Buffers](https://github.com/protocolbuffers/protobuf) and consumed by a C-based backend to generate the final accelerator instruction bitstream.

## Getting Started

End-to-end examples of using the Voyager Compiler can be found in test/test_codegen.py, which demonstrates model ingestion, quantization, compilation, and instruction generation.

## Accelerator backend adoption

The shared bufferized flow supports explicit hardware, mapping and realization
interfaces. See [compilation and target adoption](docs/compilation.md#extensible-bufferized-targets)
for the Voyager/Gemmini implementations, context API, validation and branch
publishing instructions.

## Verified Models

Validated on the following ML models:
- CNNs (ResNet, MobileNet, EfficientNet)
- Transformers (BERT, ViT, MobileBERT)
- LLMs (GPT-2, LLaMA)
- Detection and sequence models (YOLO, Mamba)

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

## Current Trainium development branch

The `trainium-isa-10-09` branch contains the consolidated compiler and physical ISA
performance model in `src/`, bufferized lowering in
`src/voyager_compiler/codegen/transform/bufferize/`, and Trainium tests in `test/`.

The benchmark workflows, exploratory scripts, profiling/report tools, hardware
results, and historical experiment outputs are kept outside this repository in the sibling
`../trainium-isa-10-09-archive/` directory. They are distributed separately from
the compiler source. The experiment-derived implementations are integrated into
`src/`; no experiment directory is required to compile or run the unit tests.
