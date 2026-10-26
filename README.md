# Voyager Compiler

The Voyager Compiler is a hardware–software co-designed machine learning (ML) compiler that efficiently maps PyTorch models onto Voyager-generated deep neural network (DNN) accelerators.

The compiler ingests PyTorch models, extracts a static compute graph using [PyTorch 2 Export (PT2E)](https://docs.pytorch.org/docs/stable/export.html), and applies [PT2E-based quantization](https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html). It incorporates a custom quantization framework that provides fine-grained control over data types—including low-bitwidth integers, floating point, posit, and NormalFloat—as well as advanced techniques such as mixed-precision and codebook quantization.

The shared bufferized pipeline lowers models through normalization, implementation search, explicit buffers/transfers, allocation and frozen target emission. Its FX program and selected plan are authoritative; protobuf and target artifacts are projections. The Voyager target retains its existing downstream bitstream flow, while other targets emit their own instruction or runtime artifacts.

Start with [documentation navigation](docs/README.md) for target registration, workload × target binding and focused role readers. The implementation and extension-point reference is [docs/compilation.md](docs/compilation.md). It defines stage ownership, the current public API, scheduler migration gaps, coverage-first objectives, target-specific model–converter debugging guidance and evidence requirements. Start there when adding a workload, hardware target or lowering recipe.

## Getting Started

Use the minimal `WorkloadSpec` / `compile_workload()` example in the [extension guide](docs/compilation.md#extension-policy-and-minimal-examples). The existing [test/test_codegen.py](test/test_codegen.py) exercises the compatible Voyager frontend. [Roles and handoffs](docs/roles.md) identify who owns each change and its validation. Public extension contracts live in [extensions.py](src/voyager_compiler/extensions.py); [pipeline.py](src/voyager_compiler/pipeline.py) owns stage order.

## Model families and validation scope

The existing frontends/examples include these model families; this is not a
claim that every family is currently validated on every target:

- CNNs (ResNet, MobileNet, EfficientNet)
- Transformers (BERT, ViT, MobileBERT)
- LLMs (GPT-2, LLaMA)
- Detection and sequence models (YOLO, Mamba)

The default Voyager target uses an 11-case pinned development gate. For other
bindings, supported operations, touched extension points and validation commands,
read the relevant [target guide](src/voyager_compiler/targets/README.md).
Compilation success, device
coverage and measured hardware utilization are separate claims. Coverage summaries
and endpoint measurements support particular adoption/performance tasks; a common
coverage subsystem, universal execution checker and fresh measurements for every
target are not shared-framework requirements.

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
