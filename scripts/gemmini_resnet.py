"""Run ResNet through the existing vision adapter and shared compiler pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from compilation.context import parse_args, write_manifest
from compilation.pipeline import compile_prepared
from PIL import Image
from test_codegen import build_parser
from torchvision import models
from utils.models import torchvision_models

from voyager_compiler import get_default_quantizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--image", type=Path, required=True)
p.add_argument("--output", type=Path, default=Path("results/gemmini/resnet50"))
p.add_argument(
    "--interstellar-cost-tradeoff",
    action=argparse.BooleanOptionalAction,
    default=False,
)
p.add_argument("--runtime-tolerance", type=float, default=None)
a = p.parse_args()
torch.manual_seed(0)
torch.set_num_threads(4)
torch.set_grad_enabled(False)
args = parse_args(
    build_parser(),
    [
        "resnet50",
        "--target_hardware",
        "gemmini",
        "--quantization_recipe",
        "INT8",
        "--model_output_dir",
        str(a.output),
        "--dump_tensors",
        "--debug",
        "--num_threads",
        "4",
    ]
    + (
        ["--no-interstellar-cost-tradeoff"]
        if not a.interstellar_cost_tradeoff
        else []
    )
    + (
        ["--runtime_tolerance", str(a.runtime_tolerance)]
        if a.runtime_tolerance is not None
        else []
    ),
)
write_manifest(args)
model = torchvision_models.load_model(args)
x = models.ResNet50_Weights.DEFAULT.transforms()(
    Image.open(a.image).convert("RGB")
).unsqueeze(0)
quantizer = get_default_quantizer(
    input_activation=args.activation,
    output_activation=args.output_activation,
    weight=args.weight,
    bias=args.bias,
    force_scale_power_of_two=args.force_scale_power_of_two,
)
patterns = args.compilation_context.backend.fusion_patterns(
    args.compilation_context.hardware
)
prepared = torchvision_models.prepare_model(
    model,
    quantizer,
    [dict(image=x, label=0)] * args.calibration_steps,
    patterns,
    args,
)
prepared.example_args = (x,)
prepared.reference = prepared.graph(x).detach().clone()
result = compile_prepared(prepared, args, patterns)
torch.testing.assert_close(result[1], result[2], rtol=0, atol=0)
reference = result[1].detach().cpu().float().numpy()
reference.tofile(a.output / "reference-output.f32.bin")
(a.output / "output.json").write_text(
    json.dumps(
        dict(
            shape=list(reference.shape),
            dtype="float32",
            reference="reference-output.f32.bin",
            image=str(a.image.resolve()),
            image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
            weights=str(models.ResNet50_Weights.DEFAULT),
            calibration_passes=args.calibration_steps,
            bufferized_reference_match=True,
        ),
        indent=2,
    )
    + "\n"
)
print("RESNET50_BUFFERIZED_REFERENCE_PASS", flush=True)
