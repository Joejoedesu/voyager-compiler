"""MobileNetV2 construction and inputs; target policy stays in the target.

Pass a local ``weights`` checkpoint and optional ``input`` tensor in workload
options. No assets are downloaded. Generated inputs are compilation probes,
not an ImageNet accuracy dataset.
"""

from pathlib import Path

import torch

from voyager_compiler.extensions import WorkloadSpec, registry


def build(options):
    from torchvision.models import mobilenet_v2

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(options.get("seed", 0))
        model = mobilenet_v2(weights=None).eval()
    if options.get("weights"):
        model.load_state_dict(
            torch.load(
                Path(options["weights"]), map_location="cpu", weights_only=True
            )
        )
    return model


def example_inputs(model, options):
    value = options.get("input")
    if value is None:
        size = options.get("image_size", 224)
        generator = torch.Generator().manual_seed(options.get("seed", 0))
        value = torch.randn(1, 3, size, size, generator=generator)
    return (value,), {}


WORKLOAD = registry.register(
    WorkloadSpec("mobilenet-v2", build, example_inputs)
)
