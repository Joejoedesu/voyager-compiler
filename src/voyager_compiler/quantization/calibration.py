"""Optional static-PTQ helpers; training's delayed observer updates stay unchanged."""

import torch

from .fake_quantize import FusedAmaxObsFakeQuantize


def observe_without_fake_quant(graph, options):
    for module in graph.modules():
        if hasattr(module, "disable_fake_quant"):
            module.disable_fake_quant()


def finalize_amax_observers(graph, options):
    """Use the final retained observation, including the last calibration batch."""
    for module in graph.modules():
        if not isinstance(module, FusedAmaxObsFakeQuantize):
            continue
        if not module.amax_history.numel():
            raise ValueError(
                "A quantization observer received no calibration data"
            )
        peak = module.amax_history.amax(dim=0)
        if not torch.isfinite(peak).all():
            raise ValueError("Calibration requires finite tensors")
        scale = torch.where(
            peak > 0, peak / module.quant_max, torch.ones_like(peak)
        )
        if module.power_2_scale:
            scale = torch.pow(2, torch.ceil(torch.log2(scale)))
        module.scale.copy_(scale)
        module.disable_observer()
