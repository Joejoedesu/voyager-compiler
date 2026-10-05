"""Floating-point Trainium policy; Voyager codebooks/scales are not inherited."""

from voyager_compiler.quantization.recipes import (
    FamilyPolicy,
    Recipe,
    register_family,
)


def configure_model(kind, model, quantizer, args, **options):
    if any(
        getattr(args, key, None)
        for key in ("activation", "weight", "bias", "residual")
    ):
        raise ValueError(
            "Initial Trainium lowering supports native FP32/BF16, not Voyager quantized operators"
        )
    quantizer.set_global(None)


def register_policy():
    register_family(
        "trainium",
        FamilyPolicy(
            recipes={
                "FP32": Recipe(
                    compilation={
                        "bf16": False,
                        "conv2d_im2col": False,
                        "use_maxpool_2x2": False,
                    }
                ),
                "BF16": Recipe(
                    compilation={
                        "bf16": True,
                        "conv2d_im2col": False,
                        "use_maxpool_2x2": False,
                    }
                ),
            },
            qconfigs={},
            configure_model=configure_model,
        ),
    )
