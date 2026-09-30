"""Voyager deployment presets, family registration, and model quantization rules."""

import math

import torch

from voyager_compiler import (
    DerivedQuantizationSpec,
    QuantizationConfig,
    QuantizationSpec,
    derive_bias_qparams_fn,
)
from voyager_compiler.quantization import parse_codebook_dtype
from voyager_compiler.quantization.recipes import (
    FamilyPolicy,
    Recipe,
    get_family,
    register_family,
)
from voyager_compiler.quantization.voyager_llm_configs import (
    QUANTIZATION_CONFIGS,
    set_kivi_attention_qconfig,
    set_qconfig,
    set_residual_attention_qconfig,
)

_ROTARY_SCOPE = r"model\.rotary_emb"


def finalize_options(args):
    """Resolve the store beat for recipe-driven Voyager invocations."""
    from voyager_compiler.codegen.node_info import dtype_byte_size

    if args.bank_width is None:
        dtype = (
            args.activation.split(",")[0]
            if args.activation
            else (torch.bfloat16 if args.bf16 else torch.float32)
        )
        args.bank_width = math.ceil(
            args.pe_array_size[1] * dtype_byte_size(dtype)
        )


def configure_model(kind, model, quantizer, args, *, qconfigs, **options):
    if kind == "torchvision":
        _vision(model, quantizer, args)
    elif kind == "vit":
        _vit(quantizer, args, options["timm_model"])
    elif kind == "bert":
        quantizer.set_module_name("pooler", None)
        quantizer.set_module_name("classifier", None)
    elif kind == "mobilebert":
        quantizer.set_module_name("classifier", None)
    elif kind == "llama":
        _llama(quantizer, args, qconfigs, options["is_decode"])


def _vision(model, quantizer, args):
    if "mobilenet" in args.model:
        quantizer.set_module_name("classifier", None)

        if args.activation is not None and "microscaling" in args.activation:
            qspec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric")

            bias_qspec = DerivedQuantizationSpec(
                derived_from=None,
                derive_qparams_fn=derive_bias_qparams_fn,
                dtype=None,
            )

            qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
            quantizer.set_module_name("features.0.0", qconfig)

    # Some designs do not support quantized fc layers
    if not args.quantize_fc:
        quantizer.set_module_name("fc", None)

    if args.residual is not None:
        qspec = QuantizationSpec.from_str(
            f"{args.residual},qs=per_tensor_symmetric"
        )
        qconfig = QuantizationConfig(qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.add.Tensor, qconfig)
        quantizer.set_object_type(torch.ops.aten.add_.Tensor, qconfig)

    # Use per-tensor instead of microscaling for conv1
    if args.activation is not None and "microscaling" in args.activation:
        dtype = args.activation.split(",")[0]
        # A lookup table's layer takes its entry dtype, and stays unquantized
        # when the entries keep the model's.
        if (codebook := parse_codebook_dtype(dtype)) is not None:
            dtype = codebook[1]
        qconfig = None
        if dtype is not None:
            qspec = QuantizationSpec.from_str(
                f"{dtype},qs=per_tensor_symmetric"
            )
            bias_qspec = DerivedQuantizationSpec(
                derived_from=None,
                derive_qparams_fn=derive_bias_qparams_fn,
                dtype=None,
            )
            qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name("^conv1$", qconfig)


def _vit(quantizer, args, timm_model):
    quantizer.set_module_name("head" if timm_model else "classifier", None)

    if args.activation is not None and "microscaling" in args.activation:
        dtype = args.activation.split(",")[0]
        # A lookup table's layer takes its entry dtype, and stays unquantized
        # when the entries keep the model's.
        if (codebook := parse_codebook_dtype(dtype)) is not None:
            dtype = codebook[1]
        qconfig = None
        if dtype is not None:
            qspec = QuantizationSpec.from_str(
                f"{dtype},qs=per_tensor_symmetric"
            )
            bias_qspec = DerivedQuantizationSpec(
                derived_from=None,
                derive_qparams_fn=derive_bias_qparams_fn,
                dtype=None,
            )
            qconfig = QuantizationConfig(qspec, None, qspec, bias_qspec)
        quantizer.set_module_name(
            (
                "^patch_embed.proj$"
                if timm_model
                else "^vit.embeddings.patch_embeddings.projection$"
            ),
            qconfig,
        )


def _llama(quantizer, args, qconfigs, is_decode):
    quantizer.set_module_name_object_type_order(
        _ROTARY_SCOPE, torch.ops.aten.matmul.default, 0, None
    )

    if args.qconfig is not None:
        set_qconfig(quantizer, qconfigs[args.qconfig])

    if args.model == "llama_decode_kivi":
        set_kivi_attention_qconfig(quantizer)
    elif is_decode:
        set_residual_attention_qconfig(quantizer)

    if args.qconfig is not None or args.model == "llama_decode_kivi":
        fp8_qspec = QuantizationSpec.from_str(
            "fp8_e4m3,qs=per_tensor_symmetric,qmax=240"
        )
        qconfig = QuantizationConfig(fp8_qspec, None, None, None)
        quantizer.set_object_type(torch.ops.aten.softmax.int, qconfig)
        quantizer.set_object_type(torch.ops.aten.layer_norm.default, qconfig)


def quantization_rules(context):
    from voyager_compiler.quantization.rules import CONCAT_INT8

    # Applicability is checked against each operation's resolved specs, so a
    # per-model precision override participates even under a different recipe.
    return (CONCAT_INT8,)


def _voyager_policy():
    compile_defaults = dict(
        layout_policy="systolic", scratchpad_size=2097152, num_banks=16
    )
    recipes = {
        "E4M3": Recipe(
            dict(activation="fp8_e4m3", weight="fp8_e4m3", bf16=True),
            compile_defaults,
        ),
        "P8_1": Recipe(
            dict(activation="posit8_1", weight="posit8_1", bf16=True),
            compile_defaults,
        ),
        "INT8": Recipe(
            dict(
                activation="int8,qs=per_tensor_symmetric",
                weight="int8,qs=per_tensor_symmetric",
                bias="int24",
                bf16=True,
                calibration_steps=3,
            ),
            compile_defaults,
        ),
        "MXINT8": Recipe(
            dict(
                activation="int8,qs=microscaling,bs=16",
                weight="int8,qs=microscaling,bs=16",
                force_scale_power_of_two=True,
                bf16=True,
            ),
            compile_defaults,
        ),
        "MXNF4": Recipe(
            dict(
                activation="lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3",
                weight="lut4_to_int6,qs=microscaling,bs=64,scale=fp8_e5m3",
                bf16=True,
                residual="fp8_e4m3",
                quantize_fc=True,
            ),
            {**compile_defaults, "conv2d_im2col": True},
        ),
    }
    return FamilyPolicy(
        recipes,
        QUANTIZATION_CONFIGS,
        configure_model,
        finalize_options,
        quantization_rules,
    )


def ensure_voyager_policy():
    """Register the built-in family once, preserving an existing policy."""
    try:
        get_family("voyager")
    except ValueError:
        register_family("voyager", _voyager_policy())
