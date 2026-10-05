"""Lean INT8 policy using the ordinary PT2E model adapters and observers."""

from voyager_compiler.quantization.recipes import (
    FamilyPolicy,
    Recipe,
    get_family,
    register_family,
)


def configure_model(kind, model, quantizer, args, *, qconfigs, **options):
    from voyager_compiler.quantization.voyager import (
        configure_model as configure,
    )

    if kind != "torchvision":
        raise NotImplementedError(
            "Lean currently supports the shared vision adapter"
        )
    configure(kind, model, quantizer, args, qconfigs=qconfigs, **options)
    import torch

    from voyager_compiler import QuantizationConfig

    qspec = quantizer.object_type_config[
        torch.ops.aten.conv2d.default
    ].input_activation
    # Lean cannot spill FP32 intermediates. Observe every ReLU output and both
    # residual operands using the existing PT2E annotators and observers.
    quantizer.STATIC_OPS = [
        "add" if op == "residual" else op for op in quantizer.STATIC_OPS
    ]
    for op in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor):
        quantizer.set_object_type(
            op, QuantizationConfig(qspec, None, None, None)
        )
    for op in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default):
        quantizer.set_object_type(
            op, QuantizationConfig(None, qspec, None, None)
        )


def finalize_options(args):
    if args.bf16 or args.bias != "int32" or not args.quantize_fc:
        raise ValueError(
            "Lean requires INT8 inputs/weights, int32 bias and quantized FC"
        )
    if args.bufferized_flow != "per_kernel":
        raise NotImplementedError(
            "Lean resident SRAM transfers are not implemented yet"
        )
    for field in ("activation", "weight", "output_activation"):
        if getattr(args, field) != "int8,qs=per_tensor_symmetric":
            raise ValueError(
                f"Lean requires {field}=int8,qs=per_tensor_symmetric"
            )
    if not args.force_scale_power_of_two:
        raise ValueError("Lean residual lowering requires power-of-two scales")


def finalize_graph(model):
    """Keep PT2E's observed scales/weights; model Lean's native INT8 rounding.

    Voyager's lookup maps first round to BF16. Lean instead converts INT32 to
    FP32, scales, rounds to nearest even, then saturates. Accumulator decoding
    must likewise retain the exact integer instead of indexing a BF16 map.
    """
    import torch

    for node in model.graph.nodes:
        if node.target == torch.ops.quantized_ops.quantize.default:
            if node.meta.get("dtype") != "int8":
                raise ValueError(
                    "Lean supports only INT8 quantized activations"
                )
            node.kwargs = {**node.kwargs, "rounding": "nearest_even_int8"}
        elif node.target == torch.ops.quantized_ops.dequantize.default:
            args = list(node.args)
            if len(args) > 5:
                args[5] = None
            node.args = tuple(args)
            if "input_qmap" in node.kwargs:
                node.kwargs = {**node.kwargs, "input_qmap": None}
    model.graph.lint()
    model.recompile()


def ensure_gemmini_policy():
    try:
        get_family("gemmini")
    except ValueError:
        register_family(
            "gemmini",
            FamilyPolicy(
                {
                    "INT8": Recipe(
                        dict(
                            activation="int8,qs=per_tensor_symmetric",
                            weight="int8,qs=per_tensor_symmetric",
                            bias="int32",
                            bf16=False,
                            quantize_fc=True,
                            force_scale_power_of_two=True,
                            output_activation="int8,qs=per_tensor_symmetric",
                            residual="int8",
                            calibration_steps=3,
                        ),
                        dict(
                            pe_array_size=(16, 16),
                            layout_policy="systolic",
                            gemv_weight_layout="ck",
                            scratchpad_size=262144,
                            num_banks=4,
                            bank_width=16,
                        ),
                    )
                },
                {},
                configure_model,
                finalize_options,
                finalize_graph=finalize_graph,
            ),
        )
