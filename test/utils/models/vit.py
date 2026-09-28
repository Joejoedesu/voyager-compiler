import torch
from compilation.pipeline import PreparedModel, compile_prepared
from tqdm import tqdm
from transformers import ViTForImageClassification
from transformers.utils import logging

from voyager_compiler import (
    convert_pt2e,
    export_model,
    prepare_pt2e,
    replace_conv2d_with_im2col,
)
from voyager_compiler.codegen import (
    pad_vit_embeddings_output,
    remove_softmax_dtype_cast,
    remove_zero_attention_mask,
)
from voyager_compiler.export_utils import get_conv_bn_layers

from .utils import configure_quantizer, get_context

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def is_timm_model(args):
    """Whether the checkpoint is a timm one, loaded through timm itself.

    A ``timm/`` repository holds a timm-native checkpoint: its config names
    a timm architecture and its weights are keyed the timm way, so
    ``ViTForImageClassification`` cannot load it.
    """
    name = args.model_name_or_path
    return name is not None and name.startswith("timm/")


class TimmEmbeddings(torch.nn.Module):
    """The patch, class-token and position embeddings of a timm ViT.

    ``pad_vit_embeddings_output`` locates the embedding output by matching a
    pattern traced from a module. timm computes the class-token concat and
    the position add in ``_pos_embed``, a method rather than a module, so
    the pattern is assembled here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model._pos_embed(self.model.patch_embed(pixel_values))


def get_logits(output):
    """timm returns the logits tensor; transformers wraps it in an output."""
    return output.logits if hasattr(output, "logits") else output


def load_model(args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_timm_model(args):
        import timm

        model = timm.create_model(
            args.model_name_or_path.removeprefix("timm/"), pretrained=True
        )
        # The array consumes the attention matmuls individually; timm's
        # fused path exports as one scaled_dot_product_attention node.
        for block in model.blocks:
            block.attn.fused_attn = False
        return model.eval().to(torch_dtype)

    model_name_or_path = (
        args.model_name_or_path or "google/vit-base-patch16-224"
    )

    return ViTForImageClassification.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
    )


def prepare_model(model, quantizer, calibration_data, vector_stages, args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        model = torch.ao.quantization.fuse_modules(
            model, modules_to_fuse, inplace=True
        )

    timm_model = is_timm_model(args)

    configure_quantizer("vit", model, quantizer, args, timm_model=timm_model)

    example_args = (calibration_data[0]["image"].to(torch_dtype),)
    vector_lanes = (
        get_context(args).hardware.vector_lanes
        if get_context(args).hardware.pe_array_size is not None
        else None
    )

    embeddings = TimmEmbeddings(model) if timm_model else model.vit.embeddings

    gm = export_model(model, example_args)
    remove_zero_attention_mask(gm, example_args)
    pad_vit_embeddings_output(gm, embeddings, example_args, unroll=vector_lanes)

    if args.conv2d_im2col:
        replace_conv2d_with_im2col(gm)

    gm = prepare_pt2e(gm, quantizer)

    remove_softmax_dtype_cast(gm)

    for i in tqdm(range(args.calibration_steps), desc="Calibrating ViT"):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = get_logits(gm(*example_args))

    return PreparedModel(
        gm,
        example_args,
        old_output,
        extract_preprocessor=True,
        output_adapter=get_logits,
    )


def quantize_and_dump_model(
    model, quantizer, calibration_data, vector_stages, args
):
    """Compatibility wrapper around preparation and the shared runner."""
    return compile_prepared(
        prepare_model(model, quantizer, calibration_data, vector_stages, args),
        args,
        vector_stages,
    )


def evaluate(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for image_label_pair in tqdm(dataset, desc="Evaluating ViT"):
            # for running the original model without the preprocessing function
            # applied to the dataset
            image = image_label_pair["image"].to(device)
            label = image_label_pair["label"]

            logits = get_logits(model(image))
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Vit Accuracy: {accuracy:.4f}")
