import torch
from compilation.pipeline import PreparedModel, compile_prepared
from torchvision import models
from tqdm import tqdm

from voyager_compiler import (
    convert_pt2e,
    export_model,
    prepare_pt2e,
    replace_conv2d_with_im2col,
)
from voyager_compiler.export_utils import get_conv_bn_layers

from .utils import configure_quantizer


def load_model(args):
    if args.model_name_or_path is None:
        args.model_name_or_path = "DEFAULT"

    try:
        model = models.__dict__[args.model](
            weights=args.model_name_or_path
        ).eval()
    except Exception:
        model = models.__dict__[args.model](pretrained=True).eval()

        if args.model_name_or_path:
            checkpoint = torch.load(args.model_name_or_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=False)

    if args.bf16:
        model.bfloat16()
    return model


def prepare_model(model, quantizer, calibration_data, vector_stages, args):
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    modules_to_fuse = get_conv_bn_layers(model)
    if len(modules_to_fuse) > 0:
        model = torch.ao.quantization.fuse_modules(
            model, modules_to_fuse, inplace=True
        )

    # Accelerator only supports 2x2 maxpool
    if args.use_maxpool_2x2:
        for module in model.modules():
            if isinstance(module, torch.nn.MaxPool2d):
                module.kernel_size = 2
                module.stride = 2
                module.padding = 0

    configure_quantizer("torchvision", model, quantizer, args)
    if "mobilenet" in args.model:
        model.features[0][0].padding = (3, 3)
        model.features[0][0].weight.data = torch.nn.functional.pad(
            model.features[0][0].weight.data, (2, 2, 2, 2)
        )

    example_args = (torch.randn(1, 3, 224, 224, dtype=torch_dtype),)
    gm = export_model(model, example_args)

    # im2col must be done before prepare_pt2e
    if args.conv2d_im2col:
        replace_conv2d_with_im2col(gm)

    gm = prepare_pt2e(gm, quantizer)

    model_name = model.__class__.__name__
    for i in tqdm(
        range(args.calibration_steps), desc=f"Calibrating {model_name}"
    ):
        inputs = calibration_data[i]["image"]
        with torch.no_grad():
            gm(inputs.to(torch_dtype))

    convert_pt2e(gm, args.bias)

    old_output = gm(*example_args)

    return PreparedModel(
        gm, example_args, old_output, extract_preprocessor=True
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
        for image_label_pair in tqdm(
            dataset, desc=f"Evaluating {model.__class__.__name__}"
        ):
            # for running the original model without the preprocessing function
            # applied to the dataset
            image = image_label_pair["image"].to(device)
            label = image_label_pair["label"]

            logits = model(image)
            prediction = torch.argmax(logits, dim=-1)
            if prediction.item() == label:
                correct_predictions += 1
            total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}")
