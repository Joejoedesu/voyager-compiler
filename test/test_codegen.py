"""CLI and model dispatch for model × hardware compilation tests."""

import argparse
import logging

import torch
from compilation.context import parse_args, write_manifest
from compilation.pipeline import compile_prepared
from torch.testing import assert_close
from torch.utils._pytree import tree_flatten

from voyager_compiler import (
    add_compile_args,
    add_quantization_args,
    get_default_quantizer,
)
from voyager_compiler.codegen.reporting import (
    coverage,
    kernel_rows,
    load_calibration,
    report,
)

logger = logging.getLogger()


def compile_adapter(adapter, args, vector_stages, **inputs):
    prepared = adapter.prepare_model(
        vector_stages=vector_stages, args=args, **inputs
    )
    return compile_prepared(prepared, args, vector_stages)


def model_input_name(gm):
    """The compiled graph's input placeholder, which names the per-sample
    tensor files the accuracy tester loads."""
    return next(n.name for n in gm.graph.nodes if n.op == "placeholder")


# Tolerance for comparing the lowered graph's output against the original's.
# Tiling reassociates a reduction, so a bfloat16 accumulation lands a few
# mantissa bits away from the reference one (an ulp is 2**-8 relative, and a
# split reduction compounds several); the two are numerically equivalent but
# not bit-identical.  Exceeding this warns rather than fails, since telling a
# real lowering error from an unlucky accumulation needs a human.
OUTPUT_RTOL = 5e-2
OUTPUT_ATOL = 1e-4


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Model or stage to compile (e.g. resnet18, llama_prefill, llama_decode).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help=(
            "Path to pretrained model or model identifier from "
            "huggingface.co/models."
        ),
    )
    parser.add_argument(
        "--task_name",
        default="sst2",
        help="Name of the task to load the dataset",
    )
    parser.add_argument(
        "--model_output_dir",
        required=True,
        help="Output directory for generated tensor files",
    )
    parser.add_argument(
        "--dump_dataset",
        action="store_true",
        help="Whether to save the dataset for later use.",
    )
    parser.add_argument(
        "--dataset_output_dir", help="Output directory for dataset files"
    )
    parser.add_argument(
        "--dump_tensors",
        action="store_true",
        help="Whether to save intermediate outputs for verification.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Re-run the lowered graph and compare its output against the "
            "pre-transform reference.  Re-running the graph is expensive on a "
            "large model, so this verification is off by default."
        ),
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Context length for the LLM decoding.",
    )
    parser.add_argument(
        "--spec_length",
        type=int,
        default=1,
        help=(
            "Tokens a llama_verify step writes at once: the draft tokens a "
            "speculative-decoding verification step checks."
        ),
    )
    parser.add_argument(
        "--residual_length",
        type=int,
        default=None,
        help=(
            "Positions of a KIVI KV cache kept in full precision while their "
            "chunk fills; a multiple of the 64-token KIVI group, 128 by "
            "default."
        ),
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help=(
            "Compile only the first N encoder/decoder layers of a Transformer "
            "(the whole model when unset).  Two is the useful minimum for "
            "calibration: a model's last layer is built differently from the "
            "ones before it -- its MLP down_proj fuses the residual quantize "
            "and stores fp8 rather than bf16 -- so a one-layer compile only "
            "ever produces the tail variant."
        ),
    )
    parser.add_argument(
        "--qconfig",
        default=None,
        help=(
            "Named per-operand qconfig table for LLMs (from "
            "the selected hardware family); also puts "
            "softmax and layer_norm outputs in fp8."
        ),
    )
    parser.add_argument(
        "--use_maxpool_2x2",
        action="store_true",
        help="Whether to use 2x2 maxpool for resnet18 and resnet50.",
    )
    parser.add_argument(
        "--conv2d_im2col",
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether to transform Conv2d operations with small input channels "
            "into linear operations using im2col."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether to run the pytorch evaluation during compilation",
    )
    parser.add_argument(
        "--quantize_attention_mask",
        action=argparse.BooleanOptionalAction,
        help="Whether to quantize Transformer attention mask to binary values.",
    )
    parser.add_argument(
        "--quantize_fc",
        action=argparse.BooleanOptionalAction,
        help="Whether to quantize the fully connected layers.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help=(
            "HuggingFace attention module the graph is built from. Only sdpa "
            "emits a scaled_dot_product_attention node, the one the flash-"
            "attention builders lower; eager decomposes it into "
            "matmul/softmax/matmul."
        ),
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=32,
        help="PyTorch CPU threads (default: 32).",
    )
    add_quantization_args(parser)
    add_compile_args(parser)
    parser.add_argument(
        "--target_hardware",
        default="voyager",
        help="Registered hardware target (default: voyager).",
    )
    parser.add_argument(
        "--quantization_recipe",
        default=None,
        help="Deployment recipe from the target's hardware family.",
    )
    parser.add_argument(
        "--single_buffer_tail",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--flash_attention_v3",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bool_mask", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv=None):
    args = parse_args(build_parser(), argv)
    run(args)


def run(args):
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, precision=10)
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    torch.set_num_threads(args.num_threads)
    torch.set_grad_enabled(False)
    context = args.compilation_context
    vector_pipeline = context.backend.fusion_patterns(context.hardware)
    write_manifest(args)

    logger.setLevel(getattr(logging, args.log_level))

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    quantizer = get_default_quantizer(
        input_activation=args.activation,
        output_activation=args.output_activation,
        weight=args.weight,
        bias=args.bias,
        force_scale_power_of_two=args.force_scale_power_of_two,
    )

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Model dependencies are loaded only after target/policy validation.
    from torchvision import models
    from utils.dataset import glue, imagenet
    from utils.models import bert, llama, mobilebert, torchvision_models, vit

    if args.model in models.__dict__:
        model = torchvision_models.load_model(args)

        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "resnet")
            if args.evaluate:
                torchvision_models.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "resnet")

        gm, old_output, new_output, preprocess_fn = compile_adapter(
            torchvision_models,
            args,
            vector_pipeline,
            model=model,
            quantizer=quantizer,
            calibration_data=imagenet_dataset,
        )

        if args.dump_dataset or args.evaluate:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir if args.dump_dataset else None,
                imagenet_dataset,
                model_input_name(gm),
                preprocess_fn,
                torch_dtype,
            )

        if args.evaluate:
            torchvision_models.evaluate(gm, preprocessed_imagenet)
    elif args.model == "mobilebert":
        model, tokenizer = mobilebert.load_model(args)

        eval_dataset, train_dataset = glue.retrieve_dataset(
            model, tokenizer, args
        )

        if args.evaluate:
            mobilebert.evaluate(model, eval_dataset)

        if args.dump_dataset or args.evaluate:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir if args.dump_dataset else None,
                eval_dataset,
                model,
            )

        gm, old_output, new_output = compile_adapter(
            mobilebert,
            args,
            vector_pipeline,
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
        )

        if args.evaluate:
            mobilebert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model == "bert":
        model, tokenizer = bert.load_model(args)

        eval_dataset, train_dataset = glue.retrieve_dataset(
            model, tokenizer, args
        )

        if args.evaluate:
            bert.evaluate(model, eval_dataset)

        if args.dump_dataset or args.evaluate:
            preprocessed_dataset = glue.dump_dataset(
                args.dataset_output_dir if args.dump_dataset else None,
                eval_dataset,
                model,
            )

        gm, old_output, new_output = compile_adapter(
            bert,
            args,
            vector_pipeline,
            model=model,
            quantizer=quantizer,
            calibration_data=train_dataset,
        )

        if args.evaluate:
            bert.evaluate_gm(gm, preprocessed_dataset)

    elif args.model in (
        "llama_prefill",
        "llama_decode",
        "llama_decode_kivi",
        "llama_verify",
    ):
        model, tokenizer = llama.load_model(args)

        gm, old_output, new_output = compile_adapter(
            llama,
            args,
            vector_pipeline,
            model=model,
            tokenizer=tokenizer,
            quantizer=quantizer,
        )
    elif args.model == "vit":
        model = vit.load_model(args)

        if args.dump_dataset or args.evaluate:
            imagenet_dataset = imagenet.retrieve_dataset(1000, "vit")
            if args.evaluate:
                vit.evaluate(model, imagenet_dataset)
        else:
            imagenet_dataset = imagenet.retrieve_dataset(10, "vit")

        gm, old_output, new_output, preprocess_fn = compile_adapter(
            vit,
            args,
            vector_pipeline,
            model=model,
            quantizer=quantizer,
            calibration_data=imagenet_dataset,
        )

        if args.dump_dataset or args.evaluate:
            preprocessed_imagenet = imagenet.dump_imagenet(
                args.dataset_output_dir if args.dump_dataset else None,
                imagenet_dataset,
                model_input_name(gm),
                preprocess_fn,
                torch_dtype,
            )

        if args.evaluate:
            vit.evaluate(gm, preprocessed_imagenet)
    else:
        raise ValueError(f"Model {args.model} not supported")

    if args.report:
        # Estimate the schedule of the graph the compile just produced, so
        # the workbook, its Calibration sheet and the emitted model all
        # describe one tiling -- which the sheet's frozen "Analytic
        # cyc/iter" requires.
        out_dir = args.report_output_dir
        if out_dir == "." and args.model_output_dir:
            out_dir = args.model_output_dir
        calibration = load_calibration(args.calib_in) if args.calib_in else None
        if calibration is not None:
            print(
                f"[report] {len(calibration.measurements)} measured groups "
                f"from {args.calib_in}"
            )
        print(f"[report] {args.model} -> {out_dir}", flush=True)
        result = report(
            gm,
            context.hardware,
            output_dir=out_dir,
            basename=args.report_basename,
            calibration=calibration,
        )
        rows = kernel_rows(result)
        # Compute kernels only: the DMA-only ones carry a key too, but they
        # are not things the RTL measures.
        groups = len({r.group for r in rows if r.group and r.ops_per_period})
        print(
            f"[report] total_latency={result.total_latency} "
            f"dram_read={result.dram_read_bytes} "
            f"dram_write={result.dram_write_bytes} "
            f"kernels={len(rows)} groups={groups} "
            f"calibrated={sum(1 for r in rows if r.calibration)} "
            f"coverage={coverage(rows, result.total_latency):.3f}",
            flush=True,
        )

    if new_output is None:
        print("Skipping output verification (pass --debug to run it)")
        return

    try:
        old_flat, old_spec = tree_flatten(old_output)
        new_flat, new_spec = tree_flatten(new_output)
        n_old, n_new = len(old_flat), len(new_flat)
        assert n_old == n_new, f"{n_old} outputs became {n_new}"
        # The reference comes from ShapeProp, which hands back the output
        # node's value and so keeps the graph's output tuple, while calling
        # the module unwraps a single-element one.  Only the leaves matter.
        if old_spec != new_spec:
            print(f"Note: output structure {old_spec} vs {new_spec}")
        worst = 0.0
        for old, new in zip(old_flat, new_flat):
            if not isinstance(old, torch.Tensor):
                assert old == new, f"non-tensor output {old!r} != {new!r}"
                continue
            deviation = (new - old).abs().to(torch.float32) / (
                old.abs().to(torch.float32) + OUTPUT_ATOL
            )
            worst = max(worst, deviation.max().item())
            assert_close(new, old, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        print(f"Results match (max deviation {worst:.2e})")
    except Exception as e:
        print(f"WARNING: output verification failed: {e}")
        print(old_output)
        print(new_output)


if __name__ == "__main__":
    main()
