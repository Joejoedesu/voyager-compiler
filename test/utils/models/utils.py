from compilation.context import resolve_context

from voyager_compiler.codegen.transform.bufferize import BufferizationOptions


def get_context(args):
    context = getattr(args, "compilation_context", None)
    if context is None:
        context = resolve_context(args)
        args.compilation_context = context
    return context


def configure_quantizer(kind, model, quantizer, args, **options):
    get_context(args).configure_quantizer(
        kind, model, quantizer, args, **options
    )


def get_transform_args(args, vector_stages):
    return {
        "patterns": vector_stages,
        "config": get_context(args).hardware,
        "layout_policy": args.layout_policy,
        "gemv_weight_layout": args.gemv_weight_layout,
        "fuse_reshape": not args.disable_reshape_fusion,
    }


def get_compile_args(args):
    return {
        "config": get_context(args).hardware,
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "runtime_tolerance": args.runtime_tolerance,
        "bufferization_options": BufferizationOptions(
            flow=getattr(args, "bufferized_flow", "per_kernel"),
            parameter_loading=getattr(args, "parameter_loading", "on_demand"),
            single_buffer_tail=getattr(args, "single_buffer_tail", False),
            flash_attention_v3=getattr(args, "flash_attention_v3", True),
            bool_mask=getattr(args, "bool_mask", True),
        ),
    }
