# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from voyager_compiler.codegen import (
    DEFAULT_ACCELERATOR_FREQUENCY_GHZ,
    DEFAULT_COMPILATION_TARGET,
    DEFAULT_DRAM_BANDWIDTH_GBPS,
    DEFAULT_EXECUTION_ORDER_MODE,
    GEMMINI_COMPILATION_TARGET,
    GEMMINI_HARDWARE_CONFIG,
    MAGNET_ALPHA_COMPILATION_TARGET,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    HardwareConfig,
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.codegen.target_backends.magnet_alpha import (
    AOV_COMPAT_HARDWARE_CONFIG,
)


def get_hardware_config(args):
    """Build a typed profile when the legacy CLI describes full hardware."""
    target = getattr(args, "target", DEFAULT_COMPILATION_TARGET)
    if (
        getattr(args, "bufferize", False)
        and target == DEFAULT_COMPILATION_TARGET
    ):
        # Generic bufferized compilation is explicitly an unscheduled
        # representation-lowering path. Its cache and unroll arguments remain
        # graph-tiling inputs rather than selecting a local scheduler.
        return None
    hardware_unrolling = getattr(args, "hardware_unrolling", None)
    cache_size = getattr(args, "cache_size", None)
    has_unrolling = hardware_unrolling is not None
    has_cache = cache_size is not None
    if target == GEMMINI_COMPILATION_TARGET:
        if has_unrolling or has_cache:
            raise ValueError(
                "gemmini uses its registered immutable hardware profile; "
                "do not pass --hardware_unrolling or --cache_size"
            )
        return GEMMINI_HARDWARE_CONFIG
    if not has_unrolling and not has_cache:
        if target == MAGNET_ALPHA_COMPILATION_TARGET:
            return AOV_COMPAT_HARDWARE_CONFIG
        return None
    if has_unrolling != has_cache:
        raise ValueError(
            "--hardware_unrolling and --cache_size must be specified together"
        )
    config_type = (
        MagnetAlphaHardwareConfig
        if target == MAGNET_ALPHA_COMPILATION_TARGET
        else HardwareConfig
    )
    legacy_args = {
        "unroll_dims": hardware_unrolling,
        "cache_size": cache_size,
        "input_buffer_size": args.input_buffer_size,
        "weight_buffer_size": args.weight_buffer_size,
        "accum_buffer_size": args.accum_buffer_size,
        "num_banks": args.num_banks,
        "bank_width": args.bank_width,
        "target": target,
    }
    if config_type is HardwareConfig:
        legacy_args.update(
            {
                "double_buffered_accum_buffer": (
                    getattr(
                        args,
                        "double_buffered_accum_buffer",
                        False,
                    )
                ),
                "double_buffered_l2": getattr(
                    args,
                    "double_buffered_l2",
                    False,
                ),
                "dram_size": getattr(args, "dram_size", None),
                "dram_bandwidth": getattr(
                    args,
                    "dram_bandwidth",
                    DEFAULT_DRAM_BANDWIDTH_GBPS,
                ),
                "frequency": getattr(
                    args,
                    "frequency",
                    DEFAULT_ACCELERATOR_FREQUENCY_GHZ,
                ),
            }
        )
    return config_type.from_legacy(
        **legacy_args,
    )


def get_transform_args(args, vector_stages, hardware_config=None):
    unroll_dims = (
        hardware_config.unroll_dims
        if hardware_config is not None
        else args.hardware_unrolling
    )
    cache_size = (
        hardware_config.global_capacity_bytes
        if hardware_config is not None
        else args.cache_size
    )
    fuse_reshape = not args.disable_reshape_fusion and (
        unroll_dims is None or max(unroll_dims) < 64
    )
    transform_args = {
        "patterns": vector_stages,
        "transform_layout": args.transform_layout,
        "transpose_fc": args.transpose_fc,
        "fuse_reshape": fuse_reshape,
        "split_spmm": args.split_spmm,
    }
    if hardware_config is not None:
        transform_args["hardware_config"] = hardware_config
    else:
        transform_args.update(
            {
                "unroll_dims": unroll_dims,
                "cache_size": cache_size,
                "num_banks": args.num_banks,
            }
        )
    return transform_args


def get_compile_args(
    args,
    hardware_config=None,
    scheduler_backend=None,
):
    compile_args = {
        "total_memory": getattr(args, "total_memory", None),
        "output_dir": args.model_output_dir,
        "output_file": args.model,
        "dump_tensors": args.dump_tensors,
        "bufferize": args.bufferize,
        "scratchpad_mapping": getattr(
            args,
            "scratchpad_mapping",
            PER_OPERATION_SCRATCHPAD_MAPPING,
        ),
        "execution_order": getattr(
            args,
            "execution_order",
            DEFAULT_EXECUTION_ORDER_MODE,
        ),
    }
    if scheduler_backend is not None:
        compile_args["scheduler_backend"] = scheduler_backend
    if hardware_config is not None:
        compile_args["hardware_config"] = hardware_config
        return compile_args

    compile_args["double_buffered_accum_buffer"] = (
        args.double_buffered_accum_buffer
    )
    compile_args.update(
        {
            "cache_size": args.cache_size,
            "num_banks": args.num_banks,
            "bank_width": args.bank_width,
            "unroll_dims": args.hardware_unrolling,
            "input_buffer_size": args.input_buffer_size,
            "weight_buffer_size": args.weight_buffer_size,
            "accum_buffer_size": args.accum_buffer_size,
        }
    )
    return compile_args
