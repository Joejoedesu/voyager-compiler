"""TargetSpec assembly: connect the public extension points, with no pass logic."""

import torch

from voyager_compiler.extensions import (
    ImplementationSpec,
    OperatorSemantics,
    TargetSpec,
    registry,
)

from . import lowering, scheduling, semantics
from .emission import emit
from .hardware import hardware
from .quantization import POLICY


def _deployment_recipes():
    # Each registration pairs candidates and cost with the builder that realizes
    # that selected mode. This is assembly, not a first-match ISA decision.
    for op in semantics.OPS:
        kind = op._schema.name.split("::")[-1]
        modes = (
            ("explicit", "hardware_loop")
            if kind == "matmul"
            else ("hardware_loop",) if kind == "conv2d" else ("dma",)
        )
        for mode in modes:
            yield ImplementationSpec(
                f"gemmini.{kind}.{mode}",
                lambda n, a, op=op: a.target == op,
                lowering.build_native_kernel,
                candidates=lambda n, c, mode=mode: scheduling.candidates(
                    n, c, mode
                ),
                estimate=scheduling.estimate_deployment,
                description="Selected tile and exact INT8 arithmetic with declared private resources",
            )


TARGET = registry.register(
    TargetSpec(
        name="gemmini",
        hardware=hardware,
        operators=tuple(
            (op, OperatorSemantics())
            for op in (torch.ops.aten._int_mm.default, *semantics.OPS)
        ),
        quantization=POLICY,
        lower=lowering.lower_operations,
        normalization=(
            ("skip_op_fusion", True),
            ("pad_to_hardware", False),
            ("tensor_layout", "nhwc"),
            ("gemm_layout", "xc,ck"),
            ("gemv_layout", "xc,ck"),
        ),
        model=scheduling.GemminiModel,
        implementations=(
            ImplementationSpec(
                "gemmini.explicit",
                lambda n, a: a.target == torch.ops.aten._int_mm.default,
                lowering.build_raw_explicit,
                estimate=scheduling.estimate_raw,
                description="Compiler-owned block-column SRAM and accumulator; explicit mvin/preload/compute/mvout.",
            ),
            ImplementationSpec(
                "gemmini.hardware_loop",
                lambda n, a: a.target == torch.ops.aten._int_mm.default,
                lowering.build_raw_loop,
                estimate=scheduling.estimate_raw,
                description="LOOP_WS macro with implicit operand DMA and private ping/pong resource requirements.",
            ),
            *_deployment_recipes(),
        ),
        refine=lowering.refine,
        emit=emit,
    )
)
