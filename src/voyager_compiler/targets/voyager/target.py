"""Voyager's registration in the shared bufferized compiler."""

from dataclasses import replace

from voyager_compiler.extensions import TargetSpec, registry
from voyager_compiler.hardware import AcceleratorConfig

from .codegen import encode
from .fusion import FUSION_PATTERNS
from .implementations import VoyagerModel
from .implementations.kernels import OPERATORS, build_pointwise


def build_gemm(node, context, parameters):
    from voyager_compiler.compiler.bufferize.pipeline import (
        build_gemm as shared_build,
    )

    # The established Voyager instruction stream retains the final shape-only
    # pass. Other targets may fold it into each reduction round.
    return shared_build(node, replace(context, preserve_reshape_tail=True))


from voyager_compiler.compiler.bufferize.builders import shared_implementations

TARGET = registry.register(
    TargetSpec(
        name="voyager",
        hardware=AcceleratorConfig,
        implementations=tuple(
            (
                replace(spec, build=build_pointwise)
                if spec.name == "voyager.pointwise"
                else (
                    replace(spec, build=build_gemm)
                    if spec.name == "voyager.gemm"
                    else spec
                )
            )
            for spec in shared_implementations()
        ),
        model=VoyagerModel,
        encode=encode,
        operators=OPERATORS,
        normalization=(("patterns", FUSION_PATTERNS),),
    )
)
