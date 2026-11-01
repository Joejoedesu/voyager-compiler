"""Voyager's registration in the shared bufferized compiler."""

from dataclasses import replace

from voyager_compiler.codegen.node_info import (
    is_depthwise_conv,
    is_grouped_conv,
)
from voyager_compiler.compiler.bufferize.bufferization import build_untiled
from voyager_compiler.compiler.bufferize.builders import shared_implementations
from voyager_compiler.extensions import ImplementationSpec, TargetSpec, registry
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


def build_grouped(node, context, parameters):
    from voyager_compiler.compiler.bufferize.pipeline import build_conv2d

    # Preserve fused MX epilogues in the convolution template; unfused nodes
    # use the identical whole-tensor fallback as before explicit registration.
    graph = build_conv2d(node, context, whole_tensor=True)
    return (
        graph if graph is not None else build_untiled(node, context, parameters)
    )


def _implementations():
    # Connectivity is classified in shared code; this target deliberately keeps
    # grouped operations whole. The dense template never owns their dispatch.
    for name, match in (
        ("depthwise", is_depthwise_conv),
        ("grouped", lambda a: is_grouped_conv(a) and not is_depthwise_conv(a)),
    ):
        yield ImplementationSpec(
            f"voyager.{name}",
            lambda n, a, match=match: match(a),
            build_grouped,
            cache_by_shape=True,
            description="Voyager whole-tensor convolution compatibility recipe",
        )
    for spec in shared_implementations():
        if spec.name == "voyager.pointwise":
            spec = replace(spec, build=build_pointwise)
        elif spec.name == "voyager.gemm":
            spec = replace(spec, build=build_gemm)
        yield replace(
            spec,
            matches=lambda n, a, match=spec.matches: not is_grouped_conv(a)
            and match(n, a),
        )


TARGET = registry.register(
    TargetSpec(
        name="voyager",
        hardware=AcceleratorConfig,
        implementations=tuple(_implementations()),
        model=VoyagerModel,
        encode=encode,
        operators=OPERATORS,
        normalization=(("patterns", FUSION_PATTERNS),),
    )
)


def with_attention_algorithms(target=TARGET):
    """Opt into fixed-region SDPA algorithm comparison on the Voyager model.

    Keep the registered default and its pinned behavior intact. Unsupported
    unquantized SDPA forms are rejected explicitly by the opt-in recipes;
    quantized attention continues to use the established recipe.
    """
    from voyager_compiler.compiler.bufferize.attention_algorithms import (
        SDPA,
        attention_implementations,
    )

    components = tuple(
        spec
        for spec in target.implementations
        if spec.name != "voyager.attention"
    )
    attention = next(
        spec
        for spec in target.implementations
        if spec.name == "voyager.attention"
    )
    retained = replace(
        attention,
        matches=lambda node, anchor: anchor.target != SDPA
        and attention.matches(node, anchor),
    )
    return replace(
        target,
        implementations=components
        + (retained,)
        + attention_implementations(components, prefix="voyager.attention"),
    )
