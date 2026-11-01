"""Trainium registration; NKI translation consumes the frozen buffer graph."""

from dataclasses import replace

from voyager_compiler.compiler.bufferize.builders import (
    SHARED_BUILDERS,
    shared_implementations,
)
from voyager_compiler.extensions import TargetSpec, registry

from .config import ATTENTION_RULE, GEMM_RULE, LAYOUT, TrainiumConfig
from .fusion import FUSION_PATTERNS
from .implementations import encode, fp8_matrix, implementations
from .model import TrainiumModel


def emit(bundle, output):
    from types import SimpleNamespace

    from .translate import translate

    data = bundle.target_data
    if data["schema_version"] != 1:
        raise ValueError("unsupported Trainium target IR")
    return translate(
        bundle.read_model(),
        str(output / "nki"),
        SimpleNamespace(moving_free_max=data["moving_free_max"]),
        {
            name: bundle.read_payload(key).item()
            for name, key in data["immediates"].items()
        },
    )


TARGET = registry.register(
    TargetSpec(
        name="trainium",
        hardware=TrainiumConfig,
        implementations=(
            *implementations(),
            *(
                (
                    replace(
                        spec,
                        matches=lambda n, a, match=spec.matches: (
                            match(n, a) and not fp8_matrix(n, a)
                        ),
                    )
                    if spec.name == "trainium.trainium_gemm"
                    else spec
                )
                for spec in shared_implementations(
                    (GEMM_RULE, ATTENTION_RULE, *SHARED_BUILDERS), "trainium"
                )
            ),
        ),
        model=TrainiumModel,
        emit=emit,
        encode=encode,
        normalization=(*LAYOUT.items(), ("patterns", FUSION_PATTERNS)),
    )
)
