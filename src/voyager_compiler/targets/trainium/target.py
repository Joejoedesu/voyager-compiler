"""Trainium registration; NKI translation consumes the frozen buffer graph."""

from dataclasses import replace

from voyager_compiler.compiler.bufferize.builders import (
    SHARED_BUILDERS,
    shared_implementations,
)
from voyager_compiler.extensions import OperatorSemantics, TargetSpec, registry

from .config import ATTENTION_RULE, GEMM_RULE, LAYOUT, TrainiumConfig
from .fusion import FUSION_PATTERNS
from .implementations import encode, fp8_matrix, implementations
from .model import TrainiumModel
from .normalization import legalize, lower, materialized_implementations
from .scan import IMPLEMENTATION as SCAN_IMPLEMENTATION, OP as SCAN_OP
from .dag import IMPLEMENTATION as DAG_IMPLEMENTATION, TAG as DAG_TAG, materialized as materialized_dag
from .compensated import IMPLEMENTATION as COMPENSATED_IMPLEMENTATION
from .scan_buffered import IMPLEMENTATION as BUFFERED_SCAN, PRIMITIVES as SCAN_PRIMITIVES


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


def _implementations():
    base = (
        *implementations(),
        replace(SCAN_IMPLEMENTATION, matches=lambda n,a:
                a.target is SCAN_OP and n.meta.get('trainium_scan_execution','native')=='native'),
        BUFFERED_SCAN,
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
    )
    from .handoff import implementations as handoff_implementations, TAG as HANDOFF_TAG
    base = tuple(replace(spec, matches=lambda n,a,match=spec.matches:
                         HANDOFF_TAG not in n.meta and match(n,a)) for spec in base)
    ordinary = tuple(replace(spec, matches=lambda n, a, match=spec.matches:
                             not n.meta.get(DAG_TAG, False) and match(n, a)) for spec in base)
    return (*ordinary, *materialized_implementations(base), DAG_IMPLEMENTATION,
            materialized_dag(base), COMPENSATED_IMPLEMENTATION, *handoff_implementations(base))


TARGET = registry.register(
    TargetSpec(
        name="trainium",
        hardware=TrainiumConfig,
        operators=tuple((op, OperatorSemantics()) for op in (SCAN_OP,*SCAN_PRIMITIVES)),
        implementations=_implementations(),
        model=TrainiumModel,
        legalize=legalize,
        lower=lower,
        emit=emit,
        encode=encode,
        normalization=(*LAYOUT.items(), ("patterns", FUSION_PATTERNS)),
    )
)
