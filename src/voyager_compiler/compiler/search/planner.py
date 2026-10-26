"""Compare implementation recipes using the common tile-search providers.

Builders expand provisional candidates during refinement. The winner's graph
is then spliced by bufferization; encoding never repeats selection.
"""

import math
from dataclasses import replace

from voyager_compiler.compiler.search.tiler import Tiler
from voyager_compiler.extensions import CandidateRejected
from voyager_compiler.ir.plan import ImplementationChoice


def candidate_key(implementation, parameters):
    from voyager_compiler.artifacts import content_digest

    return content_digest((implementation, dict(parameters)))


def _metadata_copy(value):
    # Preserve FX/tensor/immutable IR identities; isolate mutable metadata
    # containers without copying weights or whole owning GraphModules.
    if isinstance(value, dict):
        return {key: _metadata_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_metadata_copy(item) for item in value)
    if isinstance(value, set):
        return set(value)
    return value


def _candidate_nodes(node, anchor):
    nodes = [node]
    if anchor is not node:
        nodes.append(anchor)
    submodule = node.meta.get("submodule")
    if submodule is not None:
        for module in submodule.modules():
            if hasattr(module, "graph"):
                nodes.extend(
                    n
                    for n in module.graph.nodes
                    if all(n is not previous for previous in nodes)
                )
    return nodes


def select_implementation(
    node, anchor, context, implementations, *, budget=64, excluded=()
):
    candidates = []
    rejected = []
    candidate_nodes = _candidate_nodes(node, anchor)
    original = [(n, _metadata_copy(n.meta)) for n in candidate_nodes]

    def restore(snapshot):
        for n, metadata in snapshot:
            n.meta.clear()
            n.meta.update(_metadata_copy(metadata))

    count = 0
    for implementation in implementations:
        if not implementation.matches(node, anchor):
            continue
        for parameters in implementation.candidates(node, context):
            count += 1
            if count > budget:
                raise CandidateRejected(
                    f"{node.name}: implementation candidate budget {budget} exhausted"
                )
            parameters = dict(parameters)
            if candidate_key(implementation.name, parameters) in excluded:
                rejected.append(
                    f"{implementation.name}: rejected by refined allocation"
                )
                continue
            restore(original)
            trial_context = context
            try:
                profile = (
                    getattr(context.tiler.model, "profile", None)
                    if hasattr(context.tiler, "model")
                    else None
                )
                recipe_profile = profile is not None and any(
                    key == implementation.name
                    for key, _, _ in profile.coefficients
                )
                if implementation.model is not None or recipe_profile:
                    model = (
                        implementation.model(
                            context.tiler.config,
                            context.accumulate_fp32,
                            parameters,
                        )
                        if implementation.model is not None
                        else context.tiler.model.base
                    )
                    from voyager_compiler.performance.profile import (
                        CalibratedModel,
                    )

                    # Recipe-specific measurements supersede an op-family
                    # correction; never multiply both into one estimate.
                    if (
                        isinstance(context.tiler.model, CalibratedModel)
                        and not recipe_profile
                    ):
                        model = CalibratedModel(
                            model, context.tiler.model.profile
                        )
                    trial_context = replace(
                        context,
                        tiler=Tiler(
                            model,
                            context.tiler.runtime_tolerance,
                            constraints=dict(context.tiler.constraints),
                        ),
                    )
                graph = implementation.build(node, trial_context, parameters)
                if implementation.validate is not None:
                    implementation.validate(
                        node, graph, trial_context, parameters
                    )
                tile = node.meta.get("tile_result")
                cost = (
                    implementation.estimate(
                        node, graph, trial_context, parameters
                    )
                    if implementation.estimate
                    else tile.cost if tile else None
                )
                if cost is not None and recipe_profile:
                    cost = replace(
                        cost,
                        latency=profile.correct(
                            implementation.name, cost.latency
                        ),
                    )
                if cost is not None and (
                    not math.isfinite(cost.latency) or cost.latency < 0
                ):
                    raise CandidateRejected(
                        "estimated latency must be finite and nonnegative"
                    )
                choice = ImplementationChoice(
                    implementation.name,
                    tuple(sorted(parameters.items())),
                    tile,
                    cost,
                )
                candidates.append(
                    (
                        choice,
                        graph,
                        [(n, _metadata_copy(n.meta)) for n in candidate_nodes],
                    )
                )
            except CandidateRejected as error:
                rejected.append(f"{implementation.name}: {error}")
            finally:
                restore(original)
    restore(original)
    if not count:
        return None, None, (), ()
    if not candidates:
        raise CandidateRejected(
            f"{node.name}: no legal implementation: {'; '.join(rejected)}"
        )
    if len(candidates) > 1:
        if any(c[0].cost is None for c in candidates):
            raise ValueError(
                f"{node.name}: all competing implementations require modeled costs"
            )
        fastest = min(c[0].cost.latency for c in candidates)
        tolerance = context.tiler.runtime_tolerance if context.tiler else 0
        eligible = [
            c
            for c in candidates
            if c[0].cost.latency <= fastest * (1 + tolerance)
        ]
        winner = min(
            eligible,
            key=lambda c: (
                (
                    c[0].cost.dram_bytes
                    if c[0].cost.dram_bytes is not None
                    else math.inf
                ),
                c[0].cost.latency,
                c[0].implementation,
            ),
        )
    else:
        winner = candidates[0]
    choice, graph, metadata = winner
    restore(metadata)
    node.meta["implementation"] = choice
    return graph, choice, tuple(c[0] for c in candidates), tuple(rejected)
