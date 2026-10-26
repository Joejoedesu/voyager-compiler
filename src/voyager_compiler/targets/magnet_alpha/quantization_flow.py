# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
"""Alpha PTQ and format conversion, independent of workload assets and datasets."""

from dataclasses import dataclass, replace
from typing import Callable

import torch

from voyager_compiler.extensions import QuantizationPolicy

from .ptq import convert_pt2e, finalize_pt2e_calibration, prepare_pt2e
from .quantization import prepare_quantization_recalibration


@dataclass(frozen=True)
class AlphaQuantizationRecipe:
    """Binding inputs; samples() returns a fresh iterable of (args, kwargs).

    restore_profile(graph, config) optionally installs saved numerical state
    and returns JSON-compatible provenance. It owns asset interpretation, not
    the subsequent arithmetic configuration or conversion.
    """

    quantizer: object
    samples: Callable
    steps: int = 0
    bias: str | None = None
    region_qspec: object = None
    recalibrate: bool = True
    restore_profile: Callable | None = None
    expected_exact_boundaries: int | None = None
    profile: str = "fresh"
    activation_source: str = "workload calibration samples"


def _calibrate(graph, recipe, before_step=None):
    if recipe.steps == 0:
        return
    count = 0
    with torch.no_grad():
        for args, kwargs in recipe.samples():
            if count >= recipe.steps:
                raise ValueError(
                    "Alpha calibration sample count exceeds requested steps"
                )
            if before_step:
                before_step()
            graph(*args, **kwargs)
            count += 1
    if count != recipe.steps:
        raise ValueError(
            f"Alpha calibration sample count {count} != {recipe.steps}"
        )


def calibrate_and_convert(
    model, example_args, example_kwargs, config, recipe, quantizer
):
    if recipe.steps < 0:
        raise ValueError("Alpha calibration steps must be nonnegative")
    if config.target != "magnet-alpha":
        raise ValueError(
            "Alpha quantization requires an Alpha hardware configuration"
        )
    graph = prepare_pt2e(
        model,
        quantizer,
        example_args,
        example_kwargs,
        hardware_config=config,
        region_qspec=recipe.region_qspec,
    )
    configured = 0
    if recipe.restore_profile:
        report = dict(recipe.restore_profile(graph, config))
        finalize_pt2e_calibration(graph, config)
        configured = prepare_quantization_recalibration(
            graph, reset_observers=False
        )
        if (
            recipe.expected_exact_boundaries is not None
            and configured != recipe.expected_exact_boundaries
        ):
            raise ValueError(
                "Saved Alpha profile configured an unexpected number of target boundaries"
            )
    else:
        _calibrate(graph, recipe)
        if recipe.steps and recipe.recalibrate:
            finalize_pt2e_calibration(graph, config)
            configured = prepare_quantization_recalibration(
                graph, reset_observers=True
            )
            if configured:
                _calibrate(
                    graph,
                    recipe,
                    before_step=lambda: prepare_quantization_recalibration(
                        graph, reset_observers=False
                    ),
                )
        report = dict(calibration_steps=recipe.steps)
    report.update(
        profile=recipe.profile,
        activation_source=recipe.activation_source,
        configured_exact_boundaries=configured,
    )
    convert_pt2e(
        graph,
        recipe.bias,
        calibration_steps=recipe.steps,
        hardware_config=config,
    )
    graph.graph.eliminate_dead_code()
    graph.recompile()
    graph.meta["alpha_calibration_profile"] = report
    return graph, report


def run(prepared, config, options):
    recipe = options.get("alpha_quantization")
    if not isinstance(recipe, AlphaQuantizationRecipe):
        raise ValueError(
            "Alpha quantization requires an AlphaQuantizationRecipe in options['alpha_quantization']; use quantizer=None for an already quantized graph"
        )
    recipe = replace(recipe, bias=options.get("bias", recipe.bias))
    quantizer = options.get("quantizer", recipe.quantizer)
    graph, report = calibrate_and_convert(
        prepared.graph,
        prepared.args,
        prepared.kwargs,
        config,
        recipe,
        quantizer,
    )
    prepared.graph = graph
    prepared.provenance["quantization"] = dict(
        target=config.target,
        flow="alpha",
        **report,
        quantizer=f"{type(quantizer).__module__}.{type(quantizer).__qualname__}",
    )
    return prepared


POLICY = QuantizationPolicy(run=run)
