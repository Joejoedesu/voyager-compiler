"""Shared observer preparation, calibration, finalization and conversion."""

import torch

from voyager_compiler.ir.graph import PreparedWorkload
from voyager_compiler.quantization import convert_pt2e, prepare_pt2e


def quantize(prepared, target, options, *, config=None):
    policy = target.quantization
    if "quantizer" in options and options["quantizer"] is None:
        return prepared
    if policy and policy.run:
        if config is None:
            raise ValueError(
                "target quantization requires the resolved hardware config"
            )
        result = policy.run(prepared, config, options)
        if not isinstance(result, PreparedWorkload):
            raise TypeError("target quantization must return PreparedWorkload")
        if "quantization" not in result.provenance:
            raise ValueError(
                "target quantization must record quantization provenance"
            )
        with torch.no_grad():
            result.reference = result.graph(*result.args, **result.kwargs)
        return result
    quantizer = (
        options["quantizer"]
        if "quantizer" in options
        else policy.quantizer(prepared, options)
        if policy and policy.quantizer
        else None
    )
    if quantizer is None:
        return prepared
    graph = prepare_pt2e(prepared.graph, quantizer)
    if policy and policy.prepare:
        policy.prepare(graph, options)
    samples = options.get(
        "calibration",
        (
            ((prepared.args, prepared.kwargs),)
            if policy and policy.example_calibration
            else ()
        ),
    )
    count = 0
    with torch.no_grad():
        for args, kwargs in samples:
            graph(*args, **kwargs)
            count += 1
    if policy and policy.example_calibration and not count:
        raise ValueError(
            "Quantization requires at least one calibration sample"
        )
    if policy and policy.finalize:
        policy.finalize(graph, options)
    convert_pt2e(
        graph,
        options.get(
            "bias", policy.bias if policy and policy.quantizer else None
        ),
        integer_rounding=policy.integer_rounding if policy else False,
    )
    prepared.graph = graph
    prepared.provenance["quantization"] = dict(
        quantizer=type(quantizer).__module__
        + "."
        + type(quantizer).__qualname__,
        calibration_samples=count,
    )
    with torch.no_grad():
        prepared.reference = graph(*prepared.args, **prepared.kwargs)
    return prepared
