"""Optional binding-selected input/graph preparation before quantization."""

import torch

from voyager_compiler.ir.graph import PreparedWorkload


def preprocess_inputs(prepared, target, config, options):
    """Apply one selected frontend alternative, leaving constant folding separate.

    The callback receives (prepared, config, options) and returns PreparedWorkload.
    A caller's input_preprocessing option replaces the target default; None skips.
    Calibration samples are available in options before this call. If the input
    representation changes, the callback must adapt those samples as well as the
    example inputs and record the external input contract in its provenance.
    This hook neither searches alternatives nor installs a runtime input adapter.
    """
    callback = options.get("input_preprocessing", target.input_preprocessing)
    if callback is None:
        return prepared
    if not callable(callback):
        raise TypeError("input_preprocessing must be callable or None")
    provenance = dict(prepared.provenance)
    result = callback(prepared, config, options)
    if not isinstance(result, PreparedWorkload):
        raise TypeError("input preprocessing must return PreparedWorkload")
    if not isinstance(result.provenance.get("input_preprocessing"), dict):
        raise ValueError("input preprocessing must record its provenance")
    result.provenance = provenance | result.provenance
    with torch.no_grad():
        result.reference = result.graph(*result.args, **result.kwargs)
    return result
