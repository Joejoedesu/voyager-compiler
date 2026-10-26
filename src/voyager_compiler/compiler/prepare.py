"""Workload construction, asset checking and the shared export frontend."""

import hashlib
from pathlib import Path

import torch

from voyager_compiler.export_utils import export_model
from voyager_compiler.ir.graph import PreparedWorkload


def prepare(workload, options=None):
    options = {} if options is None else options
    provenance = {}
    for path, expected in workload.assets:
        path = Path(path)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected is not None and checksum != expected:
            raise ValueError(
                f"{workload.name}: asset checksum mismatch: {path}"
            )
        provenance[str(path)] = checksum
    model = workload.build(options)
    args, kwargs = workload.example_inputs(model, options)
    args, kwargs = tuple(args), dict(kwargs)
    if workload.prepare is not None:
        prepared = workload.prepare(model, args, kwargs, options)
        if not isinstance(prepared, PreparedWorkload):
            raise TypeError(
                "workload prepare hook must return PreparedWorkload"
            )
        prepared.provenance.update(provenance)
        return prepared
    if workload.preprocess is not None:
        args, kwargs = workload.preprocess(args, kwargs, options)
    with torch.no_grad():
        reference = (
            workload.reference(model, args, kwargs)
            if workload.reference
            else model(*args, **kwargs)
        )
        graph = (
            model
            if isinstance(model, torch.fx.GraphModule)
            else export_model(model, args, kwargs)
        )
    return PreparedWorkload(
        graph, tuple(args), dict(kwargs), workload.name, reference, provenance
    )
