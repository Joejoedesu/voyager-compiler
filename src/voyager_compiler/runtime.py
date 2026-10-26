"""Execution and numerical validation, distinct from modeled latency."""

import time

import torch


def run(
    result,
    args=None,
    kwargs=None,
    *,
    target=None,
    reference=None,
    rtol=5e-2,
    atol=1e-4,
):
    if args is None:
        args = result.args
        kwargs = result.kwargs if kwargs is None else kwargs
    program = result.program
    program.assert_frozen()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = (
            target.runtime(result, args, kwargs or {})
            if target and target.runtime
            else program.graph(*args, **(kwargs or {}))
        )
    elapsed = time.perf_counter() - start
    program.assert_frozen()
    if reference is not None:
        torch.testing.assert_close(outputs, reference, rtol=rtol, atol=atol)
    return outputs, dict(
        kind="target" if target and target.runtime else "host_replay",
        seconds=elapsed,
    )
