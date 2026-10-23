# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Target-neutral validation helpers for model output pytrees."""

import math

import torch
from torch.utils._pytree import tree_flatten


def _flatten_tensor_outputs(output, label):
    """Return output leaves after requiring each leaf to be a tensor."""
    leaves, _ = tree_flatten(output)
    invalid_leaves = [
        (index, type(leaf).__name__)
        for index, leaf in enumerate(leaves)
        if not isinstance(leaf, torch.Tensor)
    ]
    if invalid_leaves:
        raise TypeError(
            f"{label} has non-tensor output leaves: {invalid_leaves}"
        )
    return leaves


def _deviation_summary(reference, actual):
    """Describe absolute error for a pair of shape-compatible tensors."""
    if reference.shape != actual.shape:
        return (
            "maximum absolute deviation and RMSE unavailable for "
            "different shapes"
        )

    comparison_dtype = (
        torch.complex128
        if torch.is_complex(reference) or torch.is_complex(actual)
        else torch.float64
    )
    try:
        reference_values = reference.detach().to(comparison_dtype).reshape(-1)
        actual_values = actual.detach().to(comparison_dtype).reshape(-1)
    except (RuntimeError, TypeError, ValueError) as error:
        return f"deviation metrics unavailable: {error}"

    if reference_values.numel() == 0:
        maximum_error = 0.0
        root_mean_square_error = 0.0
    else:
        absolute_error = torch.abs(actual_values - reference_values)
        maximum_error = torch.max(absolute_error).item()
        root_mean_square_error = torch.sqrt(
            torch.mean(absolute_error.square())
        ).item()
    return (
        f"maximum absolute deviation={maximum_error:.8g}, "
        f"RMSE={root_mean_square_error:.8g}"
    )


def _empty_output_statistics():
    """Return the neutral metrics for two empty tensor leaves."""
    return {
        "cosine": 1.0,
        "sqnr_db": math.inf,
        "normalized_rmse": 0.0,
        "mean_absolute_error": 0.0,
        "max_absolute_error": 0.0,
    }


def _output_statistics(reference, actual):
    """Compute similarity and absolute-error metrics for one tensor pair."""
    reference_values = reference.detach().to(torch.float64).reshape(-1)
    actual_values = actual.detach().to(torch.float64).reshape(-1)
    if reference_values.numel() == 0:
        return _empty_output_statistics()
    if (
        not torch.isfinite(reference_values).all()
        or not torch.isfinite(actual_values).all()
    ):
        return None

    error = actual_values - reference_values
    reference_rms = torch.sqrt(torch.mean(reference_values.square())).item()
    root_mean_square_error = torch.sqrt(torch.mean(error.square())).item()
    reference_norm = torch.linalg.vector_norm(reference_values).item()
    actual_norm = torch.linalg.vector_norm(actual_values).item()

    if reference_norm == 0.0 and actual_norm == 0.0:
        cosine = 1.0
    elif reference_norm == 0.0 or actual_norm == 0.0:
        cosine = 0.0
    else:
        cosine = torch.dot(reference_values, actual_values).item()
        cosine /= reference_norm * actual_norm

    if root_mean_square_error == 0.0:
        sqnr_db = math.inf
        normalized_rmse = 0.0
    elif reference_rms == 0.0:
        sqnr_db = -math.inf
        normalized_rmse = math.inf
    else:
        sqnr_db = 20.0 * math.log10(reference_rms / root_mean_square_error)
        normalized_rmse = root_mean_square_error / reference_rms

    absolute_error = torch.abs(error)
    return {
        "cosine": cosine,
        "sqnr_db": sqnr_db,
        "normalized_rmse": normalized_rmse,
        "mean_absolute_error": torch.mean(absolute_error).item(),
        "max_absolute_error": torch.max(absolute_error).item(),
    }


def compare_output_statistics(
    reference,
    actual,
    *,
    label,
    minimum_cosine=None,
    minimum_sqnr_db=None,
):
    """Report metrics for every output and enforce optional thresholds."""
    reference_tensors = _flatten_tensor_outputs(reference, f"{label} reference")
    actual_tensors = _flatten_tensor_outputs(actual, f"{label} actual")
    if len(reference_tensors) != len(actual_tensors):
        raise AssertionError(
            f"{label} output count differs: {len(reference_tensors)} != "
            f"{len(actual_tensors)}"
        )

    failures = []
    statistics = []
    print(f"{label}: comparing {len(reference_tensors)} output tensor(s)")
    for index, (reference_tensor, actual_tensor) in enumerate(
        zip(reference_tensors, actual_tensors)
    ):
        if reference_tensor.shape != actual_tensor.shape:
            failures.append(
                f"output[{index}] shape {tuple(reference_tensor.shape)} != "
                f"{tuple(actual_tensor.shape)}"
            )
            continue

        metrics = _output_statistics(reference_tensor, actual_tensor)
        if metrics is None:
            failures.append(f"output[{index}] contains non-finite values")
            continue

        statistics.append(metrics)
        print(
            f"  output[{index}] shape={tuple(reference_tensor.shape)} "
            f"cosine={metrics['cosine']:.6f} "
            f"sqnr={metrics['sqnr_db']:.2f}dB "
            f"nrmse={metrics['normalized_rmse']:.6f} "
            f"mae={metrics['mean_absolute_error']:.6f} "
            f"max_abs={metrics['max_absolute_error']:.6f}"
        )
        if minimum_cosine is not None and metrics["cosine"] < minimum_cosine:
            failures.append(
                f"output[{index}] cosine {metrics['cosine']:.6f} < "
                f"{minimum_cosine:.6f}"
            )
        if minimum_sqnr_db is not None and metrics["sqnr_db"] < minimum_sqnr_db:
            failures.append(
                f"output[{index}] SQNR {metrics['sqnr_db']:.2f}dB < "
                f"{minimum_sqnr_db:.2f}dB"
            )

    if failures:
        raise AssertionError(f"{label} failed:\n  " + "\n  ".join(failures))
    return statistics


def assert_output_close(reference, actual, *, label, rtol, atol):
    """Require every tensor output leaf to satisfy ``assert_close``."""
    reference_tensors = _flatten_tensor_outputs(reference, f"{label} reference")
    actual_tensors = _flatten_tensor_outputs(actual, f"{label} actual")
    if len(reference_tensors) != len(actual_tensors):
        raise AssertionError(
            f"{label} output count differs: {len(reference_tensors)} != "
            f"{len(actual_tensors)}"
        )

    for index, (reference_tensor, actual_tensor) in enumerate(
        zip(reference_tensors, actual_tensors)
    ):
        try:
            torch.testing.assert_close(
                actual_tensor,
                reference_tensor,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        except AssertionError as error:
            deviation = _deviation_summary(reference_tensor, actual_tensor)
            raise AssertionError(
                f"{label} output[{index}] failed: {deviation}\n{error}"
            ) from error

    print(f"{label}: all {len(reference_tensors)} output tensor(s) match")
