# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase
from torch.fx import Node

from voyager_compiler.fake_quantize import (
    DEFAULT_QUANTIZATION_ROUNDING,
    SUPPORTED_QUANTIZATION_ROUNDING,
    FusedAmaxObsFakeQuantize,
)

__all__ = [
    "QuantizationSpec",
]


class QScheme(Enum):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    MICROSCALING = "microscaling"
    GROUP_WISE_AFFINE = "group_wise_affine"
    VSQ = "vsq"


DEFAULT_VSQ_COARSE_QSCHEME = QScheme.PER_TENSOR_SYMMETRIC


ABBREV_MAP = {
    "qmin": "quant_min",
    "qmax": "quant_max",
    "sqmax": "scale_quant_max",
    "rnd": "rounding",
    "qs": "qscheme",
    "ahl": "amax_history_len",
    "ax": "ch_axis",
    "bs": "block_size",
    "scale": "scale_dtype",
    "othr": "outlier_threshold",
    "opct": "outlier_pct",
    "coarse": "coarse_dtype",
    "cqmin": "coarse_quant_min",
    "cqmax": "coarse_quant_max",
    "csqmax": "coarse_scale_quant_max",
    "cqs": "coarse_qscheme",
    "cax": "coarse_ch_axis",
}


def parse_int_or_list(value: str):
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        parts = tuple(int(v.strip()) for v in value[1:-1].split(","))
        return parts
    return int(value)


PARAMS_TYPE = {
    "quant_min": float,
    "quant_max": float,
    "scale_quant_max": float,
    "rounding": str,
    "qscheme": QScheme,
    "amax_history_len": int,
    "ch_axis": parse_int_or_list,
    "block_size": parse_int_or_list,
    "scale_dtype": str,
    "outlier_threshold": float,
    "outlier_pct": float,
    "coarse_dtype": str,
    "coarse_quant_min": float,
    "coarse_quant_max": float,
    "coarse_scale_quant_max": float,
    "coarse_qscheme": QScheme,
    "coarse_ch_axis": int,
}


def get_quant_min_max(dtype: str):
    # Signed integers
    if match := re.fullmatch(r"int(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        max_val = 2 ** (nbits - 1) - 1
        min_val = -(2 ** (nbits - 1))
        return min_val, max_val

    # Unsigned integers
    if match := re.fullmatch(r"uint(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        return 0, 2**nbits - 1

    # Floating-point like fpN_eXmY
    if match := re.fullmatch(r"fp(\d+)_e(\d+)m(\d+)", dtype, re.IGNORECASE):
        ebits = int(match.group(2))
        mbits = int(match.group(3)) + 2
        emax = 2 ** (ebits - 1) - 1 if ebits > 4 else 2 ** (ebits - 1)

        if dtype.lower() == "fp8_e4m3":
            max_val = 2**emax * 1.75  # max mantissa (1.75)
        else:
            max_val = 2**emax * (2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)

        return -max_val, max_val

    # Posit numbers
    if match := re.fullmatch(r"posit(\d+)_(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        es = int(match.group(2))
        max_val = (2 ** (2**es)) ** (nbits - 2)
        return -max_val, max_val

    # Normalized floats (NF)
    if match := re.fullmatch(r"nf(\d+)(?:_(\d+))?", dtype, re.IGNORECASE):
        if match.group(2) is not None:
            max_val = 2 ** (int(match.group(2)) - 1) - 1
        else:
            max_val = 1
        return -max_val, max_val

    raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass(eq=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, qscheme, quant_max etc.
    """

    dtype: str
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        FusedAmaxObsFakeQuantize
    )
    quant_min: Optional[float] = None
    quant_max: Optional[float] = None
    scale_quant_max: Optional[float] = None
    rounding: str = DEFAULT_QUANTIZATION_ROUNDING
    qscheme: Optional[QScheme] = None
    amax_history_len: Optional[int] = None
    ch_axis: Optional[Union[int, List[int]]] = None
    block_size: Optional[Union[int, List[int]]] = None
    scale_dtype: Optional[str] = None
    outlier_threshold: Optional[float] = None
    outlier_pct: Optional[float] = None
    coarse_dtype: Optional[str] = None
    coarse_quant_min: Optional[float] = None
    coarse_quant_max: Optional[float] = None
    coarse_scale_quant_max: Optional[float] = None
    coarse_qscheme: Optional[QScheme] = None
    coarse_ch_axis: Optional[int] = None
    is_dynamic: bool = False  # required by sharing nodes

    @staticmethod
    def from_str(s):
        if not s:
            raise ValueError("String quantization_spec is None or empty")

        fields = re.split(r",(?![^()]*\))", s)
        params = {"dtype": fields[0]}

        for item in fields[1:]:
            if "=" not in item:
                raise ValueError(f"Expected key=value format but got '{item}'")
            key, value = item.split("=")
            key = ABBREV_MAP.get(key, key)
            if key not in PARAMS_TYPE:
                valid = ", ".join(PARAMS_TYPE.keys())
                raise ValueError(
                    f"Unknown argument '{key}'. Valid keys: {valid}"
                )
            params[key] = PARAMS_TYPE[key](value)

        if (qscheme := params.get("qscheme", None)) is not None:
            qmin, qmax = get_quant_min_max(params["dtype"])
            params.setdefault("quant_min", float(qmin))
            params.setdefault("quant_max", float(qmax))
            if qscheme in [
                QScheme.PER_TENSOR_SYMMETRIC,
                QScheme.PER_CHANNEL_SYMMETRIC,
                QScheme.VSQ,
            ]:
                params.setdefault("amax_history_len", 16)

            if qscheme == QScheme.VSQ:
                params.setdefault("coarse_qscheme", DEFAULT_VSQ_COARSE_QSCHEME)
                coarse_dtype = params.get("coarse_dtype")
                if coarse_dtype is not None:
                    coarse_qmin, coarse_qmax = get_quant_min_max(coarse_dtype)
                    params.setdefault("coarse_quant_min", float(coarse_qmin))
                    params.setdefault("coarse_quant_max", float(coarse_qmax))

        return QuantizationSpec(**params)

    def __post_init__(self):
        if self.qscheme is not None and self.quant_max is None:
            raise ValueError("quant_max is required for quantization.")
        if self.scale_quant_max is not None and (
            not math.isfinite(self.scale_quant_max) or self.scale_quant_max <= 0
        ):
            raise ValueError("scale_quant_max must be positive.")
        if self.rounding not in SUPPORTED_QUANTIZATION_ROUNDING:
            raise ValueError(
                "rounding must be one of "
                f"{sorted(SUPPORTED_QUANTIZATION_ROUNDING)}."
            )

        if (
            self.qscheme
            in [
                QScheme.MICROSCALING,
                QScheme.GROUP_WISE_AFFINE,
                QScheme.VSQ,
            ]
            and self.block_size is None
        ):
            raise ValueError("block_size is required for block quantization.")

        if self.qscheme == QScheme.VSQ:
            if self.coarse_dtype is None:
                raise ValueError("coarse_dtype is required for VSQ.")
            if self.coarse_quant_max is None:
                raise ValueError("coarse_quant_max is required for VSQ.")
            if self.coarse_scale_quant_max is not None and (
                not math.isfinite(self.coarse_scale_quant_max)
                or self.coarse_scale_quant_max <= 0
            ):
                raise ValueError("coarse_scale_quant_max must be positive.")
            if self.coarse_qscheme not in {
                QScheme.PER_TENSOR_SYMMETRIC,
                QScheme.PER_CHANNEL_SYMMETRIC,
            }:
                raise ValueError(
                    "VSQ coarse_qscheme must be per-tensor or per-channel."
                )
            if (
                self.coarse_qscheme == QScheme.PER_CHANNEL_SYMMETRIC
                and self.coarse_ch_axis is None
            ):
                raise ValueError(
                    "coarse_ch_axis is required for per-channel VSQ."
                )
            if (
                self.coarse_qscheme == QScheme.PER_TENSOR_SYMMETRIC
                and self.coarse_ch_axis is not None
            ):
                raise ValueError(
                    "coarse_ch_axis is only valid for per-channel VSQ."
                )
            has_outliers = (
                self.outlier_threshold is not None
                or self.outlier_pct is not None
            )
            if has_outliers:
                raise ValueError("Outlier filtering is not supported for VSQ.")


EdgeOrNode = Union[Tuple[Node, Node], Node]


@dataclass(eq=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    """quantization spec for the Tensors whose quantization parameters are
    derived from other Tensors
    """

    derived_from: List[EdgeOrNode]
    derive_qparams_fn: Callable[
        [List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]
    ]
    dtype: str
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[QScheme] = None
