"""Numeric formats a tensor can be quantized to.

Each module maps bfloat16 bit patterns onto the representable values of one
format, which ``fake_quantize`` turns into a lookup table.
"""

from voyager_compiler.quantization.dtypes.fp8 import quantize_to_minifloat
from voyager_compiler.quantization.dtypes.normal_float import (
    create_normal_map,
    quantize_to_nf,
)
from voyager_compiler.quantization.dtypes.posit import quantize_to_posit

__all__ = [
    "create_normal_map",
    "quantize_to_minifloat",
    "quantize_to_nf",
    "quantize_to_posit",
]
