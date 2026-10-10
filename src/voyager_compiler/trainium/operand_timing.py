"""Operand-aware completion laws for previously uncovered instruction forms.

Parameters come from isolated, numerically checked native instruction probes.
Existing calibrated laws have priority. Lookup depends on engine, dtype,
endpoints and operation function, never on a benchmark name or measured kernel.
Unknown domains stay explicit. Service uses the documented streaming pipeline;
completion adds its bounded drain/writeback and broadcast-operand overhead.
"""

from functools import lru_cache
import json
import math
from pathlib import Path
import re


@lru_cache(maxsize=1)
def parameters():
    return json.loads(
        Path(__file__).with_name("operand_characterization.json").read_text()
    )


def function_name(value):
    value = str(value).lower().removeprefix("nl.")
    return {"identity": "copy", "reciprocal_sqrt": "rsqrt", "ln": "log"}.get(
        value, value
    )


def key(d):
    result = "|".join(
        str(v) if v is not None else ""
        for v in (
            d["opcode"],
            d["engine"],
            d.get("source_dtype"),
            d["dtype"],
            d.get("source_memory"),
            d.get("destination_memory", "SBUF"),
            function_name(d.get("function", "")),
            d.get("reduction_rank", 1) if d["opcode"] == "tensor_reduce" else 0,
        )
    )
    if d["opcode"] == "tensor_tensor":
        result += "|" + "|".join(
            str(d.get(k) or "") for k in ("source1_dtype", "source1_memory")
        )
    return result


def contiguous(shape, strides):
    """Dense arbitrary-rank view, ignoring singleton axes and axis order."""
    if not shape or len(shape) != len(strides):
        return False
    axes = sorted((abs(st), n) for st, n in zip(strides, shape) if n > 1)
    expected = 1
    for st, n in axes:
        if st != expected:
            return False
        expected *= n
    return True


def signature(d):
    """Static sufficient statistics; input work differs from output cardinality."""
    reduction = d["opcode"] == "tensor_reduce"
    return dict(
        input_elements=(
            d.get("source_free", d["free"]) if reduction else d["free"]
        ),
        output_elements=d["free"],
        broadcast_reads=d.get("broadcast_reads", 0),
    )


def cost(d, occupancy):
    model = parameters()["models"].get(key(d))
    if model is None or not 1 <= d["partitions"] <= 128:
        return None
    features = signature(d)
    width = features["input_elements"]
    if not model["min_elements"] <= width <= model["max_elements"]:
        return None
    if not model.get("strided_reduction", False):
        for side in (
            ("source", "source1", "destination")
            if d["opcode"] == "tensor_tensor"
            else ("source", "destination")
        ):
            shapes, strides = d.get(side + "_shape"), d.get(side + "_strides")
            if shapes is not None and (
                shapes or d.get(side + "_memory") is not None
            ):
                if not contiguous(shapes, strides or ()):
                    return None
            elif d.get(side + "_stride") not in (None, 1):
                return None
    if (
        "fixed_broadcast_reads" in model
        and features["broadcast_reads"] != model["fixed_broadcast_reads"]
    ):
        return None
    if d["opcode"] == "tensor_tensor":
        # The generic fallback assumes two serialized SBUF reads (2N).
        # The documented Vector datapath reads PSUM+SBUF in parallel; packed,
        # contiguous BF16 add/multiply/subtract also needs only N cycles.
        # This correction applies only to the new characterized operand paths;
        # existing calibrated FP32 laws retain their own initiation model.
        parallel_ports = {d.get("source_memory"), d.get("source1_memory")} == {
            "SBUF",
            "PSUM",
        }
        packed = d.get("source_dtype") == d.get("source1_dtype") == d[
            "dtype"
        ] == "bfloat16" and function_name(d.get("function", "")) in (
            "add",
            "multiply",
            "subtract",
        )
        if parallel_ports or packed:
            occupancy *= max(64, width) / max(64, 2 * width)
    coefficients = model["completion"]
    completion = coefficients["startup_ns"] + sum(
        coefficients.get(k + "_ns", 0) * v for k, v in features.items()
    )
    # Conservative bound for these incompletely characterized initiation laws;
    # this is a modeling bound, not a universal hardware result-ready rule.
    return occupancy, max(occupancy, completion)


def evaluate(d, occupancy, forward):
    d = dict(d)
    d["function"] = function_name(d.get("function", ""))
    if d["opcode"] == "POOL":
        d["opcode"] = "tensor_reduce"
    result = cost(d, occupancy)
    return (*result, forward) if result is not None else None


def native(opcode, engine, tensors, operands, hardware):
    """Dedicated native selectors and hidden mask-register instruction."""
    from .native_timing import NativeTiming
    from .dependencies import engine_clock

    if opcode == "LOAD_MASK_SELECT":
        mask = re.search(r"\bmasks=([\d,]+)", operands)
        values = list(map(int, mask[1].split(","))) if mask else []
        if (
            engine != "VectorE"
            or len(values) != 32
            or any(not 0 <= v <= 32 for v in values)
        ):
            raise ValueError("Uncharacterized shuffle mask configuration")
        model = parameters()["mask_load"]
        return NativeTiming(
            "native.load_mask_select.VectorE",
            model["issue_ns"],
            model["completion_ns"],
        )
    dst = tensors["dst"]
    src = tensors.get("src")
    match = re.search(r"\bchannels=(\d+)", operands)
    if not match:
        raise ValueError("Native selector needs a partition count")
    width = math.prod(dst[3])
    dtype = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(
        dst[0], dst[0]
    )
    model_name = {
        "TENSOR_SCALAR_AFFINE_SELECT": "affine_select",
        "COPY_PREDICATED_SCALAR": "predicated_scalar",
        "STREAM_SHUFFLE": "stream_shuffle",
    }[opcode]
    if opcode == "COPY_PREDICATED_SCALAR":
        pred = tensors.get("pred")
        if (
            pred is None
            or math.prod(pred[3]) != width
            or pred[0] not in ("uint8", "uint16", "uint32")
        ):
            raise ValueError("Invalid predicated scalar operands")
        if not contiguous(pred[3], pred[2]):
            raise ValueError("Uncharacterized predicate strides")
        if engine != "VectorE":
            raise ValueError("Predicated copy requires VectorE")
        service = max(64, width) / engine_clock(hardware, engine)
    elif opcode == "TENSOR_SCALAR_AFFINE_SELECT":
        if engine != "GpSimdE" or src is None:
            raise ValueError("Affine select requires GpSimdE and a source")
        # GpSimd startup and lane throughput are independently characterized;
        # the TensorE clock is not assumed to be the GpSimd processing clock.
        m = parameters()["affine_service"]
        service = m["startup_ns"] + m["element_ns"] * width
    else:
        if engine != "VectorE" or src is None:
            raise ValueError("Stream shuffle requires VectorE and a source")
        service = max(64, width) / engine_clock(hardware, engine)
    d = dict(
        opcode=model_name,
        engine=engine,
        dtype=dtype,
        source_dtype=(
            {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(
                src[0], src[0]
            )
            if src
            else None
        ),
        source_memory=(
            ("PSUM" if src[1] >= 0x2000000 else "SBUF") if src else None
        ),
        destination_memory="PSUM" if dst[1] >= 0x2000000 else "SBUF",
        partitions=int(match[1]),
        free=width,
        source_free=width,
        source_shape=src[3] if src else (),
        source_strides=src[2] if src else (),
        destination_shape=dst[3],
        destination_strides=dst[2],
        function="",
    )
    if src is None:
        d.pop("source_shape")
        d.pop("source_strides")
    result = cost(d, service)
    return NativeTiming(
        "native." + model_name + "." + engine,
        service,
        result[1] if result is not None else None,
    )
