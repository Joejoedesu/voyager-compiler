"""Bounded native-only timings, separate from NKI lowering and allocation.

Scan service follows the documented two-input VectorE pipeline. Completion
includes a measured drain and, for a tensor seed, an additional operand read.
Accumulator readback has its own issue/result milestones. Values and evidence
live in native_characterization.json; no application timing enters evaluation.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from .dependencies import engine_clock


@dataclass(frozen=True)
class NativeTiming:
    implementation: str
    issue_ns: float
    completion_ns: float | None
    forward_ns: float | None = None
    read_ns: float | None = None


@lru_cache(maxsize=1)
def parameters():
    return json.loads(
        Path(__file__).with_name("native_characterization.json").read_text()
    )


def evaluate(opcode, engine, tensors, operands, hardware):
    """Return a native cost, or None when this adapter does not own the op.

    An analytical service with unknown completion remains explicitly partial.
    The characterized shapes are not legality restrictions on the compiler.
    """
    if opcode == "MOVE":
        if engine != "GpSimdE" or not re.search(
            r"\bdtype=uint32\s+\$R\[\d+\]=(?:0x[0-9a-f]+|\d+)\s*$", operands
        ):
            raise ValueError("Uncharacterized native MOVE form")
        m = parameters()["models"]["move_uint32_immediate"]
        return NativeTiming(
            "native.move.uint32.immediate.GpSimdE",
            m["issue_ns"],
            m["completion_ns"],
        )
    owned = opcode in ("TENSOR_TENSOR_SCAN", "ACTIVATION_READ_ACCUMULATOR")
    accumulate = opcode == "ACTIVATE" and bool(
        re.search(
            r"\baccumulator_cmd=(?:ZERO_ACCUMULATE|ACCUMULATE)\b", operands
        )
    )
    if not owned and not accumulate:
        return None
    dst = tensors["dst"]
    dtype = dst[0]
    free = math.prod(dst[3])
    match = re.search(r"\bchannels=(\d+)", operands)
    if not match or not 1 <= int(match[1]) <= 128:
        raise ValueError("Native timing needs a valid partition count")
    partitions = int(match[1])
    if opcode == "TENSOR_TENSOR_SCAN":
        if engine != "VectorE":
            raise ValueError("Scan timing requires VectorE")
        src0, src1 = tensors["src0"], tensors["src1"]
        if math.prod(src0[3]) != free or math.prod(src1[3]) != free:
            raise ValueError("Inconsistent native scan shapes")
        m = parameters()["models"]["scan"]
        service = max(
            m["minimum_cycles"], m["cycles_per_element"] * free
        ) / engine_clock(hardware, engine)
        characterized = (
            dtype == src0[0] == src1[0] == "fp32"
            and partitions in m["partitions"]
            and m["min_width"] <= free <= m["max_width"]
            and all(
                t[1] < 0x2000000
                and t[2][0] == 1
                and all(x == 1 for x in t[3][1:])
                for t in (src0, src1, dst)
            )
            and "ops=MULTIPLY,ADD" in operands
        )
        seed = re.search(r"\bimm=([^ ]+)", operands)
        seed_pointer = (
            re.fullmatch(r"\[fp32@(0x[0-9a-f]+)\]", seed[1]) if seed else None
        )
        tensor_seed = bool(
            seed_pointer and int(seed_pointer[1], 16) < 0x2000000
        )
        characterized &= bool(
            seed
            and (
                tensor_seed
                or seed[1] in ("0", "0.000000", "0x0", "fp32@0.000000")
            )
        )
        completion = (
            service
            + m["drain_ns"]
            + (m["tensor_seed_read_ns"] if tensor_seed else 0)
            if characterized
            else None
        )
        return NativeTiming(
            "native.tensor_tensor_scan.VectorE", service, completion
        )
    if engine != "ScalarE":
        raise ValueError("Activation accumulator timing requires ScalarE")
    m = parameters()["models"]["activation_accumulator"]
    if opcode == "ACTIVATION_READ_ACCUMULATOR":
        if free != 1:
            raise ValueError(
                "Accumulator readback must have one value per partition"
            )
        calibrated = (
            dtype == "fp32"
            and partitions in m["partitions"]
            and dst[1] < 0x2000000
        )
        service = (
            m["read_issue_ns"]
            if calibrated
            else 64 / engine_clock(hardware, engine)
        )
        return NativeTiming(
            "native.activation_read_accumulator.ScalarE",
            service,
            m["read_completion_ns"] if calibrated else None,
            read_ns=service if calibrated else None,
        )
    # A reduction-producing activation writes hidden ScalarE accumulator state.
    # Ordinary activations retain their existing timing laws.
    src = tensors.get("src")
    calibrated = (
        src is not None
        and src[0] == "fp32"
        and dtype in ("fp32", "bf16")
        and partitions in m["partitions"]
        and " EXP " in " " + operands
        and m["min_width"] <= free <= m["max_width"]
        and src[2][0] == dst[2][0] == 1
        and src[1] < 0x2000000
        and dst[1] < 0x2000000
        and all(x == 1 for x in src[3][1:] + dst[3][1:])
    )
    if not calibrated:
        return None
    service = max(64, free) / engine_clock(hardware, engine)
    return NativeTiming(
        "native.activation_accumulate.exp.ScalarE",
        service,
        service + m["producer_completion_tail_ns"],
        forward_ns=service + m["producer_forward_tail_ns"],
    )
