"""Bounded primitive completion models; source/destination geometry stays explicit."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def parameters():
    return json.loads(
        Path(__file__).with_name("isa_characterization.json").read_text()
    )


@lru_cache(maxsize=512)
def legacy_cold(width):
    from .timing import measured_timings

    law = measured_timings().operation("nki.matmul.float32")
    return law.evaluate(4 * max(64, width) / 2.4)[1]


def sectors(width, stride, element_bytes=4, sector_bytes=16):
    # Regular positive FP32 access: unique16-byte sectors, not span/sector.
    # Once stride exceeds a sector each element touches one distinct sector.
    if stride * element_bytes >= sector_bytes:
        return width
    return ((width - 1) * stride * element_bytes) // sector_bytes + 1


def legacy_evaluate(d, occupancy, latency, forward):
    if (
        d["dtype"] != "float32"
        or d["partitions"] != 128
        or d.get("source_dtype", "float32") not in ("float32", None)
    ):
        return None
    if d["opcode"] == "nc_matmul" and not d.get("transpose"):
        short = parameters()["models"].get("shortk")
        if (
            short
            and d.get("stationary") == 128
            and d.get("contraction") == 64
            and d.get("moving") == 512
            and d.get("stationary_stride") == 1
            and d.get("moving_stride") == 1
        ):
            completion = short[
                (
                    "stream_completion_ns"
                    if d.get("streaming")
                    else "fresh_completion_ns"
                )
            ]
            return short["issue_ns"], completion, short["issue_ns"]
        m = parameters()["models"].get("matmul")
        if (
            m is None
            or d.get("stationary") != 128
            or d.get("contraction") != 128
            or d.get("stationary_stride") != 1
        ):
            return None
        w = d["moving"]
        stride = d["moving_stride"]
        if (
            not min(m["widths"]) <= w <= max(m["widths"])
            or stride is None
            or not 1 <= stride <= m["max_stride"]
        ):
            return None
        features = (1, w, max(0, 2 * sectors(w, stride) - w))
        issue = sum(
            a * b for a, b in zip(features, m["coefficients"]["issue_ns"])
        )
        completion = sum(
            a * b for a, b in zip(features, m["coefficients"]["completion_ns"])
        )
        # Retain the existing cold-path bound; independent streams identify
        # steady feed throughput, not the cost of draining/restarting a pipeline.
        if not d.get("streaming"):
            completion = max(completion, legacy_cold(w), latency or 0)
        return issue, max(issue, completion), issue
    op = d["opcode"]
    key = None
    w = d["free"]
    stride = 1
    if d.get("transpose") and d["engine"] == "TensorE":
        key = "transpose"
        w = d["source_free"]
        stride = d.get("source_stride")
    elif op == "tensor_copy":
        if d["source_memory"] == "SBUF" and d["engine"] == "ScalarE":
            key = "copy"
            if (
                d.get("source_stride") is None
                or d.get("destination_stride") is None
            ):
                return None
            stride = max(
                d.get("source_stride") or 0, d.get("destination_stride") or 0
            )
            # Both independently strided operands have not been characterized.
            if not (
                d.get("source_stride") == 1 or d.get("destination_stride") == 1
            ):
                return None
        elif d["engine"] == "VectorE":
            if d["source_memory"] == "SBUF" and (
                d.get("source_stride") != 1 or d.get("destination_stride") != 1
            ):
                return None
            key = "psum_copy" if d["source_memory"] == "PSUM" else "copy_vector"
    elif op == "activation":
        key = d["function"].removeprefix("nl.")
        # Activation COPY is a ScalarE affine operation; the old "copy"
        # calibration describes tensor_copy and cannot identify this path.
        if key in ("copy", "identity"):
            return None
    elif op == "reciprocal":
        key = "reciprocal"
    elif op == "tensor_reduce":
        # Existing reduction probes produced one output per partition.
        # Multi-output and multi-axis forms use the operand characterization.
        if d["free"] != 1 or d.get("reduction_rank", 1) != 1:
            return None
        fn = d["function"].removeprefix("nl.")
        if fn not in ("add", "max"):
            return None
        key = (
            ("psum_" if d["source_memory"] == "PSUM" else "")
            + "reduce_"
            + ("sum" if fn == "add" else "max")
        )
        w = d["source_free"]
    m = parameters()["models"].get(key)
    if m is None:
        return None
    if m["template"] in ("sectors16", "sector_ramp"):
        if (
            not min(m["widths"]) <= w <= max(m["widths"])
            or stride is None
            or not 1 <= stride <= m["max_stride"]
        ):
            return None
        features = (1, w, max(0, 2 * sectors(w, stride) - w))
        if m["template"] == "sector_ramp":
            lo, hi = m["stride_ramp_bounds"]
            features += (w * (stride - lo) if lo < stride <= hi else 0,)
    else:
        if not m["min_width"] <= w <= m["max_width"]:
            return None
        features = (1, w)
    completion = sum(a * b for a, b in zip(features, m["coefficients"]))
    if key == "copy" and w == 128:
        stream_name = (
            "copy_stream"
            if d.get("destination_stride") == 1
            else "scatter_stream"
        )
        issue_model = parameters()["models"].get(stream_name)
        if issue_model and stride <= issue_model["max_stride"]:
            occupancy = sum(
                a * b for a, b in zip(features, issue_model["coefficients"])
            )
    return occupancy, completion, forward


def evaluate(d, occupancy, latency, forward):
    """Prefer previously validated laws; extend missing operand domains only."""
    cost = legacy_evaluate(d, occupancy, latency, forward)
    if cost is not None:
        return cost
    if latency is not None:
        return None
    from .operand_timing import evaluate as operand_evaluate

    return operand_evaluate(d, occupancy, forward)


def pipeline_timing(descriptors, *, startup_scenario=False):
    """Choose cold/overlapped costs using predicted readiness, never a trace.

    State lives in one graph evaluation. A command that starts after TensorE's
    last result sees a drained pipeline. Same-source instructions can therefore
    receive different costs in different legal schedules.
    """
    from dataclasses import replace

    last_result = float("-inf")
    last_was_matmul = False

    def timing(node, start):
        nonlocal last_result, last_was_matmul
        if node.resource != "TensorE":
            return node
        d = descriptors.get(node.name)
        if d is not None:
            overlap = start < last_result
            if d.get("transpose"):
                m = parameters()["models"].get("transpose_stream")
                if (
                    m
                    and d["dtype"] == "float32"
                    and d["partitions"] == 128
                    and d["source_free"] == 128
                    and d.get("source_stride") is not None
                    and 1 <= d["source_stride"] <= m["max_stride"]
                ):
                    prefix = "startup_" if startup_scenario else ""
                    completion = (
                        m[prefix + "completion_ns"]
                        if overlap
                        else node.latency_ns
                    )
                    issue = m[prefix + "issue_ns"]
                    node = replace(
                        node,
                        issue_ns=issue,
                        occupancy_ns=issue,
                        latency_ns=max(issue, completion),
                    )
            else:
                cost = evaluate(
                    dict(d, streaming=overlap and last_was_matmul),
                    node.occupancy_ns,
                    node.latency_ns,
                    node.forward_ns,
                )
                if cost is not None:
                    node = replace(
                        node,
                        issue_ns=cost[0],
                        occupancy_ns=cost[0],
                        latency_ns=cost[1],
                        forward_ns=cost[2],
                    )
        last_was_matmul = bool(
            d and d.get("opcode") == "nc_matmul" and not d.get("transpose")
        )
        last_result = max(last_result, start + node.offset("result"))
        return node

    return timing
