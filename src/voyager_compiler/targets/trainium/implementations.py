"""Searchable FP8 single-row and paired-row TensorE realizations."""

import numpy as np
import torch

from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.ir.buffer import TargetInvoke

from .gemm import build_gemm
from .model import TrainiumModel


def fp8_matrix(node, anchor):
    if "matmul" not in str(anchor.target) and "linear" not in str(
        anchor.target
    ):
        return False
    if "_mx" in str(anchor.target):
        return False
    return all(
        any(
            tag in str(n.meta.get("dtype") or n.value.dtype).lower()
            for tag in ("float8", "e4m3", "e5m2")
        )
        for n in anchor.args[:2]
    )


def model(config, accumulate_fp32, parameters):
    result = TrainiumModel(config, accumulate_fp32)
    result.row_mode = parameters["row_mode"]
    return result


def build(node, context, parameters):
    graph = build_gemm(node, context)
    if graph is None:
        raise CandidateRejected("no legal Trainium matrix tile")
    for module in graph.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for operation in module.graph.nodes:
            if operation.op == "call_function" and any(
                name in str(operation.target) for name in ("matmul", "linear")
            ):
                operation.meta["target_invoke"] = TargetInvoke(
                    f"trainium.gemm.{parameters['row_mode']}",
                    f"nc_matmul.{parameters['row_mode']}",
                    (),
                    (),
                    (),
                    resources=("matrix", "psum"),
                    scheduling_owner="hardware",
                )
    return graph


def implementations():
    return tuple(
        ImplementationSpec(
            f"trainium.gemm.{mode}",
            fp8_matrix,
            build,
            candidates=lambda n, c, mode=mode: ({"row_mode": mode},),
            model=model,
            description=(
                "Independent 128-deep FP8 contractions"
                if mode == "single_row"
                else "Paired FP8 contractions where the selected operand layout permits double_row"
            ),
        )
        for mode in ("single_row", "double_row")
    )


def encode(encoded, program, config):
    """Carry the frozen choice into the NKI projection; no late selection."""
    modes = {}
    for module in program.graph.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            invocation = node.meta.get("target_invoke")
            if invocation is None or not invocation.opcode.startswith(
                "nc_matmul."
            ):
                continue
            mode = invocation.opcode.split(".", 1)[1]
            if node.name in modes and modes[node.name] != mode:
                raise ValueError(f"ambiguous NKI primitive name {node.name}")
            modes[node.name] = mode

    def walk(message):
        if (
            message.DESCRIPTOR.full_name == "voyager.PrimOp"
            and message.name in modes
        ):
            message.kwargs["nki_row_mode"].str_value = modes[message.name]
        for field, value in message.ListFields():
            if field.message_type is not None:
                children = (
                    value.values()
                    if field.message_type.GetOptions().map_entry
                    else value if field.is_repeated else (value,)
                )
                for child in children:
                    walk(child)

    walk(encoded.model)
    from .translate import immediates

    constants = {}
    for name, value in immediates(program.graph, encoded.model).items():
        key = f"immediate-{len(constants)}"
        encoded.payloads[key] = np.asarray(value)
        constants[name] = key
    encoded.target_data = dict(
        schema_version=1,
        immediates=constants,
        moving_free_max=config.moving_free_max,
    )
    return encoded
