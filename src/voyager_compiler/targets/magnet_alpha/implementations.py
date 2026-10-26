"""Required PE/PPU and depthwise/NMP instruction regions."""

import torch

from voyager_compiler.codegen.subgraph import create_and_insert_subgraph
from voyager_compiler.ir.buffer import TargetInvoke

from .operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)


def form_execution_regions(model):
    """Keep hardware-internal accumulators inside their instruction region."""
    for node in list(model.graph.nodes):
        if node.target not in (
            INT8_CONV2D_ACCUMULATOR_OPERATOR,
            VSQ_CONV2D_OPERATOR,
        ):
            continue
        chain = [node]
        if len(node.users) != 1:
            raise ValueError(
                f"{node.name}: Alpha accumulator needs one hardware epilogue"
            )
        epilogue = next(iter(node.users))
        if epilogue.target not in (
            PPU_FIXED_RELU_OPERATOR,
            NMP_FIXED_REQUANTIZE_OPERATOR,
        ):
            raise ValueError(
                f"{node.name}: unsupported Alpha accumulator consumer {epilogue.target}"
            )
        chain.append(epilogue)
        if len(epilogue.users) == 1:
            user = next(iter(epilogue.users))
            if user.target in (PVS_QUANTIZE_OPERATOR, SATURATING_ADD_OPERATOR):
                chain.append(user)
            elif (
                epilogue.target == NMP_FIXED_REQUANTIZE_OPERATOR
                and user.target == torch.ops.aten.pad.default
            ):
                padding = tuple(user.args[1])
                if (
                    len(padding) == 6
                    and not any(padding[:5])
                    and padding[5] > 0
                    and not user.kwargs
                ):
                    # A masked NMP channel tail is part of its output encoding.
                    # Retain its exact zero-fill semantics inside the body.
                    chain.append(user)
        opcode = (
            "depthwise_conv2d"
            if epilogue.target == NMP_FIXED_REQUANTIZE_OPERATOR
            else "gemm_conv2d"
        )
        region = create_and_insert_subgraph(chain, model)
        region.meta["target_invoke"] = TargetInvoke(
            "magnet-alpha.whole",
            opcode,
            operands=tuple(n.name for n in region.all_input_nodes),
            reads=tuple(n.name for n in region.all_input_nodes),
            writes=(region.name,),
            resources=(
                ("nmp",)
                if opcode == "depthwise_conv2d"
                else ("matrix", "output_collector")
            ),
            scheduling_owner="hardware",
            allocation_owner="compiler",
        )
    model.graph.lint()
    model.recompile()
    return model
