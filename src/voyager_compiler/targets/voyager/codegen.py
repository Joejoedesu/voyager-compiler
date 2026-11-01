"""Voyager's established wire opcodes and implicit GEMM layout convention.

The shared buffer IR has explicit layouts and generic with_scratch regions.
Voyager's consumer predates these spellings: its matrix opcode implies xc,ck,
and its normalization opcodes accept named scratch operands. This projection
changes no loop, transfer, address or numerical operation.
"""


def _messages(message):
    yield message
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            values = (
                value.values()
                if field.message_type.GetOptions().map_entry
                else value
            )
            for child in values:
                yield from _messages(child)
        else:
            yield from _messages(value)


def encode(encoded, program, config):
    """Project explicit matrix layouts into the established Voyager ABI."""
    model = encoded.model
    matrix_ops = {
        "aten::linear",
        "aten::matmul",
        "quantized_ops::linear",
        "quantized_ops::matmul",
    }
    mx_ops = {"quantized_ops::linear_mx", "quantized_ops::matmul_mx"}
    for message in _messages(model):
        if (
            message.DESCRIPTOR.name != "PrimOp"
            or "layout" not in message.kwargs
        ):
            continue
        layout = message.kwargs["layout"].str_value
        if message.target in mx_ops and layout in ("xc,ck", "xc,kc"):
            message.kwargs["weight_layout"].str_value = layout.split(",")[1]
            del message.kwargs["layout"]
        elif message.target in matrix_ops and layout == "xc,ck":
            del message.kwargs["layout"]
    return encoded
