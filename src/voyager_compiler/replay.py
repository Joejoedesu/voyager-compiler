"""Optional flat-program replay images from finalized allocations.

Shared payload bounds, poison and input/reference separation; command files,
endpoint manifests and device submission remain target-owned. This helper is
for flat native programs, not a general interpreter for asynchronous kernels.
"""

from dataclasses import dataclass

import torch
from torch.fx import Interpreter
from torch.utils._pytree import tree_flatten


@dataclass
class ReplayImages:
    initial: bytearray
    expected: bytearray

    @classmethod
    def create(cls, size, *, poison=0xA5):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Replay image needs a positive byte size")
        initial = bytearray([poison]) * size
        return cls(initial, bytearray(initial))

    def store(self, start, end, payload, *, is_input=False):
        """A payload must fit both its allocation and the complete image."""
        if (
            not 0 <= start <= end <= len(self.initial)
            or len(payload) > end - start
        ):
            raise ValueError(
                "Replay payload exceeds allocation or image bounds"
            )
        if is_input:
            self.initial[start : start + len(payload)] = payload
        self.expected[start : start + len(payload)] = payload


def collect_replay_images(
    result, encode, *, alignment=1, capacity=None, poison=0xA5
):
    """Replay flat native FX; never stage computed expected bytes as inputs.

    encode(tensor) -> bytes is target-owned (dtype, endianness, physical packing).
    All captured tensors must have finalized DRAM allocations. Reject nested or
    asynchronous programs rather than silently produce incomplete images.
    """
    result.program.assert_frozen()
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError("Replay alignment must be a positive integer")
    size = (
        (result.program.plan.allocation.dram_bytes + alignment - 1)
        // alignment
        * alignment
    )
    if capacity is not None and size > capacity:
        raise ValueError("Replay image exceeds endpoint capacity")
    program = result.program.graph
    for node in program.graph.nodes:
        if node.op == "call_module" or node.target in (
            torch.ops.voyager.async_copy.default,
            torch.ops.voyager.async_wait.default,
        ):
            raise ValueError(
                "Flat native replay does not support nested/asynchronous kernels"
            )
    images = ReplayImages.create(size, poison=poison)

    def store(node, value, is_input=False):
        segment = node.meta.get("memory")
        if segment is None or node.meta.get("space") != "DRAM":
            raise ValueError(
                f"{node.name}: replay requires a finalized DRAM allocation"
            )
        images.store(
            segment.start, segment.end, encode(value), is_input=is_input
        )

    class Replay(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node.op in ("placeholder", "get_attr") and isinstance(
                value, torch.Tensor
            ):
                store(node, value, is_input=True)
            if node.target == torch.ops.voyager.insert.default:
                source, destination = node.args
                store(destination, self.env[source])
            return value

    with torch.no_grad():
        flat_inputs, _ = tree_flatten((result.args, result.kwargs))
        Replay(program).run(*flat_inputs)
    return images
