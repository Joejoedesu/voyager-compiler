"""Kernel buffering choices, not physical addresses or a second allocator.

The same immutable plan is evaluated by search and consumed by the builders.
An unspecified operand depth preserves the builder default, including resident
views. Lifetimes and physical placement remain owned by memory_planning.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelBufferPlan:
    input_slots: int | None = None
    weight_slots: int | None = None
    output_slots: int | None = None
    scratch_slots: int = 1
    bank_groups: object = None
    # Physical wide-output generations required by the target lowering.
    accumulator_copies: int = 0
    batch_tiles: int = 1

    def __post_init__(self):
        for count in (
            self.input_slots,
            self.weight_slots,
            self.output_slots,
            self.scratch_slots,
        ):
            if count is not None and count < 1:
                raise ValueError("Buffer depths must be positive")
        if not isinstance(self.batch_tiles, int) or self.batch_tiles < 1:
            raise ValueError("Batch tile count must be a positive integer")
        if (
            not isinstance(self.accumulator_copies, int)
            or self.accumulator_copies < 0
        ):
            raise ValueError("Accumulator generation count must be nonnegative")

    def apply_outputs(self, outputs):
        if self.output_slots is not None:
            for output in outputs:
                output.num_slots = self.output_slots

    def apply_inputs(self, activation, weight):
        activation.num_slots = self.input_slots
        weight.num_slots = self.weight_slots


def buffer_plan(anchor):
    meta = anchor.meta.get("tiling", {})
    return meta.get(
        "buffer_plan",
        KernelBufferPlan(scratch_slots=meta.get("scratch_slots", 1)),
    )
