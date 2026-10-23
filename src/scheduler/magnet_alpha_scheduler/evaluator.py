# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math

from scheduler.types import NUM_LOOPS, LevelAccessCounts, Loop, MappingPoint


class MagnetAlphaRuntimeCalculator:
    """Estimate Magnet Alpha runtime in vector-word and VMAC issue cycles."""

    def __init__(
        self,
        input_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        has_high_precision_vector_input: bool = False,
        has_sparse_op: bool = False,
        *,
        vector_size: int,
    ):
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        self.input_dtype_width = input_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.has_high_precision_vector_input = has_high_precision_vector_input
        self.has_sparse_op = has_sparse_op
        self.vector_size = vector_size

    def calculate_runtime(self, layer, mapping: MappingPoint) -> float:
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        if partitionings[Loop.IC][0] != self.vector_size:
            return float("inf")

        l1 = [blockings[loop][1] for loop in range(NUM_LOOPS)]
        input_width = (l1[Loop.OX] - 1) * layer.wstd + l1[Loop.FX]
        input_height = (l1[Loop.OY] - 1) * layer.hstd + l1[Loop.FY]

        input_loading_time = (
            input_width * input_height * l1[Loop.IC] * l1[Loop.ON]
        )
        weight_loading_time = (
            l1[Loop.IC] * l1[Loop.OC] * l1[Loop.FX] * l1[Loop.FY]
        )
        if self.has_sparse_op:
            weight_loading_time *= 2

        vmac_time = math.prod(l1)
        collector_drain_time = (
            l1[Loop.OC] * l1[Loop.OX] * l1[Loop.OY] * l1[Loop.ON]
        )
        if (
            self.output_dtype_width > self.input_dtype_width
            or self.has_high_precision_vector_input
        ):
            collector_drain_time *= 2

        l2_tiles = math.prod(blockings[loop][2] for loop in range(NUM_LOOPS))
        loading_time = max(input_loading_time, weight_loading_time)

        if self.double_buffered_accum_buffer:
            steady_state_time = max(
                loading_time,
                vmac_time,
                collector_drain_time,
            )
            return (
                loading_time
                + vmac_time
                + (l2_tiles - 1) * steady_state_time
                + collector_drain_time
            )

        compute_and_drain_time = vmac_time + collector_drain_time
        return (
            loading_time
            + compute_and_drain_time
            + (l2_tiles - 1) * max(loading_time, compute_and_drain_time)
        )

    def calculate_access_counts(
        self,
        layer,
        mapping: MappingPoint,
    ) -> tuple[LevelAccessCounts, ...]:
        """Estimate scalar accesses at PE, local, and global levels."""
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings
        macs = math.prod(layer.sizes)

        local_input_width = (
            blockings[Loop.OX][1] - 1
        ) * layer.wstd + blockings[Loop.FX][1]
        local_input_height = (
            blockings[Loop.OY][1] - 1
        ) * layer.hstd + blockings[Loop.FY][1]
        local_ic = blockings[Loop.IC][1] * partitionings[Loop.IC][0]
        local_oc = blockings[Loop.OC][1] * partitionings[Loop.OC][0]
        local_batch = blockings[Loop.ON][1]
        outer_tiles = math.prod(blockings[loop][2] for loop in range(NUM_LOOPS))

        local = LevelAccessCounts(
            input=(
                local_input_width
                * local_input_height
                * local_ic
                * local_batch
                * outer_tiles
            ),
            output=(
                blockings[Loop.OX][1]
                * blockings[Loop.OY][1]
                * local_oc
                * local_batch
                * outer_tiles
            ),
            weight=(
                local_ic
                * local_oc
                * blockings[Loop.FX][1]
                * blockings[Loop.FY][1]
                * outer_tiles
            ),
        )
        global_accesses = LevelAccessCounts(
            input=layer.wifm * layer.hifm * layer.nifm * layer.nimg,
            output=layer.wofm * layer.hofm * layer.nofm * layer.nimg,
            weight=layer.wfil * layer.hfil * layer.nifm * layer.nofm,
        )
        return (
            LevelAccessCounts(macs, macs, macs),
            local,
            global_accesses,
        )


__all__ = [
    "MagnetAlphaRuntimeCalculator",
]
