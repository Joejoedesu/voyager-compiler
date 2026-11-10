"""Native convolution service: row-local execution and finite in-flight work.

Five matrix IDs come from the pinned MeshWithDelays configuration. The common
controller slack is calibrated once on Lean16 row-reuse probes, rounded to an
integer cycle; no Lean32 latency or per-layer timing table is used. This is an
approximate service floor, not a cycle-accurate controller simulation.
"""

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class NativeService:
    matrix_ids: int = 5
    controller_slack: int = 8

    def row_interval(self, dim, rows):
        return max(rows, 4, (2 * dim + self.controller_slack) / self.matrix_ids)

    def convolution(self, dim, kernel, tile, launches):
        y, x, n, k = tile
        fx = fy = kernel
        batch = 1
        # LoopConv traverses each spatial row separately, and reuses weights
        # over its batches/rows/column blocks. It does not flatten x*y into M.
        intervals = [
            self.row_interval(dim, min(dim, x - col))
            for col in range(0, x, dim)
        ]
        row_service = batch * y * sum(intervals)
        # Fresh-weight preloads take a DIM-wide phase; subsequent row calls
        # use the retained weights. Initial/final pipeline effects remain
        # approximate and the existing model still prices DMA and startup.
        per_weight = dim + row_service - intervals[0]
        groups = ceil(n / dim) * ceil(k / dim) * fx * fy
        return launches * groups * per_weight
