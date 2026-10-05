"""Loop extents and transfer recurrence of the shared matrix builders."""

from dataclasses import dataclass
from interstellar import loop_enum as le


@dataclass
class MappingTraversal:
    batch: int = 1
    weight_batch: int = 1

    @staticmethod
    def _extent(mapping, loop, level):
        """Elements along ``loop`` in one tile at ``level``: the blockings
        through that level times the PE-array partition."""
        extent = mapping.loop_partitionings[loop][0]
        for blocking in mapping.loop_blockings[loop][1 : level + 1]:
            extent *= blocking
        return extent

    @staticmethod
    def _l3_blocks(mapping):
        """Total L3 (DRAM) grid steps, the IC reduction included: with IC
        innermost at L3 the grid is ``(output tiles) x num_k``, one input and
        weight load each.  Stores are ``num_k`` times fewer.
        """
        blockings = mapping.loop_blockings
        l3_blocks = 1
        for i in range(le.NUM):
            l3_blocks *= blockings[i][3]
        return l3_blocks

    @staticmethod
    def _l3_loads(mapping, dims):
        """How many times an operand spanning ``dims`` is fetched over the
        sweep.

        Order the nest outermost to innermost and let ``p`` be the position of
        the innermost loop the operand spans.  Every loop inside ``p`` re-reads
        the tile that is already there, so the operand is fetched once per
        iteration of the loops at or outside ``p``.  Ranks come off the mapping
        (``loop_orders[d][3]``, 0 = innermost), the same order the builders
        emit, so the two cannot disagree.

        A loop that is empty at L3 carries the sentinel rank and a blocking of
        1, so it can only ever multiply in as 1 -- including the case where the
        operand spans nothing tiled, which correctly gives a single fetch.
        """
        orders, blockings = mapping.loop_orders, mapping.loop_blockings
        innermost = min(orders[d][3] for d in dims)
        steps = 1
        for d in range(le.NUM):
            if orders[d][3] >= innermost:
                steps *= blockings[d][3]
        return steps

    def _batch_loads(self, mapping, dims, distinct):
        """Fetches of an operand spanning ``dims`` over the whole sweep, the
        batch loop the builder wraps the mapping in included.

        Diced inside a batch step, the operand re-reads in full on every one:
        the next step restarts the sequence, so its first block differs from
        the last one loaded and the guard never fires.  Held whole, the tile
        survives into the next step and only a change of block costs --
        ``distinct`` of them over the batch, which is fewer than ``batch``
        for an operand a group shares and 1 for one they all share.
        """
        per_step = self._l3_loads(mapping, dims) if dims else 1
        return per_step * (self.batch if per_step > 1 else distinct)
