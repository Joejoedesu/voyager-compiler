"""Place live buffer roots in clustered bank entries using shared lifetimes."""

import math

import numpy as np

from voyager_compiler.ir.plan import BankLayout


def place_bank_entries(buffers, memory):
    """Return selected entry coordinates and occupied bank ranges per root.

    Physical cells, rather than overlapping virtual byte offsets in different
    cluster modes, establish disjointness. Every live allocation is checked
    during placement; bank-wide conflict requirements are independent of size.
    """
    from voyager_compiler.compiler.allocate import MemoryCapacityError

    banks, word = memory.banks, memory.word_bytes
    if not banks or not word or memory.size.value is None:
        raise ValueError(
            f"{memory.name}: clustered placement requires bank geometry"
        )
    depth = memory.size.value // (banks * word)
    if not depth or depth * banks * word != memory.size.value:
        raise ValueError(
            f"{memory.name}: capacity is not an integral bank geometry"
        )
    occupied = np.zeros((banks, depth), dtype=np.bool_)
    active, selected, peak = [], {}, 0
    for root, buffer in sorted(
        buffers.items(), key=lambda item: (item[1].def_t, item[0].name)
    ):
        still_live = []
        for previous, end, ranges in active:
            if end < buffer.def_t:
                for bank, start, stop in ranges:
                    occupied[bank, start:stop] = False
            else:
                still_live.append((previous, end, ranges))
        active = still_live
        layout = root.meta.get("bank_layout")
        if (
            not isinstance(layout, BankLayout)
            or layout.cluster <= 0
            or banks % layout.cluster
        ):
            raise ValueError(
                f"{root.name}: invalid bank cluster for {memory.name}"
            )
        unavailable = occupied.copy()
        avoided = set(layout.reserved_banks)
        for previous, _, ranges in active:
            peer = previous.meta.get("bank_layout")
            if (
                previous.name in layout.avoid_buffers
                or root.name in peer.avoid_buffers
            ):
                avoided.update(bank for bank, _, _ in ranges)
        if any(bank < 0 or bank >= banks for bank in avoided):
            raise ValueError(
                f"{root.name}: reserved bank outside {memory.name}"
            )
        for bank in avoided:
            unavailable[bank, :] = True
        free = (
            ~unavailable.reshape(banks // layout.cluster, layout.cluster, depth)
            .any(axis=1)
            .reshape(-1)
        )
        count = math.ceil(buffer.size / (layout.cluster * word))
        transitions = np.diff(
            np.concatenate(([False], free, [False])).astype(np.int8)
        )
        starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(
            transitions == -1
        )
        candidates = starts[stops - starts >= count]
        if not len(candidates):
            raise MemoryCapacityError(
                f"{root.name}: {memory.name} has no {count}-entry cluster-{layout.cluster} placement",
                sites=(root.meta["scope"][0],) if "scope" in root.meta else (),
            )
        start = int(candidates[0])
        ranges = []
        position = start
        while position < start + count:
            group, entry = divmod(position, depth)
            run = min(depth - entry, start + count - position)
            for lane in range(layout.cluster):
                bank = group * layout.cluster + lane
                if occupied[bank, entry : entry + run].any():
                    raise AssertionError(
                        "bank allocator selected occupied cells"
                    )
                occupied[bank, entry : entry + run] = True
                ranges.append((bank, entry, entry + run))
            position += run
        selected[root] = (start, count, tuple(ranges))
        active.append((root, buffer.last_t, tuple(ranges)))
        peak = max(peak, int(occupied.sum()) * word)
    return selected, peak
