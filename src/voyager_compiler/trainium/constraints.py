"""Partition-aware placement using the shared alias/lifetime/best-fit planner."""

import math


def place_local_buffers(model, bufs, config):
    from voyager_compiler.codegen.transform.bufferize.memory_planning import (
        MemoryLevel,
        Segment,
        _greedy_best_fit,
        _slots,
        _val,
    )
    from voyager_compiler.codegen.transform.tiling.cost import get_dtype_width

    layouts = {}
    for root, buf in bufs.items():
        value = _val(root)
        slots = _slots(root) or 1
        shape = tuple(value.shape[1:] if _slots(root) else value.shape)
        partitions = min(128, shape[-1] if shape else 1)
        bits = get_dtype_width(root.meta.get("dtype") or value.dtype)
        per_partition = math.ceil(math.prod(shape) / partitions) * bits / 8
        # A partition's byte offset must be 16-byte aligned. Conservatively
        # reserve all 128 partitions even for a narrower logical tile.
        pitch = math.ceil(per_partition / 16) * 16 * 128
        layouts[root] = (slots, pitch)
    bases, total = _greedy_best_fit(
        [
            (root, slots * pitch, buf.def_t, buf.last_t, 16 * 128)
            for root, buf in bufs.items()
            for slots, pitch in [layouts[root]]
        ]
    )
    for root, buf in bufs.items():
        slots, pitch = layouts[root]
        base = config.scratchpad_offset + bases[root]
        segment = Segment(base, base + slots * pitch, MemoryLevel.SRAM, root)
        for member in buf.members:
            member.meta["scratchpad"] = segment
            if _slots(member):
                member.meta["bank_group_stride"] = pitch
    model.meta["trainium_placement"] = {
        "arena_bytes": total,
        "reserved_bytes": config.scratchpad_offset,
        "partitions": 128,
        "physical_assignment": "NKI compiler; shared allocation is a capacity certificate",
    }
    return total + config.scratchpad_offset, model.meta["trainium_placement"]
