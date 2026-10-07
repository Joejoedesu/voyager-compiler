"""Partition-aware placement using the shared alias/lifetime/best-fit planner."""

import math


def place_local_buffers(model, bufs, config, tuning):
    from voyager_compiler.codegen.transform.bufferize.memory_planning import (
        MemoryLevel,
        Segment,
        _greedy_best_fit,
        _slots,
        _val,
    )
    from voyager_compiler.codegen.transform.tiling.cost import get_dtype_width

    from .lowering import RECIPES, operation_name, reduction_workspace
    reserve = tuning.sbuf_reserve_bytes
    pool_rows = 0
    for module in model.modules():
        if not hasattr(module, "graph"):
            continue
        for node in module.graph.nodes:
            if node.op == "call_function":
                name = operation_name(node.target)
                if name == "max_pool2d":
                    from voyager_compiler.codegen.node_info import get_arg_value, _pair
                    pool_rows=max(pool_rows,_pair(get_arg_value(node,1,"kernel_size"))[0])
                # Named reduction scratch is already present in bufs. The
                # final instruction allocator replaces it with exact ISA
                # lifetimes; do not reserve the entire recipe again here.
    layouts = {}
    for root, buf in bufs.items():
        value = _val(root)
        slots = _slots(root) or 1
        shape = tuple(value.shape[1:] if _slots(root) else value.shape)
        partitions = min(128, shape[-1] if shape else 1)
        bits = get_dtype_width(root.meta.get("dtype") or value.dtype)
        per_partition = math.prod(shape[:-1]) * math.ceil((shape[-1] if shape else 1) / partitions) * bits / 8
        if pool_rows and len(shape) == 4 and shape[0] == shape[-1] == 1:
            per_partition=pool_rows*shape[2]*bits/8
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
        base = reserve + bases[root]
        segment = Segment(base, base + slots * pitch, MemoryLevel.SRAM, root)
        for member in buf.members:
            member.meta["scratchpad"] = segment
            if _slots(member):
                member.meta["bank_group_stride"] = pitch
    model.meta["trainium_placement"] = {
        "arena_bytes": total,
        "reserved_bytes": reserve,
        "partitions": 128,
        "physical_assignment": "NKI compiler; shared allocation is a capacity certificate",
    }
    return total + reserve, model.meta["trainium_placement"]
