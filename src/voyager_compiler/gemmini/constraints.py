"""Legality/placement extensions to Voyager's existing search and allocator."""

import math


def accumulator_slot_bytes(config, payload, *, separate_banks=True):
    """Separate buffered matrix outputs at the accumulator read-port scope.

    Accumulating a result and writing back the preceding result both read the
    accumulator. A single read port per bank requires disjoint bank groups to
    realize the search's overlap assumption. This is a placement policy, not a
    claim that crossing a bank is an illegal hardware instruction.
    """
    memory = config.memory_instance("accumulator")
    read = memory.port("read")
    quantum = (
        memory.size.value // memory.banks
        if separate_banks and read.scope == "bank" and read.count == 1
        else memory.row_bytes
    )
    return math.ceil(payload / quantum) * quantum


def place_local_buffers(model, bufs, config):
    import torch

    sp_memory = config.memory_instance("scratchpad")
    acc_memory = config.memory_instance("accumulator")
    SP_BANKS = sp_memory.banks
    SP_BANK_BYTES = sp_memory.size.value // SP_BANKS
    RESERVED_BYTES = config.scratchpad_offset

    def bank_span(payload, reserved=0):
        return math.ceil((payload + reserved) / SP_BANK_BYTES) * SP_BANK_BYTES

    from voyager_compiler.codegen.node_info import get_arg_value
    from voyager_compiler.codegen.transform.bufferize.memory_planning import (
        MemoryLevel,
        MemoryPlanningError,
        Segment,
        _buffer_identity,
        _greedy_best_fit,
        _slot_payload,
        _slots,
        _val,
        _walk,
    )
    from voyager_compiler.codegen.transform.tiling.cost import get_dtype_width

    # Reuse the shared alias and lifetime model. DMA ingress identifies real
    # scratchpad storage; produced values are accumulator-backed in Lean.
    aliases = _buffer_identity(model)
    ingress = set()
    for node in _walk(model):
        if node.target is torch.ops.voyager.async_copy.default:
            dst = get_arg_value(node, 1, "dst")
            ingress.add(aliases.get(dst, dst))
    sp, acc = {}, {}
    for root, buf in bufs.items():
        dtype = root.meta.get("dtype") or _val(root).dtype
        (sp if root in ingress and get_dtype_width(dtype) <= 8 else acc)[
            root
        ] = buf

    layouts = {}
    for root, buf in sp.items():
        _, slots = _slot_payload(buf, config)
        raw = _val(root).numel() // slots
        scope = root.meta.get("scope", (None, None))
        schema = getattr(scope[1], "_schema", None)
        matrix = schema is not None and schema.name.split("::")[-1] in (
            "matmul",
            "linear",
            "conv2d",
        )
        layouts[root] = (raw, slots, 0 if matrix else RESERVED_BYTES)
    bases, total = _greedy_best_fit(
        [
            (
                r,
                slots * bank_span(payload, guard),
                b.def_t,
                b.last_t,
                SP_BANK_BYTES,
            )
            for r, b in sp.items()
            for payload, slots, guard in [layouts[r]]
        ]
    )
    if total > SP_BANKS * SP_BANK_BYTES:
        raise MemoryPlanningError(
            f"Gemmini bank placement needs {total // SP_BANK_BYTES} scratchpad banks; only {SP_BANKS} exist"
        )
    for root, buf in sp.items():
        payload, slots, guard = layouts[root]
        pitch = bank_span(payload, guard)
        base = bases[root] + guard
        segment = Segment(
            base, base + (slots - 1) * pitch + payload, MemoryLevel.SRAM, root
        )
        for member in buf.members:
            member.meta["scratchpad"] = segment
            if _slots(member):
                member.meta["bank_group_stride"] = pitch

    acc_bases, acc_total = _greedy_best_fit(
        [
            (r, b.size, b.def_t, b.last_t, acc_memory.row_bytes)
            for r, b in acc.items()
        ]
    )
    if acc_total > acc_memory.size.value:
        raise MemoryPlanningError(
            f"Gemmini logical accumulator arena needs {acc_total} > {acc_memory.size.value} bytes"
        )
    for root, buf in acc.items():
        base = sp_memory.size.value + acc_bases[root]
        segment = Segment(base, base + buf.size, MemoryLevel.SRAM, root)
        for member in buf.members:
            member.meta["scratchpad"] = segment
    model.meta["gemmini_local_bytes"] = dict(
        scratchpad=total, accumulator=acc_total, banked_input_buffers=len(sp)
    )
    return (
        max(
            RESERVED_BYTES,
            total,
            sp_memory.size.value + acc_total if acc_total else 0,
        ),
        model.meta["gemmini_local_bytes"],
    )


def matrix_storage(
    config, input_bytes, weight_bytes, output_elements, plan, tuning
):
    """Physical footprints used by both early fit pruning and candidate scoring.

    Accumulator slot sizing is also used by the converter's placement. No
    fictitious scratchpad capacity is used to represent wide accumulators.
    """
    from voyager_compiler.codegen.transform.tiling.contracts import (
        StorageRequirement,
    )

    sp = config.memory_instance("scratchpad")
    bank = sp.size.value // sp.banks
    acc = accumulator_slot_bytes(
        config,
        output_elements * config.accumulator_element_bytes,
        separate_banks=tuning.separate_accumulator_banks,
    )
    return (
        StorageRequirement("scratchpad", input_bytes, plan.input_slots, bank),
        StorageRequirement("scratchpad", weight_bytes, plan.weight_slots, bank),
        StorageRequirement("accumulator", acc, plan.accumulator_copies),
    )
