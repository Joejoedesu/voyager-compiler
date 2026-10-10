"""Static accounting for packed DMA descriptors with unequal partition loads."""

import ast
from collections import Counter


def partition_bytes(descriptor, store, total_bytes):
    """Decode the observed contiguous byte-block descriptor form exactly.

    A descriptor can repeat a partition, e.g. four 1 KiB blocks on three
    partitions. Dividing its total bytes by the number of distinct partitions
    is invalid. Unknown descriptor layouts remain unsupported.
    """
    side = "source" if store else "dest"
    shapes = descriptor["read_shape" if store else "write_shape"]
    offsets = ast.literal_eval(descriptor[side + "_offset"])
    steps = ast.literal_eval("[" + descriptor[side + "_steps"] + "]")
    if not (len(shapes) == len(offsets) == len(steps)):
        raise ValueError("Unsupported fragmented DMA descriptor cardinality")
    loads = Counter()
    for shape, address, stride in zip(shapes, offsets, steps):
        if store:
            if len(address) != 1 or len(stride) != 1:
                raise ValueError("Unsupported fragmented DMA source block")
            address, stride = address[0], stride[0]
        if len(shape) != 2 or stride != [1, 262144]:
            raise ValueError("Unsupported fragmented DMA byte strides")
        free, count = shape
        if free <= 0 or count <= 0:
            raise ValueError("Empty fragmented DMA block")
        for index in range(count):
            start = address + index * 262144
            part, offset = divmod(start, 262144)
            if not 0 <= part < 128 or offset + free > 262144:
                raise ValueError(
                    "Fragmented DMA block exceeds an SBUF partition"
                )
            loads[part] += free
    if (
        sum(loads.values()) != total_bytes
        or len(loads) != descriptor[side + "_num_sb_partitions"]
    ):
        raise ValueError(
            "Fragmented DMA descriptor does not match static byte accounting"
        )
    return dict(loads)


def payload_ns(loads, bandwidth):
    """Work on the busiest eight-partition group among sixteen DMA engines."""
    engines = Counter()
    for partition, nbytes in loads.items():
        engines[partition // 8] += nbytes
    return max(engines.values()) / (bandwidth / 16)
