"""Pinned LoopWS transfers, distinct from padded SRAM reservations.

Search uses aligned base estimates; pass finalized operand addresses to audit
actual packet traffic. This describes the pinned LoadB tail behavior, not a
portable promise about other Gemmini revisions. Maximum read size is 64 bytes;
native LoopWS load commands group at most 64 INT8 columns.
"""

from functools import lru_cache
from math import ceil, gcd


@lru_cache(maxsize=8192)
def packet_bytes(offset, count, bus=16, maximum=64):
    if bus <= 0 or maximum < bus or maximum % bus or bus & (bus - 1):
        raise ValueError("Packet widths must be ordered powers of two")
    if count < 0:
        raise ValueError("Negative transfer size")
    sizes = tuple(bus << i for i in range((maximum // bus).bit_length()))
    result = 0
    while count:
        size = max(sizes, key=lambda s: min(s - offset % s, count))
        used = min(size - offset % size, count)
        result += size
        count -= used
        offset += used
    return result


@lru_cache(maxsize=32768)
def row_bytes(address, rows, stride, columns, chunk=64, bus=16):
    # Packet phases repeat after at most 64 rows. Avoid expanding large
    # activations for every candidate evaluated by the search engine.
    period = 64 // gcd(stride, 64)
    phases = [
        sum(
            packet_bytes(
                (address + r * stride + c) % 64, min(chunk, columns - c), bus
            )
            for c in range(0, columns, chunk)
        )
        for r in range(min(rows, period))
    ]
    return sum(phases) * (rows // period) + sum(phases[: rows % period])


def segments(length, extent, stride=1):
    """Multiplicity of full/tail DMA windows, grouped by 64-byte phase."""
    full, tail = divmod(length, extent)
    period = 64 // gcd(extent * stride, 64)
    result = []
    for index in range(min(full, period)):
        result.append(
            (
                (index * extent * stride) % 64,
                extent,
                (full - 1 - index) // period + 1,
            )
        )
    if tail:
        result.append(((full * extent * stride) % 64, tail, 1))
    return result


@lru_cache(maxsize=8192)
def matrix_records(m, n, k, tile, dim):
    """LoopWS A/B/bias/output bytes and work per launch, with multiplicity.

    Operand bases are assumed 64-byte aligned during search. Final addresses
    may change packet traffic. DIM-padded B rows match the pinned LoadB tail
    behavior; output bytes are logical writes, not padded SRAM reservations.
    """
    tm, tn, tk = tile
    records = []
    for rp, mr, rc in segments(m, tm, k):
        for cp, nr, cc in segments(n, tn):
            bias = ceil(mr / dim) * row_bytes(
                cp * 4, 1, 0, nr * 4, chunk=dim * 4
            )
            for kp, kr, kc in segments(k, tk):
                a = row_bytes(rp + kp, mr, k, kr)
                b = row_bytes(kp * n + cp, ceil(kr / dim) * dim, n, nr)
                records.append((mr, nr, kr, a, b, rc * cc * kc))
            # Bias and output occur once per output tile, across all K blocks.
            records.append((mr, nr, 0, bias, mr * nr, rc * cc))
    return tuple(records)
