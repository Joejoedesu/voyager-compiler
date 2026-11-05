"""Enumerate physical BF16 decode-cache capacities using existing attention search.

This is input/state representation selection, before model export. It neither
changes the logical context nor forces a tile into the compiled program. The
ordinary attention search still owns the final tile and product mappings.
"""
import math

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from voyager_compiler.compiler.search.attention_tiles import attention_op_tiling
from voyager_compiler.compiler.search.tiler import Tiler
from voyager_compiler.extensions import NoLegalMapping
from voyager_compiler.shape_prop import set_node_value
from .model import TrainiumModel


def capacity_domain(required, alignment, config):
    """All aligned capacities up to the last undominated tile's first multiple.

    A score tile has two parities, each using ceil(width/bank_width) PSUM
    bank-widths per query block. Even with one query block and no other buffers,
    width cannot exceed floor(banks/2)*bank_width. Actual fitting is stricter.
    For each allowed width, its first capacity >= required is a multiple of
    lcm(width, alignment). Extra multiples repeat the same tile with more steps,
    traffic and storage; they cannot lower this single-token attention cost.
    We enumerate every aligned capacity through the largest such first multiple,
    including dominated interior points, rather than sampling preferred shapes.
    """
    if required <= 0 or alignment <= 0 or alignment % config.partitions:
        raise ValueError('positive capacity and partition-aligned cache alignment required')
    first = math.ceil(required / alignment) * alignment
    maximum_tile = (config.psum_banks // 2) * config.moving_free_max
    widths = range(config.partitions, maximum_tile + 1, config.partitions)
    last = max(math.ceil(required / math.lcm(w, alignment)) * math.lcm(w, alignment)
               for w in widths)
    return range(first, last + 1, alignment), maximum_tile


def decode_attention_node(capacity, heads, kv_heads, head_dim):
    """Shape-only normalized single-token GQA with an explicit additive mask."""
    graph = torch.fx.Graph()
    with FakeTensorMode():
        def tensor(name, shape, dtype=torch.bfloat16):
            n = graph.placeholder(name)
            set_node_value(n, torch.empty(shape, dtype=dtype))
            return n
        q = tensor('query', (1, heads, head_dim, 1))
        k = tensor('key', (1, kv_heads, head_dim, capacity))
        v = tensor('value', (1, kv_heads, capacity, head_dim))
        mask = tensor('mask', (1, 1, 1, capacity), torch.float32)
        n = graph.call_function(torch.ops.quantized_ops.sdpa_kmajor.default,
                                (q, k, v, mask), {'enable_gqa': True})
        set_node_value(n, torch.empty((1, heads, 1, head_dim), dtype=torch.bfloat16))
    return n


class _RecordedAttention:
    """Observe the real fitter/scorer without changing enumeration or selection."""
    def __init__(self, op):
        self.op, self.rows = op, {}

    def __getattr__(self, name):
        return getattr(self.op, name)

    def fit(self, tiling):
        placement = self.op.fit(tiling)
        self.rows[tiling.tile_sizes] = dict(tile=list(tiling.tile_sizes), grid=list(tiling.grid),
            status='product_mapping_rejected' if placement is not None else 'resource_fit_rejected')
        return placement

    def cost(self, tiling, placement):
        cost = self.op.cost(tiling, placement)
        self.rows[tiling.tile_sizes].update(status='scored', cycles=cost.latency,
            attention_ms=cost.latency / self.op.model.config.frequency / 1e6,
            dram_bytes=cost.dram_bytes)
        return cost


def search_decode_capacity(config, *, required, alignment, heads, kv_heads,
                           head_dim, layers, max_kv_bytes=None):
    """Strict latency selection, then traffic/storage ties; report every trial.

    The objective is total attention latency over homogeneous layers. Other
    full-model operators and host setup are not included. Persistent KV bytes
    are a constraint and tie-break, not an invented latency penalty. No model
    weights or actual KV tensors are allocated during this search.
    """
    if min(heads, kv_heads, head_dim, layers) <= 0 or heads % kv_heads:
        raise ValueError('positive homogeneous GQA dimensions required')
    capacities, tile_bound = capacity_domain(required, alignment, config)
    if max_kv_bytes is not None and max_kv_bytes <= 0:
        raise ValueError('max_kv_bytes must be positive')
    model = TrainiumModel(config, False)
    tiler = Tiler(model, runtime_tolerance=0.)
    report = dict(policy='search', objective='attention_latency_only', runtime_tolerance=0.,
        required=required, alignment=alignment, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, layers=layers, dtype='bfloat16', batch=1,
        capacity_min=capacities.start, capacity_max=capacities.stop-alignment,
        score_only_tile_upper_bound=tile_bound, max_kv_bytes=max_kv_bytes,
        bound='first aligned capacity for every PSUM-bounded tile; extra full tiles are dominated',
        candidates=[])
    viable = []
    for capacity in capacities:
        kv_bytes = 2 * layers * kv_heads * capacity * head_dim * 2
        row = dict(capacity=capacity, persistent_kv_bytes=kv_bytes)
        report['candidates'].append(row)
        if max_kv_bytes is not None and kv_bytes > max_kv_bytes:
            row.update(status='kv_memory_budget_rejected', tiles=[])
            continue
        node = decode_attention_node(capacity, heads, kv_heads, head_dim)
        op = _RecordedAttention(model.prepare_attention(node, torch.float32, False))
        try:
            attention_op_tiling(node, op, tiler, sq_eff=heads//kv_heads, kv_batch=(1, kv_heads))
        except NoLegalMapping as error:
            row.update(status='no_legal_attention_mapping', reason=str(error), tiles=list(op.rows.values()))
            continue
        selected = node.meta['tile_result']
        row.update(status='scored', tiles=list(op.rows.values()),
            selected_tile=list(selected.tiling.tile_sizes), selected_grid=list(selected.tiling.grid),
            attention_cycles_per_layer=selected.cost.latency,
            attention_ms=layers * selected.cost.latency / config.frequency / 1e6,
            attention_dram_bytes=layers * selected.cost.dram_bytes)
        viable.append(row)
    if not viable:
        raise NoLegalMapping('No decode capacity fits the attention and KV-memory budgets')
    winner = min(viable, key=lambda r: (r['attention_ms'], r['attention_dram_bytes'], r['persistent_kv_bytes']))
    report.update(selected_capacity=winner['capacity'], selected_tile=winner['selected_tile'],
                  selected_attention_ms=winner['attention_ms'],
                  selected_persistent_kv_bytes=winner['persistent_kv_bytes'])
    return report
