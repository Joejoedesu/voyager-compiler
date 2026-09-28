"""
Bufferized FX lowering.

Rewrites tiled FX nodes into an explicit, executable *bufferized FX graph* that
contains tile loops (``torch.ops.higher_order.while_loop``) and explicit memory
primitives in the ``voyager`` torch.library namespace (``voyager.alloc``,
``voyager.async_copy``, ``voyager.insert``), then generates protobuf /
graphviz / text from that graph.

Modules: ``ops`` (voyager.* primitives), ``utils`` (shared builder helpers +
layout projections), ``pipeline`` (the unified Pallas-style software-pipelining
scheduler and its GEMM / conv2d / pointwise / pool builders),
``bufferization`` (the rewrite pass), ``codegen`` (loop-aware output).
"""

from dataclasses import asdict, dataclass

import torch
from torch.fx.node import has_side_effect

# Registers the voyager.* torch.library ops.
from voyager_compiler.codegen.transform.bufferize import ops  # noqa: F401
from voyager_compiler.codegen.transform.bufferize.bufferization import (
    annotate_tensor_spaces,
    bufferize_graph,
)
from voyager_compiler.codegen.transform.bufferize.emit import (
    flush_tensor_files,
    gen_code_bufferized,
    gen_compute_graph,
    print_bufferized_graph,
    print_layer_table,
)
from voyager_compiler.codegen.transform.bufferize.memory_planning import (
    MemoryPlan,
    plan_memory,
)

# Mark the in-place / DMA primitives side-effecting so DCE never drops them.
has_side_effect(torch.ops.higher_order.cond)
has_side_effect(torch.ops.voyager.insert.default)
has_side_effect(torch.ops.voyager.async_copy.default)
has_side_effect(torch.ops.voyager.async_wait.default)
has_side_effect(torch.ops.higher_order.commit)

__all__ = [
    "bufferize_graph",
    "BufferizationOptions",
    "annotate_tensor_spaces",
    "print_layer_table",
    "flush_tensor_files",
    "gen_code_bufferized",
    "gen_compute_graph",
    "print_bufferized_graph",
    "plan_memory",
    "MemoryPlan",
]


@dataclass(frozen=True)
class BufferizationOptions:
    """Algorithm choices, separate from hardware buffer capacities."""

    single_buffer_tail: bool = False
    flash_attention_v3: bool = True
    bool_mask: bool = True

    def kwargs(self):
        return asdict(self)
