"""Shared candidate engines and search state.

``interstellar`` searches hierarchical GEMM/conv mappings, ``tile_shapes``
searches dimension-wise tiles, and ``attention_tiles`` searches coupled
query/key blocks. ``csr_slice_planning`` couples outlier-CSR slice widths.
Models implement the contracts in ``extensions`` and live under ``targets``;
selected records live in ``ir.plan``. ``planner`` compares registered recipes.
Builders call these engines before emitting the selected bufferized program.
"""

from voyager_compiler.compiler.search.attention_tiles import (
    attention_op_tiling,
)
from voyager_compiler.compiler.search.csr_slice_planning import (
    DEFAULT_MIN_CSR_SLICE,
    plan_csr_slices,
)
from voyager_compiler.compiler.search.interstellar import (
    CONV_L3_ORDER,
    GEMM_L3_ORDER,
    get_tiling,
    prefetch_tilings,
)
from voyager_compiler.compiler.search.tile_shapes import (
    gemv_op_tiling,
    pool_op_tiling,
    vector_op_tiling,
)
from voyager_compiler.compiler.search.tiler import (
    DEFAULT_RUNTIME_TOLERANCE,
    TileConstraint,
    Tiler,
    build_tiler,
)
from voyager_compiler.compiler.search.traversal import (
    attention_crosses,
    attention_kv_last,
    attention_mask_dyn,
)
from voyager_compiler.extensions import OpModel, RuntimeModel
from voyager_compiler.ir.plan import (
    AttentionSpace,
    AttentionTiling,
    Cost,
    Fused,
    GridSpace,
    GridTiling,
    KernelInput,
    MatrixSpace,
    Placement,
    Role,
    TileResult,
    pad_head,
)

__all__ = [
    "AttentionSpace",
    "AttentionTiling",
    "CONV_L3_ORDER",
    "Cost",
    "DEFAULT_MIN_CSR_SLICE",
    "DEFAULT_RUNTIME_TOLERANCE",
    "Fused",
    "GEMM_L3_ORDER",
    "GridSpace",
    "GridTiling",
    "KernelInput",
    "MatrixSpace",
    "OpModel",
    "Placement",
    "Role",
    "RuntimeModel",
    "TileConstraint",
    "TileResult",
    "Tiler",
    "attention_crosses",
    "attention_kv_last",
    "attention_mask_dyn",
    "attention_op_tiling",
    "build_tiler",
    "gemv_op_tiling",
    "get_tiling",
    "pad_head",
    "plan_csr_slices",
    "pool_op_tiling",
    "prefetch_tilings",
    "vector_op_tiling",
]
