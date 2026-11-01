"""The runtime model a backend plugs into the shared tile search.

The shared search helpers -- interstellar for GEMM / conv, the grid
enumeration for vector, GEMV and pool ops, the attention search -- generate
candidate tilings and pick the winner.  What depends on the hardware comes
from the backend's :class:`RuntimeModel`: for each op it prepares an
:class:`OpModel`, whose ``space`` says which candidates exist and whose
``fit`` and ``cost`` judge each one.  A GEMM / conv's tiling is
interstellar's own ``MappingPoint``, read in place; a grid op's is a
:class:`GridTiling` and a flash attention's an :class:`AttentionTiling`.  The
winner is stamped on the node as a :class:`TileResult`, which the builders,
the memory planner, emit and reporting read.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable


class Role(Enum):
    """A buffer of a tiled op, by the part it plays: an operand of a GEMM /
    conv anchor, the op's output, or the scratch its kernel keeps."""

    INPUT = "input"
    CSR = "csr"
    INPUT_SCALE = "input_scale"
    WEIGHT = "weight"
    WEIGHT_SCALE = "weight_scale"
    BIAS = "bias"
    OUTPUT = "output"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class Fused:
    """The ``index``-th operand a GEMM / conv's fused tail adds
    (``fused_operand_nodes``)."""

    index: int


@dataclass(frozen=True)
class KernelInput:
    """The ``index``-th of a grid op's ``all_input_nodes``."""

    index: int


@dataclass(frozen=True)
class Placement:
    """Where a tiling's buffers go on chip.

    Attributes:
        groups: The bank groups, each a frozenset of the :class:`Role`,
            :class:`Fused` and :class:`KernelInput` members sharing its banks;
            ``None`` when the model imposes no grouping.
        scratch_slots: The regions the reduction scratch is allocated with.
        tier: How much extra bank sharing the tiling needed to fit, 0 for
            none; the grid search keeps the lowest.
        scratch: The regions the kernel keeps beside the op's tiles that no
            FX node names (``OpModel.scratch``), which its builder allocates
            as placed and hands the op by name (``with_scratch``).

    A backend whose memory has levels the shared code does not know
    subclasses it to place buffers there too.
    """

    groups: tuple | None = None
    scratch_slots: int = 1
    tier: int = 0
    scratch: tuple = ()


@dataclass(frozen=True)
class Cost:
    """What a tiling costs, in the backend's cycles.

    Attributes:
        latency: The whole op.
        dram_bytes: Its DRAM traffic, which breaks near-ties in latency;
            ``None`` when the model does not count it.
        compute: One step of its grid, which attention's products and
            reporting charge; ``None`` when nothing reads it.
    """

    latency: float
    dram_bytes: float | None = None
    compute: float | None = None


@dataclass(frozen=True)
class MatrixSpace:
    """The GEMM / conv tilings interstellar searches; its candidates are
    interstellar ``MappingPoint``s.

    Attributes:
        architecture: The interstellar ``Resource``: the memory levels, their
            capacities and the PE array.
        schedule: The interstellar ``Schedule``: the loops and orders the
            target/recipe permits, not the default target's schedule.
        mapping_filter: Optional target predicate on a complete MappingPoint,
            including loop orders. Capacity/extent-only checks belong in fit;
            this predicate runs after ordering and before cost. Keep it pure
            and usable in forked workers. Changes must change the model key.
        fit_level: Internal memory level checked by fit and TileConstraint.
            Default 2 retains the shared four-level builder contract; native
            recipes may describe another hierarchy (for example three levels).
    """

    architecture: object
    schedule: object
    mapping_filter: Callable | None = None
    fit_level: int = 2


@dataclass(frozen=True)
class GridSpace:
    """The vector, GEMV and pool tilings the grid enumeration searches.

    Attributes:
        lanes: The multiple a tile's innermost dim takes.
        ranked: Price every tiling that fits; else the first that fits, in
            the enumeration's order, wins unpriced.
        relative_ties: Bytes break a near-tie in latency only share for
            share (``pick``).
    """

    lanes: int
    ranked: bool
    relative_ties: bool = False
    split_inner: bool = True


@dataclass(frozen=True)
class AttentionSpace:
    """The flash-attention tilings the attention search enumerates.

    Attributes:
        unit: The multiple a query / key tile takes.
        head_multiple: The kernel holds the head dim padded to a multiple of
            this (:func:`pad_head`).
        relative_ties: Bytes break a near-tie in latency only share for
            share (``pick``).
    """

    unit: int
    head_multiple: int
    relative_ties: bool = False


@dataclass(frozen=True)
class GridTiling:
    """A vector, GEMV or pool op cut into a grid of tiles, per dim it tiles.

    Attributes:
        tile_sizes: The elements one tile spans.
        grid: The tiles along the dim.
    """

    tile_sizes: tuple
    grid: tuple


@dataclass(frozen=True)
class AttentionTiling:
    """A flash attention cut into query and key tiles.

    Attributes:
        tile_sizes: ``(tq, tkv)``, the query and key rows of one tile.
        grid: The key / value batch dims, then the query and key block
            counts.
        products: Each product's :class:`TileResult`, ``"scores"`` and
            ``"context"``; ``None`` while the search checks the fit, before
            it maps them.
    """

    tile_sizes: tuple
    grid: tuple
    products: dict | None = None


@dataclass(frozen=True)
class TileResult:
    """A search's winner, stamped as ``meta["tile_result"]``.

    Attributes:
        tiling: The winning tiling: interstellar's ``MappingPoint`` for a
            GEMM / conv, else a :class:`GridTiling` or
            :class:`AttentionTiling`.
        placement: Its :class:`Placement`.
        cost: Its :class:`Cost`; ``None`` from an unranked grid search.
        access_list: Interstellar's per-level ``(input, output, weight)``
            access counts, which emit reports; ``None`` off the matrix
            search. These are logical Interstellar accesses, not a native DMA
            trace or measured traffic; physical estimates belong in Cost.
    """

    tiling: object
    placement: Placement
    cost: Cost | None
    access_list: tuple | None = None


def pad_head(head_dim, space):
    """``head_dim`` rounded up to the multiple an :class:`AttentionSpace`
    holds the head at."""
    return math.ceil(head_dim / space.head_multiple) * space.head_multiple


@dataclass(frozen=True)
class BufferRequirement:
    """A candidate's resource demand before concrete buffer placement."""

    name: str
    space: str
    bytes: int
    slots: int = 1
    alignment: int = 1


@dataclass(frozen=True)
class ImplementationChoice:
    """An implementation identity and its modeled, refined schedule."""

    implementation: str
    parameters: tuple = ()
    tile: TileResult | None = None
    cost: Cost | None = None
    requirements: tuple[BufferRequirement, ...] = ()
    # Ordered component choices for a region lowered as sequential kernels.
    # The parent's cost covers the region; components are attribution, not
    # additional work to charge again.
    components: tuple[tuple[str, "ImplementationChoice"], ...] = ()


@dataclass(frozen=True)
class BankLayout:
    """Consecutive entries striped over fixed clusters of physical banks.

    The allocator owns placement; the target owns the meaning of one cluster
    and bank conflicts. Offsets are entries within a bank, not flat bytes.
    """

    cluster: int = 1
    reserved_banks: tuple[int, ...] = ()
    avoid_buffers: tuple[str, ...] = ()


@dataclass(frozen=True)
class BufferComponent:
    """A logical buffer encoded in a component of another physical backing.

    ``backing`` is a qualified BufferProgram allocation name. The target owns
    the packing format; the common allocator combines the component lifetimes
    and reserves the backing once. This is not a byte-offset tensor view.
    """

    backing: str
    layout: str
    component: str

    def __post_init__(self):
        if not self.backing or not self.layout or not self.component:
            raise ValueError(
                "buffer components require backing, layout and component identities"
            )


@dataclass(frozen=True)
class BufferAllocation:
    """One actual buffer root; native ownership intentionally has no address."""

    buffer: str
    space: str
    start: int | None
    size: int
    slots: int = 1
    slot_stride: int | None = None
    owner: str = "compiler"
    memory_instance: str | None = None
    bank_ranges: tuple = ()
    layout: BankLayout | None = None
    component: BufferComponent | None = None
    physical_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class AllocationPlan:
    dram_bytes: int
    scratchpad_bytes: int
    buffers: tuple[BufferAllocation, ...] = ()
    arenas: tuple[tuple[str, int], ...] = ()


@dataclass
class CompilationPlan:
    """Mutable during refinement, frozen only after actual allocation succeeds."""

    target: str
    choices: dict[str, ImplementationChoice] = field(default_factory=dict)
    alternatives: dict[str, tuple[ImplementationChoice, ...]] = field(
        default_factory=dict
    )
    rejected: dict[str, tuple[str, ...]] = field(default_factory=dict)
    allocation: AllocationPlan | None = None
    frozen: bool = False

    def __setattr__(self, name, value):
        if getattr(self, "frozen", False):
            raise RuntimeError("cannot change a frozen compilation plan")
        object.__setattr__(self, name, value)

    def select(self, site, choice, alternatives=(), rejected=()):
        if self.frozen:
            raise RuntimeError("cannot change a frozen compilation plan")
        self.choices[site] = choice
        self.alternatives[site] = tuple(alternatives)
        self.rejected[site] = tuple(rejected)

    def freeze(self, allocation):
        if self.frozen:
            raise RuntimeError("compilation plan is already frozen")
        self.allocation = allocation
        self.choices = MappingProxyType(dict(self.choices))
        self.alternatives = MappingProxyType(dict(self.alternatives))
        self.rejected = MappingProxyType(dict(self.rejected))
        self.frozen = True


@dataclass(frozen=True)
class ExecutionOptions:
    """Compiler policy, separate from the immutable physical hardware graph."""

    buffer_slots: int = 2
    async_pipeline: bool = True

    def __post_init__(self):
        if self.buffer_slots not in (1, 2):
            raise ValueError(
                "shared templates currently support one or two buffer slots"
            )
