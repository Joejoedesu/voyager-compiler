"""Endpoint-constrained data-movement recipes and bounded whole-plan search.

This module contains legal transformations, not measured nanosecond constants.
A request fixes values/layout/dtype at both ends. Intermediate placement, ISA
chains, tile geometry and traversal are implementation choices. The hardware
profile supplies timing; the selected dependency graph supplies contention.
"""

from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import math
import logging
import time


@dataclass(frozen=True)
class Endpoint:
    memory: str
    dtype: str = "float32"
    axes: tuple = (0, 1)

    def __post_init__(self):
        if self.memory not in ("HBM", "SBUF", "PSUM") or self.axes not in (
            (0, 1),
            (1, 0),
        ):
            raise ValueError("Unsupported movement endpoint")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Unsupported movement dtype")


@dataclass(frozen=True)
class Request:
    source: Endpoint
    destination: Endpoint
    shape: tuple
    role: str = "movement"
    materialize: bool = True

    def __post_init__(self):
        if len(self.shape) != 2 or any(
            type(n) is not int or n <= 0 for n in self.shape
        ):
            raise ValueError(
                "Movement requires a positive two-dimensional tile"
            )
        for endpoint in (self.source, self.destination):
            if (
                endpoint.memory != "HBM"
                and self.physical_shape(endpoint)[0] > 128
            ):
                raise ValueError(
                    "Local movement endpoint exceeds partition capacity"
                )

    def physical_shape(self, endpoint):
        return tuple(self.shape[a] for a in endpoint.axes)

    @property
    def key(self):
        return ":".join(
            (
                self.role,
                self.source.memory,
                self.destination.memory,
                self.source.dtype,
                self.destination.dtype,
                "swap" if self.source.axes != self.destination.axes else "same",
                "x".join(map(str, self.shape)),
            )
        )


@dataclass(frozen=True)
class Mode:
    name: str
    opcode: str
    engine: str
    source_memory: str
    destination_memory: str
    transpose: bool = False
    cast: bool = False
    partition_limit: int = 128
    free_limit: int = 4096
    exact_tile: bool = False
    dtypes: tuple = ("float32", "bfloat16", "float16")
    storage_encoding: str = ""


# These are physical/SDK capabilities; search policy does not alter them.
MODES = (
    Mode("dma_load", "dma_copy", "DMA", "HBM", "SBUF"),
    Mode("dma_store", "dma_copy", "DMA", "SBUF", "HBM"),
    Mode("sbuf_scalar", "tensor_copy", "ScalarE", "SBUF", "SBUF", cast=True),
    Mode("sbuf_vector", "tensor_copy", "VectorE", "SBUF", "SBUF", cast=True),
    Mode("psum_scalar", "tensor_copy", "ScalarE", "PSUM", "SBUF", cast=True),
    Mode("psum_vector", "tensor_copy", "VectorE", "PSUM", "SBUF", cast=True),
    Mode(
        "tensor_transpose",
        "nc_matmul",
        "TensorE",
        "SBUF",
        "PSUM",
        True,
        free_limit=128,
    ),
    # The NKI capability is larger than the characterized subset. Only this
    # exact subset participates in measured search until more probes exist.
    Mode(
        "stream_transpose",
        "nc_transpose",
        "VectorE",
        "SBUF",
        "SBUF",
        True,
        partition_limit=32,
        free_limit=32,
        exact_tile=True,
        dtypes=("float32",),
        storage_encoding="disjoint_arenas",
    ),
)
BY_NAME = {m.name: m for m in MODES}


@dataclass(frozen=True)
class Chain:
    modes: tuple[str, ...]
    tile: tuple[int, int]
    order: str = "row"
    schedule: str = "tile"

    @property
    def name(self):
        return (
            "+".join(self.modes)
            + "/"
            + "x".join(map(str, self.tile))
            + "/"
            + self.order
            + "/"
            + self.schedule
        )

    def record(self):
        return asdict(self) | {"name": self.name}


@lru_cache(maxsize=2048)
def enumerate_chains(request, max_steps=4):
    """Search paths through memory/layout/dtype states, then legal tilings.

    HBM is an endpoint only: there is no implicit spill workspace. Revisited
    intermediate states are pruned; identity copies remain legal terminals.
    A dtype conversion may occur only on a copy, directly to the requested
    dtype, so paths cannot hide lossy downcast/upcast cycles.
    """
    paths = []

    def visit(state, path, seen):
        if path and state == request.destination:
            paths.append(path)
            return
        if len(path) == max_steps:
            return
        for mode in MODES:
            if (
                mode.source_memory != state.memory
                or state.dtype not in mode.dtypes
            ):
                continue
            if (
                mode.destination_memory == "HBM"
                and request.destination.memory != "HBM"
            ):
                continue
            dtypes = (
                tuple(dict.fromkeys((state.dtype, request.destination.dtype)))
                if mode.cast
                else (state.dtype,)
            )
            for dtype in dtypes:
                if dtype not in mode.dtypes:
                    continue
                axes = (
                    tuple(reversed(state.axes))
                    if mode.transpose
                    else state.axes
                )
                end = Endpoint(mode.destination_memory, dtype, axes)
                if end in seen and end != request.destination:
                    continue
                if end == state and end != request.destination:
                    continue
                visit(end, path + (mode.name,), seen | {end})

    if not request.materialize and request.source == request.destination:
        return (Chain((), request.shape),)
    visit(request.source, (), {request.source})
    candidates = {}
    for path in paths:
        has_transpose = any(BY_NAME[n].transpose for n in path)
        sizes = (
            ((128, 128), (64, 64), (32, 32))
            if has_transpose
            else ((128, 4096), (128, 512), (128, 128), (32, 128))
        )
        for limit in sizes:
            tile = tuple(min(a, b) for a, b in zip(request.shape, limit))
            for order in ("row", "column"):
                chain = Chain(path, tile, order)
                if legal(request, chain):
                    # A traversal choice only matters when both axes split.
                    effective_order = (
                        order
                        if all(a > b for a, b in zip(request.shape, tile))
                        else "row"
                    )
                    chain = Chain(path, tile, effective_order)
                    candidates[chain.name] = chain
                    if len(path) > 1:
                        staged = Chain(path, tile, effective_order, "stage")
                        if legal(request, staged):
                            candidates[staged.name] = staged
    return tuple(candidates.values())


def stage_tile(request, state, mode, chain):
    if chain.schedule == "tile" or mode.transpose:
        return chain.tile
    limits = [0, 0]
    limits[state.axes[0]] = mode.partition_limit
    limits[state.axes[1]] = mode.free_limit
    return tuple(min(n, limit) for n, limit in zip(request.shape, limits))


def legal(request, chain):
    if (
        chain.schedule not in ("tile", "stage")
        or chain.order not in ("row", "column")
        or any(n <= 0 for n in chain.tile)
    ):
        return False
    state = request.source
    for name in chain.modes:
        mode = BY_NAME[name]
        if state.memory != mode.source_memory or state.dtype not in mode.dtypes:
            return False
        logical_tile = stage_tile(request, state, mode, chain)
        shape = tuple(logical_tile[a] for a in state.axes)
        if shape[0] > mode.partition_limit or shape[1] > mode.free_limit:
            return False
        if mode.exact_tile and (
            shape != (mode.partition_limit, mode.free_limit)
            or any(a % b for a, b in zip(request.shape, logical_tile))
        ):
            return False
        dtype = request.destination.dtype if mode.cast else state.dtype
        state = Endpoint(
            mode.destination_memory,
            dtype,
            tuple(reversed(state.axes)) if mode.transpose else state.axes,
        )
        if chain.schedule == "stage" and state.memory != "HBM":
            full_shape = request.physical_shape(state)
            if full_shape[0] > 128:
                return False
            if (
                state.memory == "PSUM"
                and full_shape[1] * (4 if dtype == "float32" else 2) > 2048
            ):
                return False
    return state == request.destination and (
        bool(chain.modes) or not request.materialize
    )


def missing_timing(request, chain, hardware):
    """Only new movement laws gate selection; unrelated compute gaps persist."""
    state = request.source
    missing = []
    for name in chain.modes:
        mode = BY_NAME[name]
        dtype = request.destination.dtype if mode.cast else state.dtype
        if mode.engine == "DMA":
            law = (
                hardware.timing_profile.load
                if name == "dma_load"
                else hardware.timing_profile.store
            )
            if law is None:
                missing.append(name)
        else:
            if name == "tensor_transpose":
                law_name = f"nki.transpose_copy.{state.dtype}.ScalarE"
            elif name == "stream_transpose":
                law_name = (
                    f"nki.stream_transpose.{state.dtype}.SBUF.VectorE.switch"
                )
            else:
                law_name = f"nki.copy.{state.memory}.{dtype}.{mode.engine}"
            if hardware.timing_profile.operation(law_name) is None:
                missing.append(law_name)
        state = Endpoint(
            mode.destination_memory,
            dtype,
            tuple(reversed(state.axes)) if mode.transpose else state.axes,
        )
    return tuple(missing)


def coordinates(builder, expr, selection=None):
    """Resolve only the requested view coordinates through index/reshape chains.

    Reshaping broadcast root grids can allocate the entire HBM tensor. Instead,
    map the small selected coordinates backwards through each reshape.
    """
    import numpy as np

    def grid(shape):
        return tuple(
            np.broadcast_to(
                np.arange(n).reshape(
                    tuple(n if i == axis else 1 for i in range(len(shape)))
                ),
                shape,
            )
            for axis, n in enumerate(shape)
        )

    def resolve(e, points):
        if e.op == "name":
            tensor = builder.program.tensors[e.value]
            if tensor.alias:
                return resolve(tensor.view, points)
            return tensor.name, points
        if e.op == "index":
            base = e.args[0]
            shape = builder.value(base).shape
            index = builder.value(e.args[1])
            index = index if isinstance(index, tuple) else (index,)
            if len(index) == len(shape) and all(
                isinstance(i, (int, np.integer, np.ndarray)) for i in index
            ):
                # Advanced integer indices already ARE the base coordinates.
                # A flattened HBM root can have hundreds of millions of
                # elements; even arange(root.shape[0]) is prohibitively large.
                mapped = tuple(a[points] for a in np.broadcast_arrays(*index))
            else:
                mapped = tuple(a[index][points] for a in grid(shape))
            return resolve(base, mapped)
        if (
            e.op == "call"
            and e.args[0].op == "attr"
            and e.args[0].value == "reshape"
        ):
            base = e.args[0].args[0]
            shape = builder.value(e).shape
            original = builder.value(base).shape
            flat = sum(
                a * math.prod(shape[i + 1 :]) for i, a in enumerate(points)
            )
            mapped = tuple(
                (flat // math.prod(original[i + 1 :])) % n
                for i, n in enumerate(original)
            )
            return resolve(base, mapped)
        raise ValueError("Movement requires an addressable tensor view")

    points = grid(builder.value(expr).shape)
    if selection is not None:
        points = tuple(a[selection] for a in points)
    return resolve(expr, points)


def slice_view(builder, expression, row, col, nr, nc):
    """Emit a root-relative access, avoiding nested SDK access-pattern objects."""
    import numpy as np
    from .instruction_plan import Expr

    name, arrays = coordinates(
        builder,
        Expr.parse(expression),
        (slice(row, row + nr), slice(col, col + nc)),
    )
    indices = []
    for values in arrays:
        a = values
        start = int(a[0, 0])
        dr = int(a[1, 0]) - start if nr > 1 else 0
        dc = int(a[0, 1]) - start if nc > 1 else 0
        affine = (
            start + dr * np.arange(nr)[:, None] + dc * np.arange(nc)[None, :]
        )
        if not np.array_equal(a, affine):
            raise ValueError(
                "Movement tile has a non-affine or wrapped access pattern"
            )
        terms = [str(start)]
        if dr:
            terms.append(f"nl.arange({nr})[:,None]*{dr}")
        if dc:
            terms.append(f"nl.arange({nc})[None,:]*{dc}")
        indices.append("(" + "+".join(terms) + ")")
    # Builder address expressions can refer to indices but not create arange
    # as an instruction. NKI evaluates these at compile time.
    return name + "[" + ",".join(indices) + "]"


def emit_chain(planner, request, chain, source, destination=None):
    """Realize a selected chain; never choose a different route here."""
    from .instruction_plan import Tensor

    if not legal(request, chain):
        raise ValueError("Illegal selected movement chain")
    for name in chain.modes:
        if BY_NAME[name].storage_encoding:
            planner.required_movement_storage = BY_NAME[name].storage_encoding
    if not chain.modes:
        return source
    shape = request.physical_shape(request.destination)
    if destination is None:
        memory = {"HBM": "shared_hbm", "SBUF": "sbuf", "PSUM": "psum"}[
            request.destination.memory
        ]
        destination = planner.tmp(
            f"nl.ndarray({shape},dtype=nl.{request.destination.dtype},buffer=nl.{memory})"
        )
    if chain.schedule == "stage" and len(chain.modes) > 1:
        current = source
        state = request.source
        for i, name in enumerate(chain.modes):
            mode = BY_NAME[name]
            end = Endpoint(
                mode.destination_memory,
                request.destination.dtype if mode.cast else state.dtype,
                tuple(reversed(state.axes)) if mode.transpose else state.axes,
            )
            step_request = Request(state, end, request.shape, request.role)
            step = Chain(
                (name,), stage_tile(request, state, mode, chain), chain.order
            )
            current = emit_chain(
                planner,
                step_request,
                step,
                current,
                destination if i == len(chain.modes) - 1 else None,
            )
            state = end
        return destination
    coords = list(
        product(
            range(0, request.shape[0], chain.tile[0]),
            range(0, request.shape[1], chain.tile[1]),
        )
    )
    if chain.order == "column":
        coords.sort(key=lambda x: (x[1], x[0]))
    for row, col in coords:
        extent = (
            min(chain.tile[0], request.shape[0] - row),
            min(chain.tile[1], request.shape[1] - col),
        )
        offset = (row, col)
        axes = request.source.axes
        current = slice_view(
            planner.builder,
            source,
            offset[axes[0]],
            offset[axes[1]],
            extent[axes[0]],
            extent[axes[1]],
        )
        state = request.source
        for i, name in enumerate(chain.modes):
            mode = BY_NAME[name]
            dtype = request.destination.dtype if mode.cast else state.dtype
            axes = tuple(reversed(state.axes)) if mode.transpose else state.axes
            out_shape = tuple(extent[a] for a in axes)
            last = i == len(chain.modes) - 1
            target = (
                slice_view(
                    planner.builder,
                    destination,
                    offset[axes[0]],
                    offset[axes[1]],
                    *out_shape,
                )
                if last
                else None
            )
            engine = {
                "VectorE": "vector",
                "ScalarE": "scalar",
                "TensorE": "tensor",
            }.get(mode.engine)
            if mode.opcode == "nc_matmul":
                identity = "identity_" + state.dtype
                if identity not in planner.builder.program.tensors:
                    hbm = identity + "_hbm"
                    planner.builder.program.tensors[hbm] = Tensor(
                        hbm,
                        (128, 128),
                        state.dtype,
                        "HBM",
                        "contiguous",
                        constant="identity",
                    )
                    planner.builder.add(
                        f"{identity}=nl.ndarray((128,128),dtype=nl.{state.dtype},buffer=nl.sbuf)"
                    )
                    planner.builder.add(
                        f"nisa.dma_copy(dst={identity},src={hbm})"
                    )
                    planner.stats["isa_dma_panels"] += 1
                p = extent[state.axes[0]]
                call = f"nisa.nc_matmul({current},{identity}[:{p},:{p}],is_transpose=True)"
                planner.stats["isa_transposes"] += 1
                planner.expanded_isa.update(
                    {"LDWEIGHTS": 1, "MATMUL_TRANSPOSE": 1}
                )
            elif mode.opcode == "nc_transpose":
                call = f"nisa.nc_transpose({current},engine=nisa.vector_engine)"
                planner.stats["isa_transposes"] += 1
                planner.expanded_isa["STREAM_TRANSPOSE"] += 1
            elif mode.opcode == "tensor_copy":
                call = f"nisa.tensor_copy({current},dtype=nl.{dtype},engine=nisa.{engine}_engine)"
                planner.stats[f"isa_{engine}_copies"] += 1
            else:
                if target is None:
                    memory = (
                        "sbuf"
                        if mode.destination_memory == "SBUF"
                        else "shared_hbm"
                    )
                    target = planner.tmp(
                        f"nl.ndarray({out_shape},dtype=nl.{dtype},buffer=nl.{memory})"
                    )
                planner.emit(f"nisa.dma_copy(dst={target},src={current})")
                planner.stats["isa_dma_panels"] += 1
                current = target
                state = Endpoint(mode.destination_memory, dtype, axes)
                continue
            if target is not None:
                if mode.opcode == "nc_matmul":
                    planner.emit(
                        f"{target}=nisa.memset({out_shape},value=0,dtype=nl.{dtype},engine=nisa.vector_engine)"
                    )
                planner.emit(f"{target}={call}")
                current = target
            else:
                current = planner.tmp(call)
            state = Endpoint(mode.destination_memory, dtype, axes)
    return destination


class MovementSelector:
    """Bindings choose recipes; observations expose the remaining design space."""

    def __init__(self, hardware, bindings=None):
        self.hardware = hardware
        self.bindings = dict(bindings or {})
        self.requests = {}
        self._admitted = {}

    def choose(self, request, preferred_engine="ScalarE"):
        if request.key not in self.requests:
            choices = enumerate_chains(request)
            admitted = [
                c
                for c in choices
                if not missing_timing(request, c, self.hardware)
            ]
            self._admitted[request.key] = admitted
            self.requests[request.key] = dict(
                request=asdict(request),
                count=0,
                choices=[c.record() for c in admitted],
                rejected=[
                    c.record()
                    | {
                        "reason": "uncalibrated movement timing",
                        "missing_laws": missing_timing(
                            request, c, self.hardware
                        ),
                    }
                    for c in choices
                    if c not in admitted
                ],
            )
        admitted = self._admitted[request.key]
        self.requests[request.key]["count"] += 1
        if not admitted:
            raise ValueError(f"No calibrated movement chain for {request.key}")
        if request.key in self.bindings:
            selected = next(
                (c for c in admitted if c.name == self.bindings[request.key]),
                None,
            )
            if selected is None:
                raise ValueError(f"Illegal movement binding for {request.key}")
        else:

            def prior(c):
                modes = [BY_NAME[n] for n in c.modes]
                # Baseline only seeds the search; it is not a cost term.
                return (
                    any(m.name == "stream_transpose" for m in modes),
                    sum(
                        m.opcode == "tensor_copy"
                        and m.engine != preferred_engine
                        for m in modes
                    ),
                    len(c.modes),
                    -math.prod(c.tile),
                    c.order != "row",
                    c.schedule != "tile",
                )

            selected = min(admitted, key=prior)
        self.requests[request.key]["selected"] = selected.name
        return selected


def search(build, hardware, *, budget=32, beam_width=2):
    """Beam search over endpoint request bindings, scored after allocation.

    `build(bindings)` returns a planner and its allocated selected program.
    The full program, not sums of isolated copy rates, determines the score.
    """
    from .program_analysis import analyze_selected

    if budget < 1 or beam_width < 1:
        raise ValueError("Movement search budget/beam must be positive")
    records = []
    seen = set()
    best = None
    beam = [{}]
    while beam and len(records) < budget:
        frontier = []
        for bindings in beam:
            key = tuple(sorted(bindings.items()))
            if key in seen or len(records) >= budget:
                continue
            seen.add(key)
            started = time.monotonic()
            try:
                planner, program = build(bindings)
                program.validate()
                timing = analyze_selected(program, hardware)
                score = timing["prediction_ns"]
                record = dict(
                    bindings=bindings,
                    predicted_ns=score,
                    status="legal",
                    instruction_count=len(program.instructions),
                    storage_encoding=program.encoding_storage,
                    allocation_policy=getattr(
                        planner, "selected_allocation_policy", "external"
                    ),
                    unknown_completion_count=len(timing["unknown_completion"]),
                    sbuf_bytes=max(
                        (
                            p.byte_address + p.bytes_per_partition
                            for p in program.placements.values()
                            if p.memory == "SBUF"
                        ),
                        default=0,
                    )
                    * 128,
                    psum_banks=len(
                        {
                            p.bank
                            for p in program.placements.values()
                            if p.memory == "PSUM"
                        }
                    ),
                )
                canonical = {
                    k: v["selected"]
                    for k, v in planner.movement_selector.requests.items()
                }
                record["selected_bindings"] = canonical
                seen.add(tuple(sorted(canonical.items())))
                frontier.append(
                    (score, canonical, planner.movement_selector.requests)
                )
                if best is None or score < best[0]:
                    best = (score, planner, program, record)
            except (ValueError, NotImplementedError) as error:
                record = dict(
                    bindings=bindings, status="rejected", reason=str(error)
                )
            record["search_seconds"] = time.monotonic() - started
            records.append(record)
            logging.getLogger(__name__).info(
                "Movement candidate %d/%d: %s, predicted_us=%s, seconds=%.2f%s",
                len(records),
                budget,
                record["status"],
                record.get("predicted_ns", 0) / 1000,
                record["search_seconds"],
                " reason=" + record["reason"] if "reason" in record else "",
            )
        beam = []
        for _, bindings, requests in sorted(frontier, key=lambda x: x[0])[
            :beam_width
        ]:
            # Round-robin across endpoint classes so a large load design
            # space cannot consume the entire budget before eviction/staging.
            alternatives = [
                (key, [c for c in req["choices"] if c["name"] != bindings[key]])
                for key, req in requests.items()
            ]
            for index in range(
                max((len(v) for _, v in alternatives), default=0)
            ):
                for key, choices in alternatives:
                    if index < len(choices):
                        alternate = bindings | {key: choices[index]["name"]}
                        if tuple(sorted(alternate.items())) not in seen:
                            beam.append(alternate)
        # One layer contains the neighbors of the selected beam parents.
        # Keep all neighbors for evaluation, not just the first beam_width.
    if best is None:
        raise ValueError("No legal movement candidate: " + str(records))
    requests = best[1].movement_selector.requests
    report = dict(
        strategy="bounded beam search over endpoint-constrained instruction chains",
        capabilities=[asdict(mode) for mode in MODES],
        budget=budget,
        beam_width=beam_width,
        evaluated=len(records),
        nominal_combinations=math.prod(
            len(v["choices"]) for v in requests.values()
        ),
        selected=best[3],
        requests=requests,
        candidates=records,
        scope="Fixed shared software schedule; ISA chain/tiling/engine and physical allocation are searched. No global optimum claim.",
    )
    return best[1], best[2], report
