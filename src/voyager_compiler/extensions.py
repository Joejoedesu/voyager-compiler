"""Public extension contracts; IR records have a single owner in ir/."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class OpModel(ABC):
    """A backend's model of one op: which tilings exist, whether one fits
    and what it costs.

    A matrix op's model runs in a forked search worker, so it holds plain
    values only -- no FX node, op object or tensor -- and dispatches no torch
    op.

    Attributes:
        space: How the shared search generates the op's candidates, which
            also picks the search: a :class:`MatrixSpace`,
            :class:`GridSpace` or :class:`AttentionSpace`.
        key: What the answers depend on: matrix ops with equal keys share one
            search; ``None`` never shares.
    """

    space: object
    key: object

    @abstractmethod
    def fit(self, tiling):
        """The :class:`Placement` ``tiling`` takes on chip, or ``None`` when
        it does not fit.

        A matrix op's fit is asked at interstellar's fit check, on a
        ``MappingPoint`` with no loop orders whose L0 blocking counts that
        level's steps and copies together (its partitionings all 1), and its
        answer is reused for every partitioning and loop order of that
        blocking.  So it reads only each level's total extent (blocking x
        partitioning) and the blockings above L0, never the loop orders or
        how L0 splits into steps and copies.
        """

    @abstractmethod
    def cost(self, tiling, placement):
        """The :class:`Cost` of ``tiling`` laid out as ``placement``, the
        answer ``fit`` gave."""

    def scratch(self, out_tile):
        """The regions the backend's kernel keeps beside the op's tiles that
        no FX node names, for an output tile ``out_tile`` (one per output
        for several): ``(name, shape, dtype, slots, space)`` each, ``space``
        the ``MemoryLevel`` it is allocated in.  The fit places
        them for its tiling's tile, and a builder sizing its own nest, or an
        op built whole, asks for its own.  None by default.
        """
        return ()


class RuntimeModel(ABC):
    """A backend's hardware, as the shared tile search sees it.

    Built once per compile from the backend's config
    (``config.runtime_model``), it answers in that backend's cycles.  Times
    rank candidates against each other, so only their ratios matter to the
    search; a reported latency is cycles at the backend's clock.

    Attributes:
        config: The hardware description the model was built from.
        accumulate_fp32: A GEMM split along its reduction accumulates in
            float32.
    """

    def __init__(self, config, accumulate_fp32):
        self.config = config
        self.accumulate_fp32 = accumulate_fp32

    @abstractmethod
    def prepare(self, node):
        """The :class:`OpModel` of ``node`` -- a GEMM / conv, vector, GEMV or
        pool op, bare or fused -- or ``None`` to leave it untiled.

        Raises:
            NotImplementedError: The backend does not lower ``node``.
        """

    def prepare_attention(self, node, acc_dtype, bool_mask):
        """The :class:`OpModel` of a flash-attention ``node``.

        Args:
            node: The attention node.
            acc_dtype: The dtype the kernel keeps its softmax state in.
            bool_mask: The causal mask is int1 codes plus a scale, rather
                than an additive fill.

        Raises:
            NotImplementedError: The backend does not run fused attention.
        """
        raise NotImplementedError(f"{type(self).__name__}: fused attention")


def _one_candidate(node, context):
    return ({},)


@dataclass(frozen=True)
class ImplementationSpec:
    """One searchable lowering, with legality, model and builder kept together.

    ``candidates`` yields named parameter dictionaries. ``build`` refines one
    into executable FX, using the paired runtime model's shared tile search.
    ``estimate`` may supply a whole-region Cost; otherwise the selected tile's
    Cost is used. Two competing implementations must both provide a cost.
    A builder may reject a candidate with ``CandidateRejected``; programming
    errors propagate rather than silently hiding a broken implementation.
    """

    name: str
    matches: Callable
    build: Callable
    candidates: Callable = _one_candidate
    model: Callable | None = None
    estimate: Callable | None = None
    validate: Callable | None = None
    description: str = ""
    cache_by_shape: bool = False


class CandidateRejected(ValueError):
    """Expected legality/fit/refinement rejection, recorded in search evidence."""


class NoLegalMapping(CandidateRejected, RuntimeError):
    """A supported search space has no fitting mapping for this candidate.

    Retain RuntimeError compatibility for existing search callers, while the
    implementation planner can reject this algorithm and try another. Model or
    builder programming errors must not be translated into this exception.
    """


@dataclass(frozen=True)
class OperatorSemantics:
    """Shared classification for a registered target operator.

    Aliasing and writes still come from its Torch schema; registration never
    licenses treating arbitrary target calls as pure or address-only views.
    """

    compute: bool = True
    elementwise: bool = False


@dataclass(frozen=True)
class QuantizationPolicy:
    """Shared PT2E defaults or a target-owned quantization flow.

    ``run(prepared, config, options) -> PreparedWorkload`` optionally replaces
    this stage only, including target format conversion and recalibration.
    It must honor an explicit caller quantizer, preserve the input contract,
    and return an executable graph plus quantization provenance. The shared
    driver refreshes its reference and never converts it a second time.
    Explicit ``options["quantizer"] = None`` disables either flow.

    ``quantizer(prepared, options)`` supplies a default; an explicit options
    quantizer (including None) wins. ``prepare(graph, options)`` sees observers;
    ``finalize(graph, options)`` runs after calibration, before shared conversion.
    Both mutate in place. Neither hook lowers target operations or chooses layout.
    """

    quantizer: Callable | None = None
    prepare: Callable | None = None
    finalize: Callable | None = None
    bias: str | None = None
    example_calibration: bool = False
    integer_rounding: bool = False
    run: Callable | None = None


@dataclass(frozen=True)
class TargetSpec:
    """Bindings at shared compiler boundaries; never a replacement pipeline.

    ``operators`` declares semantics; ``input_preprocessing`` optionally rewrites
    the exported graph and its inputs before quantization. ``quantization``
    configures shared PT2E or overrides that stage;
    ``legalize`` rewrites it before shared ``normalization``; ``lower`` then maps
    normalized tensor operations to custom target operations. ``implementations``
    pair candidate search/model/estimate with a buffer builder. ``refine`` may
    finalize target execution/storage declarations before shared allocation.
    ``encode(EncodedIR, program, config)`` projects frozen decisions into common
    IR, target records and payloads, then the driver seals the bundle.
    ``emit(SavedBundle, output_directory)`` produces target collateral using only
    saved inputs, without a live graph or renewed search. Optional ``runtime``
    and ``characterize`` own execution and measurement, not compilation. A missing optional callback
    means no specialization of that boundary, not an unimplemented target.

    See docs/extensions.md for signatures, consumers and decision boundaries.
    """

    name: str
    hardware: Callable
    implementations: tuple[ImplementationSpec, ...]
    model: Callable
    quantization: QuantizationPolicy | None = None
    legalize: Callable | None = None
    lower: Callable | None = None
    refine: Callable | None = None
    emit: Callable | None = None
    runtime: Callable | None = None
    characterize: Callable | None = None
    encode: Callable | None = None
    operators: tuple = ()
    normalization: tuple = ()
    input_preprocessing: Callable | None = None

    def __post_init__(self):
        names = [i.name for i in self.implementations]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{self.name}: duplicate implementation identities"
            )


@dataclass(frozen=True)
class WorkloadSpec:
    """The minimum workload extension is construction plus example inputs.

    Optional hooks supply real workload semantics. Asset paths/checksums,
    stateful inputs and custom references may live in this same module.
    ``prepare`` is a frontend hook returning PreparedWorkload, never a compiler.
    """

    name: str
    build: Callable
    example_inputs: Callable
    prepare: Callable | None = None
    preprocess: Callable | None = None
    reference: Callable | None = None
    calibration: Callable | None = None
    assets: tuple = ()
    cases: tuple = ()


class Registry:
    """Explicit registration, deterministic discovery, no target-name switches."""

    def __init__(self):
        self.targets: dict[str, TargetSpec] = {}
        self.workloads: dict[str, WorkloadSpec] = {}
        self.operators: dict[object, OperatorSemantics] = {}

    def register(self, spec):
        if isinstance(spec, TargetSpec):
            table = self.targets
        elif isinstance(spec, WorkloadSpec):
            table = self.workloads
        else:
            raise TypeError(
                f"expected TargetSpec or WorkloadSpec, got {type(spec)}"
            )
        if spec.name in table:
            raise ValueError(f"extension already registered: {spec.name}")
        if isinstance(spec, TargetSpec):
            for operator, semantics in spec.operators:
                previous = self.operators.get(operator)
                if previous is not None and previous != semantics:
                    raise ValueError(
                        f"conflicting operator semantics: {operator}"
                    )
            self.operators.update(spec.operators)
        table[spec.name] = spec
        return spec

    def target(self, name):
        """Load a registered package or installed entry point on first use."""
        if name not in self.targets:
            import importlib
            from importlib.metadata import entry_points

            entries = tuple(entry_points(group="voyager.targets", name=name))
            if entries:
                if len(entries) != 1:
                    raise ValueError(f"ambiguous target entry point: {name}")
                spec = entries[0].load()
                if name not in self.targets:
                    self.register(spec)
            else:
                if (
                    not isinstance(name, str)
                    or not name.replace("-", "_").isidentifier()
                ):
                    raise ValueError(f"invalid target name: {name!r}")
                importlib.import_module(
                    f"voyager_compiler.targets.{name.replace('-', '_')}.target"
                )
        return self.targets[name]

    def workload(self, name):
        if name not in self.workloads:
            import importlib
            from importlib.metadata import entry_points

            entries = tuple(entry_points(group="voyager.workloads", name=name))
            if len(entries) > 1:
                raise ValueError(f"ambiguous workload entry point: {name}")
            if entries:
                spec = entries[0].load()
                if name not in self.workloads:
                    self.register(spec)
            else:
                module = name.replace("-", "_")
                if not module.isidentifier():
                    raise ValueError(f"invalid workload name: {name!r}")
                importlib.import_module(f"voyager_compiler.workloads.{module}")
        return self.workloads[name]


def describe_target(target):
    """Machine-readable discovery of supported recipes and owned boundaries."""

    def binding(callback):
        if callback is None:
            return None
        # Partial functions and callable objects are legal callbacks too.
        callback = getattr(callback, "func", callback)
        owner = (
            callback if hasattr(callback, "__qualname__") else type(callback)
        )
        return f"{owner.__module__}.{owner.__qualname__}"

    return dict(
        name=target.name,
        bindings={
            name: binding(getattr(target, name))
            for name in (
                "hardware",
                "model",
                "input_preprocessing",
                "legalize",
                "lower",
                "refine",
                "encode",
                "emit",
                "runtime",
                "characterize",
            )
        },
        quantization=(
            dict(
                run=binding(target.quantization.run),
                quantizer=binding(target.quantization.quantizer),
                prepare=binding(target.quantization.prepare),
                finalize=binding(target.quantization.finalize),
                bias=target.quantization.bias,
                example_calibration=target.quantization.example_calibration,
                integer_rounding=target.quantization.integer_rounding,
            )
            if target.quantization
            else None
        ),
        implementations=[
            dict(
                name=spec.name,
                description=spec.description,
                candidate_parameters=getattr(
                    spec.candidates, "__qualname__", binding(spec.candidates)
                ),
                paired_model=spec.model is not None,
                explicit_estimate=spec.estimate is not None,
                bindings={
                    name: binding(getattr(spec, name))
                    for name in (
                        "matches",
                        "candidates",
                        "model",
                        "build",
                        "estimate",
                        "validate",
                    )
                },
            )
            for spec in target.implementations
        ],
        normalization={
            name: (
                [
                    [
                        dict(
                            targets=[str(op) for op in matcher.targets],
                            predicate=binding(matcher.predicate),
                            keeps_input=matcher.keeps_input,
                        )
                        for matcher in pattern
                    ]
                    for pattern in value
                ]
                if name == "patterns" and value is not None
                else value
            )
            for name, value in target.normalization
        },
        operator_contracts=[str(op) for op, _ in target.operators],
        characterization=target.characterize is not None,
        runtime=target.runtime is not None,
        emission=target.emit is not None,
    )


registry = Registry()
