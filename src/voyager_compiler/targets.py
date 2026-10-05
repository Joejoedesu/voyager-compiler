"""Named hardware targets, family policies, and compiler backend dispatch.

A target describes an instance; its family supplies deployment recipes. A
backend owns lowering, allocation and emission, and need not use Voyager IR.
Registration is explicit: describing hardware alone does not add a compiler.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from voyager_compiler.hardware_config import AcceleratorConfig


class Backend(Protocol):
    def validate(self, config): ...
    def fusion_patterns(self, config): ...
    def transform(
        self, model, example_args, example_kwargs=None, **options
    ): ...
    def compile(self, model, example_args, example_kwargs=None, **options): ...


class BufferizedBackend(Backend, Protocol):
    """Extra hooks for backends opting into the shared bufferized flow."""

    uses_bufferized_flow: bool

    def mapping_policy(self, config): ...
    def restore_mapping_policy(self, config, options): ...
    def realize(self, root, context, **options): ...
    def interstellar_memory(self, config): ...
    def prepare_graph(self, model, config): ...

    skip_rgb_padding: bool


class BufferizedPolicy(Protocol):
    """Fixed hooks used by shared matrix/nonmatrix search and placement.

    prepare_matrix supplies size/runtime callbacks; evaluate returns the
    selected plan and its resource estimate. Policies do not replace shared
    builders, dependency tracking or the allocation lifetime analysis.
    """

    config: AcceleratorConfig
    search_fully_connected: bool
    matrix_only_fusion: bool
    speed_only: bool

    def schedule(self): ...
    def options(self): ...
    def partition(self, architecture, size_fn, layer, mapping): ...
    def prepare_matrix(self, problem, tiler): ...
    def evaluate(
        self, runtime, architecture, layer, mapping, bank_groups, scratch_slots
    ): ...
    def nonmatrix_cost(self, kind, node, default): ...
    def nonmatrix_footprint(self, node, shapes, sharing, default): ...
    def nonmatrix_slot_size(self, node, slots): ...
    def vector_limits(self, anchor, limits): ...
    def place_local_buffers(self, model, bufs): ...


@dataclass(frozen=True)
class Target:
    name: str
    family: str
    backend: str
    hardware: Callable
    # Instance-level recipe overrides; family recipes remain the fallback.
    recipes: Mapping = field(default_factory=dict)


_TARGETS = {}
_BACKENDS = {}


def register_backend(name: str, backend: Backend):
    if name in _BACKENDS:
        raise ValueError(f"Backend already registered: {name}")
    _BACKENDS[name] = backend


def get_backend(name):
    if name not in _BACKENDS:
        raise NotImplementedError(
            f"No compiler backend registered for {name!r}"
        )
    return _BACKENDS[name]


def validate_bufferized_target(config):
    """Validate a backend opting into the shared bufferized compiler."""
    backend = get_backend(config.backend)
    if not getattr(backend, "uses_bufferized_flow", False):
        raise NotImplementedError(
            f"{config.backend} does not implement bufferized lowering"
        )
    backend.validate(config)


def register_target(target: Target):
    if target.name in _TARGETS:
        raise ValueError(f"Target already registered: {target.name}")
    if not target.name or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in target.name
    ):
        raise ValueError("Target names must be lowercase path-safe identifiers")
    _TARGETS[target.name] = target


def get_target(name="voyager"):
    if name not in _TARGETS:
        raise ValueError(
            f"Unknown target {name!r}; available: {', '.join(_TARGETS)}"
        )
    return _TARGETS[name]


class VoyagerBackend:
    uses_bufferized_flow = True

    def restore_mapping_policy(self, config, options):
        if options:
            raise ValueError("Unknown Voyager mapping policy options")
        return self.mapping_policy(config)

    def realize(self, root, context, **options):
        from pathlib import Path

        if options:
            raise ValueError("Voyager realization accepts no extra options")
        context.check_artifacts(root)
        path = Path(root) / "model.txt"
        if not path.exists():
            raise ValueError("Missing Voyager model.txt")
        return path

    def mapping_policy(self, config):
        from voyager_compiler.codegen.transform.tiling.policy import (
            VoyagerMappingPolicy,
        )

        return VoyagerMappingPolicy(config)

    def interstellar_memory(self, config):
        from voyager_compiler.voyager_adapter import interstellar_memory

        return interstellar_memory(config)

    skip_rgb_padding = True

    def prepare_graph(self, model, config):
        return None

    def validate(self, config):
        from voyager_compiler.voyager_adapter import interstellar_memory

        interstellar_memory(config)

    def fusion_patterns(self, config):
        from voyager_compiler.voyager_adapter import fusion_patterns

        return fusion_patterns(config)

    def transform(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _transform_voyager

        return _transform_voyager(
            model, example_args, example_kwargs, **options
        )

    def compile(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _compile_voyager

        return _compile_voyager(model, example_args, example_kwargs, **options)


register_backend("voyager", VoyagerBackend())
register_target(
    Target("voyager", "voyager", "voyager", AcceleratorConfig.from_args)
)

from voyager_compiler.gemmini.backend import GemminiBackend
from voyager_compiler.gemmini.hardware import lean_config
from voyager_compiler.quantization.gemmini import ensure_gemmini_policy

ensure_gemmini_policy()
register_backend("gemmini", GemminiBackend())
register_target(Target("gemmini", "gemmini", "gemmini", lean_config))
