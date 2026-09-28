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
