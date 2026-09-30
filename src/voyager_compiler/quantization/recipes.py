"""Deployment recipes selected by hardware family and optional model/stage.

Quantization settings and compilation defaults are separate dictionaries. CLI
values explicitly supplied by the caller take precedence over both. A named
per-operand qconfig is a separate override, preserving the existing LLM CLI.
"""

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class Recipe:
    quantization: Mapping = field(default_factory=dict)
    compilation: Mapping = field(default_factory=dict)
    models: Mapping = field(default_factory=dict)

    def options(self, model):
        return {
            **self.quantization,
            **self.compilation,
            **self.models.get(model, {}),
        }


@dataclass(frozen=True)
class FamilyPolicy:
    recipes: Mapping
    qconfigs: Mapping
    configure_model: Callable
    finalize_options: Callable | None = None
    quantization_rules: Callable | None = None


_FAMILIES = {}


def register_family(name, policy):
    if name in _FAMILIES:
        raise ValueError(f"Quantization family already registered: {name}")
    _FAMILIES[name] = policy


def get_family(name):
    if name not in _FAMILIES:
        raise ValueError(
            f"No quantization policy registered for family {name!r}"
        )
    return _FAMILIES[name]


def get_recipe(target, name):
    family = get_family(target.family)
    if name in target.recipes:
        return target.recipes[name]
    if name not in family.recipes:
        raise ValueError(
            f"Unknown quantization recipe {name!r} for {target.name}"
        )
    return family.recipes[name]
