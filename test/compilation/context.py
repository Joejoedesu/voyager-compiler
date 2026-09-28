"""Resolve one model/target deployment before loading model or dataset assets."""

import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path

from voyager_compiler.quantization.recipes import (
    get_family,
    get_recipe,
)
from voyager_compiler.quantization.voyager import ensure_voyager_policy
from voyager_compiler.targets import get_backend, get_target


@dataclass(frozen=True)
class CompilationContext:
    target: object
    hardware: object
    backend: object
    policy: object
    recipe: str | None

    def configure_quantizer(self, kind, model, quantizer, args, **options):
        self.policy.configure_model(
            kind,
            model,
            quantizer,
            args,
            qconfigs=self.policy.qconfigs,
            **options,
        )


def resolve_context(args):
    ensure_voyager_policy()
    target = get_target(getattr(args, "target_hardware", "voyager"))
    backend = get_backend(target.backend)
    policy = get_family(target.family)
    qconfig = getattr(args, "qconfig", None)
    if qconfig is not None and qconfig not in policy.qconfigs:
        raise ValueError(
            f"Unknown qconfig {qconfig!r} for family {target.family}"
        )
    hardware = target.hardware(args)
    if hardware.backend != target.backend:
        raise ValueError("Target hardware and backend disagree")
    backend.validate(hardware)
    return CompilationContext(
        target,
        hardware,
        backend,
        policy,
        getattr(args, "quantization_recipe", None),
    )


def parse_args(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    ensure_voyager_policy()
    try:
        target = get_target(args.target_hardware)
        get_backend(target.backend)
        if args.quantization_recipe:
            recipe = get_recipe(target, args.quantization_recipe)
            defaults = recipe.options(args.model)
            destinations = {a.dest for a in parser._actions}
            if unknown := defaults.keys() - destinations:
                raise ValueError(
                    f"Recipe contains unknown options: {sorted(unknown)}"
                )
            # Parsing into a prefilled namespace preserves explicit CLI values,
            # including --no-* flags, without changing shared parser defaults.
            import argparse

            args = parser.parse_args(
                argv, namespace=argparse.Namespace(**defaults)
            )
            policy = get_family(target.family)
            if policy.finalize_options is not None:
                policy.finalize_options(args)
        args.compilation_context = resolve_context(args)
    except (ValueError, NotImplementedError, KeyError) as exc:
        parser.error(str(exc))
    return args


def write_manifest(args):
    context = args.compilation_context
    options = {
        key: value
        for key, value in vars(args).items()
        if key != "compilation_context"
    }
    record = dict(
        model=args.model,
        target=context.target.name,
        family=context.target.family,
        backend=context.target.backend,
        quantization_recipe=context.recipe,
        verification_stage="final_lowered_graph" if args.debug else None,
        hardware=context.hardware,
        options=options,
    )

    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        raise TypeError(f"Cannot serialize {type(value)}")

    path = Path(args.model_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "compilation.json").write_text(
        json.dumps(record, default=encode, indent=2) + "\n"
    )
