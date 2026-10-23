"""Fixed Voyager CLI configuration and metadata; no target registration."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path

from voyager_compiler.hardware_config import AcceleratorConfig
from voyager_compiler.quantization.voyager import (
    PRESETS,
    QUANTIZATION_CONFIGS,
    configure_model,
    finalize_options,
)
from voyager_compiler.voyager_adapter import interstellar_memory


@dataclass(frozen=True)
class CompilationContext:
    hardware: AcceleratorConfig

    def configure_quantizer(self, kind, model, quantizer, args, **options):
        configure_model(kind, model, quantizer, args, **options)
        quantizer._share_concat_scales = True


def resolve_context(args):
    if getattr(args, "target_hardware", "voyager") != "voyager":
        raise ValueError("This compiler implements Voyager only")
    qconfig = getattr(args, "qconfig", None)
    if qconfig is not None and qconfig not in QUANTIZATION_CONFIGS:
        raise ValueError(f"Unknown Voyager qconfig {qconfig!r}")
    hardware = AcceleratorConfig.from_args(args)
    interstellar_memory(hardware)
    return CompilationContext(hardware)


def parse_args(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    try:
        if args.quantization_recipe:
            if args.quantization_recipe not in PRESETS:
                raise ValueError(
                    f"Unknown Voyager preset {args.quantization_recipe!r}"
                )
            defaults = PRESETS[args.quantization_recipe]
            args = parser.parse_args(
                argv, namespace=argparse.Namespace(**defaults)
            )
            finalize_options(args)
        if (
            getattr(args, "parameter_loading", "on_demand") == "preload"
            and getattr(args, "bufferized_flow", "per_kernel") != "resident"
        ):
            raise ValueError(
                "Parameter preload requires --bufferized_flow resident"
            )
        args.compilation_context = resolve_context(args)
    except (ValueError, NotImplementedError, KeyError) as exc:
        parser.error(str(exc))
    return args


def write_manifest(args):
    record = dict(
        model=args.model,
        target="voyager",
        family="voyager",
        backend="voyager",
        quantization_recipe=args.quantization_recipe,
        verification_stage="final_lowered_graph" if args.debug else None,
        hardware=args.compilation_context.hardware,
        options={
            k: v for k, v in vars(args).items() if k != "compilation_context"
        },
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
