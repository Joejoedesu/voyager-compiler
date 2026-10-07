"""Re-select ISA from a saved shared schedule, preserving original evidence."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.timing import (
    TrainiumTimings,
    PrimitiveTiming,
    DmaTiming,
)
from voyager_compiler.trainium.planning import select_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--allocation-policy",
        choices=("best_fit", "size_classes"),
        default="best_fit",
    )
    args = parser.parse_args()
    for name in args.cases:
        source = args.inputs.resolve() / name
        root = args.output.resolve() / name
        if root.exists():
            parser.error(f"Output already exists: {root}")
        root.mkdir(parents=True)
        for filename in ("model.txt", "hardware.json", "compilation.json"):
            shutil.copy2(source / filename, root / filename)
        reference = (
            (args.reference_root.resolve() / name)
            if args.reference_root
            else source
        )
        (root / "reference.npz").symlink_to(reference / "reference.npz")
        generation = json.loads((reference / "generation.json").read_text())
        generation["reselection_origin"] = str(source)
        generation["allocation_diagnostic"] = args.allocation_policy
        if args.reference_root:
            generation["diagnostic_tile"] = None
        (root / "generation.json").write_text(
            json.dumps(generation, indent=2) + "\n"
        )
        raw = json.loads((root / "hardware.json").read_text())["hardware"][
            "timing_profile"
        ]
        raw["primitives"] = tuple(
            PrimitiveTiming(**p) for p in raw["primitives"]
        )
        for key in ("load", "store"):
            raw[key] = DmaTiming(**raw[key]) if raw[key] else None
        hardware = replace(
            neuron_core(3), timing_profile=TrainiumTimings(**raw)
        )
        context = CompilerContext.from_artifacts(root, hardware)
        select_plan(
            root, context=context, allocation_policy=args.allocation_policy
        )
        context.realize(root)
        print(
            "RESELECTED",
            name,
            (root / "nki/program.py").stat().st_size,
            flush=True,
        )


if __name__ == "__main__":
    main()
