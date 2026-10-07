"""Search data movement on a fixed shared schedule, preserving input artifacts."""

import argparse
import json
import logging
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from voyager_compiler.compilation import CompilerContext
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import TrainiumMappingPolicy
from voyager_compiler.trainium.planning import select_plan


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--beam", type=int, default=2)
    p.add_argument(
        "--bindings",
        type=Path,
        help="Replay explicit candidate bindings for a diagnostic comparison",
    )
    p.add_argument(
        "--allocation-policy",
        choices=("best_fit", "size_classes"),
        default="best_fit",
    )
    a = p.parse_args()
    if a.output.exists():
        p.error("Output already exists")
    a.output.mkdir(parents=True)
    for name in (
        "model.txt",
        "hardware.json",
        "compilation.json",
        "generation.json",
    ):
        shutil.copy2(a.input / name, a.output / name)
    (a.output / "reference.npz").symlink_to(
        (a.input / "reference.npz").resolve()
    )
    record = json.loads((a.output / "compilation.json").read_text())
    options = record["compiler"]["policy"]
    tuning = replace(
        TrainiumTuning(**options),
        movement_search_budget=a.budget,
        movement_search_beam=a.beam,
    )
    hw = neuron_core(3)
    context = CompilerContext.resolve(hw, TrainiumMappingPolicy(hw, tuning))
    context.write(a.output)
    hardware = json.loads((a.output / "hardware.json").read_text())
    hardware["hardware"] = asdict(hw)
    (a.output / "hardware.json").write_text(
        json.dumps(hardware, default=lambda x: sorted(x), indent=2)
    )
    generation = json.loads((a.output / "generation.json").read_text())
    generation["movement_search_origin"] = str(a.input.resolve())
    (a.output / "generation.json").write_text(json.dumps(generation, indent=2))
    result = select_plan(
        a.output,
        context=context,
        allocation_policy=a.allocation_policy,
        movement_bindings=(
            json.loads(a.bindings.read_text()) if a.bindings else None
        ),
    )
    context.realize(a.output)
    print(
        json.dumps(
            dict(
                predicted_us=result["program_analysis"][
                    "whole_program_prediction_ns"
                ]
                / 1000,
                stats=result["stats"],
                selected=result.get("movement_search", {}).get("selected"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
