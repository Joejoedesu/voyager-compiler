"""Select compact encodings for saved plans without repeating mapping search."""

import argparse
from dataclasses import replace
import hashlib
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
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.plan_emitter import emit
from voyager_compiler.trainium.compact_encoding import select


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--storage-binding", choices=("arena", "disjoint_arenas"))
    a = p.parse_args()
    for name in a.cases:
        source = a.inputs.resolve() / name
        root = a.output.resolve() / name
        if source == root:
            p.error(
                "Use a separate output directory to preserve the original artifacts"
            )
        root.mkdir(parents=True, exist_ok=True)
        for filename in (
            "model.txt",
            "hardware.json",
            "compilation.json",
            "generation.json",
        ):
            shutil.copy2(source / filename, root / filename)
        ref = root / "reference.npz"
        if not ref.exists():
            ref.symlink_to(source / "reference.npz")
        program = Program.load(
            json.loads((source / "instructions.json").read_text())
        )
        if a.storage_binding == "disjoint_arenas":
            program.bind_disjoint_regions()
        elif a.storage_binding == "arena":
            program.encoding_storage = "arena"
            program.storage_regions = []
        program.encoding_loops = []
        lines = []
        emit(program, _capture=lines)
        program.encoding_loops = select(program, lines)
        (root / "instructions.json").write_text(
            json.dumps(program.record(), separators=(",", ":"))
        )
        manifest = json.loads((source / "selection.json").read_text())
        manifest["instructions_sha256"] = hashlib.sha256(
            (root / "instructions.json").read_bytes()
        ).hexdigest()
        manifest["encoding_origin"] = str(source)
        manifest["encoding_loops"] = len(program.encoding_loops)
        (root / "selection.json").write_text(json.dumps(manifest, indent=2))
        raw = json.loads((root / "hardware.json").read_text())["hardware"][
            "timing_profile"
        ]
        raw["primitives"] = tuple(
            PrimitiveTiming(**x) for x in raw["primitives"]
        )
        for key in ("load", "store"):
            raw[key] = DmaTiming(**raw[key]) if raw[key] else None
        hardware = replace(
            neuron_core(3), timing_profile=TrainiumTimings(**raw)
        )
        context = CompilerContext.from_artifacts(root, hardware)
        context.realize(root)
        generation = json.loads((root / "generation.json").read_text())
        generation["program_sha256"] = hashlib.sha256(
            (root / "nki/program.py").read_bytes()
        ).hexdigest()
        generation["encoding_origin"] = str(source)
        (root / "generation.json").write_text(json.dumps(generation, indent=2))
        print(
            name,
            "loops",
            len(program.encoding_loops),
            "source bytes",
            (root / "nki/program.py").stat().st_size,
            flush=True,
        )


if __name__ == "__main__":
    main()
