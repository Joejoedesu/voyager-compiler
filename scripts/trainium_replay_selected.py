"""Re-evaluate authenticated instructions without changing measured artifacts."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.program_analysis import analyze_selected


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("artifacts", type=Path, nargs="+")
    a = p.parse_args()
    code = Path(__file__).resolve().parents[1] / "src/voyager_compiler/trainium"
    sources = {path.name: digest(path) for path in sorted(code.glob("*.py"))}
    sources["timing_trainium2.json"] = digest(code / "timing_trainium2.json")
    hardware = neuron_core(3)
    for folder in a.artifacts:
        for path in sorted(folder.glob("*/instructions.json")):
            root = path.parent
            if not (root / "result.json").exists():
                continue
            result = json.loads((root / "result.json").read_text())
            if result["program_sha256"] != digest(root / "nki/program.py"):
                continue
            program = Program.load(json.loads(path.read_text()))
            validation_error = None
            added = 0
            try:
                program.validate()
            except ValueError as error:
                validation_error = str(error)
                if "missing physical reuse completion" not in validation_error:
                    raise
                # Historical records can omit independent-reader retirement.
                # Audit a repaired DAG in memory; do not alter the measured
                # source, saved plan, or claim the original plan was complete.
                extra = {}
                for previous, current in program.reuse_edges():
                    if (
                        previous
                        not in program.instructions[current].dependencies
                    ):
                        extra.setdefault(current, set()).add(previous)
                added = sum(map(len, extra.values()))
                program.instructions = [
                    replace(
                        ins,
                        dependencies=tuple(
                            sorted(set(ins.dependencies) | extra.get(i, set()))
                        ),
                    )
                    for i, ins in enumerate(program.instructions)
                ]
                program.validate()
            analysis = analyze_selected(program, hardware)
            analysis.update(
                recorded_plan_valid=validation_error is None,
                recorded_plan_validation_error=validation_error,
                analysis_completion_edges_added=added,
            )
            analysis.update(
                instructions_sha256=digest(path),
                source_sha256=sources,
                hardware=asdict(hardware),
                evidence="Reanalysis of the unchanged measured instruction plan; no new execution",
            )
            (root / "selected_analysis.json").write_text(
                json.dumps(
                    analysis, indent=2, default=lambda value: sorted(value)
                )
                + "\n"
            )
            print(root, analysis["prediction_ns"] / 1000, flush=True)


if __name__ == "__main__":
    main()
