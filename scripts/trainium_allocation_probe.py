"""Diagnostic: isolate SBUF address reuse without changing the selected ISA.

Consumes a strict artifact. Reconstructs logical dependencies, preserves PSUM
placement, and assigns each SBUF generation distinct storage. This is a bounded
experiment, not an allocation policy or a second production compiler.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.plan_emitter import emit
from voyager_compiler.trainium.program_analysis import analyze_selected


def prepare(source, destination):
    p = Program.load(json.loads((source / "instructions.json").read_text()))
    if p.encoding_storage == "compiler":
        raise ValueError("Expected a strict allocated plan")
    original = analyze_selected(p, neuron_core(3))
    old_extent = max(
        x.byte_address + x.bytes_per_partition
        for x in p.placements.values() if x.memory == "SBUF"
    ) * 128
    offset = 0
    for name, placement in p.placements.items():
        if placement.memory == "SBUF":
            p.placements[name] = replace(placement, byte_address=offset)
            offset += placement.bytes_per_partition
    if offset * 128 > neuron_core(3).scratchpad_size:
        raise ValueError(f"Distinct SBUF generations need {offset * 128} bytes")
    # Discard only placement-induced edges, reconstructing the same logical
    # RAW/WAR/WAW rules as Builder before adding the remaining PSUM reuse.
    last_write, readers, instructions = {}, {}, []
    for index, ins in enumerate(p.instructions):
        reads = {p.root(n) for n in ins.reads}
        writes = {p.root(n) for n in ins.writes}
        required = {last_write[n] for n in reads | writes if n in last_write}
        for n in writes:
            required.update(readers.get(n, ()))
            last_write[n] = index
            readers[n] = set()
        for n in reads:
            readers.setdefault(n, set()).add(index)
        instructions.append(replace(ins, dependencies=tuple(sorted(required))))
    extra = {}
    for previous, current in p.reuse_edges():
        extra.setdefault(current, set()).add(previous)
    p.instructions = [
        replace(i, dependencies=tuple(sorted(set(i.dependencies) | extra.get(n, set()))))
        for n, i in enumerate(instructions)
    ]
    p.encoding_storage = "arena"
    p.storage_regions = []
    p.validate(neuron_core(3).scratchpad_size)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("reference.npz", "hardware.json", "model.txt", "compilation.json", "generation.json"):
        shutil.copy2(source / name, destination / name)
    (destination / "nki").mkdir(exist_ok=True)
    (destination / "nki/program.py").write_text(emit(p))
    (destination / "instructions.json").write_text(json.dumps(p.record(), separators=(",", ":")))
    selected = analyze_selected(p, neuron_core(3))
    report = dict(
        experiment="Unique SBUF generations; unchanged ISA order, operands and PSUM banks",
        source=str(source.resolve()),
        source_instructions_sha256=hashlib.sha256((source / "instructions.json").read_bytes()).hexdigest(),
        original_sbuf_bytes=old_extent,
        distinct_sbuf_bytes=offset * 128,
        original_analysis=original,
        selected_analysis=selected,
    )
    (destination / "allocation-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    # Preserve the semantic/workspace ABI, but do not copy the source's stale
    # physical allocation or instruction-plan hashes into this experiment.
    selection = json.loads((source / "selection.json").read_text())
    selection["instructions_sha256"] = hashlib.sha256((destination / "instructions.json").read_bytes()).hexdigest()
    selection["physical_placement_strategy"] = "diagnostic_distinct_sbuf"
    selection["program_analysis"]["selected_instruction_analysis"] = selected
    selection["program_analysis"]["whole_program_prediction_ns"] = selected["prediction_ns"]
    selection["allocation_probe"] = report
    (destination / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (destination / "nki/plan.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(dict(case=source.name, **report)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["gemm128", "gemm512"])
    args = parser.parse_args()
    for case in args.cases:
        prepare(args.source / case, args.output / case)


if __name__ == "__main__":
    main()
