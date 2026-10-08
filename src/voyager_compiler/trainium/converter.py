"""Deterministic formal conversion of a validated selected instruction plan.

This module never invokes target lowering, scheduling, transfer subdivision or
allocation. Regenerate collaterals with compile() when the plan is absent/stale.
"""

import hashlib
import json
from pathlib import Path
from .instruction_plan import Program
from .plan_emitter import emit


def convert(root, output=None, target=None, *, context=None):
    from .hardware import TARGETS, neuron_core
    from voyager_compiler.compilation import CompilerContext

    root = Path(root)
    if context is None:
        recorded = json.loads((root / "hardware.json").read_text())["hardware"][
            "name"
        ]
        context = CompilerContext.from_artifacts(
            root, neuron_core(TARGETS[recorded])
        )
    if target is not None and target != context.hardware.name:
        raise ValueError("Target differs from recorded compiler context")
    context.check_artifacts(root)
    if not context.policy.tuning.isa_lowering:
        if context.policy.options().get("isa_lowering") is not False:
            raise ValueError(
                "Legacy conversion requires explicit isa_lowering=False; "
                "recompile artifacts with an incomplete instruction policy"
            )
        from .legacy_converter import convert as legacy

        return legacy(root, output, target, context=context)
    if not (root / "instructions.json").is_file():
        raise ValueError(
            "Missing selected ISA plan instructions.json; compile before conversion"
        )
    manifest = json.loads((root / "selection.json").read_text())
    if (
        hashlib.sha256((root / "hardware.json").read_bytes()).hexdigest()
        != manifest["hardware_record_sha256"]
    ):
        raise ValueError(
            "Selected dependency templates or hardware metadata changed after instruction selection"
        )
    for filename, field in (
        ("model.txt", "source_sha256"),
        ("instructions.json", "instructions_sha256"),
    ):
        if (
            hashlib.sha256((root / filename).read_bytes()).hexdigest()
            != manifest[field]
        ):
            raise ValueError(f"Selected plan does not match {filename}")
    program = Program.load(json.loads((root / "instructions.json").read_text()))
    if (program.encoding_storage != "compiler") != context.policy.tuning.strict_realization:
        raise ValueError("Selected physical allocation differs from strict_realization policy")
    from dataclasses import asdict

    contracts = {c.name: asdict(c) for c in context.hardware.isa_instructions}
    if json.dumps(contracts, sort_keys=True) != json.dumps(
        program.contracts, sort_keys=True
    ):
        raise ValueError("Selected ISA implementations differ from hardware IR")
    source = emit(program)
    output = Path(output) if output else root / "nki"
    output.mkdir(parents=True, exist_ok=True)
    (output / "program.py").write_text(source)
    (output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
