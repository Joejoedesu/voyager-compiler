"""Rank complete bufferized candidates after physical ISA expansion and placement.

No native compiler, hardware timings or compact-cost shortlist enter selection.
This is a bounded search: largest legal row divisors, invariant residency, and
uniform matrix orientation. Region boundaries, layouts and mixed per-operation
orientations are not silently claimed as searched dimensions.
"""
import gc
import hashlib
import shutil
import itertools
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from voyager_compiler.codegen.transform.bufferize.residency import clone_graph
from voyager_compiler.codegen.transform.bufferize.stream_regions import plan_stream_regions
from voyager_compiler.compilation import CompilerContext


def enumerate_candidates(records, *, row_count, orientations, budget=None):
    """Capacity pruning only; compact prediction never orders the candidates."""
    choices, coverage = [], []
    for record in records:
        legal = [c for c in record.get("candidates", []) if c.get("legal")]
        if not legal:
            choices.append([None])
            coverage.append(dict(status="fallback", reason=record.get("reason"), candidate_count=1))
            continue
        rows = sorted({c["tile_rows"] for c in legal}, reverse=True)
        retained = rows[:row_count]
        selected = sorted(
            [dict(tile_rows=c["tile_rows"], weight_residency=c["weight_residency"])
             for c in legal if c["tile_rows"] in retained],
            key=lambda c: (-c["tile_rows"], c["weight_residency"]),
        )
        choices.append(selected)
        coverage.append(dict(candidate_count=len(selected), legal_rows=rows, expanded_rows=retained,
                             omitted_rows=rows[row_count:]))
    # Orientation is the innermost loop: a tight budget still compares both
    # orientations for the same logical schedule before moving to another tile.
    specs = (dict(regions=list(combo), orientation=o)
             for combo in itertools.product(*choices) for o in orientations)
    return list(specs if budget is None else itertools.islice(specs, budget)), coverage


def choose_winner(records):
    valid = [r for r in records if r["status"] == "valid"
             and math.isfinite(r["physical_prediction_ns"])]
    if not valid:
        raise ValueError("No region candidate passed physical ISA placement and analysis")
    return min(valid, key=lambda r: (r["physical_prediction_ns"], r["id"]))


def compile_expanded_regions(backend, model, args, kwargs=None, **options):
    root = Path(options["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    buffers = options["bufferization_options"]
    context = options.get("context") or CompilerContext.resolve(
        options["config"], options.get("mapping_policy"),
        cost_tradeoff=options.get("interstellar_cost_tradeoff", True),
        runtime_tolerance=options.get("runtime_tolerance"),
    )
    if not context.policy.tuning.isa_lowering or not context.policy.tuning.strict_realization:
        raise ValueError("Expanded region search requires strict physical ISA realization")
    options = dict(options, context=context)
    discovery = clone_graph(model)
    discovery.meta.pop("stream_regions", None)
    plan_stream_regions(discovery, SimpleNamespace(mapping_policy=context.policy),
                        discovery_only=True)
    records = discovery.meta.get("stream_regions", [])
    orientation = context.policy.tuning.matmul_orientation
    orientations = ("weights", "activations") if orientation == "auto" else (orientation,)
    specs, coverage = enumerate_candidates(
        records, row_count=buffers.stream_region_row_candidates, orientations=orientations,
        budget=buffers.stream_region_search_budget
    )
    if not any(r["status"] == "selected" for r in records):
        # Ordinary lowering is still a valid fallback, not a claimed region win.
        from dataclasses import replace
        return backend.compile(model, args, kwargs, **dict(
            options, bufferization_options=replace(buffers, stream_region_search="compact")
        ))
    enumerated_count = math.prod(c["candidate_count"] for c in coverage) * len(orientations)
    report = dict(
        objective="minimum selected physical ISA prediction after placement",
        selection_uses_compact_prediction=False, selection_uses_measurement=False,
        native_compilation_required=False,
        scope="whole program; fixed maximal regions; uniform matrix orientation",
        row_coverage=coverage, enumerated_candidates=enumerated_count,
        budget=buffers.stream_region_search_budget,
        row_candidate_count=buffers.stream_region_row_candidates,
        omitted_by_budget=max(0, enumerated_count-buffers.stream_region_search_budget),
        limitations=["Only the configured largest legal row divisors are expanded.",
                     "No selective HBM cuts, feature-chunk reductions, or mixed matrix orientations.",
                     "Existing physical timing laws retain incomplete completion terms."],
        candidates=[],
    )
    def save():
        (root / "region-search.json").write_text(json.dumps(report, indent=2)+"\n")
    save()
    start = time.monotonic()
    for index, spec in enumerate(specs[:buffers.stream_region_search_budget]):
        path = root / "region-candidates" / f"{index:03d}"
        trial = clone_graph(model)
        trial.meta.pop("stream_regions", None)
        trial.meta["stream_region_choices"] = spec["regions"]
        trial.meta["stream_search_trial"] = spec
        record = dict(id=index, spec=spec, path=str(path))
        tick = time.monotonic()
        try:
            backend.compile(trial, args, kwargs, **dict(
                options, output_dir=path, dump_tensors=False, before_emit=None
            ))
            manifest = json.loads((path / "selection.json").read_text())
            physical = manifest["program_analysis"]["selected_instruction_analysis"]
            hw = json.loads((path / "hardware.json").read_text())
            score = physical["prediction_ns"]
            if not math.isfinite(score):
                raise ValueError("Non-finite physical prediction")
            record.update(
                status="valid", physical_prediction_ns=score,
                compact_reference_prediction_ns=sum(r["selected"]["prediction_ns"]
                                          for r in hw["stream_regions"] if r["status"] == "selected"),
                compact_reference_orientation=context.policy.tuning.matmul_orientation,
                stats=manifest["stats"], instruction_count=physical["instruction_count"],
                hbm_bytes=physical["hbm_bytes"],
                unknown_completion_count=len(physical["unknown_completion"]),
                instructions_sha256=manifest["instructions_sha256"],
                sbuf_reserved_bytes=manifest["temporary_buffering"]["sbuf_reserved_bytes"],
            )
        except (ValueError, RuntimeError, NotImplementedError, AssertionError) as exc:
            record.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.monotonic() - tick
        report["candidates"].append(record)
        print("REGION CANDIDATE", index, record["status"],
              record.get("physical_prediction_ns", record.get("reason")), flush=True)
        save()
        del trial
        gc.collect()
    winner = choose_winner(report["candidates"])
    report["selected"] = winner
    report["search_seconds"] = time.monotonic()-start
    save()  # Persist prediction-only selection before final emission/hardware.
    model.meta["stream_region_choices"] = winner["spec"]["regions"]
    model.meta["stream_search_trial"] = winner["spec"]
    model.meta["stream_selected_plan_root"] = winner["path"]
    result = backend.compile(model, args, kwargs, **options)
    final = json.loads((root / "selection.json").read_text())
    if final["instructions_sha256"] != winner["instructions_sha256"]:
        raise ValueError("Final region realization differs from the scored physical ISA")
    report["final_instructions_sha256"] = final["instructions_sha256"]
    report["realization_matches_scored_candidate"] = True
    save()
    return result


def reuse_scored_plan(root, source, context):
    """Reuse a selected physical program only for identical emitted inputs.

    This is finalization within a search, not a cross-version compiler cache.
    The caller additionally checks the selected instruction digest against the
    prediction record. Formal conversion still validates the copied program.
    """
    root, source = Path(root), Path(source)
    manifest = json.loads((source / "selection.json").read_text())
    if context.policy.tuning.physical_model != "baseline":
        from .physical_context import identity
        saved=manifest["program_analysis"]["selected_instruction_analysis"].get("physical_model_record")
        if saved != identity(context.policy.tuning.physical_model):
            raise ValueError("Scored physical model contents differ from final compilation")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest["compiler"] != context.record():
        raise ValueError("Scored plan compiler context differs from final compilation")
    for filename, key in (("model.txt", "source_sha256"),
                          ("hardware.json", "hardware_record_sha256")):
        if digest(source / filename) != manifest[key] or digest(root / filename) != manifest[key]:
            raise ValueError(f"Final compilation differs from scored {filename}")
    if digest(source / "instructions.json") != manifest["instructions_sha256"]:
        raise ValueError("Scored physical instruction artifact changed")
    for name in ("instructions.json", "selection.json", "movement-search.json"):
        if (source / name).exists():
            shutil.copy2(source / name, root / name)
    return manifest
