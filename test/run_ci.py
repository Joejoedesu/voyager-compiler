#!/usr/bin/env python
"""Local pre-push CI for the voyager codegen commands.

Runs every actively-used ``test_codegen.py`` invocation (the ones reached
through the accelerator's ``codegen.mk``) without dumping tensors, and drops
each run's artifacts into a date+time folder under a user-supplied output
location.  If a prior run exists in that location, the freshly produced
``model.txt`` of each command is compared against the previous run's and any
mismatch (or compile failure) is reported.

Usage (from the repo root, with the conda env active)::

    python test/run_ci.py <output_dir>

Exits non-zero if any command fails to compile or any ``model.txt`` differs
from the previous run, so it can gate a pre-push hook.
"""

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from threading import Event

# Repo root = parent of this file's directory (test/run_ci.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CODEGEN = REPO_ROOT / "test" / "test_codegen.py"

# Number of unified-diff lines shown in the report for a mismatch.
DIFF_EXCERPT_LINES = 60

# ---------------------------------------------------------------------------
# Command table.
#
# Each Command is expanded at runtime into a test_codegen.py argv:
#     <python> test_codegen.py <model> --target_hardware <target>
#         --quantization_recipe <scheme>
#         --pe_array_size <unrolling> <extra>
#         --model_output_dir <run_dir>/<label>
# Voyager presets live in quantization.voyager; recipes supplies family lookup.
# To add coverage,
# add a Command and, if needed, register the target/family/backend.
# ---------------------------------------------------------------------------

# Recipes are resolved by the selected hardware family inside test_codegen.
# The CI matrix owns only model, target, recipe, geometry and case overrides.

# Reused per-command extra-flag groups.
_SINGLE = "--num_hidden_layers 1"
_LLM = "--context_length 1024 --num_hidden_layers 1 --quantize_attention_mask"
_LLM_MP = _LLM + " --qconfig mxnf4_attn_head_int6"
_LLM_SPMM = _LLM + " --qconfig mxnf4_outlier"
_DB = "--double_buffered_l2"
_LLM_DB = _LLM + " " + _DB
_LLM_SPMM_DB = _LLM_SPMM + " " + _DB


@dataclass(frozen=True)
class Command:
    """One supported model × hardware × quantization invocation."""

    model: str  # test_codegen.py positional argument
    scheme: str  # key into the family recipe registry
    # Voyager --pe_array_size; optional for other targets.
    unrolling: str | None = None
    network: str = ""  # output/label name (defaults to model)
    extra: str = ""  # any per-command extra flags
    target_hardware: str = "voyager"


COMMANDS = [
    # -- E4M3 --
    Command("resnet18", "E4M3", "16,16"),
    Command("resnet18", "E4M3", "32,64"),
    Command("resnet18", "E4M3", "4,8"),
    Command("resnet50", "E4M3", "32,64"),
    Command("mobilebert", "E4M3", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilebert", "E4M3", "4,8", "mobilebert_encoder", _SINGLE),
    # -- P8_1 --
    Command("resnet18", "P8_1", "16,16"),
    Command("mobilebert", "P8_1", "16,16", "mobilebert_encoder", _SINGLE),
    # -- INT8 --
    Command("resnet18", "INT8", "16,16"),
    Command("mobilebert", "INT8", "16,16", "mobilebert_encoder", _SINGLE),
    # -- MXINT8 --
    Command("resnet18", "MXINT8", "16,16"),
    Command("mobilebert", "MXINT8", "16,16", "mobilebert_encoder", _SINGLE),
    Command("mobilenet_v2", "MXINT8", "16,16"),
    # -- MXNF4 (llama) --
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill", _LLM),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_mp", _LLM_MP),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_spmm", _LLM_SPMM),
    Command("llama_decode", "MXNF4", "64,64", "llama_decode", _LLM),
    Command("llama_decode_kivi", "MXNF4", "64,64", "llama_decode_kivi", _LLM),
    # -- MXNF4 (vision / bert) --
    Command("resnet18", "MXNF4", "64,64"),
    Command("resnet50", "MXNF4", "64,64"),
    Command("vit", "MXNF4", "64,64"),
    Command("bert", "MXNF4", "64,64"),
    # -- double-buffered L2 (conv / prefill / decode / sparse prefill each
    # pipeline apart) --
    Command("resnet18", "MXNF4", "64,64", "resnet18_db", _DB),
    Command("llama_prefill", "MXNF4", "64,64", "llama_prefill_db", _LLM_DB),
    Command("llama_decode", "MXNF4", "64,64", "llama_decode_db", _LLM_DB),
    Command(
        "llama_prefill", "MXNF4", "64,64", "llama_prefill_spmm_db", _LLM_SPMM_DB
    ),
]

# Timestamp folder format; lexicographic sort == chronological sort.
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _label(command):
    """``<target>/<network>/<scheme>/<unrolling>`` — unique per command (unrolling
    disambiguates commands sharing a network/scheme, e.g. resnet18 E4M3)."""
    network = command.network or command.model
    unroll = (
        command.unrolling.replace(",", "x") if command.unrolling else "default"
    )
    return f"{command.target_hardware}/{network}/{command.scheme}/{unroll}"


assert len({_label(c) for c in COMMANDS}) == len(COMMANDS), (
    "COMMANDS have duplicate <target>/<network>/<scheme>/<unrolling> labels; give "
    "colliding commands distinct 'network' names"
)


def _matches(label, pattern):
    """Does ``pattern`` select ``label`` (``network/scheme/unrolling``)?

    Matching is anchored at ``/`` segment boundaries, not raw substring: a
    single token matches a whole segment by prefix (so ``bert`` selects
    ``bert`` but not ``mobilebert_encoder``, while ``mobilebert`` still
    selects ``mobilebert_encoder`` and ``resnet`` selects both resnets); a
    ``a/b`` token matches as an anchored path prefix.
    """
    pattern = pattern.lower().strip("/")
    if (
        label.startswith("voyager/")
        and "/" in pattern
        and not pattern.startswith("voyager/")
    ):
        label = label.removeprefix("voyager/")
    segs = label.lower().split("/")
    parts = pattern.split("/")
    if len(parts) == 1:
        return any(s.startswith(parts[0]) for s in segs)
    return len(parts) <= len(segs) and all(
        segs[i].startswith(parts[i]) for i in range(len(parts))
    )


def _build(command, run_dir, threads_per_job=None):
    """Expand a Command into ``(label, dest, argv)`` for this run.

    The argv runs this repo's test_codegen.py with --model_output_dir pointed
    into the timestamped run folder; --dump_tensors is never added.
    ``--debug`` is always added: without it test_codegen never evaluates the
    lowered graph, so the run checks codegen only and no numeric comparison
    happens at all.
    """
    label = _label(command)
    dest = run_dir / label

    argv = [sys.executable, str(TEST_CODEGEN), command.model, "--debug"]
    argv += [
        "--target_hardware",
        command.target_hardware,
        "--quantization_recipe",
        command.scheme,
    ]
    if command.unrolling is not None:
        argv += ["--pe_array_size", command.unrolling]
    argv += shlex.split(command.extra)
    if threads_per_job is not None:
        argv += ["--num_threads", str(threads_per_job)]
    argv += ["--model_output_dir", str(dest)]
    return label, dest, argv


def _run_one(label, dest, argv, stop=None):
    """Run one command; capture combined output to ``dest/run.log``.

    Returns a status string: ``ok`` / ``error`` (nonzero exit) /
    ``no_output`` (exited 0 but no model.txt) / ``numeric_drift`` (the
    lowered graph's output left test_codegen's tolerance) / ``unverified``
    (the model ran no comparison despite --debug).  test_codegen only warns
    about the last two, so they have to be read back out of the log; neither
    fails the run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    log_path = dest / "run.log"
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(shlex.quote(a) for a in argv) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while proc.poll() is None:
                if stop is not None and stop.is_set():
                    return "error"
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    if proc.returncode != 0:
        return "error"
    if not (dest / "model.txt").exists():
        return "no_output"
    log = log_path.read_text()
    if "Skipping output verification" in log:
        return "unverified"
    if "Results match" not in log:
        return "numeric_drift"
    return "ok"


def _run_cases(commands, run_dir, prev_run, jobs=1, threads_per_job=None):
    """Execute independent cases concurrently; keep report order deterministic."""
    stop = Event()
    results = [None] * len(commands)

    def run(command):
        if stop.is_set():
            return None
        label, dest, argv = _build(command, run_dir, threads_per_job)
        print(f"starting {label}", flush=True)
        status = _run_one(label, dest, argv, stop)
        verdict, excerpt = _compare(label, status, run_dir, prev_run)
        return label, status, verdict, excerpt

    pool = ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = {
            pool.submit(run, command): idx
            for idx, command in enumerate(commands)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[futures[future]] = result
            label, status, verdict, _ = result
            suffix = "" if status == "ok" else f" ({status})"
            print(
                f"[{completed}/{len(commands)}] {label}: {verdict}{suffix}",
                flush=True,
            )
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def _find_previous_run(out_dir, current):
    """Newest timestamp dir under ``out_dir`` that isn't ``current``."""
    runs = sorted(
        p.name
        for p in out_dir.iterdir()
        if p.is_dir() and TS_RE.match(p.name) and p.name != current
    )
    return out_dir / runs[-1] if runs else None


def _compare(label, status, run_dir, prev_run):
    """Classify one label's model.txt vs the previous run.

    Returns ``(verdict, diff_excerpt)``.  ``diff_excerpt`` is non-empty only
    for a MISMATCH.
    """
    if status in ("error", "no_output"):
        return "FAILED", ""

    cur = (run_dir / label / "model.txt").read_text()

    if prev_run is None:
        return "NEW", ""
    prev_path = prev_run / label / "model.txt"
    if not prev_path.exists() and label.startswith("voyager/"):
        prev_path = prev_run / label.removeprefix("voyager/") / "model.txt"
    if not prev_path.exists():
        return "NEW", ""

    prev = prev_path.read_text()
    if cur == prev:
        return "MATCH", ""

    diff = difflib.unified_diff(
        prev.splitlines(),
        cur.splitlines(),
        fromfile=f"{prev_run.name}/{label}/model.txt",
        tofile=f"{run_dir.name}/{label}/model.txt",
        lineterm="",
    )
    excerpt = list(diff)[:DIFF_EXCERPT_LINES]
    return "MISMATCH", "\n".join(excerpt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Location for dated CI artifacts (created if missing).",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SUBSTR",
        help="Run only commands whose <network>/<scheme>/<unrolling> label "
        "contains SUBSTR (case-insensitive; repeatable).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List command labels and exit.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against this fixed timestamp directory instead of the latest run.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="File of exact case labels (legacy Voyager labels accepted).",
    )
    parser.add_argument(
        "--target_hardware",
        action="append",
        help="Select target hardware (repeatable).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent model processes (default: 1).",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        help="PyTorch CPU threads per process; parallel runs default to an equal share of available CPUs, capped at 32.",
    )
    parser.add_argument("--bufferized-flow", choices=("per_kernel", "resident"))
    parser.add_argument("--parameter-loading", choices=("preload", "on_demand"))
    args = parser.parse_args()
    if args.jobs < 1 or (
        args.threads_per_job is not None and args.threads_per_job < 1
    ):
        parser.error("jobs and threads-per-job must be positive")

    commands = COMMANDS
    if args.only:
        commands = [
            c
            for c in COMMANDS
            if any(_matches(_label(c), p) for p in args.only)
        ]

    if args.target_hardware:
        commands = [
            c for c in commands if c.target_hardware in args.target_hardware
        ]
    if args.suite:
        requested = {
            line.split("#", 1)[0].strip()
            for line in args.suite.read_text().splitlines()
        } - {""}
        available = {_label(c): c for c in COMMANDS}
        canonical = {
            label if label in available else "voyager/" + label
            for label in requested
        }
        if missing := canonical - available.keys():
            parser.error(f"Unknown suite labels: {sorted(missing)}")
        commands = [c for c in commands if _label(c) in canonical]
    if not commands:
        parser.error("No compilation cases matched the selection")
    if (
        args.parameter_loading == "preload"
        and args.bufferized_flow != "resident"
    ):
        parser.error("preload requires --bufferized-flow resident")
    overrides = ""
    if args.bufferized_flow:
        overrides += " --bufferized_flow " + args.bufferized_flow
    if args.parameter_loading:
        overrides += " --parameter_loading " + args.parameter_loading
    if overrides:
        commands = [replace(c, extra=c.extra + overrides) for c in commands]
    jobs = min(args.jobs, len(commands))
    threads_per_job = args.threads_per_job
    if threads_per_job is None and jobs > 1:
        cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        threads_per_job = max(1, min(32, cpus // jobs))

    if args.list:
        for command in commands:
            _, _, argv = _build(command, Path("<run_dir>"), threads_per_job)
            print(" ".join(shlex.quote(a) for a in argv))
        return 0

    if not args.output_dir:
        parser.error("output_dir is required (unless --list)")
    if not commands:
        parser.error(f"--only {args.only} matched no commands")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(TS_FORMAT)
    run_dir = out_dir / timestamp
    run_dir.mkdir()

    prev_run = (
        args.baseline.resolve()
        if args.baseline
        else _find_previous_run(out_dir, timestamp)
    )
    if args.baseline and not (prev_run / "report.txt").is_file():
        parser.error(f"Baseline is missing a completed report: {prev_run}")

    print(f"voyager codegen CI -> {run_dir}")
    if prev_run is not None:
        print(f"comparing model.txt against previous run: {prev_run.name}")
    else:
        print("no previous run found; this run is the baseline")
    print(
        f"running {len(commands)} commands with {jobs} workers ({threads_per_job or 32} CPU threads each)\n"
    )

    results = _run_cases(commands, run_dir, prev_run, jobs, threads_per_job)

    report = _build_report(results, run_dir, prev_run)
    (run_dir / "report.txt").write_text(report)
    print("\n" + report)

    # Gate: fail on any compile failure or model.txt mismatch.
    bad = [r for r in results if r[2] in ("FAILED", "MISMATCH")]
    return 1 if bad else 0


def _build_report(results, run_dir, prev_run):
    """Render the human-readable summary written to report.txt + stdout."""
    counts = {}
    for _, _, verdict, _ in results:
        counts[verdict] = counts.get(verdict, 0) + 1

    lines = []
    lines.append("=" * 70)
    lines.append(f"voyager codegen CI report  ({run_dir.name})")
    if prev_run is not None:
        lines.append(f"compared against: {prev_run.name}")
    else:
        lines.append("compared against: (none - baseline run)")
    lines.append("=" * 70)

    order = ["MATCH", "NEW", "MISMATCH", "FAILED", "MISSING"]
    summary = "  ".join(f"{v}={counts[v]}" for v in order if v in counts)
    lines.append(f"totals: {summary}")
    lines.append("")

    # Detail any non-clean verdicts.
    for label, status, verdict, excerpt in results:
        if verdict in ("MATCH", "NEW"):
            continue
        detail = f"  - {verdict}: {label}"
        if status != "ok":
            detail += f" [{status}; see {label}/run.log]"
        lines.append(detail)
        if excerpt:
            for dl in excerpt.splitlines():
                lines.append(f"      {dl}")

    # Numeric warnings: reported, never gated.
    warned = [r for r in results if r[1] in ("numeric_drift", "unverified")]
    if warned:
        lines.append("")
        lines.append("numeric warnings (not gated):")
        for label, status, _, _ in warned:
            lines.append(f"  - {status}: {label} [see {label}/run.log]")

    # Labels present in the previous run but absent now.
    if prev_run is not None:
        current_labels = {r[0] for r in results}
        for p in sorted(prev_run.rglob("model.txt")):
            label = str(p.parent.relative_to(prev_run))
            canonical = (
                "voyager/" + label
                if len(p.parent.relative_to(prev_run).parts) == 3
                else label
            )
            if canonical not in current_labels:
                lines.append(f"  - MISSING: {label} (in {prev_run.name})")

    bad = [r for r in results if r[2] in ("FAILED", "MISMATCH")]
    lines.append("")
    lines.append("RESULT: FAIL" if bad else "RESULT: PASS")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
