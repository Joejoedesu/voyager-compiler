"""Task-local adoption work record; no evaluator, compiler pass, or job launcher.

Run from a compiler checkout: python tools/adoption.py --help.
Edit the JSON with normal file tools; one writer owns it. Evidence remains in
referenced artifacts. Validation checks record shape, not the truth of claims.
"""

import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate(record):
    require(isinstance(record, dict), "record must be an object")
    require(record.get("version") == 1, "expected version 1")
    scope = record.get("scope", {})
    require(isinstance(scope, dict), "scope must be an object")
    require(scope.get("route") in {"registration", "binding"}, "unknown route")
    for key in ("target", "objective", "constraints"):
        require(nonempty(scope.get(key)), f"scope.{key} is required")
    if scope["route"] == "binding":
        require(nonempty(scope.get("workload")), "binding needs scope.workload")
    best = record.get("best")
    if best is not None:
        require(isinstance(best, dict), "best must be an object or null")
        require(nonempty(best.get("artifact")), "best needs an artifact identity")
        require(nonempty(best.get("claim")), "best needs its validated claim/scope")
        require(isinstance(best.get("evidence"), list) and best["evidence"]
                and all(nonempty(item) for item in best["evidence"]),
                "best needs evidence references")
        metric = best.get("metric")
        if metric is not None:
            require(isinstance(metric, dict), "metric must be an object")
            for key in ("name", "unit", "boundary"):
                require(nonempty(metric.get(key)), f"metric.{key} is required")
            require(metric.get("kind") in {"measured", "estimated"},
                    "metric.kind must be measured or estimated")
            value = metric.get("value")
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "metric.value must be finite and nonnegative")
    for group, states in (
        ("opportunities", {"open", "active", "resolved", "deferred"}),
        ("checks", {"pending", "running", "passed", "failed", "cancelled"}),
    ):
        rows = record.get(group)
        require(isinstance(rows, list), f"{group} must be a list")
        seen = set()
        for row in rows:
            require(isinstance(row, dict), f"{group} entries must be objects")
            for key in ("id", "question"):
                require(nonempty(row.get(key)), f"{group}.{key} is required")
            require(row["id"] not in seen, f"duplicate {group} id: {row['id']}")
            seen.add(row["id"])
            require(row.get("state") in states, f"invalid {group} state")
            evidence = row.get("evidence", [])
            require(isinstance(evidence, list) and all(nonempty(x) for x in evidence),
                    f"{group}.evidence must contain references")
            if group == "checks":
                require(nonempty(row.get("artifact")), "check needs artifact identity")
                if row["state"] in {"passed", "failed"}:
                    require(bool(evidence), "completed check needs evidence")
            elif row["state"] in {"open", "active"}:
                require(nonempty(row.get("next")), "open opportunity needs next action")
            else:
                require(nonempty(row.get("reason")), "closed opportunity needs reason")
    return record


def summarize(record):
    validate(record)
    scope = record["scope"]
    lines = [f"{scope['route']}: {scope['target']} / {scope.get('workload') or '-'}",
             f"Objective: {scope['objective']}", f"Constraints: {scope['constraints']}"]
    best = record.get("best")
    if best is None:
        lines.append("Best: no validated candidate recorded")
    else:
        lines.append(f"Best: {best['artifact']} — {best['claim']}")
        if metric := best.get("metric"):
            lines.append(f"  {metric['kind']} {metric['name']}: {metric['value']} "
                         f"{metric['unit']} ({metric['boundary']})")
        lines.append("  Evidence: " + ", ".join(best["evidence"]))
    for group in ("opportunities", "checks"):
        for row in record[group]:
            lines.append(f"{group}/{row['id']} [{row['state']}]: {row['question']}")
            for field in ("artifact", "owner", "job", "next", "reason"):
                if row.get(field):
                    lines.append(f"  {field}: {row[field]}")
            if row.get("evidence"):
                lines.append("  evidence: " + ", ".join(row["evidence"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a task record without overwriting")
    init.add_argument("path", type=Path)
    init.add_argument("--route", choices=("registration", "binding"), required=True)
    init.add_argument("--target", required=True)
    init.add_argument("--workload")
    init.add_argument("--objective", required=True)
    init.add_argument("--constraints", required=True)
    for name in ("check", "show"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init":
            record = validate({
                "version": 1,
                "scope": {key: getattr(args, key) for key in
                          ("route", "target", "workload", "objective", "constraints")},
                "best": None, "opportunities": [], "checks": [],
            })
            args.path.parent.mkdir(parents=True, exist_ok=True)
            with args.path.open("x") as stream:
                stream.write(json.dumps(record, indent=2) + "\n")
        else:
            record = validate(json.loads(args.path.read_text()))
        print(summarize(record) if args.command == "show" else f"Record valid: {args.path}")
    except (ValueError, OSError) as error:
        parser.exit(1, f"adoption: {error}\n")


if __name__ == "__main__":
    main()
