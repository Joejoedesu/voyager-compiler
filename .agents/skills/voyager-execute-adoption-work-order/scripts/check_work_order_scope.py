#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Gate a work-order handoff against its declared repository write scope."""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

import yaml

BASELINE_SCHEMA_VERSION = 1
HANDOFF_SCHEMA_VERSION = 4
ROLE_CONTRACT_SCHEMA_VERSION = 1
WORK_ORDER_SCHEMA_VERSION = 5
ROLE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
ROLE_CONTRACT_DIRECTORY = "docs/backend-adoption/roles"
REGISTRATION_MODE = "register_target"
BINDING_MODE = "bind_workload_target"
INNER_CONTROL_MODE = "deepen_inner_control"
ADOPTION_MODES = frozenset(
    {REGISTRATION_MODE, BINDING_MODE, INNER_CONTROL_MODE}
)
X_STEPS = ("bit_accurate_core_lowering", "inner_control")
REFERENCE_FIDELITY = "reference_tolerance"
OUTPUT_CODE_LSB_FIDELITY = "output_code_lsb"
EXACT_FIDELITY = "bit_exact_endpoint"
CONTROL_CONSUMPTION = "control_consumption"
EVIDENCE_BOUNDARIES = frozenset(
    {
        "serialization",
        "backend_translation",
        "operation_execution",
        "connected_execution",
        "performance",
        CONTROL_CONSUMPTION,
    }
)


class ScopeError(Exception):
    pass


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _run_git(repository, arguments, allow_failure=False):
    process = subprocess.Popen(
        ["git", "-C", str(repository)] + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    if process.returncode and not allow_failure:
        raise ScopeError(
            "git command failed ({}): {}".format(
                " ".join(arguments),
                stderr.decode("utf-8", "replace").strip(),
            )
        )
    return process.returncode, stdout, stderr


def _load_yaml(path, label):
    try:
        with path.open("r") as source:
            value = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as error:
        raise ScopeError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise ScopeError(
            "{} must contain one YAML mapping: {}".format(label, path)
        )
    return value


def _require_schema_version(value, expected, label):
    actual = value.get("schema_version")
    if type(actual) is not int or actual != expected:
        raise ScopeError(
            "{} schema_version must be {}: got {!r}".format(
                label, expected, actual
            )
        )


def _require_mapping(value, key, label):
    result = value.get(key)
    if not isinstance(result, dict):
        raise ScopeError("{} requires a {} mapping".format(label, key))
    return result


def _require_concrete_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ScopeError("{} must be a concrete string".format(label))
    result = value.strip()
    if "<" in result or ">" in result:
        raise ScopeError("{} contains an unresolved placeholder".format(label))
    return result


def _require_optional_string(value, label):
    if value is None:
        return None
    return _require_concrete_string(value, label)


def _require_string_list(value, label, allow_empty=True):
    if not isinstance(value, list):
        raise ScopeError("{} must be a list".format(label))
    if not value and not allow_empty:
        raise ScopeError("{} must not be empty".format(label))
    result = []
    for index, item in enumerate(value):
        result.append(
            _require_concrete_string(item, "{}[{}]".format(label, index))
        )
    if len(result) != len(set(result)):
        raise ScopeError("{} must not contain duplicates".format(label))
    return result


def _require_tolerance(value, label):
    if not isinstance(value, dict):
        raise ScopeError("{} must be a mapping".format(label))
    required_keys = {"absolute", "relative"}
    if set(value) != required_keys:
        raise ScopeError(
            "{} must contain exactly absolute and relative".format(label)
        )
    result = {}
    for key in sorted(required_keys):
        item = value[key]
        if type(item) not in (int, float) or item < 0:
            raise ScopeError(
                "{}.{} must be a nonnegative number".format(label, key)
            )
        result[key] = item
    return result


def _parse_adoption(value, label, require_intended_claim):
    adoption = _require_mapping(value, "adoption", label)
    mode = adoption.get("mode")
    if mode not in ADOPTION_MODES:
        raise ScopeError(
            "{}.adoption.mode must be one of {}".format(
                label, ", ".join(sorted(ADOPTION_MODES))
            )
        )
    claim_id = adoption.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ScopeError("{}.adoption.claim_id must be concrete".format(label))
    if require_intended_claim:
        intended = adoption.get("intended_claim")
        if not isinstance(intended, str) or not intended.strip():
            raise ScopeError(
                "{}.adoption.intended_claim must be concrete".format(label)
            )
    x_step = adoption.get("x_step")
    if mode == INNER_CONTROL_MODE:
        if x_step not in X_STEPS:
            raise ScopeError(
                "{}.adoption.x_step must be one of {}".format(
                    label, ", ".join(X_STEPS)
                )
            )
    elif x_step is not None:
        raise ScopeError(
            "{}.adoption.x_step must be null outside "
            "deepen_inner_control".format(label)
        )
    return {"mode": mode, "claim_id": claim_id.strip(), "x_step": x_step}


def _validate_owned_output(work_order):
    _require_concrete_string(
        work_order.get("objective"), "work order.objective"
    )
    outputs = work_order.get("owned_outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ScopeError(
            "work order.owned_outputs must contain exactly one output"
        )
    output = outputs[0]
    if not isinstance(output, dict):
        raise ScopeError("work order.owned_outputs[0] must be a mapping")
    _require_concrete_string(
        output.get("artifact"), "work order.owned_outputs[0].artifact"
    )
    _normalize_relative_path(
        output.get("path"), "work order.owned_outputs[0].path"
    )
    _require_concrete_string(
        output.get("consumer"), "work order.owned_outputs[0].consumer"
    )


def _validate_run_guide(work_order, adoption):
    guide = _require_mapping(work_order, "run_guide", "work order")
    _require_concrete_string(guide.get("path"), "run_guide.path")
    _require_concrete_string(
        guide.get("revision_or_hash"), "run_guide.revision_or_hash"
    )
    registration = _require_optional_string(
        guide.get("registration_claim_id"),
        "run_guide.registration_claim_id",
    )
    if adoption["mode"] == REGISTRATION_MODE:
        if registration is not None:
            raise ScopeError(
                "run_guide.registration_claim_id must be null for a "
                "register_target order, which establishes it"
            )
    elif registration is None:
        raise ScopeError(
            "run_guide.registration_claim_id must name the R claim that "
            "established the target configuration and injection contract"
        )


def _validate_acceptance(work_order, adoption):
    acceptance = _require_mapping(work_order, "acceptance", "work order")
    _require_concrete_string(acceptance.get("oracle"), "acceptance.oracle")
    _require_concrete_string(
        acceptance.get("comparison_scope"), "acceptance.comparison_scope"
    )
    tolerance = _require_tolerance(
        acceptance.get("tolerance"), "acceptance.tolerance"
    )
    fidelity = acceptance.get("fidelity")
    evidence = _require_string_list(
        acceptance.get("required_evidence"), "acceptance.required_evidence"
    )
    unknown = sorted(set(evidence) - EVIDENCE_BOUNDARIES)
    if unknown:
        raise ScopeError(
            "acceptance.required_evidence has unknown boundaries: {}".format(
                ", ".join(unknown)
            )
        )

    inner = _require_mapping(work_order, "inner_control", "work order")
    decisions = _require_string_list(
        inner.get("voyager_owned_decisions"),
        "inner_control.voyager_owned_decisions",
    )
    consumed = _require_string_list(
        inner.get("endpoint_consumed_fields"),
        "inner_control.endpoint_consumed_fields",
    )
    mode = adoption["mode"]
    has_control_evidence = CONTROL_CONSUMPTION in evidence
    control_parts = (bool(decisions), bool(consumed), has_control_evidence)
    if (
        mode != INNER_CONTROL_MODE
        and any(control_parts)
        and not all(control_parts)
    ):
        raise ScopeError(
            "mapping-control evidence requires Voyager-owned decisions, "
            "endpoint-consumed fields, and control_consumption together"
        )

    if mode != INNER_CONTROL_MODE:
        expected_fidelity = (
            REFERENCE_FIDELITY
            if mode == REGISTRATION_MODE
            else OUTPUT_CODE_LSB_FIDELITY
        )
        if fidelity != expected_fidelity:
            raise ScopeError(
                "{} requires acceptance.fidelity={}".format(
                    mode, expected_fidelity
                )
            )
        if mode == BINDING_MODE and tolerance != {
            "absolute": 1,
            "relative": 0,
        }:
            raise ScopeError(
                "bind_workload_target requires acceptance.tolerance "
                "absolute=1 and relative=0"
            )
        # R and W may use either a partial- or full-control injection point.
        # A complete declaration records mapping-control evidence without
        # classifying the point or promoting the achieved stage.
        return

    if fidelity != EXACT_FIDELITY:
        raise ScopeError(
            "deepen_inner_control requires acceptance.fidelity={}".format(
                EXACT_FIDELITY
            )
        )
    if any(value != 0 for value in tolerance.values()):
        raise ScopeError(
            "deepen_inner_control requires zero acceptance.tolerance"
        )

    if adoption["x_step"] == "bit_accurate_core_lowering":
        if any(control_parts):
            raise ScopeError(
                "a bit_accurate_core_lowering order establishes exactness "
                "only; it cannot claim mapping-control evidence"
            )
        return

    if not decisions:
        raise ScopeError(
            "an inner_control order requires "
            "inner_control.voyager_owned_decisions"
        )
    if not consumed:
        raise ScopeError(
            "an inner_control order requires "
            "inner_control.endpoint_consumed_fields"
        )
    if CONTROL_CONSUMPTION not in evidence:
        raise ScopeError(
            "an inner_control order requires {} evidence proving the "
            "endpoint consumed each claimed decision".format(
                CONTROL_CONSUMPTION
            )
        )


def _parse_role_contract(repository, work_order):
    role_id = work_order.get("role_id")
    if not isinstance(role_id, str) or not ROLE_ID_PATTERN.fullmatch(role_id):
        raise ScopeError("work order role_id must be a concrete kebab-case ID")

    contract = _normalize_relative_path(
        work_order.get("role_contract"), "work order role_contract"
    )
    expected = "{}/{}.md".format(ROLE_CONTRACT_DIRECTORY, role_id)
    if contract != expected:
        raise ScopeError(
            "work order role_contract must be {} for role_id {}".format(
                expected, role_id
            )
        )

    contract_scope = {"path": contract, "subtree": False}
    _validate_scope_location(repository, [contract_scope], "role contract path")
    contract_path = repository / contract
    try:
        mode = contract_path.lstat().st_mode
    except OSError as error:
        raise ScopeError(
            "role contract must be an existing regular file: {}: {}".format(
                contract, error
            )
        )
    if not stat.S_ISREG(mode):
        raise ScopeError(
            "role contract must be an existing regular file: {}".format(
                contract
            )
        )

    try:
        payload = contract_path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ScopeError(
            "cannot read role contract {}: {}".format(contract, error)
        )
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ScopeError(
            "role contract requires YAML frontmatter: {}".format(contract)
        )
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        raise ScopeError(
            "role contract YAML frontmatter is not terminated: {}".format(
                contract
            )
        )
    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        raise ScopeError(
            "invalid role contract YAML frontmatter {}: {}".format(
                contract, error
            )
        )
    if not isinstance(frontmatter, dict):
        raise ScopeError(
            "role contract frontmatter must be a YAML mapping: {}".format(
                contract
            )
        )
    _require_schema_version(
        frontmatter, ROLE_CONTRACT_SCHEMA_VERSION, "role contract"
    )
    if frontmatter.get("role_id") != role_id:
        raise ScopeError(
            "role contract role_id must match work order role_id: {}".format(
                contract
            )
        )
    return {
        "role_id": role_id,
        "role_contract": contract,
        "role_contract_digest": _sha256_bytes(payload),
    }


def _repository_root(work_order):
    workspace = work_order.get("workspace")
    if not isinstance(workspace, dict):
        raise ScopeError("work order requires a workspace mapping")
    repository_value = workspace.get("repository")
    if not isinstance(repository_value, str) or not repository_value.strip():
        raise ScopeError("workspace.repository must be a concrete path")
    repository = Path(repository_value).expanduser().resolve()
    _, stdout, _ = _run_git(repository, ["rev-parse", "--show-toplevel"])
    discovered = Path(
        stdout.decode("utf-8", "surrogateescape").strip()
    ).resolve()
    if discovered != repository:
        raise ScopeError(
            "workspace.repository must be the Git root: {}".format(discovered)
        )
    return repository, workspace


def _normalize_relative_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ScopeError("{} entries must be non-empty strings".format(label))
    value = value.strip().replace("\\", "/")
    if value.startswith("/"):
        raise ScopeError(
            "{} must be repository-relative: {}".format(label, value)
        )
    if "<" in value or ">" in value:
        raise ScopeError(
            "{} contains an unresolved placeholder: {}".format(label, value)
        )
    path = PurePosixPath(value)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ScopeError("{} is not normalized: {}".format(label, value))
    normalized = str(path)
    if (
        normalized in ("", ".")
        or normalized == ".git"
        or normalized.startswith(".git/")
    ):
        raise ScopeError(
            "{} cannot name the repository root or .git: {}".format(
                label, value
            )
        )
    return normalized


def _parse_scope(value, label):
    if not isinstance(value, str):
        raise ScopeError("{} entries must be strings".format(label))
    raw = value.strip().replace("\\", "/")
    subtree = raw.endswith("/") or raw.endswith("/**")
    if raw.endswith("/**"):
        raw = raw[:-3]
    raw = raw.rstrip("/")
    if any(character in raw for character in ("*", "?", "[", "]")):
        raise ScopeError(
            "{} supports only exact paths, trailing '/', or '/**': {}".format(
                label, value
            )
        )
    return {
        "path": _normalize_relative_path(raw, label),
        "subtree": subtree,
    }


def _parse_scopes(workspace, key):
    values = workspace.get(key, [])
    if not isinstance(values, list):
        raise ScopeError("workspace.{} must be a list".format(key))
    return [_parse_scope(value, "workspace.{}".format(key)) for value in values]


def _parse_readonly_scopes(repository, workspace):
    values = workspace.get("readonly_dependencies", [])
    if not isinstance(values, list):
        raise ScopeError("workspace.readonly_dependencies must be a list")
    scopes = []
    for value in values:
        if not isinstance(value, str):
            raise ScopeError(
                "workspace.readonly_dependencies entries must be strings"
            )
        raw = value.strip().replace("\\", "/")
        if os.path.isabs(raw):
            subtree = raw.endswith("/") or raw.endswith("/**")
            if raw.endswith("/**"):
                raw = raw[:-3]
            candidate = Path(raw.rstrip("/")).resolve()
            try:
                common = os.path.commonpath([str(repository), str(candidate)])
            except ValueError:
                common = None
            if common != str(repository):
                continue
            relative = candidate.relative_to(repository).as_posix()
            value = relative + ("/**" if subtree else "")
        scopes.append(_parse_scope(value, "workspace.readonly_dependencies"))
    return scopes


def _matches_scope(path, scope):
    prefix = scope["path"]
    return path == prefix or (
        scope["subtree"] and path.startswith(prefix + "/")
    )


def _matches_any(path, scopes):
    return any(_matches_scope(path, scope) for scope in scopes)


def _scopes_overlap(left, right):
    return _matches_scope(left["path"], right) or _matches_scope(
        right["path"], left
    )


def _validate_scope_location(repository, scopes, label):
    repository_text = str(repository)
    for scope in scopes:
        candidate = (repository / scope["path"]).resolve()
        try:
            common = os.path.commonpath([repository_text, str(candidate)])
        except ValueError:
            common = None
        if common != repository_text:
            raise ScopeError(
                "{} escapes the repository through a symlink: {}".format(
                    label, scope["path"]
                )
            )


def _is_nested_git_root(path):
    if not path.is_dir():
        return False
    returncode, stdout, _ = _run_git(
        path, ["rev-parse", "--show-toplevel"], allow_failure=True
    )
    if returncode:
        return False
    discovered = Path(
        stdout.decode("utf-8", "surrogateescape").strip()
    ).resolve()
    return discovered == path.resolve()


def _directory_digest(path):
    digest = hashlib.sha256()
    for root, directories, files in os.walk(str(path), followlinks=False):
        directories[:] = sorted(name for name in directories if name != ".git")
        files = sorted(name for name in files if name != ".git")
        root_path = Path(root)
        for name in list(directories):
            child = root_path / name
            if child.is_symlink():
                relative = child.relative_to(path).as_posix()
                digest.update(("L\0" + relative + "\0").encode("utf-8"))
                target = os.readlink(str(child)).encode(
                    "utf-8", "surrogateescape"
                )
                digest.update(target)
                directories.remove(name)
        for name in files:
            child = root_path / name
            relative = child.relative_to(path).as_posix()
            if child.is_symlink():
                digest.update(("L\0" + relative + "\0").encode("utf-8"))
                target = os.readlink(str(child)).encode(
                    "utf-8", "surrogateescape"
                )
                digest.update(target)
            else:
                mode = stat.S_IMODE(child.lstat().st_mode)
                digest.update(
                    (
                        "F\0{}\0{:o}\0{}".format(
                            relative, mode, _sha256_file(child)
                        )
                    ).encode("utf-8")
                )
    return digest.hexdigest()


def _nested_repository_digest(path):
    _, head, _ = _run_git(path, ["rev-parse", "HEAD"])
    state = {
        "head": head.decode("ascii", "replace").strip(),
        "dirty": _collect_dirty_state(path),
    }
    return _sha256_bytes(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _path_digest(repository, relative_path):
    path = repository / relative_path
    try:
        mode = path.lstat().st_mode
    except OSError:
        return "missing"
    permissions = stat.S_IMODE(mode)
    if stat.S_ISLNK(mode):
        value = os.readlink(str(path)).encode("utf-8", "surrogateescape")
        return "symlink:{:o}:{}".format(permissions, _sha256_bytes(value))
    if stat.S_ISREG(mode):
        return "file:{:o}:{}".format(permissions, _sha256_file(path))
    if stat.S_ISDIR(mode):
        if _is_nested_git_root(path):
            return "gitdir:{:o}:{}".format(
                permissions, _nested_repository_digest(path)
            )
        return "dir:{:o}:{}".format(permissions, _directory_digest(path))
    return "special:{:o}".format(permissions)


def _collect_dirty_state(repository):
    _, stdout, _ = _run_git(
        repository,
        [
            "-c",
            "status.renames=false",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
    )
    state = {}
    for record in stdout.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ScopeError(
                "cannot parse Git status record: {!r}".format(record)
            )
        status_value = record[:2].decode("ascii", "replace")
        path = record[3:].decode("utf-8", "surrogateescape").rstrip("/")
        path = _normalize_relative_path(path, "Git status path")
        state[path] = {
            "status": status_value,
            "digest": _path_digest(repository, path),
        }
    return state


def _collect_visible_state(repository):
    _, stdout, _ = _run_git(
        repository,
        ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
    )
    state = {}
    for raw_path in stdout.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", "surrogateescape").rstrip("/")
        path = _normalize_relative_path(path, "Git-visible path")
        state[path] = _path_digest(repository, path)
    return state


def _index_digest(repository):
    _, stdout, _ = _run_git(repository, ["ls-files", "--stage", "-z"])
    return _sha256_bytes(stdout)


def _work_order_context(work_order_path):
    work_order = _load_yaml(work_order_path, "work order")
    _require_schema_version(work_order, WORK_ORDER_SCHEMA_VERSION, "work order")
    task_id = work_order.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ScopeError("work order requires a concrete task_id")
    adoption = _parse_adoption(work_order, "work order", True)
    _validate_owned_output(work_order)
    _validate_run_guide(work_order, adoption)
    _validate_acceptance(work_order, adoption)
    repository, workspace = _repository_root(work_order)
    role = _parse_role_contract(repository, work_order)
    writable = _parse_scopes(workspace, "writable_paths")
    readonly = _parse_readonly_scopes(repository, workspace)
    forbidden = _parse_scopes(workspace, "forbidden_paths")
    allowed_dirty = _parse_scopes(workspace, "allow_preexisting_dirty_paths")
    _validate_scope_location(repository, writable, "writable scope")
    _validate_scope_location(repository, readonly, "read-only scope")
    _validate_scope_location(repository, forbidden, "forbidden scope")
    _validate_scope_location(repository, allowed_dirty, "allowed dirty scope")
    for writable_scope in writable:
        for readonly_scope in readonly:
            if _scopes_overlap(writable_scope, readonly_scope):
                raise ScopeError(
                    (
                        "writable_paths overlaps readonly_dependencies: "
                        "{} and {}"
                    ).format(writable_scope["path"], readonly_scope["path"])
                )
    for scope in allowed_dirty:
        if not any(
            _matches_scope(scope["path"], writable_scope)
            for writable_scope in writable
        ):
            raise ScopeError(
                (
                    "allow_preexisting_dirty_paths must be within "
                    "writable_paths: {}"
                ).format(scope["path"])
            )
        if any(_scopes_overlap(scope, value) for value in readonly + forbidden):
            raise ScopeError(
                (
                    "allow_preexisting_dirty_paths cannot authorize read-only "
                    "or forbidden paths: {}"
                ).format(scope["path"])
            )
    return {
        "work_order": work_order,
        "task_id": task_id.strip(),
        "repository": repository,
        "writable": writable,
        "readonly": readonly,
        "forbidden": forbidden,
        "allowed_dirty": allowed_dirty,
        "work_order_digest": _sha256_file(work_order_path),
        "role_id": role["role_id"],
        "role_contract": role["role_contract"],
        "role_contract_digest": role["role_contract_digest"],
        "adoption_mode": adoption["mode"],
        "claim_id": adoption["claim_id"],
        "x_step": adoption["x_step"],
    }


def _head(repository):
    _, stdout, _ = _run_git(repository, ["rev-parse", "HEAD"])
    return stdout.decode("ascii", "replace").strip()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(str(temporary), str(path))


def _read_json(path):
    try:
        with path.open("r") as source:
            value = json.load(source)
    except (OSError, ValueError) as error:
        raise ScopeError("cannot read baseline {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise ScopeError(
            "baseline must contain one JSON mapping: {}".format(path)
        )
    _require_schema_version(value, BASELINE_SCHEMA_VERSION, "baseline")
    recorded_digest = value.get("manifest_digest")
    unsigned = dict(value)
    unsigned.pop("manifest_digest", None)
    actual_digest = _sha256_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if not recorded_digest or recorded_digest != actual_digest:
        raise ScopeError("baseline manifest digest does not match its payload")
    return value


def _require_baseline_outside_repository(repository, baseline_path):
    repository_text = str(repository)
    baseline_text = str(baseline_path.resolve())
    try:
        common = os.path.commonpath([repository_text, baseline_text])
    except ValueError:
        common = None
    if common == repository_text:
        raise ScopeError(
            "baseline must be outside the repository, for example under /tmp"
        )


def snapshot(work_order_path, baseline_path):
    context = _work_order_context(work_order_path)
    repository = context["repository"]
    _require_baseline_outside_repository(repository, baseline_path)
    dirty = _collect_dirty_state(repository)
    collisions = sorted(
        path
        for path in dirty
        if _matches_any(path, context["writable"])
        and not _matches_any(path, context["allowed_dirty"])
    )
    if collisions:
        raise ScopeError(
            "writable paths already contain unapproved dirty state: {}".format(
                ", ".join(collisions)
            )
        )
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "task_id": context["task_id"],
        "role_id": context["role_id"],
        "role_contract": context["role_contract"],
        "role_contract_digest": context["role_contract_digest"],
        "repository": str(repository),
        "head": _head(repository),
        "index_digest": _index_digest(repository),
        "work_order_digest": context["work_order_digest"],
        "dirty_state": dirty,
        "visible_state": _collect_visible_state(repository),
    }
    baseline["manifest_digest"] = _sha256_bytes(
        json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    _write_json(baseline_path, baseline)
    print(
        json.dumps(
            {
                "status": "snapshotted",
                "task_id": context["task_id"],
                "baseline": str(baseline_path.resolve()),
                "preexisting_dirty_paths": sorted(dirty),
            },
            sort_keys=True,
        )
    )


def _handoff_changed_paths(handoff_path, context):
    handoff = _load_yaml(handoff_path, "handoff")
    _require_schema_version(handoff, HANDOFF_SCHEMA_VERSION, "handoff")
    if handoff.get("task_id") != context["task_id"]:
        raise ScopeError("handoff task_id does not match work order")
    adoption = _parse_adoption(handoff, "handoff", False)
    if adoption["mode"] != context["adoption_mode"]:
        raise ScopeError("handoff adoption mode does not match work order")
    if adoption["claim_id"] != context["claim_id"]:
        raise ScopeError("handoff claim_id does not match work order")
    if adoption["x_step"] != context["x_step"]:
        raise ScopeError("handoff adoption x_step does not match work order")
    paths = handoff.get("changed_paths")
    if not isinstance(paths, list):
        raise ScopeError("handoff.changed_paths must be a list")
    normalized = []
    for value in paths:
        path = _normalize_relative_path(value, "handoff.changed_paths")
        if path.endswith("/**") or path.endswith("/"):
            raise ScopeError("handoff.changed_paths must list exact Git paths")
        normalized.append(path)
    if len(set(normalized)) != len(normalized):
        raise ScopeError("handoff.changed_paths contains duplicates")
    return sorted(normalized)


def verify(work_order_path, baseline_path, handoff_path):
    context = _work_order_context(work_order_path)
    repository = context["repository"]
    baseline = _read_json(baseline_path)
    if baseline.get("task_id") != context["task_id"]:
        raise ScopeError("baseline task_id does not match work order")
    if Path(baseline.get("repository", "")).resolve() != repository:
        raise ScopeError("baseline repository does not match work order")
    if baseline.get("work_order_digest") != context["work_order_digest"]:
        raise ScopeError("work order changed after the baseline was captured")
    if baseline.get("role_id") != context["role_id"]:
        raise ScopeError("baseline role_id does not match work order")
    if baseline.get("role_contract") != context["role_contract"]:
        raise ScopeError("baseline role_contract does not match work order")
    if baseline.get("role_contract_digest") != context["role_contract_digest"]:
        raise ScopeError(
            "role contract changed after the baseline was captured"
        )
    if baseline.get("head") != _head(repository):
        raise ScopeError(
            "repository HEAD changed after the baseline was captured"
        )
    if baseline.get("index_digest") != _index_digest(repository):
        raise ScopeError("Git index changed after the baseline was captured")

    before = baseline.get("visible_state")
    if not isinstance(before, dict):
        raise ScopeError("baseline visible_state is invalid")
    after = _collect_visible_state(repository)
    changed = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    unauthorized = sorted(
        path
        for path in changed
        if not _matches_any(path, context["writable"])
        or _matches_any(path, context["readonly"])
        or _matches_any(path, context["forbidden"])
    )
    reported = _handoff_changed_paths(handoff_path, context)
    errors = []
    if unauthorized:
        errors.append(
            "out-of-scope changes: {}".format(", ".join(unauthorized))
        )
    missing = sorted(set(changed) - set(reported))
    extra = sorted(set(reported) - set(changed))
    if missing:
        errors.append(
            "handoff omitted changed paths: {}".format(", ".join(missing))
        )
    if extra:
        errors.append(
            "handoff reported unchanged paths: {}".format(", ".join(extra))
        )
    if errors:
        raise ScopeError("; ".join(errors))
    print(
        json.dumps(
            {
                "status": "pass",
                "task_id": context["task_id"],
                "changed_paths": changed,
            },
            sort_keys=True,
        )
    )


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot or verify a Voyager adoption work-order write scope."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--work-order", required=True)
    snapshot_parser.add_argument("--baseline", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--work-order", required=True)
    verify_parser.add_argument("--baseline", required=True)
    verify_parser.add_argument("--handoff", required=True)
    return parser


def main():
    parser = _parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "snapshot":
            snapshot(Path(arguments.work_order), Path(arguments.baseline))
        elif arguments.command == "verify":
            verify(
                Path(arguments.work_order),
                Path(arguments.baseline),
                Path(arguments.handoff),
            )
        else:
            parser.error("a command is required")
    except ScopeError as error:
        print("scope check failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
