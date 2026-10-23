# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCOPE_CHECK = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "voyager-execute-adoption-work-order"
    / "scripts"
    / "check_work_order_scope.py"
)
_UNSET = object()


class WorkOrderScopeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self._run(["git", "init", "-q"], cwd=self.repository)
        self._run(
            ["git", "config", "user.email", "scope-test@example.com"],
            cwd=self.repository,
        )
        self._run(
            ["git", "config", "user.name", "Scope Test"],
            cwd=self.repository,
        )
        (self.repository / "src").mkdir()
        self._write("src/allowed.py", "allowed = 1\n")
        self._write("src/locked.py", "locked = 1\n")
        self._write("user.txt", "user = 1\n")
        self.role_id = "scope-test"
        self.role_contract = "docs/backend-adoption/roles/scope-test.md"
        self._write_role_contract()
        self._run(["git", "add", "."], cwd=self.repository)
        self._run(["git", "commit", "-qm", "initial"], cwd=self.repository)
        self.work_order = self.root / "work-order.yaml"
        self.baseline = self.root / "baseline.json"
        self.handoff = self.root / "handoff.yaml"

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, command, cwd=None):
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        return (
            process.returncode,
            stdout.decode("utf-8"),
            stderr.decode("utf-8"),
        )

    def _write(self, relative_path, value):
        path = self.repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def _write_role_contract(
        self, schema_version=1, role_id=None, body="# Scope test role\n"
    ):
        value = (
            "---\n" "schema_version: {}\n" "role_id: {}\n" "---\n" "{}"
        ).format(schema_version, role_id or self.role_id, body)
        self._write(self.role_contract, value)

    def _write_work_order(
        self,
        writable=None,
        forbidden=None,
        allow_preexisting_dirty=None,
        readonly=None,
        role_id=None,
        role_contract=None,
        schema_version=5,
        adoption_mode="bind_workload_target",
        claim_id="scope-test-claim",
        intended_claim="exercise scoped work-order validation",
        x_step=_UNSET,
        fidelity=None,
        tolerance=None,
        required_evidence=None,
        decisions=None,
        consumed_fields=None,
        registration_claim_id=_UNSET,
    ):
        is_inner_control = adoption_mode == "deepen_inner_control"
        if x_step is _UNSET:
            x_step = "bit_accurate_core_lowering" if is_inner_control else None
        if fidelity is None:
            if is_inner_control:
                fidelity = "bit_exact_endpoint"
            elif adoption_mode == "bind_workload_target":
                fidelity = "output_code_lsb"
            else:
                fidelity = "reference_tolerance"
        if tolerance is None:
            if is_inner_control:
                tolerance = {"absolute": 0, "relative": 0}
            elif adoption_mode == "bind_workload_target":
                tolerance = {"absolute": 1, "relative": 0}
            else:
                tolerance = {"absolute": 0.01, "relative": 0.01}
        claims_control = x_step == "inner_control"
        if required_evidence is None:
            required_evidence = (
                ["control_consumption"] if claims_control else []
            )
        if decisions is None:
            decisions = ["schedule"] if claims_control else []
        if consumed_fields is None:
            consumed_fields = ["instruction_order"] if claims_control else []
        if registration_claim_id is _UNSET:
            registration_claim_id = (
                None
                if adoption_mode == "register_target"
                else "r-registration-claim"
            )
        value = {
            "schema_version": schema_version,
            "task_id": "scope-test",
            "objective": "produce one scoped output",
            "role_id": role_id or self.role_id,
            "role_contract": role_contract or self.role_contract,
            "adoption": {
                "mode": adoption_mode,
                "claim_id": claim_id,
                "intended_claim": intended_claim,
                "x_step": x_step,
            },
            "run_guide": {
                "path": str(self.root / "run-guide.md"),
                "revision_or_hash": "run-guide-hash",
                "registration_claim_id": registration_claim_id,
            },
            "acceptance": {
                "fidelity": fidelity,
                "tolerance": tolerance,
                "oracle": "PyTorch reference",
                "comparison_scope": "complete output",
                "required_evidence": required_evidence,
                "tests": [],
                "regression_tests": [],
            },
            "inner_control": {
                "voyager_owned_decisions": decisions,
                "endpoint_consumed_fields": consumed_fields,
            },
            "workspace": {
                "repository": str(self.repository),
                "writable_paths": writable or [],
                "readonly_dependencies": readonly or [],
                "forbidden_paths": forbidden or [],
                "allow_preexisting_dirty_paths": allow_preexisting_dirty or [],
            },
            "owned_outputs": [
                {
                    "artifact": "scoped code change",
                    "path": "src/allowed.py",
                    "consumer": "scope test",
                }
            ],
        }
        self.work_order.write_text(
            yaml.safe_dump(value, default_flow_style=False)
        )

    def _write_handoff(
        self,
        changed_paths,
        schema_version=4,
        adoption_mode="bind_workload_target",
        claim_id="scope-test-claim",
        x_step=_UNSET,
    ):
        if x_step is _UNSET:
            x_step = (
                "bit_accurate_core_lowering"
                if adoption_mode == "deepen_inner_control"
                else None
            )
        value = {
            "task_id": "scope-test",
            "status": "complete",
            "changed_paths": changed_paths,
            "adoption": {
                "mode": adoption_mode,
                "claim_id": claim_id,
                "x_step": x_step,
            },
        }
        if schema_version is not None:
            value["schema_version"] = schema_version
        self.handoff.write_text(yaml.safe_dump(value, default_flow_style=False))

    def _snapshot(self):
        return self._run(
            [
                sys.executable,
                str(SCOPE_CHECK),
                "snapshot",
                "--work-order",
                str(self.work_order),
                "--baseline",
                str(self.baseline),
            ]
        )

    def _verify(self):
        return self._run(
            [
                sys.executable,
                str(SCOPE_CHECK),
                "verify",
                "--work-order",
                str(self.work_order),
                "--baseline",
                str(self.baseline),
                "--handoff",
                str(self.handoff),
            ]
        )

    def test_requires_work_order_schema_version_five(self):
        for schema_version in (None, True, 0, 1, 2, 3, 4):
            with self.subTest(schema_version=schema_version):
                self._write_work_order()
                value = yaml.safe_load(self.work_order.read_text())
                if schema_version is None:
                    value.pop("schema_version")
                else:
                    value["schema_version"] = schema_version
                self.work_order.write_text(
                    yaml.safe_dump(value, default_flow_style=False)
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("work order schema_version must be 5", stderr)

    def test_requires_concrete_adoption_contract(self):
        cases = (
            ("bad_mode", "scope-test-claim", "claim", "adoption.mode"),
            (
                "bind_workload_target",
                "",
                "claim",
                "adoption.claim_id",
            ),
            (
                "bind_workload_target",
                "scope-test-claim",
                "",
                "adoption.intended_claim",
            ),
        )
        for mode, claim_id, intended, expected in cases:
            with self.subTest(expected=expected):
                self._write_work_order(
                    adoption_mode=mode,
                    claim_id=claim_id,
                    intended_claim=intended,
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(expected, stderr)

    def test_accepts_all_three_documented_adoption_modes(self):
        for mode in (
            "register_target",
            "bind_workload_target",
            "deepen_inner_control",
        ):
            with self.subTest(mode=mode):
                self._write_work_order(adoption_mode=mode)
                self.assertEqual(self._snapshot()[0], 0)

    def test_requires_one_objective_and_owned_output(self):
        self._write_work_order()
        value = yaml.safe_load(self.work_order.read_text())
        value["objective"] = ""
        self.work_order.write_text(yaml.safe_dump(value))
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("work order.objective", stderr)

        for outputs in ([], [{}, {}]):
            with self.subTest(outputs=outputs):
                self._write_work_order()
                value = yaml.safe_load(self.work_order.read_text())
                value["owned_outputs"] = outputs
                self.work_order.write_text(yaml.safe_dump(value))
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("must contain exactly one output", stderr)

    def test_run_guide_supplies_campaign_parameters(self):
        for key in ("path", "revision_or_hash"):
            with self.subTest(key=key):
                self._write_work_order()
                value = yaml.safe_load(self.work_order.read_text())
                value["run_guide"][key] = ""
                self.work_order.write_text(yaml.safe_dump(value))
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("run_guide.{}".format(key), stderr)

    def test_registration_establishes_its_own_claim(self):
        self._write_work_order(
            adoption_mode="register_target",
            registration_claim_id="r-registration-claim",
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("must be null for a ", stderr)

    def test_later_modes_bind_the_registration_claim(self):
        for mode in ("bind_workload_target", "deepen_inner_control"):
            with self.subTest(mode=mode):
                self._write_work_order(
                    adoption_mode=mode, registration_claim_id=None
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("must name the R claim", stderr)

    def test_rejects_unknown_evidence_boundary(self):
        self._write_work_order(required_evidence=["made_up_boundary"])
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("unknown boundaries: made_up_boundary", stderr)

    def test_r_requires_reference_tolerance(self):
        for fidelity in ("output_code_lsb", "bit_exact_endpoint"):
            with self.subTest(fidelity=fidelity):
                self._write_work_order(
                    adoption_mode="register_target", fidelity=fidelity
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("acceptance.fidelity=reference_tolerance", stderr)

    def test_w_requires_output_code_lsb_fidelity(self):
        for fidelity in ("reference_tolerance", "bit_exact_endpoint"):
            with self.subTest(fidelity=fidelity):
                self._write_work_order(
                    adoption_mode="bind_workload_target", fidelity=fidelity
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("acceptance.fidelity=output_code_lsb", stderr)

    def test_w_requires_one_lsb_absolute_and_zero_relative_tolerance(self):
        for tolerance in (
            {"absolute": 0, "relative": 0},
            {"absolute": 0.5, "relative": 0},
            {"absolute": 2, "relative": 0},
            {"absolute": 1, "relative": 0.01},
        ):
            with self.subTest(tolerance=tolerance):
                self._write_work_order(
                    adoption_mode="bind_workload_target",
                    tolerance=tolerance,
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(
                    "acceptance.tolerance absolute=1 and relative=0",
                    stderr,
                )

    def test_r_and_w_accept_empty_or_complete_mapping_control_evidence(self):
        for mode in ("register_target", "bind_workload_target"):
            with self.subTest(mode=mode, evidence="empty"):
                self._write_work_order(adoption_mode=mode)
                self.assertEqual(self._snapshot()[0], 0)

            with self.subTest(mode=mode, evidence="complete"):
                self._write_work_order(
                    adoption_mode=mode,
                    decisions=["schedule"],
                    consumed_fields=["instruction_order"],
                    required_evidence=["control_consumption"],
                )
                self.assertEqual(self._snapshot()[0], 0)

    def test_rejects_incomplete_mapping_control_evidence(self):
        cases = (
            {"decisions": ["schedule"]},
            {"consumed_fields": ["instruction_order"]},
            {"required_evidence": ["control_consumption"]},
        )
        for mode in ("register_target", "bind_workload_target"):
            for arguments in cases:
                with self.subTest(mode=mode, arguments=arguments):
                    self._write_work_order(adoption_mode=mode, **arguments)
                    result, _, stderr = self._snapshot()
                    self.assertEqual(result, 1)
                    self.assertIn("mapping-control evidence requires", stderr)

    def test_x_step_belongs_to_inner_control_mode_only(self):
        self._write_work_order(
            adoption_mode="bind_workload_target",
            x_step="bit_accurate_core_lowering",
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("x_step must be null outside", stderr)

        self._write_work_order(
            adoption_mode="deepen_inner_control", x_step=None
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("adoption.x_step must be one of", stderr)

    def test_x_requires_exact_fidelity_and_zero_tolerance(self):
        cases = (
            (
                {"fidelity": "reference_tolerance"},
                "acceptance.fidelity=bit_exact_endpoint",
            ),
            (
                {"tolerance": {"absolute": 0.01, "relative": 0}},
                "requires zero acceptance.tolerance",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self._write_work_order(
                    adoption_mode="deepen_inner_control",
                    x_step="inner_control",
                    **arguments,
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(expected, stderr)

    def test_x_core_lowering_is_a_bounded_first_work_order(self):
        self._write_work_order(adoption_mode="deepen_inner_control")
        self.assertEqual(self._snapshot()[0], 0)

        cases = (
            {"decisions": ["schedule"]},
            {"required_evidence": ["control_consumption"]},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self._write_work_order(
                    adoption_mode="deepen_inner_control",
                    x_step="bit_accurate_core_lowering",
                    **arguments,
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn("mapping-control evidence", stderr)

    def test_x_inner_control_requires_consumed_decisions(self):
        cases = (
            ({"decisions": []}, "voyager_owned_decisions"),
            ({"consumed_fields": []}, "endpoint_consumed_fields"),
            (
                {"required_evidence": []},
                "requires control_consumption evidence",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self._write_work_order(
                    adoption_mode="deepen_inner_control",
                    x_step="inner_control",
                    **arguments,
                )
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(expected, stderr)

        self._write_work_order(
            adoption_mode="deepen_inner_control", x_step="inner_control"
        )
        self.assertEqual(self._snapshot()[0], 0)

    def test_rejects_non_kebab_role_id(self):
        for role_id in ("Scope-Test", "scope_test", "scope--test"):
            with self.subTest(role_id=role_id):
                self._write_work_order(role_id=role_id)
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(
                    "role_id must be a concrete kebab-case ID", stderr
                )

    def test_role_contract_path_must_match_role_id(self):
        self._write_work_order(
            role_contract="docs/backend-adoption/roles/other-role.md"
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn(
            "role_contract must be "
            "docs/backend-adoption/roles/scope-test.md",
            stderr,
        )

    def test_role_contract_must_be_regular_file(self):
        contract = self.repository / self.role_contract
        contract.unlink()
        contract.symlink_to(self.repository / "src" / "allowed.py")
        self._write_work_order()
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("role contract must be an existing regular file", stderr)

    def test_rejects_invalid_role_contract_frontmatter(self):
        cases = (
            (
                "# Missing frontmatter\n",
                "role contract requires YAML frontmatter",
            ),
            (
                "---\nschema_version: 2\nrole_id: scope-test\n---\n",
                "role contract schema_version must be 1",
            ),
            (
                "---\nschema_version: 1\nrole_id: other-role\n---\n",
                "role contract role_id must match work order role_id",
            ),
        )
        for contract_text, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                self._write(self.role_contract, contract_text)
                self._write_work_order()
                result, _, stderr = self._snapshot()
                self.assertEqual(result, 1)
                self.assertIn(expected_error, stderr)

    def test_snapshot_records_role_contract_binding(self):
        self._write_work_order()
        self.assertEqual(self._snapshot()[0], 0)
        value = json.loads(self.baseline.read_text())
        expected_digest = hashlib.sha256(
            (self.repository / self.role_contract).read_bytes()
        ).hexdigest()
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["role_id"], self.role_id)
        self.assertEqual(value["role_contract"], self.role_contract)
        self.assertEqual(value["role_contract_digest"], expected_digest)

    def test_role_contract_is_frozen_even_when_writable(self):
        self._write_work_order(writable=[self.role_contract])
        self.assertEqual(self._snapshot()[0], 0)
        self._write_role_contract(body="# Changed role contract\n")
        self._write_handoff([self.role_contract])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn(
            "role contract changed after the baseline was captured", stderr
        )

    def test_allows_exact_reported_change(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")
        self._write_handoff(["src/allowed.py"])
        result, stdout, stderr = self._verify()
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn('"status": "pass"', stdout)

    def test_rejects_change_outside_writable_paths(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("user.txt", "user = 2\n")
        self._write_handoff(["user.txt"])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("out-of-scope changes: user.txt", stderr)

    def test_forbidden_path_overrides_writable_subtree(self):
        self._write_work_order(
            writable=["src/**"],
            forbidden=["src/locked.py"],
        )
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/locked.py", "locked = 2\n")
        self._write_handoff(["src/locked.py"])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("out-of-scope changes: src/locked.py", stderr)

    def test_preserves_unchanged_preexisting_dirty_path(self):
        self._write("user.txt", "user-owned = true\n")
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")
        self._write_handoff(["src/allowed.py"])
        self.assertEqual(self._verify()[0], 0)

    def test_detects_additional_edit_to_preexisting_dirty_path(self):
        self._write("user.txt", "user-owned = true\n")
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("user.txt", "user-owned = overwritten\n")
        self._write_handoff(["user.txt"])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("out-of-scope changes: user.txt", stderr)

    def test_dirty_writable_path_requires_explicit_approval(self):
        self._write("src/allowed.py", "prior = true\n")
        self._write_work_order(writable=["src/allowed.py"])
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("unapproved dirty state", stderr)

        self._write_work_order(
            writable=["src/allowed.py"],
            allow_preexisting_dirty=["src/allowed.py"],
        )
        self.assertEqual(self._snapshot()[0], 0)

    def test_handoff_must_match_actual_changed_paths(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")
        self._write_handoff([])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("handoff omitted changed paths: src/allowed.py", stderr)

    def test_requires_handoff_schema_version_four(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")
        for schema_version in (None, True, 0, 1, 2, 3):
            with self.subTest(schema_version=schema_version):
                self._write_handoff(
                    ["src/allowed.py"], schema_version=schema_version
                )
                result, _, stderr = self._verify()
                self.assertEqual(result, 1)
                self.assertIn("handoff schema_version must be 4", stderr)

    def test_handoff_adoption_must_match_work_order(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")

        self._write_handoff(["src/allowed.py"], adoption_mode="register_target")
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("adoption mode does not match", stderr)

        self._write_handoff(["src/allowed.py"], claim_id="other-claim")
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("claim_id does not match", stderr)

    def test_x_handoff_reports_the_work_order_x_step(self):
        self._write_work_order(
            writable=["src/allowed.py"],
            adoption_mode="deepen_inner_control",
            x_step="inner_control",
        )
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")

        # A handoff omitting x_step, or naming the wrong step, is rejected.
        self._write_handoff(
            ["src/allowed.py"],
            adoption_mode="deepen_inner_control",
            x_step=None,
        )
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("adoption.x_step must be one of", stderr)

        self._write_handoff(
            ["src/allowed.py"],
            adoption_mode="deepen_inner_control",
            x_step="bit_accurate_core_lowering",
        )
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("x_step does not match work order", stderr)

        self._write_handoff(
            ["src/allowed.py"],
            adoption_mode="deepen_inner_control",
            x_step="inner_control",
        )
        self.assertEqual(self._verify()[0], 0)

    def test_work_order_cannot_change_after_snapshot(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write_work_order(writable=["src/**"])
        self._write_handoff([])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("work order changed after the baseline", stderr)

    def test_allows_new_untracked_file_in_writable_subtree(self):
        self._write_work_order(writable=["src/**"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/new.py", "new = true\n")
        self._write_handoff(["src/new.py"])
        self.assertEqual(self._verify()[0], 0)

    def test_rejects_writable_readonly_overlap(self):
        self._write_work_order(
            writable=["src/**"],
            readonly=["src/locked.py"],
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("writable_paths overlaps readonly_dependencies", stderr)

    def test_rejects_absolute_repository_readonly_overlap(self):
        self._write_work_order(
            writable=["src/**"],
            readonly=[str(self.repository / "src" / "locked.py")],
        )
        result, _, stderr = self._snapshot()
        self.assertEqual(result, 1)
        self.assertIn("writable_paths overlaps readonly_dependencies", stderr)

    def test_rejects_index_changes(self):
        self._write_work_order(writable=["src/allowed.py"])
        self.assertEqual(self._snapshot()[0], 0)
        self._write("src/allowed.py", "allowed = 2\n")
        self._run(["git", "add", "src/allowed.py"], cwd=self.repository)
        self._write_handoff(["src/allowed.py"])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("Git index changed", stderr)

    def test_rename_requires_both_paths_to_be_writable(self):
        self._write_work_order(writable=["src/new.py"])
        self.assertEqual(self._snapshot()[0], 0)
        (self.repository / "src" / "allowed.py").rename(
            self.repository / "src" / "new.py"
        )
        self._write_handoff(["src/allowed.py", "src/new.py"])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("out-of-scope changes: src/allowed.py", stderr)

    def test_rejects_tampered_baseline(self):
        self._write_work_order(writable=[])
        self.assertEqual(self._snapshot()[0], 0)
        value = json.loads(self.baseline.read_text())
        value["head"] = "tampered"
        self.baseline.write_text(json.dumps(value))
        self._write_handoff([])
        result, _, stderr = self._verify()
        self.assertEqual(result, 1)
        self.assertIn("manifest digest does not match", stderr)

    def test_requires_baseline_schema_version_one(self):
        self._write_work_order(writable=[])
        self.assertEqual(self._snapshot()[0], 0)
        original = json.loads(self.baseline.read_text())
        self._write_handoff([])
        for schema_version in (None, True, 0, 2):
            with self.subTest(schema_version=schema_version):
                value = dict(original)
                if schema_version is None:
                    value.pop("schema_version")
                else:
                    value["schema_version"] = schema_version
                self.baseline.write_text(json.dumps(value))
                result, _, stderr = self._verify()
                self.assertEqual(result, 1)
                self.assertIn("baseline schema_version must be 1", stderr)

    def test_rejects_path_traversal_and_placeholders(self):
        for writable in (["../outside.py"], ["<target-path>"]):
            self._write_work_order(writable=writable)
            result, _, stderr = self._snapshot()
            self.assertEqual(result, 1)
            self.assertTrue(
                "not normalized" in stderr or "unresolved placeholder" in stderr
            )


if __name__ == "__main__":
    unittest.main()
