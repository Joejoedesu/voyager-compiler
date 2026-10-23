# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import re
import unittest
from collections import Counter
from pathlib import Path

import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ROOT_INSTRUCTIONS = _REPOSITORY_ROOT / "AGENTS.md"
_SKILLS_DIRECTORY = _REPOSITORY_ROOT / ".agents" / "skills"
_ROLE_DIRECTORY = _REPOSITORY_ROOT / "docs" / "backend-adoption" / "roles"
_ROLE_REGISTRY = _ROLE_DIRECTORY / "README.md"
_ADOPTER_SKILL = _SKILLS_DIRECTORY / "voyager-adopt-backend" / "SKILL.md"
_COORDINATOR_CONTRACT = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "contracts"
    / "coordinator.md"
)
_PERFORMANCE_CONTRACT = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "contracts"
    / "performance.md"
)
_ADOPTION_LEVELS = (
    _REPOSITORY_ROOT / "docs" / "backend-adoption" / "adoption-levels.md"
)
_RUN_GUIDE_TEMPLATE = (
    _REPOSITORY_ROOT / "docs" / "backend-adoption" / "run-guide-template.md"
)
_OPERATION_INVENTORY_TEMPLATE = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "templates"
    / "operation-inventory.yaml"
)
_GAP_REPORT_TEMPLATE = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "templates"
    / "gap-report.yaml"
)
_HANDOFF_TEMPLATE = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "templates"
    / "handoff.yaml"
)
_WORK_ORDER_TEMPLATE = (
    _REPOSITORY_ROOT
    / "docs"
    / "backend-adoption"
    / "templates"
    / "work-order.yaml"
)
_TARGET_CONTRACT_DIRECTORY = _REPOSITORY_ROOT / "docs" / "targets"

_ADOPTION_SKILLS = (
    "voyager-adopt-backend",
    "voyager-execute-adoption-work-order",
    "voyager-gate-regressions",
)
_FORBIDDEN_EAGER_REFERENCES = (
    "architecture.md",
    "agent-development-guide.md",
    "common-agent-contract.md",
    "agent-charters.md",
    "docs/backend-adoption/readme.md",
)

_MAX_ROOT_WORDS = 300
_MAX_SKILL_WORDS = 450
_MAX_ROLE_REGISTRY_WORDS = 600
_MAX_ROLE_WORDS = 225

_SPDX_HTML_HEADER = re.compile(
    r"<!--\s*"
    r"SPDX-FileCopyrightText:[^\n]*\n"
    r"SPDX-License-Identifier:[^\n]*"
    r"\s*-->\s*"
)


def _word_count(path):
    source = _SPDX_HTML_HEADER.sub("", path.read_text())
    return len(source.split())


def _role_files():
    return sorted(
        path for path in _ROLE_DIRECTORY.glob("*.md") if path != _ROLE_REGISTRY
    )


def _adoption_markdown_files():
    roots = (
        _REPOSITORY_ROOT / "docs" / "backend-adoption",
        _TARGET_CONTRACT_DIRECTORY,
    )
    return sorted(path for root in roots for path in root.rglob("*.md"))


def _frontmatter(path):
    match = re.match(
        r"\A---\s*\n(.*?)\n---(?:\s*\n|\Z)",
        path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return yaml.safe_load(match.group(1))


class AgentContextRoutingTest(unittest.TestCase):
    def test_context_entrypoints_are_compact(self):
        self.assertLessEqual(
            _word_count(_ROOT_INSTRUCTIONS),
            _MAX_ROOT_WORDS,
        )

        skill_files = sorted(_SKILLS_DIRECTORY.glob("*/SKILL.md"))
        self.assertNotEqual(skill_files, [])
        for path in skill_files:
            with self.subTest(skill=path.parent.name):
                self.assertLessEqual(
                    _word_count(path),
                    _MAX_SKILL_WORDS,
                )

        self.assertLessEqual(
            _word_count(_ROLE_REGISTRY),
            _MAX_ROLE_REGISTRY_WORDS,
        )
        for path in _role_files():
            with self.subTest(role=path.stem):
                self.assertLessEqual(
                    _word_count(path),
                    _MAX_ROLE_WORDS,
                )

    def test_role_frontmatter_matches_filename(self):
        role_files = _role_files()
        self.assertNotEqual(role_files, [])

        for path in role_files:
            with self.subTest(role=path.stem):
                metadata = _frontmatter(path)
                self.assertIsInstance(metadata, dict)
                self.assertEqual(metadata.get("schema_version"), 1)
                self.assertEqual(metadata.get("role_id"), path.stem)

    def test_registry_links_every_role_contract(self):
        link_targets = re.findall(
            r"\[[^\]]+\]\(([^)]+)\)",
            _ROLE_REGISTRY.read_text(),
        )
        linked_files = [
            Path(target.split("#", 1)[0]).name
            for target in link_targets
            if target.split("#", 1)[0].endswith(".md")
        ]
        expected_files = {path.name for path in _role_files()}
        self.assertEqual(Counter(linked_files), Counter(expected_files))

    def test_adoption_skills_do_not_eagerly_load_catalogs(self):
        for skill_name in _ADOPTION_SKILLS:
            path = _SKILLS_DIRECTORY / skill_name / "SKILL.md"
            with self.subTest(skill=skill_name):
                self.assertTrue(path.is_file())
                source = path.read_text().lower()
                for reference in _FORBIDDEN_EAGER_REFERENCES:
                    self.assertNotIn(reference, source)

    def test_adoption_guides_do_not_reintroduce_parallel_execution_ir(self):
        paths = [
            *_adoption_markdown_files(),
            *sorted(_SKILLS_DIRECTORY.glob("*/SKILL.md")),
            *sorted(
                (
                    _REPOSITORY_ROOT / "docs" / "backend-adoption" / "templates"
                ).glob("*.yaml")
            ),
        ]
        forbidden = (
            "high_level_composite",
            "backend_call_plan",
            "strict_fusion_scope",
            "outer_tiling_scope",
            "accelerator_call_plan_scope",
            "isa_fused_instruction",
            "accelerator_resident_region",
            "fused_into_owner",
            "host_boundary_permitted",
        )
        for path in paths:
            source = path.read_text().lower()
            for label in forbidden:
                with self.subTest(path=path, label=label):
                    self.assertNotIn(label, source)

        model = _ADOPTION_LEVELS.read_text()
        self.assertIn(
            "Final FX and `model.txt` are the semantic source of truth",
            model,
        )
        self.assertIn("### Multi-call tiling", model)

    def test_workload_binding_uses_w_without_colliding_with_backend_alias(self):
        model = _ADOPTION_LEVELS.read_text()
        self.assertIn("## R: register a target", model)
        self.assertIn("## W: bind and refine a workload-target pair", model)
        self.assertIn(
            "## X: bit-accurate full-control workload binding",
            model,
        )
        self.assertIn("`X => W => R`", model)
        self.assertIn(
            "Report the strongest completed stage as the achieved stage.",
            model,
        )

    def test_injection_control_and_x_implication_are_explicit(self):
        model = re.sub(r"\s+", " ", _ADOPTION_LEVELS.read_text())
        for requirement in (
            "## Injection points and control",
            "is where Voyager hands emitted work to a target path",
            "**partial-control injection point**",
            "**full-control injection point**",
            "performance-based, not a universal field checklist",
            "R and W may use either class of injection point",
            "X requires a full-control point",
            "target actual output == frozen workload-semantic reference output",
            "No second numerical \"bridge\" comparison is required",
            "A full-control point is X-capable, not proof",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, model)
        self.assertRegex(
            model,
            r"Incomplete X work does not advance the achieved stage",
        )
        self.assertRegex(
            model,
            r"it remains W only when W was already complete",
        )
        guide = re.sub(r"\s+", " ", _RUN_GUIDE_TEMPLATE.read_text())
        self.assertIn(
            "same workload-semantic reference already frozen for W",
            guide,
        )
        self.assertIn(
            "X packaged-stream provenance to the same W reference",
            guide,
        )

        paths = [
            _ROOT_INSTRUCTIONS,
            *sorted((_REPOSITORY_ROOT / "docs").glob("*.md")),
            *_adoption_markdown_files(),
            *sorted((_REPOSITORY_ROOT / "conversion").glob("*/*.md")),
            *sorted(
                (
                    _REPOSITORY_ROOT / "docs" / "backend-adoption" / "templates"
                ).glob("*.yaml")
            ),
            *sorted(_TARGET_CONTRACT_DIRECTORY.rglob("*.yaml")),
            *sorted(_SKILLS_DIRECTORY.glob("*/SKILL.md")),
            *sorted(_SKILLS_DIRECTORY.glob("*/agents/openai.yaml")),
        ]
        stale_lifecycle_labels = (
            r"\bR/B(?:/X)?\b",
            r"\bB\s*(?:/|\+)\s*X\b",
            r"(?i)\bB\s+plus\s+X\b",
            r"\bB-(?:only|level)\b",
            r"(?i)\b(?:phase|level)\s+`?B`?\b",
            r"(?i)\b(?:at|ordinary)\s+`?B`?\b",
            r"(?i)\bbinding\s*\(`?B`?\)",
            r"\bR\s*(?:and|,)\s*B\b",
            r"\bR\s*,\s*B\s*,?\s*and\s*X\b",
            r"(?m)^#{1,4}\s+B:",
        )
        for path in paths:
            source = path.read_text()
            for pattern in stale_lifecycle_labels:
                with self.subTest(path=path, pattern=pattern):
                    self.assertIsNone(re.search(pattern, source))

    def test_adopter_selects_an_explicit_adoption_mode(self):
        metadata = _frontmatter(_ADOPTER_SKILL)
        description = metadata["description"].lower()
        self.assertIn("target registration", description)
        self.assertIn("workload-target binding", description)
        self.assertIn("full-control workload binding", description)

        skill = _ADOPTER_SKILL.read_text().lower()
        contract = _COORDINATOR_CONTRACT.read_text().lower()
        for mode in (
            "register_target",
            "bind_workload_target",
            "deepen_inner_control",
        ):
            with self.subTest(mode=mode):
                self.assertIn(f"`{mode}`", skill)
                self.assertIn(f"`{mode}`", contract)
        self.assertIn("agent orchestration", contract)
        self.assertIn("candidate assignments", contract)
        self.assertIn("final execution assignments", contract)
        self.assertIn("injection-point control classification", contract)
        self.assertIn("reference provenance", contract)

    def test_exact_pair_routing_keeps_pair_code_conditional(self):
        contract_markers = (
            "(workload, target, injection point)",
            "<workload>_<target>.py",
            "test/test_codegen.py",
            "workload_target_registry",
            "workload-target-overlay",
            "integration",
            "serialize",
            "execution.md",
        )
        skill = re.sub(r"\s+", " ", _ADOPTER_SKILL.read_text().lower())
        contract = re.sub(
            r"\s+", " ", _COORDINATOR_CONTRACT.read_text().lower()
        )
        for marker in contract_markers:
            with self.subTest(marker=marker):
                self.assertIn(marker, contract)
        self.assertNotIn("model_registry", skill)
        self.assertNotIn("model_registry", contract)

        self.assertIn(
            "combined pair adapter only for genuinely pair-specific",
            skill,
        )
        self.assertIn("exact tuple remains the evidence unit", skill)
        self.assertIn(
            "target-independent workload helper is optional",
            contract,
        )

        integration = (_ROLE_DIRECTORY / "integration.md").read_text().lower()
        combined = (
            (_ROLE_DIRECTORY / "workload-target-overlay.md").read_text().lower()
        )
        self.assertIn("exact tuple", integration)
        self.assertIn("workload_target_registry", integration)
        self.assertIn("target-independent", combined)
        self.assertRegex(combined, r"own registry(?:/| or )cli routing")
        self.assertIn(
            "does not require a policy-only overlay or new file",
            combined,
        )
        self.assertIn("yolo5_face_nv_magnet_alpha.py", combined)

    def test_performance_contract_preserves_qualified_w_estimation(self):
        contract = _PERFORMANCE_CONTRACT.read_text()
        for marker in (
            "checked_accelerator_coverage(x)",
            "estimated_calls(x)",
            "estimated_total_time(x)",
            "coverage_gain",
            "estimated_call_reduction",
            "estimated_time_reduction",
            "estimated_speedup",
            "estimated_unresolved_time",
            "maximize checked accelerator coverage",
            "minimize estimated total time",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, contract)

        gate = (
            _SKILLS_DIRECTORY / "voyager-gate-regressions" / "SKILL.md"
        ).read_text()
        performance = gate.split("## Performance", 1)[1].split("## ", 1)[0]
        self.assertIn("qscheme candidates", performance)
        self.assertIn("checked accelerator", performance)
        self.assertIn("estimated total time", performance)
        self.assertIn("Estimated calls", performance)
        self.assertIn("per-call launch latency", performance)
        self.assertIn("conversion cost", performance)
        self.assertIn("never call coarse W estimates exact", performance)

    def test_gap_schema_covers_every_in_scope_nonpass_state(self):
        gap = yaml.safe_load(_GAP_REPORT_TEMPLATE.read_text())
        self.assertEqual(gap["schema_version"], 2)
        states = gap["evidence_state"].strip("<>").split("|")
        self.assertEqual(
            set(states),
            {
                "unsupported",
                "approximated",
                "fail",
                "blocked",
                "unverified",
            },
        )
        locus = gap["blocker"]
        self.assertEqual(
            set(locus),
            {"boundary", "owner", "authority"},
        )
        self.assertIn("affected", gap)
        self.assertIn("details", gap)
        self.assertIn("next_action", gap)

    def test_adoption_artifact_versions_cover_staged_execution(self):
        templates = {
            _WORK_ORDER_TEMPLATE: 5,
            _HANDOFF_TEMPLATE: 4,
            _GAP_REPORT_TEMPLATE: 2,
        }
        for path, expected_version in templates.items():
            with self.subTest(template=path.name):
                artifact = yaml.safe_load(path.read_text())
                self.assertEqual(artifact["schema_version"], expected_version)

        work_order = yaml.safe_load(_WORK_ORDER_TEMPLATE.read_text())
        adoption = work_order["adoption"]
        for key in ("mode", "claim_id", "intended_claim", "x_step"):
            self.assertIn(key, adoption)

        run_guide = work_order["run_guide"]
        for key in ("path", "revision_or_hash", "registration_claim_id"):
            self.assertIn(key, run_guide)

        acceptance = work_order["acceptance"]
        for key in (
            "fidelity",
            "oracle",
            "comparison_scope",
            "required_evidence",
            "tests",
            "regression_tests",
        ):
            self.assertIn(key, acceptance)
        self.assertEqual(set(acceptance["tolerance"]), {"absolute", "relative"})

        inner_control = work_order["inner_control"]
        self.assertIn("voyager_owned_decisions", inner_control)
        self.assertIn("endpoint_consumed_fields", inner_control)

        workspace = work_order["workspace"]
        for key in (
            "repository",
            "writable_paths",
            "readonly_dependencies",
            "forbidden_paths",
            "allow_preexisting_dirty_paths",
        ):
            self.assertIn(key, workspace)
        self.assertEqual(len(work_order["owned_outputs"]), 1)

        # Campaign parameters belong to the instantiated run guide. Keep them
        # out of the work order so one assignment stays one assignment.
        for retired in (
            "run_parameters",
            "target_configuration",
            "operation_registration",
            "workload_binding",
            "assignment_scope",
            "binding_requirements",
            "injection_contract",
            "numerical_fidelity",
            "performance_scope",
            "inner_control_scope",
            "external_evidence_scope",
            "native_baselines",
            "calibration",
            "quantization_scope",
        ):
            self.assertNotIn(retired, work_order)

        handoff = yaml.safe_load(_HANDOFF_TEMPLATE.read_text())
        self.assertNotIn("implicit_cpu_fallback", _HANDOFF_TEMPLATE.read_text())
        self.assertIn("operational_state", handoff)
        handoff_adoption = handoff["adoption"]
        for key in ("mode", "claim_id", "x_step", "claim_state"):
            self.assertIn(key, handoff_adoption)
        self.assertIn("semantic_support_evidence", handoff)
        self.assertIn("boundary_check_evidence", handoff)
        self.assertIn("registration_evidence", handoff)
        self.assertIn("compiler_artifacts", handoff)
        self.assertIn("connected_execution_evidence", handoff)
        self.assertIn("checks", handoff)

        handoff_assignment = handoff["assignment_evidence"]
        handoff_candidate = handoff_assignment["candidate_assignment"]
        handoff_final = handoff_assignment["final_execution_assignment"]
        self.assertIn("unsupported_occurrence_ids", handoff_candidate)
        self.assertNotIn("cpu_candidate_occurrence_ids", handoff_candidate)
        self.assertIn("unsupported_model_operation_ids", handoff_final)
        self.assertNotIn("cpu_model_operation_ids", handoff_final)

        performance_evidence = handoff["performance_evidence"]
        self.assertIn(
            "coverage_denominator",
            performance_evidence["frozen_comparison"],
        )
        self.assertIn("deltas", performance_evidence)
        self.assertIn("measured_validation", performance_evidence)
        self.assertIn("bottleneck", performance_evidence)
        # One reusable result shape replaces duplicated baseline/candidate
        # blocks that had to be edited in lockstep.
        for retired in ("baseline", "candidate"):
            self.assertNotIn(retired, performance_evidence)
        result = performance_evidence["results"][0]
        for key in (
            "id",
            "quantization_identity",
            "checked_accelerator_coverage",
            "estimated_calls",
            "estimated_total_time",
            "cost_components",
        ):
            self.assertIn(key, result)
        self.assertEqual(
            set(result["cost_components"][0]),
            {"component", "value", "provenance"},
        )

        exact = handoff["bit_accurate_core_lowering_evidence"]
        self.assertIn("canonical_compiler_artifact", exact)
        self.assertIn("physical_qparam_artifact", exact)
        self.assertIn("bit_exact_output", exact)
        # Zero tolerance and forbidden perturbation are contract rules. A
        # handoff cannot report anything else, so it does not carry them.
        for retired in (
            "required_fidelity",
            "tolerance",
            "numerical_perturbation_permitted",
        ):
            self.assertNotIn(retired, exact)

        inner = handoff["inner_control_evidence"]
        self.assertIn("control_variation_artifacts", inner)
        self.assertIn(
            "control_variation_preserved_bit_exact_output",
            inner,
        )

        self.assertIn("target_operation_execution_lanes", handoff)
        operation_lane = handoff["target_operation_execution_lanes"][0]
        for key in (
            "native_reference",
            "injection_contract_hash",
            "boundary_source",
            "compiler_artifact",
            "converter_artifact",
            "endpoint_glue_artifact",
            "target_artifact",
            "emitted_call_artifact",
            "actual_execution_trace",
            "actual_accelerator_call_count",
            "semantic_accounting_complete",
            "all_emitted_calls_executed",
            "unsupported_excluded_from_accelerator_coverage",
            "claimed_accelerator_work_executed_on_target",
            "observed_fidelity",
        ):
            self.assertIn(key, operation_lane)

        # Command lists merged into checks[]; the summary ID lists duplicated
        # semantic_support_evidence.operation_records states.
        for retired in (
            "shared_unit_tests",
            "target_integration_lanes",
            "temporary_diagnostics",
            "supported",
            "unsupported",
            "unverified",
            "approximations",
        ):
            self.assertNotIn(retired, handoff)

    def test_operation_inventories_report_named_support(self):
        template = yaml.safe_load(_OPERATION_INVENTORY_TEMPLATE.read_text())
        self.assertEqual(template["schema_version"], 6)
        self.assertIn("regression_status", template)
        self.assertIn("endpoint", template)
        self.assertIn("quantization", template)
        self.assertIn("semantic_fidelity", template)
        self.assertIn("capabilities", template)
        self.assertIn("performance_observations", template)
        self.assertIn("gaps", template)
        self.assertIn("status", template["regression_status"])
        self.assertIn("control", template["endpoint"])
        self.assertNotIn("requires_inner_control", template["endpoint"])

        inventories = sorted(
            _TARGET_CONTRACT_DIRECTORY.glob("*/operations.yaml")
        )
        self.assertNotEqual(inventories, [])

        statuses = {"supported", "unsupported", "unverified"}
        kinds = {"operation", "composite"}
        check_states = {
            "pass",
            "fail",
            "blocked",
            "unverified",
            "not_in_scope",
        }
        regression_states = {"pass", "fail", "blocked", "unverified"}
        control_classes = {"partial-control", "full-control"}

        def assert_locator(locator):
            self.assertIsInstance(locator, str)
            self.assertTrue(locator)
            source, separator, symbol = locator.partition("::")
            if "://" in source or Path(source).is_absolute():
                return
            source_path = _REPOSITORY_ROOT / source
            self.assertTrue(source_path.exists(), locator)
            if separator:
                self.assertTrue(symbol)
                symbol_name = symbol.split(".")[-1]
                if source_path.suffix == ".py":
                    declaration = re.compile(
                        r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+{}\b|"
                        r"^[ \t]*class[ \t]+{}\b|"
                        r"^[ \t]*{}[ \t]*(?::[^=\n]+)?=".format(
                            *(re.escape(symbol_name),) * 3
                        )
                    )
                    self.assertRegex(source_path.read_text(), declaration)
                else:
                    self.assertIn(symbol_name, source_path.read_text())

        for path in inventories:
            with self.subTest(inventory=path.parent.name):
                inventory = yaml.safe_load(path.read_text())
                self.assertEqual(inventory["schema_version"], 6)
                self.assertEqual(inventory["backend"], path.parent.name)
                revision = inventory["source_revision"]
                if revision is not None:
                    self.assertRegex(revision, r"^[0-9a-f]{40}$")
                endpoint = inventory["endpoint"]
                self.assertTrue(endpoint["endpoint_id"])
                self.assertTrue(endpoint["accepted_artifact"])
                self.assertIn(endpoint["control"], control_classes)
                self.assertEqual(
                    set(endpoint["ownership"]), {"voyager", "endpoint"}
                )
                assert_locator(endpoint["adapter"])
                assert_locator(endpoint["interface"])

                fidelity = inventory["semantic_fidelity"]
                self.assertIn(fidelity["stage"], {"R", "W", "X"})
                self.assertTrue(fidelity["reference"])
                self.assertEqual(
                    set(fidelity["tolerance"]),
                    {"absolute", "relative"},
                )
                if fidelity["stage"] == "R":
                    self.assertEqual(
                        fidelity["required"],
                        "reference_tolerance",
                    )
                    self.assertIs(
                        fidelity["numerical_perturbation_permitted"],
                        True,
                    )
                elif fidelity["stage"] == "W":
                    self.assertEqual(
                        fidelity["required"],
                        "output_code_lsb",
                    )
                    self.assertIs(
                        fidelity["numerical_perturbation_permitted"],
                        True,
                    )
                    self.assertEqual(
                        fidelity["tolerance"],
                        {"absolute": 1, "relative": 0},
                    )
                else:
                    self.assertEqual(
                        fidelity["required"],
                        "bit_exact_endpoint",
                    )
                    self.assertIs(
                        fidelity["numerical_perturbation_permitted"],
                        False,
                    )
                    self.assertEqual(
                        fidelity["tolerance"],
                        {"absolute": 0, "relative": 0},
                    )

                regression = inventory["regression_status"]
                self.assertEqual(regression["stage"], fidelity["stage"])
                self.assertIn(regression["status"], regression_states)
                if regression["stage"] == "X":
                    self.assertEqual(endpoint["control"], "full-control")
                self.assertTrue(regression["scope"])
                details = _REPOSITORY_ROOT / regression["details"]
                self.assertTrue(details.is_file(), details)
                regression_checks = regression["checks"]
                self.assertTrue(regression_checks)
                check_ids = [check["check_id"] for check in regression_checks]
                self.assertEqual(len(check_ids), len(set(check_ids)))
                observed_boundaries = []
                for check in regression_checks:
                    collateral = check["voyager_collateral"]
                    self.assertEqual(set(collateral), {"artifact", "sha256"})
                    self.assertTrue(collateral["artifact"])
                    self.assertRegex(collateral["sha256"], r"^[0-9a-f]{64}$")
                    conversion = check["conversion"]
                    self.assertEqual(
                        set(conversion), {"passed", "total", "unit"}
                    )
                    self.assertGreaterEqual(conversion["passed"], 0)
                    self.assertLessEqual(
                        conversion["passed"], conversion["total"]
                    )
                    self.assertTrue(conversion["unit"])
                    boundaries = check["boundaries"]
                    self.assertEqual(
                        set(boundaries),
                        {"compiler", "conversion", "execution", "comparison"},
                    )
                    self.assertTrue(
                        all(
                            state in check_states
                            for state in boundaries.values()
                        )
                    )
                    observed_boundaries.append(boundaries)
                    if boundaries["conversion"] == "not_in_scope":
                        self.assertEqual(
                            conversion,
                            {"passed": 0, "total": 0, "unit": "not_in_scope"},
                        )
                    else:
                        self.assertNotEqual(conversion["unit"], "not_in_scope")

                if regression["status"] == "pass":
                    unresolved = {"fail", "blocked", "unverified"}
                    self.assertFalse(
                        any(
                            state in unresolved
                            for boundaries in observed_boundaries
                            for state in boundaries.values()
                        )
                    )
                    self.assertTrue(
                        any(
                            boundaries["compiler"] == "pass"
                            and boundaries["execution"] == "pass"
                            and boundaries["comparison"] == "pass"
                            for boundaries in observed_boundaries
                        ),
                        "a passing gate needs one compiled, executed, and "
                        "compared retained check",
                    )

                schemes = inventory["quantization"]["schemes"]
                qdq_forms = inventory["quantization"]["qdq_forms"]
                capabilities = inventory["capabilities"]
                gaps = inventory["gaps"]

                scheme_ids = [item["scheme_id"] for item in schemes]
                qdq_ids = [item["capability_id"] for item in qdq_forms]
                capability_ids = [
                    item["capability_id"] for item in capabilities
                ]
                gap_ids = [item["gap_id"] for item in gaps]
                for values in (scheme_ids, qdq_ids, capability_ids, gap_ids):
                    self.assertTrue(all(values))
                    self.assertEqual(len(values), len(set(values)))
                all_registration_ids = {
                    endpoint["endpoint_id"],
                    *scheme_ids,
                    *qdq_ids,
                    *capability_ids,
                }
                self.assertEqual(
                    len(all_registration_ids),
                    1 + len(scheme_ids) + len(qdq_ids) + len(capability_ids),
                )

                for scheme in schemes:
                    self.assertIn(scheme["status"], statuses)
                    self.assertTrue(scheme["summary"])
                    if scheme["status"] == "supported":
                        self.assertTrue(scheme["implementation"])
                        self.assertTrue(scheme["semantic_evidence"])
                    for key in (
                        "implementation",
                        "semantic_evidence",
                        "adapter",
                    ):
                        for locator in scheme[key]:
                            assert_locator(locator)
                    self.assertEqual(
                        set(scheme["checks"]),
                        {"serialization", "adapter", "execution", "comparison"},
                    )
                    self.assertTrue(
                        set(scheme["checks"].values()).issubset(check_states)
                    )

                for item in qdq_forms:
                    self.assertIn(item["status"], statuses)
                    self.assertTrue(item["summary"])
                    if item["status"] == "supported":
                        self.assertTrue(item["implementation"])
                        self.assertTrue(item["semantic_evidence"])
                    for key in (
                        "implementation",
                        "semantic_evidence",
                        "adapter",
                    ):
                        for locator in item[key]:
                            assert_locator(locator)
                    self.assertTrue(
                        set(item["checks"].values()).issubset(check_states)
                    )

                for item in capabilities:
                    self.assertIn(item["kind"], kinds)
                    self.assertIn(item["status"], statuses)
                    self.assertTrue(item["summary"])
                    if item["status"] == "supported":
                        self.assertTrue(item["implementation"])
                        self.assertTrue(item["semantic_evidence"])
                    for key in (
                        "recognition",
                        "implementation",
                        "semantic_evidence",
                        "adapter",
                    ):
                        for locator in item[key]:
                            assert_locator(locator)
                    self.assertTrue(
                        set(item["checks"].values()).issubset(check_states)
                    )

                affected = set()
                for gap in gaps:
                    self.assertTrue(gap["statement"])
                    self.assertTrue(gap["affects"])
                    self.assertTrue(
                        set(gap["affects"]).issubset(all_registration_ids)
                    )
                    affected.update(gap["affects"])
                self.assertTrue(
                    {
                        item["capability_id"]
                        for item in capabilities
                        if item["status"] == "unverified"
                    }.issubset(affected)
                )
                self.assertTrue(
                    {
                        item["capability_id"]
                        for item in qdq_forms
                        if item["status"] == "unverified"
                    }.issubset(affected)
                )
                self.assertTrue(
                    {
                        item["scheme_id"]
                        for item in schemes
                        if item["status"] == "unverified"
                    }.issubset(affected)
                )

                observations = inventory["performance_observations"]
                self.assertIsInstance(observations["available"], dict)
                self.assertIsInstance(observations["unavailable"], list)
                for locator in observations["available"].values():
                    assert_locator(locator)

    def test_target_contracts_do_not_delegate_to_skills(self):
        readmes = [
            _TARGET_CONTRACT_DIRECTORY / "README.md",
            *sorted(_TARGET_CONTRACT_DIRECTORY.glob("*/README.md")),
        ]
        self.assertNotEqual(readmes, [])
        for path in readmes:
            self.assertTrue(path.is_file())
            with self.subTest(target=path.parent.name):
                source = path.read_text().lower()
                self.assertNotIn("skill.md", source)
                self.assertNotIn(".agents/skills", source)

    def test_adoption_document_links_stay_inside_repository(self):
        for path in _adoption_markdown_files():
            for target in re.findall(
                r"\[[^\]]+\]\(([^)]+)\)", path.read_text()
            ):
                target = target.split("#", 1)[0]
                if not target or "://" in target or "<" in target:
                    continue
                resolved = (path.parent / target).resolve()
                with self.subTest(document=path, target=target):
                    self.assertTrue(
                        resolved == _REPOSITORY_ROOT
                        or _REPOSITORY_ROOT in resolved.parents
                    )
                    self.assertTrue(resolved.exists())

    def test_role_contracts_use_real_test_boundaries(self):
        paths = [
            _ADOPTER_SKILL,
            _COORDINATOR_CONTRACT,
            _ROLE_REGISTRY,
            *_role_files(),
        ]
        for path in paths:
            with self.subTest(path=path):
                source = path.read_text().lower()
                self.assertNotIn("t/scenarios", source)
                self.assertNotIn("test/targets", source)

        workload = (_ROLE_DIRECTORY / "workload-target-overlay.md").read_text()
        integration = (_ROLE_DIRECTORY / "integration.md").read_text()
        coordinator = _COORDINATOR_CONTRACT.read_text()
        self.assertIn("examples/<workload>", workload)
        self.assertIn("test/utils/models/", workload)
        self.assertNotIn("test/test_codegen.py", workload)
        self.assertIn("WORKLOAD_TARGET_REGISTRY", integration)
        self.assertIn("test/test_codegen.py", integration)


if __name__ == "__main__":
    unittest.main()
