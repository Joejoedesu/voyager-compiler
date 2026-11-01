"""Coordination checks run without importing the compiler or device libraries."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/adoption.py"
spec = importlib.util.spec_from_file_location("adoption", SCRIPT)
adoption = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adoption)


class AdoptionHarnessTest(unittest.TestCase):
    def setUp(self):
        self.record = {
            "version": 1,
            "scope": {"route": "binding", "target": "test", "workload": "cnn",
                      "objective": "Reduce latency", "constraints": "fixed accuracy"},
            "best": {"artifact": "candidate-a:source-and-artifact-hash",
                     "claim": "connected numerical check passed", "evidence": ["a/result.json"],
                     "metric": {"name": "latency", "value": 1.2, "unit": "ms",
                                "kind": "measured", "boundary": "device completion"}},
            "opportunities": [{"id": "layout", "question": "Can another layout reduce copies?",
                               "state": "open", "next": "Compile and compare another layout"}],
            "checks": [{"id": "accuracy-b", "artifact": "candidate-b:hash",
                        "question": "Are connected outputs within tolerance?", "state": "passed",
                        "evidence": ["b/numerics.json"]}],
        }

    def test_device_time_needs_no_clock_and_validation_does_not_promote(self):
        before = copy.deepcopy(self.record)
        self.assertIn("1.2 ms", adoption.summarize(self.record))
        self.assertEqual(before, self.record)
        self.assertEqual(self.record["opportunities"][0]["state"], "open")

    def test_reject_unattributable_completed_check(self):
        for field in ("artifact", "evidence"):
            broken = copy.deepcopy(self.record)
            del broken["checks"][0][field]
            with self.assertRaises(ValueError):
                adoption.validate(broken)

    def test_duplicate_ids_and_invalid_metric(self):
        self.record["checks"].append(copy.deepcopy(self.record["checks"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            adoption.validate(self.record)
        self.record["checks"].pop()
        for value in (float("nan"), -1, True):
            self.record["best"]["metric"]["value"] = value
            with self.assertRaises(ValueError):
                adoption.validate(self.record)

    def test_cli_registration_no_overwrite_and_binding_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "work.json"
            command = [sys.executable, str(SCRIPT), "init", str(path), "--route",
                       "registration", "--target", "test", "--objective", "Bind matmul",
                       "--constraints", "hardware spec"]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            content = path.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(path.read_bytes(), content)
            record = json.loads(content)
            record["scope"]["route"] = "binding"
            with self.assertRaisesRegex(ValueError, "workload"):
                adoption.validate(record)


if __name__ == "__main__":
    unittest.main()
