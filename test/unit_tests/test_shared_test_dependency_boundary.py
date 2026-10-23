# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import ast
import re
import unittest
from pathlib import Path

from voyager_compiler.codegen.targets import (
    COMPILATION_TARGETS,
    DEFAULT_COMPILATION_TARGET,
)

_UNIT_TEST_DIRECTORY = Path(__file__).resolve().parent
_FORBIDDEN_IMPORT_ROOTS = (
    "conversion",
    "test.utils.models",
)
_TARGET_BACKEND_PACKAGE = "voyager_compiler.codegen.target_backends"


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module


def _is_module_or_child(module, root):
    return module == root or module.startswith(f"{root}.")


class SharedTestDependencyBoundaryTest(unittest.TestCase):
    def test_shared_tests_do_not_reference_production_targets(self):
        target_names = tuple(
            target.lower()
            for target in COMPILATION_TARGETS
            if target != DEFAULT_COMPILATION_TARGET
        )
        violations = []

        for path in sorted(_UNIT_TEST_DIRECTORY.rglob("*.py")):
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
            relative_path = path.relative_to(_UNIT_TEST_DIRECTORY)

            for module in _imported_modules(tree):
                forbidden_root = next(
                    (
                        root
                        for root in _FORBIDDEN_IMPORT_ROOTS
                        if _is_module_or_child(module, root)
                    ),
                    None,
                )
                target_backend = next(
                    (
                        target
                        for target in target_names
                        if _is_module_or_child(
                            module,
                            f"{_TARGET_BACKEND_PACKAGE}.{target}",
                        )
                    ),
                    None,
                )
                if forbidden_root is not None or target_backend is not None:
                    violations.append(
                        f"{relative_path}: imports non-shared module {module}"
                    )

            lowered = source.lower()
            for target in target_names:
                pattern = rf"(?<![a-z0-9_]){re.escape(target)}(?![a-z0-9_])"
                if re.search(pattern, lowered):
                    violations.append(
                        f"{relative_path}: references target {target!r}"
                    )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
