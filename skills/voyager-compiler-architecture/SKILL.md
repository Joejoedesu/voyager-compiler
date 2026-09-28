---
name: voyager-compiler-architecture
description: Navigate and modify the Voyager compiler architecture, model-by-hardware compilation tests, hardware-family quantization policies, and bufferized lowering; validate compiler changes against run_ci baselines.
---

# Voyager compiler architecture

Locate `AGEN-voyager/voyager-base` from the current workspace (or use the
user-selected checkout). Read its local instructions and current working-tree
changes before modifying overlapping files. Paths below are relative to that
repository, not to the skill installation directory.

Read `docs/compilation.md` for current stage and policy ownership and
`docs/hardware-ir.md` for the typed hardware graph. Treat source as authoritative
when documentation and implementation disagree.

For an architectural change, use the stage map in `docs/compilation.md` to identify
where the behavior belongs. Keep model preparation, family quantization policy,
hardware capacities and backend algorithms distinct. `voyager` is the default
named target. A new hardware description does not imply a working backend.

For compiler edits, use the regression procedure in `docs/compilation.md`.
Honor the selected test scope; this workspace uses the marked cases in
`test/regression_suite.txt` unless broader coverage is requested.
Preserve a fixed pre-change comparator and inspect emitted-program differences,
compile failures and numerical results separately. The suite may contain known
failures; identify them from the current reference rather than assuming a clean
exit or hardcoding permanent exceptions.

Keep explicit session constraints, including review/commit requirements. This
skill does not grant permission to commit, publish or contact other people.
