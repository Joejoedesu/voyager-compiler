<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Global instructions

**Scope:** this file holds repo-agnostic authoring style only — formatting,
imports, defaults, and commit messages. It deliberately contains nothing
specific to Voyager.

Project-specific routing, ownership, and validation policy live in
[`AGENTS.md`](AGENTS.md), which is the entry point for work in this
repository. Read both: `AGENTS.md` tells you *what* to change and how to
validate it; this file tells you *how to write* the resulting code.

## Python formatting (all projects, all sessions)

Write Python at **≤80 columns**. Apply this even if the repo has no formatter config and longer existing lines.

- **Format code with black:** `python -m black --line-length 80 --target-version py312 <file>`. black's wrapping is the desired style:
  - whole call/def on one line if it fits;
  - else **Option A** — args on one wrapped line when they fit at that indent;
  - else one-arg-per-line (trailing comma);
  - **ternaries each clause on its own line** in parens (value / `if cond` / `else val`).
- **Never leave an inline (trailing) comment that exceeds 80** — move it to its own line *above* the statement (a long trailing comment makes black explode the code to fit it, which is ugly). Inline comments ≤80 are fine.
- black does **not** reflow comments/docstrings — reflow those to ≤80 by hand (paragraph- and bullet-aware; preserve `` ``code`` `` markup; keep the summary on the opening `"""` line, reserving 3 cols for the quotes).
- After formatting, verify: no line >80, `black --check` clean, and tests still pass (formatting must not change behavior).

## Imports (all projects, all sessions)

Put imports at the **top of the file**, not inside functions/methods. Only use a local (in-function) import when it is genuinely required — to break a real circular import, or to defer an optional/heavy dependency — and add a short comment saying which. Default to top-level; do not scatter imports mid-function for convenience.

## Default values (all projects, all sessions)

A default value must have a **single source of truth**.

- If a default is used in **only one place**, define it directly in the function signature.
- If it is **shared across multiple places** (re-specified at several call layers, or duplicated in a dataclass + its constructors + a convenience wrapper), define it **once as a named constant** and reference that constant everywhere.

Do not repeat the same literal default at every layer of a call hierarchy — that is multiple sources of truth and they drift. Prefer making inner/lower-level functions **require** the parameter (no default) so the default lives only at the outermost entry point or in the named constant.

## Commit messages (all projects, all sessions)

- Format: `type: summary` on the first line, a blank line, then bullets grouped by functionality/fix (not per-file).
- **Wrap every line at ≤80 columns** — never let a line run on forever; wrap continuations onto an indented line.
- Keep each bullet to **one line, at most two**.
- **Never mention Claude / AI / co-authorship** anywhere in the message (no attribution trailer).
