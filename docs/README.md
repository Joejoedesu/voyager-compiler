<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager compiler documentation

Use this directory through progressive disclosure. A document's presence in
the repository does not make it mandatory context.

## Route by task

- [Architecture](architecture.md) records shared stages, representation
  boundaries, sources of truth, and the target-backend binding contract. It
  routes target-specific implementation and evidence to the selected target
  documentation. Read it end to end only for a genuinely cross-layer change.
- [Agent development guide](agent-development-guide.md) owns shared coding and
  validation policy. Load the relevant test, packaging, or handoff section.
- [Backend adoption](backend-adoption/README.md) is a human-facing overview.
  Adoption agents should use the compact contracts and exact role selected by
  their skill rather than loading every adoption document. Start with the
  [glossary](backend-adoption/glossary.md), which defines `R`, `W`, and `X`
  with worked examples; the
  [adoption model](backend-adoption/adoption-levels.md) is the normative owner
  of the phases and the invariants that hold across them.
- [Target contracts](targets/README.md) route backend-specific compile,
  conversion, simulator, and runtime evidence. Load only affected targets.

The architecture document describes what exists. Focused design documents may
describe unfinished work. Keep that distinction explicit.

## Agent context policy

Repository `AGENTS.md` files are short routing instructions. A `SKILL.md`
contains only the reusable procedure needed after that skill triggers.
File-specific ownership lives under
[backend-adoption/roles](backend-adoption/roles/README.md), and common
execution/evidence rules live under
[backend-adoption/contracts](backend-adoption/contracts/README.md). A work
order names exactly one role contract.

Do not duplicate the same rule in `AGENTS.md`, a skill, a role, and a design
document. Put the source of truth at the narrowest durable owner and link to it
conditionally. For adoption specifically, the owners are fixed: vocabulary in
[glossary](backend-adoption/glossary.md), phases and invariants in
[adoption-levels](backend-adoption/adoption-levels.md), file ownership in
[roles](backend-adoption/roles/README.md), and everything else in the one
[contract](backend-adoption/contracts/README.md) for that responsibility. A
skill or role that restates one of those is drifting; replace it with a link.

Agent procedures are routed by the repository `AGENTS.md`; durable ownership
and validation requirements remain in the linked repository documents.

## Maintenance

Update `architecture.md` when stage order, subsystem ownership, a persistent
IR, a target, or a scheduler registration changes. Update the development
guide when repository-wide coding or validation policy changes. Keep local
machine paths and transient collateral out of both documents.
