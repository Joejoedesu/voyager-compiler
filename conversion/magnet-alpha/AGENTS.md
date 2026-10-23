<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Magnet Alpha conversion instructions

Before changing converter code in this directory, read the repository
`AGENTS.md`, `CLAUDE.md`, and
`docs/targets/magnet-alpha/regression.md`.

Keep every CLI-supplied AoV or Alpha reference-collateral root read-only. Write
only to a fresh output directory selected for the current run.

Use only a fresh Magnet Alpha dump compiled with AoV's fixed mixed W4A4/W8A8 policy.
Default-W8/W8A8 dumps and dated reports are historical evidence.

`convert.py` must remain a fail-closed external-schema adapter. Treat
`model.txt` as semantic truth. The converter may derive target routing and
converter-recognized Q/DQ absorption from that dataflow and translate
registered ordinary or composite forms into Alpha collateral. It must account
for every executable semantic member, accept only Alpha-implemented or
correctly absorbed members, reject an unsupported sequence, and never change
numerical semantics.
CPU-routed work is a separate accelerator-module `unsupported` assignment
outside the Alpha composite; the converter never implements its CPU path.

The converter may validate finalized schedules, addresses, payloads, and
quantization operands required by the target file interface, but it must not
select, relocate, numerically reconstruct, or repair them. Do not claim one
target call or hardware fusion from a `fused_op`; report emitted layers only
as translation evidence.
Legacy `cmod_*` fields are non-authoritative.

Conversion success proves only schema-valid, internally consistent target
collateral. It does not prove target execution, numerical parity, accuracy, or
performance. Report the input artifact identity, conversion output, warnings,
and first unverified boundary.
