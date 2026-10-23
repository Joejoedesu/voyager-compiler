# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable operator traits for the Gemmini registration anchor."""

import torch

from voyager_compiler.codegen.target_backends.spec import TargetOperatorTraits
from voyager_compiler.quantization_contracts import (
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainTransferRule,
)


def _is_global_adaptive_average(node: torch.fx.Node) -> bool:
    output_size = (
        node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size")
    )
    if isinstance(output_size, int):
        return output_size == 1
    if not isinstance(output_size, (tuple, list, torch.Size)):
        return False
    return tuple(output_size) == (1, 1)


# TODO: Remove this forcing rule when the public Gemmini global-average ABI
# accepts an explicit output scale. Until then, the prepare hook shares
# calibration but retains explicit Q/DQ so tiled_global_average_auto sees one
# representable input/output domain without changing PT2E rounding semantics.
PREPARED_DOMAIN_TRANSFER_RULES = (
    PreparedDomainTransferRule(
        target=torch.ops.aten.adaptive_avg_pool2d.default,
        input_format=PreparedDomainInputFormat.FIRST,
        relation=PreparedDomainRelation.ALIAS,
        region_root=True,
        forcing_root=True,
        predicate=_is_global_adaptive_average,
    ),
)

# The target keeps shared operator identities. The native ``_auto`` helper
# owns local convolution control, so Gemmini adds no target-only GEMM traits.
GEMMINI_OPERATOR_TRAITS = TargetOperatorTraits(
    prepared_domain_transfer_rules=PREPARED_DOMAIN_TRANSFER_RULES,
)

__all__ = [
    "GEMMINI_OPERATOR_TRAITS",
]
