"""Versioned performance corrections; physical fit and legality are unchanged."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from voyager_compiler.extensions import OpModel, RuntimeModel


@dataclass(frozen=True)
class PerformanceProfile:
    name: str
    target: str
    hardware_digest: str
    coefficients: tuple[tuple[str, float, float], ...]
    training_cases: tuple[str, ...]
    held_out_cases: tuple[str, ...]
    evidence_kind: str
    version: int = 1

    @property
    def identity(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def correct(self, family, cycles):
        for key, scale, bias in self.coefficients:
            if key == family:
                return scale * cycles + bias
        return cycles


class _CalibratedOp(OpModel):
    def __init__(self, base, profile):
        self.base, self.profile = base, profile
        self.space = base.space
        self.key = None if base.key is None else (base.key, profile.identity)

    def fit(self, tiling):
        return self.base.fit(tiling)

    def cost(self, tiling, placement):
        cost = self.base.cost(tiling, placement)
        return replace(
            cost,
            latency=self.profile.correct(
                type(self.base).__name__, cost.latency
            ),
        )

    def scratch(self, out_tile):
        return self.base.scratch(out_tile)


class CalibratedModel(RuntimeModel):
    def __init__(self, base, profile):
        super().__init__(base.config, base.accumulate_fp32)
        self.base, self.profile = base, profile

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prepare(self, node):
        op = self.base.prepare(node)
        return None if op is None else _CalibratedOp(op, self.profile)

    def prepare_attention(self, node, acc_dtype, bool_mask):
        return _CalibratedOp(
            self.base.prepare_attention(node, acc_dtype, bool_mask),
            self.profile,
        )
