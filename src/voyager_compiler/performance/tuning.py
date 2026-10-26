"""Fit measured device/RTL timings and evaluate disjoint held-out cases.

Capture measurements through the normal compile/runtime flow. Host replay
timings are correctness evidence and cannot calibrate an accelerator model.
"""

import math
from dataclasses import dataclass

import numpy as np

from .profile import PerformanceProfile


@dataclass(frozen=True)
class Measurement:
    case: str
    family: str
    target: str
    hardware_digest: str
    estimated_cycles: float
    measured_seconds: float
    clock_ghz: float
    evidence_kind: str
    provenance: str

    def __post_init__(self):
        if self.evidence_kind not in (
            "device",
            "rtl",
            "cycle_accurate_simulator",
        ):
            raise ValueError(
                "performance fitting requires device, RTL or cycle-accurate simulation evidence"
            )
        if not self.case or not self.family or not self.provenance:
            raise ValueError(
                "measurements need case, implementation family and provenance"
            )
        if any(
            not math.isfinite(v) or v <= 0
            for v in (
                self.estimated_cycles,
                self.measured_seconds,
                self.clock_ghz,
            )
        ):
            raise ValueError("timing and frequency must be finite and positive")

    @property
    def measured_cycles(self):
        return self.measured_seconds * self.clock_ghz * 1e9


def fit_profile(name, training, held_out):
    training, held_out = tuple(training), tuple(held_out)
    if not training or not held_out:
        raise ValueError(
            "provide training measurements and a disjoint held-out set"
        )
    if {m.case for m in training} & {m.case for m in held_out}:
        raise ValueError(
            "training and held-out workload cases must be disjoint"
        )
    identities = {
        (m.target, m.hardware_digest, m.evidence_kind)
        for m in training + held_out
    }
    if len(identities) != 1:
        raise ValueError(
            "one fit must use one target, hardware identity and evidence kind"
        )
    coefficients = []
    families = sorted({m.family for m in training})
    if {m.family for m in held_out} != set(families):
        raise ValueError("every fitted family needs held-out coverage")
    for family in families:
        rows = [m for m in training if m.family == family]
        x = np.array([m.estimated_cycles for m in rows])
        y = np.array([m.measured_cycles for m in rows])
        if len(set(x)) < 2:
            raise ValueError(
                f"{family}: at least two distinct modeled timings are needed"
            )
        slope, bias = np.linalg.lstsq(
            np.column_stack((x, np.ones_like(x))), y, rcond=None
        )[0]
        options = [
            (0.0, float(y.mean())),
            (float(np.dot(x, y) / np.dot(x, x)), 0.0),
        ]
        if slope >= 0 and bias >= 0:
            options.append((float(slope), float(bias)))
        slope, bias = min(
            options, key=lambda c: np.square(c[0] * x + c[1] - y).sum()
        )
        coefficients.append((family, slope, bias))
    target, hardware, kind = identities.pop()
    profile = PerformanceProfile(
        name,
        target,
        hardware,
        tuple(coefficients),
        tuple(sorted({m.case for m in training})),
        tuple(sorted({m.case for m in held_out})),
        kind,
    )
    errors = [
        abs(
            profile.correct(m.family, m.estimated_cycles) / m.measured_cycles
            - 1
        )
        for m in held_out
    ]
    evidence = dict(
        profile=profile.identity,
        held_out_samples=len(held_out),
        mean_relative_error=float(np.mean(errors)),
        max_relative_error=max(errors),
        evidence_kind=kind,
    )
    return profile, evidence
