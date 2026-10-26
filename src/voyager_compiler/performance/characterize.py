"""Characterize real candidates through normal compilation and target timing."""

import json
from dataclasses import dataclass
from pathlib import Path

from voyager_compiler.artifacts import hardware_digest, to_data
from voyager_compiler.pipeline import compile_workload

from .tuning import Measurement, fit_profile


@dataclass(frozen=True)
class CharacterizationCase:
    name: str
    workload: object
    family: str
    options: tuple = ()
    held_out: bool = False


def characterize(target, cases, output_dir, *, config=None, measure=None):
    """Compile each case, collect attributed timings, then fit/hold out.

    ``measure(result, case, config)`` returns a Measurement from an accelerator,
    RTL, or cycle-accurate simulator. It may be the target's registered hook.
    No host wall time is promoted to accelerator performance evidence.
    """
    cases = tuple(cases)
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("characterization case names must be unique")
    if (
        not cases
        or not any(c.held_out for c in cases)
        or all(c.held_out for c in cases)
    ):
        raise ValueError(
            "characterization requires training and held-out cases"
        )
    measure = measure or target.characterize
    if measure is None:
        raise ValueError(
            "provide an accelerator measurement adapter or register TargetSpec.characterize"
        )
    config = config or target.hardware()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    training, held_out = [], []
    for index, case in enumerate(cases):
        result = compile_workload(
            case.workload,
            target,
            output / f"case-{index:04d}",
            config=config,
            options=dict(case.options),
        )
        measurement = measure(result, case, config)
        result.program.assert_frozen()
        if not isinstance(measurement, Measurement):
            raise TypeError("measurement adapter must return a Measurement")
        if (
            measurement.case,
            measurement.family,
            measurement.target,
            measurement.hardware_digest,
        ) != (case.name, case.family, target.name, hardware_digest(config)):
            raise ValueError(
                "measurement identity differs from the compiled case"
            )
        (held_out if case.held_out else training).append(measurement)
        (output / "measurements.json").write_text(
            json.dumps(to_data(training + held_out), indent=2) + "\n"
        )
    profile, validation = fit_profile(output.name, training, held_out)
    (output / "profile.json").write_text(
        json.dumps(to_data(profile), indent=2) + "\n"
    )
    (output / "held-out.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    return profile, validation
