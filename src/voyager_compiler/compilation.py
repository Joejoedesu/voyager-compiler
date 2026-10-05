"""One resolved hardware/policy contract across compilation and realization.

The record is run metadata in compilation.json, not an executable collateral.
Restoration uses registered backends only; metadata cannot import arbitrary code.
"""

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path


def _encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def hardware_fingerprint(hardware):
    return hashlib.sha256(
        json.dumps(hardware, default=_encode, sort_keys=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CompilerContext:
    hardware: object
    policy: object
    cost_tradeoff: bool = True
    runtime_tolerance: float = 0.02

    def __post_init__(self):
        from .targets import validate_bufferized_target

        validate_bufferized_target(self.hardware)
        if self.policy.config != self.hardware:
            raise ValueError(
                "Mapping policy hardware differs from compilation hardware"
            )
        if type(self.cost_tradeoff) is not bool:
            raise TypeError("cost_tradeoff must be boolean")
        object.__setattr__(
            self,
            "cost_tradeoff",
            self.cost_tradeoff and not self.policy.speed_only,
        )
        if self.runtime_tolerance < 0 or not math.isfinite(
            self.runtime_tolerance
        ):
            raise ValueError("Runtime tolerance must be finite and nonnegative")

    @classmethod
    def resolve(
        cls,
        hardware,
        policy=None,
        *,
        cost_tradeoff=True,
        runtime_tolerance=None,
    ):
        from .targets import get_backend

        policy = policy or get_backend(hardware.backend).mapping_policy(
            hardware
        )
        return cls(
            hardware,
            policy,
            cost_tradeoff and not policy.speed_only,
            0.02 if runtime_tolerance is None else runtime_tolerance,
        )

    def record(self):
        return dict(
            version=1,
            backend=self.hardware.backend,
            hardware_sha256=hardware_fingerprint(self.hardware),
            policy_type=type(self.policy).__module__
            + "."
            + type(self.policy).__qualname__,
            policy=self.policy.options(),
            cost_tradeoff=self.cost_tradeoff,
            runtime_tolerance=self.runtime_tolerance,
        )

    def write(self, root):
        path = Path(root) / "compilation.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        record["compiler"] = self.record()
        path.write_text(json.dumps(record, default=_encode, indent=2) + "\n")

    @classmethod
    def from_artifacts(cls, root, hardware):
        path = Path(root) / "compilation.json"
        record = (
            json.loads(path.read_text()).get("compiler")
            if path.exists()
            else None
        )
        if record is None:
            raise ValueError(
                "Missing resolved compiler policy in compilation.json; supply an explicit context for legacy artifacts"
            )
        if (
            record["version"] != 1
            or record["backend"] != hardware.backend
            or record["hardware_sha256"] != hardware_fingerprint(hardware)
        ):
            raise ValueError(
                "Artifact hardware/backend contract differs from converter hardware"
            )
        from .targets import get_backend

        policy = get_backend(hardware.backend).restore_mapping_policy(
            hardware, record["policy"]
        )
        context = cls.resolve(
            hardware,
            policy,
            cost_tradeoff=record["cost_tradeoff"],
            runtime_tolerance=record["runtime_tolerance"],
        )
        if context.record() != record:
            raise ValueError(
                "Artifact policy cannot be restored by the registered backend"
            )
        return context

    def check_artifacts(self, root):
        path = Path(root) / "compilation.json"
        recorded = (
            json.loads(path.read_text()).get("compiler")
            if path.exists()
            else None
        )
        if recorded is not None and recorded != self.record():
            raise ValueError(
                "Converter context differs from the policy used to compile these artifacts"
            )

    def realize(self, root, **options):
        from .targets import get_backend

        self.check_artifacts(root)
        return get_backend(self.hardware.backend).realize(root, self, **options)
