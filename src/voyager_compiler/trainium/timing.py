"""Shape-dependent timing laws owned by the Trainium hardware description.

Calibration is from isolated primitives, never application latency. Values are
ns; service_scale multiplies the documented shape-dependent service estimate.
An absent law stays explicitly unknown. Backend allocation policy is separate.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrimitiveTiming:
    implementation: str
    issue_floor_ns: float
    service_scale: float
    completion_base_ns: float
    completion_service_scale: float = 1.0

    def __post_init__(self):
        if not self.implementation or any(
            not math.isfinite(x) or x < 0
            for x in (
                self.issue_floor_ns,
                self.service_scale,
                self.completion_base_ns,
                self.completion_service_scale,
            )
        ):
            raise ValueError("Invalid primitive timing law")

    def evaluate(self, service):
        occupancy = max(self.issue_floor_ns, self.service_scale * service)
        return occupancy, max(
            occupancy,
            self.completion_base_ns + self.completion_service_scale * service,
        )


@dataclass(frozen=True)
class DmaTiming:
    issue_ns: float
    dispatch_ns: float
    payload_floor_ns: float
    payload_scale: float
    notification_ns: float
    payload_base_ns: float = 0

    def __post_init__(self):
        if any(
            not math.isfinite(x) or x < 0
            for x in (
                self.issue_ns,
                self.dispatch_ns,
                self.payload_floor_ns,
                self.payload_scale,
                self.notification_ns,
                self.payload_base_ns,
            )
        ):
            raise ValueError("Invalid DMA timing law")

    def payload(self, ideal):
        return max(
            self.payload_floor_ns,
            self.payload_base_ns + ideal * self.payload_scale,
        )


@dataclass(frozen=True)
class TrainiumTimings:
    name: str = "uncharacterized"
    compiler: str = "neuronx-cc 2.22.12471 / Trainium2"
    primitives: tuple[PrimitiveTiming, ...] = ()
    load: DmaTiming | None = None
    store: DmaTiming | None = None
    fixed_kernel_ns: float = 0
    evidence: str = ""

    def __post_init__(self):
        if len({x.implementation for x in self.primitives}) != len(
            self.primitives
        ):
            raise ValueError("Duplicate primitive timing law")
        if not math.isfinite(self.fixed_kernel_ns) or self.fixed_kernel_ns < 0:
            raise ValueError("Invalid kernel overhead")

    def operation(self, name):
        return next(
            (x for x in self.primitives if x.implementation == name), None
        )


def measured_timings():
    """Load the versioned, reproducible primitive characterization record."""
    import json
    from pathlib import Path

    path = Path(__file__).with_name("timing_trainium2.json")
    if not path.exists():
        return TrainiumTimings()
    record = json.loads(path.read_text())
    record["primitives"] = tuple(
        PrimitiveTiming(**x) for x in record["primitives"]
    )
    for key in ("load", "store"):
        record[key] = DmaTiming(**record[key]) if record.get(key) else None
    return TrainiumTimings(**record)


def matmul_timing_name(m, n, k, dtype):
    """Select a measured geometry without changing the physical ISA recipe.

    The short-K law is only characterized for FP32 stationary (64,128),
    moving (64,512). Other shapes retain the existing analytical profile.
    """
    if dtype == "float32" and (m, n, k) == (512, 128, 64):
        return "nki.matmul.float32.K64_N128_M512"
    return f"nki.matmul.{dtype}"


@dataclass
class StreamTransposeTiming:
    """Context in compiled Vector-engine issue order, not execution time.

    Only contiguous FP32 32x32 SBUF tiles have measured laws. Control waits
    preserve the previous quadrant pair; other Vector payloads reset it.
    A switch is an observed context cost, not a claimed bank-conflict model.
    """

    previous_quadrants: tuple[int, int] | None = None

    def reset(self):
        self.previous_quadrants = None

    def select(
        self,
        *,
        dtype,
        source_dtype,
        engine,
        partitions,
        source_free,
        destination_free,
        source_address,
        destination_address,
        strides,
    ):
        if not (
            dtype == source_dtype == "float32"
            and engine == "VectorE"
            and partitions == source_free == destination_free == 32
            and 0 <= source_address < 0x2000000
            and 0 <= destination_address < 0x2000000
            and strides
            and all(s == 1 for s in strides)
        ):
            self.reset()
            return None
        quadrants = (
            source_address // (32 * 262144),
            destination_address // (32 * 262144),
        )
        kind = "steady" if quadrants == self.previous_quadrants else "switch"
        self.previous_quadrants = quadrants
        return f"nki.stream_transpose.float32.SBUF.VectorE.{kind}"
