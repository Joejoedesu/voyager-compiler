"""NKI ISA expansion contract for the pinned neuronx-cc 2.22 / Trainium2 path.

Counts describe lowering, not latency: TensorE pipelines LDWEIGHTS and MATMUL.
The documented back-to-back cost already includes that pipeline and FP32 work.
It must not be multiplied again by the expanded instruction count. Completion,
SDK prologue/epilogue, and issue costs are not supplied by those throughput
formulas; reports explicitly distinguish this analytical service estimate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Expansion:
    instructions: tuple
    tensor_cycles: float = 0
    vector_cycles: float = 0
    shared_constant_bytes: int = 0
    scalar_cycles: float = 0
    implementation: str = ""
    timing_implementation: str = ""


def matmul(m, n, k, bits, hardware=None, dtype=None):
    if not (0 < m <= 512 and 0 < n <= 128 and 0 < k <= 128):
        raise ValueError("Invalid TensorE instruction shape")
    if bits not in (16, 32):
        raise ValueError("ISA timing currently supports FP32/FP16/BF16")
    from .operations import implementation

    dtype = dtype or ("float32" if bits == 32 else "bfloat16")
    if dtype not in (("float32",) if bits == 32 else ("float16", "bfloat16")):
        raise ValueError("Matmul dtype does not match operand width")
    from .timing import matmul_timing_name

    contract = implementation(f"nki.matmul.{dtype}", hardware)
    return Expansion(
        contract.instruction_counts(),
        tensor_cycles=(4 if bits == 32 else 1) * max(min(64, n), m),
        implementation=contract.name,
        timing_implementation=matmul_timing_name(m, n, k, dtype),
    )


def transpose(
    partitions, free, hardware=None, engine="ScalarE", dtype="float32"
):
    from .operations import implementation

    contract = implementation(f"nki.transpose_copy.{dtype}.{engine}", hardware)
    if not (0 < partitions <= 128 and 0 < free <= 128):
        raise ValueError("Invalid TensorE transpose shape")
    # Native nc_transpose on the pinned v3 compiler uses one shared 128x128
    # byte identity constant. Verified in NEFF, including FP32 input kernels.
    return Expansion(
        contract.instruction_counts(),
        tensor_cycles=max(partitions, min(64, free)),
        scalar_cycles=max(64, partitions),
        shared_constant_bytes=contract.shared_constant_bytes,
        implementation=contract.name,
    )


def copy(partitions, free):
    if not (0 < partitions <= 128 and free > 0):
        raise ValueError("Invalid VectorE copy shape")
    return Expansion((("COPY_VECTOR", 1),), vector_cycles=max(64, free))
