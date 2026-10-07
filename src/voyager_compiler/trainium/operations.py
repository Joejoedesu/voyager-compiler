"""Pinned NKI operation implementations, represented in the shared hardware IR."""

from functools import lru_cache
from dataclasses import dataclass
from voyager_compiler.hardware_config import (
    DataType,
    ImplementationValue as Value,
    ImplementationStep as Step,
    OperationImplementation,
)

APPLICABILITY = "Trainium2 / neuronx-cc 2.22.12471 / nki-isa-v1"


@dataclass(frozen=True)
class ISAInstruction:
    """Pinned explicit ISA interface, including legal physical layout."""

    name: str
    opcode: str
    engine: str
    result_memories: tuple
    layout: str = "partition_free"
    max_partitions: int = 128
    applicability: str = APPLICABILITY


def isa_instructions():
    result = []
    for opcode, engines, memories in (
        ("dma_copy", ("DMA",), ("SBUF", "HBM")),
        ("nc_matmul", ("TensorE",), ("PSUM",)),
        ("nc_transpose", ("VectorE",), ("SBUF",)),
        ("memset", ("VectorE",), ("SBUF", "PSUM")),
        ("tensor_copy", ("VectorE", "ScalarE"), ("SBUF",)),
        ("tensor_tensor", ("VectorE",), ("SBUF",)),
        ("tensor_scalar", ("VectorE", "ScalarE"), ("SBUF",)),
        ("tensor_reduce", ("VectorE",), ("SBUF",)),
        ("activation", ("ScalarE",), ("SBUF",)),
        ("reciprocal", ("VectorE",), ("SBUF",)),
    ):
        result.extend(
            ISAInstruction(
                f"nki.isa.{opcode}.{engine}", f"nisa.{opcode}", engine, memories
            )
            for engine in engines
        )
    return tuple(result)


@lru_cache(None)
def implementations():
    result = []
    fp32 = DataType("float32", 32)
    for dtype in (fp32, DataType("bfloat16", 16), DataType("float16", 16)):
        passes = 2 if dtype.bits == 32 else 1
        result.append(
            OperationImplementation(
                f"nki.matmul.{dtype.name}",
                "matmul",
                (
                    Value("stationary", dtype, "SBUF"),
                    Value("moving", dtype, "SBUF"),
                    Value("result", fp32, "PSUM"),
                ),
                ("stationary", "moving"),
                ("result",),
                (
                    Step(
                        "matmul",
                        "TensorE",
                        "dense",
                        "matmul",
                        (
                            ("stationary", "stationary"),
                            ("moving", "moving"),
                            ("result", "result"),
                        ),
                        (("LDWEIGHTS", passes), ("MATMUL_REGULAR", passes)),
                    ),
                ),
                APPLICABILITY,
            )
        )
        for engine in ("ScalarE", "VectorE"):
            for memory in ("SBUF", "PSUM"):
                result.append(
                    OperationImplementation(
                        f"nki.copy.{memory}.{dtype.name}.{engine}",
                        "copy",
                        (
                            Value(
                                "input",
                                fp32 if memory == "PSUM" else dtype,
                                memory,
                            ),
                            Value("result", dtype, "SBUF"),
                        ),
                        ("input",),
                        ("result",),
                        (
                            Step(
                                "copy",
                                engine,
                                "copy",
                                "copy",
                                (("src", "input"), ("result", "result")),
                                (
                                    (
                                        (
                                            "COPY_SCALAR"
                                            if engine == "ScalarE"
                                            else "COPY_VECTOR"
                                        ),
                                        1,
                                    ),
                                ),
                            ),
                        ),
                        APPLICABILITY,
                    )
                )
            result.append(
                OperationImplementation(
                    f"nki.transpose_copy.{dtype.name}.{engine}",
                    "transpose_copy",
                    (
                        Value("input", dtype, "SBUF"),
                        Value("psum", dtype, "PSUM"),
                        Value("result", dtype, "SBUF"),
                    ),
                    ("input",),
                    ("result",),
                    (
                        Step(
                            "transpose",
                            "TensorE",
                            "transpose",
                            "transpose",
                            (("src", "input"), ("result", "psum")),
                            (("LDWEIGHTS", 1), ("MATMUL_TRANSPOSE", 1)),
                        ),
                        Step(
                            "copy",
                            engine,
                            "copy",
                            "copy",
                            (("src", "psum"), ("result", "result")),
                            (
                                (
                                    (
                                        "COPY_SCALAR"
                                        if engine == "ScalarE"
                                        else "COPY_VECTOR"
                                    ),
                                    1,
                                ),
                            ),
                        ),
                    ),
                    APPLICABILITY,
                    shared_constant_bytes=128 * 128 * dtype.bits // 8,
                )
            )
    from .lowering import RECIPES

    for name, recipe in RECIPES.items():
        inputs = tuple(
            dict.fromkeys(
                x
                for step in recipe
                for x in step.inputs
                if x not in {s.name for s in recipe}
            )
        )
        values = tuple(
            Value(x, fp32, "SBUF", "row_partition")
            for x in (*inputs, *(s.name for s in recipe))
        )
        steps = []
        for step in recipe:
            engine = (
                "ScalarE" if step.instruction == "activation" else "VectorE"
            )
            op = {
                "binary": "binary",
                "scalar": "scalar",
                "scale_epsilon": "scalar",
                "reduce": "reduce",
                "activation": "activation",
                "reciprocal": "reciprocal",
                "parameter": "scalar_tensor",
            }[step.instruction]
            bindings = [("src", step.inputs[0]), ("result", step.name)]
            if op in ("binary", "scalar_tensor"):
                bindings.append(("rhs", step.inputs[1]))
            # Scalar-broadcast instructions have a named optional row operand.
            if op == "scalar" and len(step.inputs) > 1:
                op = "scalar_tensor"
                bindings.append(("rhs", step.inputs[1]))
            steps.append(
                Step(step.name, engine, "arithmetic", op, tuple(bindings))
            )
        result.append(
            OperationImplementation(
                f"nki.{name}.float32",
                name,
                values,
                inputs,
                ("result",),
                tuple(steps),
                APPLICABILITY,
            )
        )
    return tuple(result)


def implementation(name, hardware=None):
    if hardware is not None:
        return hardware.operation_implementation(name)
    for impl in implementations():
        if impl.name == name:
            return impl
    raise ValueError(f"Unknown NKI implementation {name}")


def pool_reduction_steps(kh, kw, width):
    """Selected separable rectangle maximum: vertical strips, horizontal views.

    Each entry fixes the result name, two (value, free-axis offset) operands,
    and physical free extent. Search and typed instruction lowering consume
    this same expansion; there are kh + kw - 2 programmable maxima.
    """
    current = "row0"
    for row in range(1, kh):
        name = "result" if row == kh - 1 and kw == 1 else f"vertical{row}"
        yield name, (current, 0), (f"row{row}", 0), width
        current = name
    vertical = current
    for column in range(1, kw):
        name = "result" if column == kw - 1 else f"horizontal{column}"
        yield name, (current, 0), (vertical, column), width - kw + 1
        current = name
