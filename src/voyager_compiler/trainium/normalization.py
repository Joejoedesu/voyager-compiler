"""Selectable algebraic LayerNorm recipes; no timing or benchmark identities."""

ALGORITHMS = ("centered", "moments", "shifted_moments")


def layernorm_recipe(
    step, *, algorithm="centered", fused=False, square_engine="vector"
):
    if algorithm not in ALGORITHMS or square_engine not in ("vector", "scalar"):
        raise ValueError("Unknown LayerNorm rewrite")

    def square(name, src, shape="tile"):
        return (
            step(name, "activation", (src,), shape, "square")
            if square_engine == "scalar"
            else step(name, "binary", (src, src), shape, "multiply")
        )

    s = []
    source = "input"
    if algorithm == "shifted_moments":
        s += [
            step("anchor", "first", (source,), "row"),
            step("shifted", "scalar", (source, "anchor"), op="subtract"),
        ]
        source = "shifted"
    s += [
        step("sum", "reduce", (source,), "row", "add"),
        step("mean", "scalar", ("sum",), "row", "multiply", "inverse_width"),
    ]
    if algorithm == "centered":
        s += [
            step("centered", "scalar", (source, "mean"), op="subtract"),
            square("square", "centered"),
            step("variance_sum", "reduce", ("square",), "row", "add"),
            step("variance", "scale_epsilon", ("variance_sum",), "row"),
        ]
    else:
        s += [
            square("square", source),
            step("square_sum", "reduce", ("square",), "row", "add"),
            step(
                "square_mean",
                "scalar",
                ("square_sum",),
                "row",
                "multiply",
                "inverse_width",
            ),
            square("mean_square", "mean", "row"),
            step(
                "variance",
                "difference_epsilon",
                ("square_mean", "mean_square"),
                "row",
            ),
        ]
    s += [step("inverse", "activation", ("variance",), "row", "rsqrt")]
    if fused:
        s += [step("normalized", "center_scale", (source, "mean", "inverse"))]
    else:
        if algorithm != "centered":
            s += [step("centered", "scalar", (source, "mean"), op="subtract")]
        s += [
            step("normalized", "scalar", ("centered", "inverse"), op="multiply")
        ]
    s += [
        step("scaled", "parameter", ("normalized", "weight"), op="multiply"),
        step("result", "parameter", ("scaled", "bias"), op="add"),
    ]
    return tuple(s)
