"""TensorE operand choices shared by search and selected-instruction planning.

These are local ISA panels inside a shared software tile, not new HBM tiles.
The conversion graph uses existing primitive laws and records unknown latency.
"""

from dataclasses import replace
from functools import lru_cache
from collections import Counter

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    RepeatedGraph,
)
from .execution import matmul_panels
from . import isa


@lru_cache(maxsize=2048)
def matrix_choice(
    config,
    m,
    n,
    k,
    bits,
    transposed,
    tuning,
    dtype=None,
    output_dtype=None,
    input_row=False,
    output_row=False,
    weight_layout="generic",
):
    from .dependencies import evaluate_graph

    choices = (
        ("weights", "activations")
        if tuning.matmul_orientation == "auto" and tuning.isa_lowering
        else (
            "weights" if not tuning.isa_lowering else tuning.matmul_orientation,
        )
    )
    candidates = []
    for orientation in choices:
        fixed = replace(tuning, matmul_orientation=orientation)
        graph, counts = orientation_graph(
            config,
            m,
            n,
            k,
            bits,
            transposed,
            fixed,
            dtype,
            output_dtype,
            input_row,
            output_row,
            weight_layout,
        )
        timing = evaluate_graph(graph)
        candidates.append(
            dict(
                orientation=orientation,
                prediction_ns=timing.duration_ns,
                matmul_calls=sum(1 for _ in matmul_panels(m, n, k, orientation))
                * ((k + 127) // 128),
                transpose_calls=counts.get("MATMUL_TRANSPOSE", 0),
                unknown_completion=list(timing.unknown_latency),
            )
        )
    selected = min(candidates, key=lambda c: c["prediction_ns"])
    return dict(
        m=m,
        n=n,
        k=k,
        transposed=transposed,
        input_row=input_row,
        output_row=output_row,
        weight_layout=weight_layout,
        orientation=selected["orientation"],
        candidates=candidates,
    )


@lru_cache(maxsize=2048)
def orientation_graph(
    config,
    m,
    n,
    k,
    bits,
    transposed,
    tuning,
    dtype=None,
    output_dtype=None,
    input_row=False,
    output_row=False,
    weight_layout="generic",
):
    from .dependencies import timed_event, engine_clock, _weights_compute_graph

    dtype = dtype or ("float32" if bits == 32 else "bfloat16")
    output_dtype = output_dtype or dtype
    orientation = tuning.matmul_orientation
    if weight_layout not in ("generic", "k_partitioned"):
        raise ValueError("Unknown physical weight layout")
    if (
        orientation == "weights"
        and not input_row
        and not output_row
        and weight_layout == "generic"
    ):
        return _weights_compute_graph(
            config, m, n, k, bits, transposed, tuning, dtype, output_dtype
        )
    if orientation not in ("weights", "activations"):
        raise ValueError(
            "Conversion graph requires an explicit operand orientation"
        )
    nodes, counts = [], Counter()
    last_tensor_matmul = False
    last_tensor_index = -1
    engine = "ScalarE" if tuning.copy_policy == "scalar" else "VectorE"

    def emit(
        resource,
        service,
        deps=(),
        implementation="",
        step=0,
        role="copy",
        timing="",
        timing_override=None,
    ):
        nonlocal last_tensor_matmul, last_tensor_index
        if resource == "TensorE":
            last_tensor_matmul = implementation.startswith("nki.matmul.")
            last_tensor_index = len(nodes)
        idx = len(nodes)
        nodes.append(
            timed_event(
                config,
                f"{role}_{idx}",
                resource,
                service,
                tuple(Dependency(d) for d in deps),
                implementation=implementation,
                step=(
                    None if implementation == "nki.isa.memset.VectorE" else step
                ),
                timing_implementation=timing,
                timing_override=timing_override,
            )
        )
        return idx

    def copy(free, deps=(), memory="SBUF", out_dtype=dtype, resource="VectorE"):
        return emit(
            resource,
            max(64, free) / engine_clock(config, resource),
            deps,
            f"nki.copy.{memory}.{out_dtype}.{resource}",
        )

    def transpose(partitions, free, deps=(), out_dtype=dtype):
        expansion = isa.transpose(partitions, free, config, engine, out_dtype)
        counts.update(
            {
                a: b
                for a, b in expansion.instructions
                if a in ("LDWEIGHTS", "MATMUL_TRANSPOSE")
            }
        )
        clear = emit(
            "VectorE",
            max(64, partitions) / engine_clock(config, "VectorE"),
            deps,
            "nki.isa.memset.VectorE",
            role="psum_clear",
        )
        t = emit(
            "TensorE",
            expansion.tensor_cycles / config.frequency,
            (clear,),
            expansion.implementation,
            role="transpose",
        )
        return emit(
            engine,
            expansion.scalar_cycles / engine_clock(config, engine),
            (t,),
            expansion.implementation,
            step=1,
        )

    weights, activations = {}, {}
    reuse = tuning.matmul_operands == "reuse"
    staged = tuning.matmul_operands == "staged"
    for mi, ni, mm, nn, kpanels in matmul_panels(m, n, k, orientation):
        moving, stationary = (mm, nn) if orientation == "weights" else (nn, mm)
        acc = emit(
            "VectorE",
            max(64, moving) / engine_clock(config, "VectorE"),
            implementation="nki.isa.memset.VectorE",
            role="psum_clear",
        )
        for ki, kk in kpanels:
            deps = [acc]
            ar = None
            key = mi, ki
            if input_row:
                if reuse and key in activations:
                    ar = activations[key]
                else:
                    ar = transpose(mm, kk, (copy(kk),))
                    if reuse:
                        activations[key] = ar
                deps.append(ar)
            if staged and (m > 512 or n > 128 or k > 128):
                deps.append(copy(mm, () if ar is None else (ar,)))
            key = ni, ki
            if weight_layout == "k_partitioned":
                # DMA already produced a K-partitioned resident operand.
                # The view is valid across row tiles until its buffer is written.
                if staged and (m > 512 or n > 128 or k > 128):
                    deps.append(copy(nn))
            elif reuse and key in weights:
                deps.append(weights[key])
            else:
                ready = []
                for offset in range(0, nn, 128):
                    width = min(128, nn - offset)
                    wd = (
                        (copy(kk),)
                        if staged and (m > 512 or n > 128 or k > 128)
                        else ()
                    )
                    if not transposed:
                        wd = (transpose(width, kk, wd),)
                    if nn > 128:
                        wd = (copy(width, wd),)
                    ready.extend(wd)
                deps.extend(ready)
                if reuse and ready:
                    # Assemble the moving panel before its consumer; each subcopy
                    # writes a disjoint range of the selected SBUF tensor.
                    weights[key] = tuple(ready)
            # The cached moving panel can have several producers.
            flat = []
            for dep in deps:
                flat.extend(dep if isinstance(dep, tuple) else (dep,))
            a_stride = (
                1
                if input_row or (staged and (m > 512 or n > 128 or k > 128))
                else (k + 127) // 128
            )
            b_stride = (
                (k + 127) // 128
                if transposed
                and weight_layout == "generic"
                and not staged
                and (orientation == "weights" or nn <= 128)
                else 1
            )
            moving_stride, stationary_stride = (
                (a_stride, b_stride)
                if orientation == "weights"
                else (b_stride, a_stride)
            )
            expansion = isa.matmul(
                moving,
                stationary,
                kk,
                bits,
                config,
                dtype,
                moving_stride=moving_stride,
                stationary_stride=stationary_stride,
                streaming=last_tensor_matmul
                and not any(d > last_tensor_index for d in flat),
            )
            counts.update(dict(expansion.instructions))
            idx = emit(
                "TensorE",
                expansion.tensor_cycles / config.frequency,
                flat,
                expansion.implementation,
                role="matmul",
                timing=expansion.timing_implementation,
                timing_override=expansion.timing_override,
            )
            # PSUM forwarding is distinct from completion of an operand read.
            nodes[idx] = replace(
                nodes[idx],
                dependencies=tuple(
                    (
                        replace(d, milestone="forward")
                        if d.source == acc
                        and nodes[acc].name.startswith("matmul")
                        else d
                    )
                    for d in nodes[idx].dependencies
                ),
            )
            acc = idx
        evict = copy(moving, (acc,), "PSUM", output_dtype, engine)
        if orientation == "weights" and output_row:
            for row in range(0, mm, 128):
                r = min(128, mm - row)
                t = transpose(
                    nn,
                    r,
                    (copy(r, (evict,), out_dtype=output_dtype),),
                    output_dtype,
                )
                copy(nn, (t,), out_dtype=output_dtype)
        elif orientation == "activations" and not output_row:
            for col in range(0, nn, 128):
                c = min(128, nn - col)
                transpose(
                    mm,
                    c,
                    (copy(c, (evict,), out_dtype=output_dtype),),
                    output_dtype,
                )
        elif orientation == "activations" and output_row:
            copy(nn, (evict,), out_dtype=output_dtype)
    return RepeatedGraph(tuple(nodes)), counts
