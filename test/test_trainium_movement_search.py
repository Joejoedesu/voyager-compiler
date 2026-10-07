"""Endpoint legality, chain composition, selected realization and search."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import pytest

from voyager_compiler.trainium.movement_search import (
    Endpoint,
    Request,
    enumerate_chains,
    emit_chain,
    MovementSelector,
    search,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Builder
from voyager_compiler.trainium.program_analysis import analyze_selected
from voyager_compiler.trainium.plan_emitter import emit


class Harness:
    def __init__(self, hardware):
        self.hardware = hardware
        self.builder = Builder(
            [SimpleNamespace(shape=(128, 128), dtype="float32")], hardware
        )
        self.stats = Counter()
        self.expanded_isa = Counter()
        self.serial = 0

    def allocate(self):
        self.builder.program.allocate(
            strategy=(
                "size_classes"
                if getattr(self, "required_movement_storage", "")
                else "best_fit"
            )
        )
        if getattr(self, "required_movement_storage", ""):
            self.builder.program.bind_disjoint_regions()

    def emit(self, code):
        self.builder.add(code)

    def tmp(self, code):
        self.serial += 1
        name = f"t{self.serial}"
        self.emit(f"{name}={code}")
        return name


def request(
    src="HBM", dst="SBUF", swap=True, shape=(128, 128), dtype="float32"
):
    return Request(
        Endpoint(src, dtype),
        Endpoint(dst, dtype, (1, 0) if swap else (0, 1)),
        shape,
    )


def test_routes_are_constrained_by_both_endpoints():
    chains = enumerate_chains(request())
    routes = {c.modes for c in chains}
    assert ("dma_load", "stream_transpose") in routes
    assert ("dma_load", "tensor_transpose", "psum_scalar") in routes
    assert ("dma_load", "tensor_transpose", "psum_vector") in routes
    assert all(c.modes[0] == "dma_load" for c in chains)
    assert ("psum_scalar", "dma_store") in {
        c.modes for c in enumerate_chains(request("PSUM", "HBM", False))
    }
    assert ("dma_store",) not in {
        c.modes for c in enumerate_chains(request("PSUM", "HBM", False))
    }


def test_layout_dtype_tile_and_materialization_are_constraints():
    assert not any(
        "stream_transpose" in c.modes
        for c in enumerate_chains(request(shape=(16, 128)))
    )
    assert not any(
        "stream_transpose" in c.modes
        for c in enumerate_chains(request(dtype="bfloat16"))
    )
    assert all(
        c.tile == (32, 32)
        for c in enumerate_chains(request())
        if "stream_transpose" in c.modes
    )
    r = request("SBUF", "SBUF", False)
    assert all(c.modes for c in enumerate_chains(r))
    assert enumerate_chains(replace(r, materialize=False))[0].modes == ()
    with pytest.raises(ValueError, match="partition"):
        request(shape=(128, 256))
    cast = Request(Endpoint("PSUM"), Endpoint("SBUF", "bfloat16"), (128, 128))
    assert ("psum_scalar",) in {c.modes for c in enumerate_chains(cast)}


@pytest.mark.parametrize(
    "route",
    [
        ("dma_load", "tensor_transpose", "psum_scalar"),
        ("dma_load", "stream_transpose"),
    ],
)
def test_realized_chain_is_typed_allocated_and_timed(route):
    h = Harness(neuron_core(3))
    r = request()
    c = next(c for c in enumerate_chains(r) if c.modes == route)
    y = emit_chain(h, r, c, "a0")
    h.emit("out=nl.ndarray((128,128),dtype=nl.float32,buffer=nl.shared_hbm)")
    h.emit(f"nisa.dma_copy(dst=out,src={y})")
    h.builder.program.outputs = ("out",)
    h.allocate()
    code = emit(h.builder.program)
    timing = analyze_selected(h.builder.program, h.hardware)
    assert timing["hbm_write_bytes"] == 128 * 128 * 4
    assert not timing["unknown_completion"]
    assert ("nisa.nc_transpose" in code) == ("stream_transpose" in route)
    assert ("identity_float32" in h.builder.program.tensors) == (
        "tensor_transpose" in route
    )


def test_uncalibrated_chain_is_reported_not_silently_scored_as_zero():
    hw = neuron_core(3)
    r = request()
    selector = MovementSelector(hw)
    selector.choose(r)
    rejected = selector.requests[r.key]["rejected"]
    assert any(
        "nki.copy.PSUM.float32.VectorE" in x["missing_laws"] for x in rejected
    )
    assert all(
        "psum_vector" not in c["modes"]
        for c in selector.requests[r.key]["choices"]
    )


def test_search_evaluates_full_graph_and_preserves_selected_program():
    hw = neuron_core(3)
    r = request()

    def build(bindings):
        h = Harness(hw)
        h.movement_selector = MovementSelector(hw, bindings)
        choice = h.movement_selector.choose(r)
        y = emit_chain(h, r, choice, "a0")
        h.emit(
            "out=nl.ndarray((128,128),dtype=nl.float32,buffer=nl.shared_hbm)"
        )
        h.emit(f"nisa.dma_copy(dst=out,src={y})")
        h.builder.program.outputs = ("out",)
        h.allocate()
        return h, h.builder.program

    h, program, record = search(build, hw, budget=16)
    assert record["selected"]["predicted_ns"] == min(
        c["predicted_ns"]
        for c in record["candidates"]
        if c["status"] == "legal"
    )
    assert (
        analyze_selected(program, hw)["prediction_ns"]
        == record["selected"]["predicted_ns"]
    )
    assert record["nominal_combinations"] > 1
    from voyager_compiler.trainium.instruction_plan import Program

    assert emit(program) == emit(Program.load(program.record()))


def test_stage_tiling_keeps_dma_coarse_and_layout_fine():
    h = Harness(neuron_core(3))
    r = request()
    c = next(
        c
        for c in enumerate_chains(r)
        if c.modes == ("dma_load", "stream_transpose")
        and c.schedule == "stage"
        and c.order == "row"
    )
    emit_chain(h, r, c, "a0")
    counts = Counter(i.opcode for i in h.builder.program.instructions)
    assert counts["nisa.dma_copy"] == 1
    assert counts["nisa.nc_transpose"] == 16
    tiled = Harness(neuron_core(3))
    emit_chain(tiled, r, replace(c, schedule="tile"), "a0")
    counts = Counter(i.opcode for i in tiled.builder.program.instructions)
    assert counts["nisa.dma_copy"] == 16
    assert counts["nisa.nc_transpose"] == 16


def test_hardware_profile_changes_route_selection_without_search_constants():
    from voyager_compiler.trainium.instruction_plan import Program

    def selected(stream_ns):
        original = neuron_core(3)
        laws = tuple(
            (
                replace(
                    p, issue_floor_ns=stream_ns, completion_base_ns=stream_ns
                )
                if p.implementation.startswith("nki.stream_transpose.")
                else p
            )
            for p in original.timing_profile.primitives
        )
        hw = replace(
            original,
            timing_profile=replace(original.timing_profile, primitives=laws),
        )
        r = request()

        def build(bindings):
            h = Harness(hw)
            h.movement_selector = MovementSelector(hw, bindings)
            y = emit_chain(h, r, h.movement_selector.choose(r), "a0")
            h.emit(
                "out=nl.ndarray((128,128),dtype=nl.float32,buffer=nl.shared_hbm)"
            )
            h.emit(f"nisa.dma_copy(dst=out,src={y})")
            h.builder.program.outputs = ("out",)
            h.allocate()
            return h, h.builder.program

        _, program, report = search(build, hw, budget=32)
        assert emit(program) == emit(Program.load(program.record()))
        return report["selected"]["selected_bindings"][r.key]

    assert "stream_transpose" in selected(1)
    assert "stream_transpose" not in selected(100000)


def test_stream_rejects_wrong_destination_shape_in_selected_ir():
    h = Harness(neuron_core(3))
    h.emit("x=nl.ndarray((32,32),dtype=nl.float32,buffer=nl.sbuf)")
    h.emit("y=nl.ndarray((32,64),dtype=nl.float32,buffer=nl.sbuf)")
    h.emit("y[...]=nisa.nc_transpose(x,engine=nisa.vector_engine)")
    with pytest.raises(ValueError, match="32x32"):
        h.allocate()


def test_encoder_rejects_unrealizable_monolithic_stream_storage():
    h = Harness(neuron_core(3))
    r = request()
    c = next(
        c
        for c in enumerate_chains(r)
        if c.modes == ("dma_load", "stream_transpose")
    )
    emit_chain(h, r, c, "a0")
    h.builder.program.allocate()
    with pytest.raises(ValueError, match="disjoint arena"):
        emit(h.builder.program)


def test_full_intermediate_storage_must_fit_its_partition_geometry():
    r = request("SBUF", "SBUF", False, shape=(128, 512))
    chains = enumerate_chains(r)
    assert any("tensor_transpose" in c.modes for c in chains)
    assert not any(
        "tensor_transpose" in c.modes and c.schedule == "stage" for c in chains
    )


def test_strided_movement_view_preserves_root_coordinates():
    import numpy as np
    from voyager_compiler.trainium.instruction_plan import Expr
    from voyager_compiler.trainium.movement_search import (
        coordinates,
        slice_view,
    )

    h = Harness(neuron_core(3))
    h.emit("buf=nl.ndarray((128,1024),dtype=nl.float32,buffer=nl.sbuf)")
    view = "buf[nl.arange(128)[:,None],nl.arange(128)[None,:]*8]"
    sliced = slice_view(h.builder, view, 32, 16, 32, 32)
    root, actual = coordinates(h.builder, Expr.parse(sliced))
    _, expected = coordinates(h.builder, Expr.parse(view))
    assert root == "buf"
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e[32:64, 16:48])
    wrapped = "buf[nl.arange(128)[:,None],(nl.arange(128)[None,:]*8)%512]"
    with pytest.raises(ValueError, match="wrapped"):
        slice_view(h.builder, wrapped, 0, 0, 128, 128)


def test_movement_coordinates_through_large_hbm_reshape():
    import numpy as np
    from voyager_compiler.trainium.instruction_plan import Expr
    from voyager_compiler.trainium.movement_search import coordinates

    h = Harness(neuron_core(3))
    h.emit(
        "large=nl.ndarray((16,4096,4096),dtype=nl.float32,buffer=nl.shared_hbm)"
    )
    view = Expr.parse(
        "large.reshape((65536,4096))[nl.arange(64)[:,None]+8192,nl.arange(128)[None,:]+256]"
    )
    root, points = coordinates(h.builder, view)
    assert root == "large"
    assert all(p.shape == (64, 128) for p in points)
    np.testing.assert_array_equal(points[0], np.full((64, 128), 2))
    np.testing.assert_array_equal(
        points[1], np.broadcast_to(np.arange(64)[:, None], (64, 128))
    )
    np.testing.assert_array_equal(
        points[2], np.broadcast_to(np.arange(128)[None, :] + 256, (64, 128))
    )


def test_movement_coordinates_through_flattened_hbm_root():
    import numpy as np
    from voyager_compiler.trainium.instruction_plan import Expr
    from voyager_compiler.trainium.movement_search import coordinates

    h = Harness(neuron_core(3))
    h.emit(
        "large=nl.ndarray((16,4096,4096),dtype=nl.float32,buffer=nl.shared_hbm)"
    )
    flat = (
        2 * 4096 * 4096
        + np.arange(128)[:, None] * 8192
        + np.arange(64)[None, :]
    )
    view = Expr.parse(
        "large.reshape((268435456,))[33554432+nl.arange(128)[:,None]*8192+nl.arange(64)[None,:]]"
    )
    root, points = coordinates(h.builder, view)
    assert root == "large"
    for actual, expected in zip(
        points, np.unravel_index(flat, (16, 4096, 4096))
    ):
        np.testing.assert_array_equal(actual, expected)
