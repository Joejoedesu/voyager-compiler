"""Selected-plan invariants, independent of generated source spelling."""

from dataclasses import replace
from types import SimpleNamespace
import json
import pytest

from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Builder, Program
from voyager_compiler.trainium.plan_emitter import emit


def selected():
    b = Builder(
        [SimpleNamespace(shape=(32, 64), dtype="float32")], neuron_core(3)
    )
    b.add("x = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    b.add(
        "y = nisa.tensor_scalar(x, op0=nl.multiply, operand0=2, engine=nisa.vector_engine)"
    )
    b.add("out = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.shared_hbm)")
    b.add("nisa.dma_copy(dst=out, src=y)")
    b.program.outputs = ("out",)
    b.program.allocate()
    return b.program


def test_serialized_plan_has_no_generated_source_and_emits_deterministically():
    p = selected()
    record = json.dumps(p.record())
    assert "def kernel" not in record
    assert emit(p) == emit(Program.load(json.loads(record)))
    assert "ncc.sbuf.alloc" in emit(p)
    assert p.placements["x"].byte_address != p.placements["y"].byte_address


@pytest.mark.parametrize(
    "fault",
    [
        "layout",
        "layout_memory",
        "memory",
        "placement",
        "dependency",
        "implementation",
        "operand",
        "engine",
        "overlap",
    ],
)
def test_incomplete_or_inconsistent_selection_is_rejected(fault):
    p = selected()
    if fault == "layout":
        p.tensors["x"] = replace(p.tensors["x"], layout="")
    elif fault == "layout_memory":
        p.tensors["x"] = replace(p.tensors["x"], layout="contiguous")
    elif fault == "memory":
        p.tensors["x"] = replace(p.tensors["x"], memory="unknown")
    elif fault == "placement":
        del p.placements["x"]
    elif fault == "dependency":
        p.instructions[1] = replace(p.instructions[1], dependencies=())
    elif fault == "implementation":
        p.instructions[1] = replace(p.instructions[1], implementation="")
    elif fault == "operand":
        p.instructions[1] = replace(p.instructions[1], reads=())
    elif fault == "engine":
        p.instructions[1] = replace(
            p.instructions[1], implementation="nki.isa.tensor_scalar.ScalarE"
        )
    else:
        p.placements["y"] = replace(
            p.placements["y"], byte_address=p.placements["x"].byte_address
        )
    with pytest.raises(ValueError):
        emit(p)


def test_physical_reuse_requires_completion_of_previous_reader():
    p = selected()
    # x and y overlap in lifetime even though x's final read defines y.
    p.placements["y"] = replace(
        p.placements["y"], byte_address=p.placements["x"].byte_address
    )
    with pytest.raises(ValueError, match="overlap"):
        p.validate()


def test_hidden_language_tensor_operations_are_not_accepted():
    b = Builder([], neuron_core(3))
    with pytest.raises(ValueError, match="language operation"):
        b.add("x = nl.full((32,64), 0, dtype=nl.float32)")


def test_layernorm_parameter_dma_is_retained_across_tiles():
    from voyager_compiler.trainium.lowering import tile_graph
    from voyager_compiler.trainium.execution import TrainiumTuning
    from voyager_compiler.trainium.dependencies import _active_count

    g = tile_graph(
        neuron_core(3),
        ("layer_norm",),
        (128, 8192),
        ((128, 8192), (8192,), (8192,)),
        TrainiumTuning(),
        repetitions=32,
    )
    # Two data panels and two output panels per tile, four parameter loads once.
    assert (
        sum(_active_count(n, 32) for n in g.nodes if n.resource == "DMAIssue")
        == 132
    )
    assert any(n.resource == "TensorE" and n.period == 32 for n in g.nodes)


def test_pool_uses_three_vertical_halos_and_one_output_store():
    from voyager_compiler.trainium.lowering import spatial_pool_graph
    from voyager_compiler.trainium.execution import TrainiumTuning

    g = spatial_pool_graph(
        neuron_core(3), (1, 30, 192, 1), (1, 32, 194, 1), TrainiumTuning(), 9
    )
    assert sum(n.resource == "DMAIssue" for n in g.nodes) == 4
    assert sum(n.resource == "VectorE" for n in g.nodes) == 4


def test_shared_allocator_keeps_original_choices_with_lifetime_index():
    import random
    from voyager_compiler.codegen.transform.bufferize.memory_planning import (
        _greedy_best_fit,
    )

    rng = random.Random(29)
    for count in (0, 1, 32, 300):
        items = []
        for key in range(count):
            first = rng.randrange(1000)
            items.append(
                (
                    key,
                    rng.randrange(1, 256),
                    first,
                    first + rng.randrange(100),
                    rng.choice((1, 16, 32)),
                )
            )
        placed, expected, total = [], {}, 0
        for key, size, lo, hi, align in sorted(
            items, key=lambda it: (-it[1], it[2])
        ):
            off = 0
            for start, end in sorted(
                (s, e) for a, b, s, e in placed if a <= hi and lo <= b
            ):
                if ((off + align - 1) // align) * align + size <= start:
                    break
                off = max(off, end)
            off = ((off + align - 1) // align) * align
            placed.append((lo, hi, off, off + size))
            expected[key] = off
            total = max(total, off + size)
        assert _greedy_best_fit(items) == (expected, total)


def test_physical_reuse_chain_keeps_partial_range_owners():
    from voyager_compiler.trainium.instruction_plan import Placement

    p = Program({}, {}, [], (), ())

    def slot(start, size, first, last):
        return Placement("SBUF", 0, start, None, size, first, last)

    p.placements = {
        "a": slot(0, 64, 0, 2),
        "b": slot(0, 32, 3, 4),
        "c": slot(32, 32, 5, 6),
        "d": slot(0, 64, 7, 8),
    }
    assert set(p.reuse_edges()) == {(2, 3), (2, 5), (4, 7), (6, 7)}


def test_transpose_preserves_bfloat16_psum_type():
    b = Builder(
        [SimpleNamespace(shape=(32, 64), dtype="bfloat16")], neuron_core(3)
    )
    b.add("x = nl.ndarray((32,64), dtype=nl.bfloat16, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    b.add("y = nisa.nc_matmul(x, x, is_transpose=True)")
    assert b.program.tensors["y"].dtype == "bfloat16"


def test_recorded_loop_encoding_only_changes_affine_hbm_addresses():
    from voyager_compiler.trainium.compact_encoding import select, encode

    p = selected()
    p.repeated_regions = [
        dict(name="tiles", iterations=[(i, i + 1) for i in range(8)])
    ]
    lines = [
        f"nisa.dma_copy(dst=x, src=a0[{i*32}:({i*32}+32), :])" for i in range(8)
    ]
    p.encoding_loops = select(p, lines)
    assert len(p.encoding_loops) == 1
    assert any("nl.sequential_range(8)" in line for line in encode(p, lines))
    changed = list(lines)
    changed[-1] = changed[-1].replace("224", "225")
    with pytest.raises(ValueError, match="expansion"):
        encode(p, changed)
    local = [
        f"nisa.dma_copy(dst=x[{i*32}:({i*32}+32), :], src=a0)" for i in range(8)
    ]
    assert not select(p, local)


def test_compact_address_rules_preserve_traversal_wraps():
    from voyager_compiler.trainium.compact_encoding import fit, shift

    values = [48 + 16 * i - 64 * ((i + 1) // 4) for i in range(32)]
    rule = fit(values)
    assert rule is not None
    assert [values[0] + shift(rule, i) for i in range(32)] == values
    assert fit([0, 1, 4, 9, 16, 25, 36, 49]) is None


def test_psum_placement_uses_idle_banks_before_reuse():
    b = Builder(
        [SimpleNamespace(shape=(32, 32), dtype="float32")], neuron_core(3)
    )
    b.add("x = nl.ndarray((32,32), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    for i in range(12):
        b.add(f"p{i} = nisa.nc_matmul(x, x)")
        b.add(f"y{i} = nisa.tensor_copy(p{i}, engine=nisa.scalar_engine)")
    b.program.allocate()
    banks = [b.program.placements[f"p{i}"].bank for i in range(12)]
    assert banks == list(range(8)) + list(range(4))
    b.program.validate()


def test_physical_reuse_retires_independent_readers_on_different_engines():
    b = Builder(
        [SimpleNamespace(shape=(32, 64), dtype="float32")], neuron_core(3)
    )
    b.add("x = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    b.add("y = nisa.tensor_copy(x, engine=nisa.scalar_engine)")
    b.add("v = nisa.tensor_copy(x, engine=nisa.vector_engine)")
    b.add("z = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=z, src=a0)")
    b.add("out = nisa.tensor_tensor(y, v, op=nl.add)")
    p = b.program
    p.allocate()
    assert p.placements["x"].byte_address == p.placements["z"].byte_address
    assert {1, 2}.issubset(p.instructions[3].dependencies)
    p.instructions[3] = replace(
        p.instructions[3],
        dependencies=tuple(d for d in p.instructions[3].dependencies if d != 1),
    )
    with pytest.raises(ValueError, match="physical reuse completion"):
        p.validate()


def test_separable_pool_expansion_preserves_rectangular_maximum():
    import numpy as np
    from voyager_compiler.trainium.operations import pool_reduction_steps

    rng = np.random.default_rng(29)
    for kh, kw in ((1, 1), (1, 3), (3, 1), (3, 3), (2, 5)):
        raw = rng.normal(size=(kh, 7, 19))
        values = {f"row{i}": raw[i] for i in range(kh)}
        for name, left, right, width in pool_reduction_steps(kh, kw, 19):
            values[name] = np.maximum(
                values[left[0]][:, left[1] : left[1] + width],
                values[right[0]][:, right[1] : right[1] + width],
            )
        actual = values.get("result", values["row0"])
        expected = np.maximum.reduce(
            [raw[r, :, c : c + 20 - kw] for r in range(kh) for c in range(kw)]
        )
        np.testing.assert_array_equal(actual, expected)


def test_compact_encoding_preserves_nested_outer_traversal_wraps():
    from voyager_compiler.trainium.compact_encoding import fit, shift

    values = [
        17 + 64 * i - 512 * ((i + 1) // 8) + 2048 * ((i + 9) // 128)
        for i in range(512)
    ]
    rule = fit(values)
    assert rule is not None
    assert [values[0] + shift(rule, i) for i in range(len(values))] == values


def test_disjoint_storage_binding_keeps_addresses_and_instructions():
    from copy import deepcopy

    p = selected()
    placements = deepcopy(p.placements)
    instructions = list(p.instructions)
    p.bind_disjoint_regions()
    p.validate()
    assert len(p.storage_regions) > 1
    assert p.placements == placements and p.instructions == instructions
    source = emit(p)
    assert source.count("buffer=ncc.sbuf.alloc") == len(p.storage_regions)
    assert emit(Program.load(p.record())) == source
    p.storage_regions[1]["start"] = p.storage_regions[0]["start"]
    with pytest.raises(ValueError, match="overlapping selected storage"):
        emit(p)


def test_size_class_placement_preserves_instructions_and_separates_slot_sizes():
    b = Builder(
        [SimpleNamespace(shape=(32, 64), dtype="float32")], neuron_core(3)
    )
    b.add("x = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    b.add("y = nisa.tensor_reduce(x, op=nl.add, axis=[1], dtype=nl.float32)")
    b.add("out = nl.ndarray((32,1), dtype=nl.float32, buffer=nl.shared_hbm)")
    b.add("nisa.dma_copy(dst=out, src=y)")
    b.program.outputs = ("out",)
    original = tuple(
        (i.opcode, i.args, i.kwargs, i.destination)
        for i in b.program.instructions
    )
    b.program.allocate(strategy="size_classes")
    b.program.bind_disjoint_regions()
    b.program.validate()
    assert original == tuple(
        (i.opcode, i.args, i.kwargs, i.destination)
        for i in b.program.instructions
    )
    slots = list(b.program.placements.values())
    assert len(b.program.storage_regions) >= 2
    for i, left in enumerate(slots):
        for right in slots[i + 1 :]:
            if left.bytes_per_partition != right.bytes_per_partition:
                assert not b.program.overlap(left, right)
    assert emit(b.program) == emit(Program.load(b.program.record()))
    with pytest.raises(ValueError, match="unplaced"):
        b.program.allocate(strategy="size_classes")
    fresh = Program.load(b.program.record())
    fresh.placements = {}
    with pytest.raises(ValueError, match="does not fit"):
        fresh.allocate(sbuf_bytes=128, strategy="size_classes")


def test_compiler_allocation_preserves_logical_validation():
    p = selected()
    p.placements.clear()
    p.encoding_storage = "compiler"
    restored = Program.load(json.loads(json.dumps(p.record())))
    source = emit(restored)
    assert "buffer=nl.sbuf" in source and "ncc.sbuf.alloc" not in source
    assert "y[...] = nisa.tensor_scalar" in source
    restored.instructions[1] = replace(restored.instructions[1], dependencies=())
    with pytest.raises(ValueError, match="missing operand/reuse"):
        emit(restored)


def test_compiler_allocation_rejects_fixed_placements():
    p = selected()
    p.encoding_storage = "compiler"
    with pytest.raises(ValueError, match="cannot carry physical placements"):
        emit(p)


def test_native_tensor_transpose_requires_compiler_owned_psum():
    b = Builder([SimpleNamespace(shape=(32, 64), dtype="float32")], neuron_core(3))
    b.program.encoding_storage = "compiler"
    b.add("x = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
    b.add("nisa.dma_copy(dst=x, src=a0)")
    b.add("t = nisa.nc_transpose(x, engine=nisa.tensor_engine)")
    b.add("y = nisa.tensor_copy(t, engine=nisa.scalar_engine)")
    b.add("out = nl.ndarray((64,32), dtype=nl.float32, buffer=nl.shared_hbm)")
    b.add("nisa.dma_copy(dst=out, src=y)")
    b.program.outputs = ("out",)
    b.program.validate()
    assert b.program.tensors["t"].memory == "PSUM"
    assert b.program.tensors["t"].shape == (64, 32)
    from voyager_compiler.trainium.program_analysis import analyze_selected

    analysis = analyze_selected(b.program, neuron_core(3))
    assert analysis["hbm_read_bytes"] == 32 * 64 * 4 + 128 * 128
    assert not analysis["unknown_completion"]
    b.program.encoding_storage = "arena"
    with pytest.raises(ValueError, match="compiler-managed"):
        b.program.validate()


def temporary_chain(count):
    b = Builder([SimpleNamespace(shape=(32, 64), dtype="float32")], neuron_core(3))
    b.add("out = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.shared_hbm)")
    for i in range(count):
        b.add(f"x{i} = nl.ndarray((32,64), dtype=nl.float32, buffer=nl.sbuf)")
        b.add(f"nisa.dma_copy(dst=x{i}, src=a0)")
        b.add(f"y{i} = nisa.tensor_scalar(x{i}, op0=nl.multiply, operand0=2, engine=nisa.vector_engine)")
        b.add(f"nisa.dma_copy(dst=out, src=y{i})")
    b.program.outputs = ("out",)
    return b.program


def test_bounded_temporary_storage_does_not_grow_with_repetitions():
    extents = []
    for count in (16, 160):
        p = temporary_chain(count)
        p.allocate(temporary_buffer_depth=4)
        extents.append(max(x.byte_address + x.bytes_per_partition for x in p.placements.values()))
        assert len({x.byte_address for x in p.placements.values()}) == 8
        # Logical copies do not authorize address reuse until previous readers
        # complete; every reuse must remain represented in the selected DAG.
        for previous, current in p.reuse_edges():
            assert previous in p.instructions[current].dependencies
        p.validate()
    assert extents[0] == extents[1]


def test_bounded_temporary_storage_rejects_capacity_overflow():
    p = temporary_chain(16)
    with pytest.raises(ValueError, match="does not fit"):
        p.allocate(sbuf_bytes=128 << 10, temporary_buffer_depth=4)


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_invalid_temporary_depth_is_rejected(depth):
    from voyager_compiler.trainium.execution import TrainiumTuning

    with pytest.raises(ValueError, match="positive integer"):
        TrainiumTuning(temporary_buffer_depth=depth)
    with pytest.raises(ValueError, match="positive integer"):
        temporary_chain(1).allocate(temporary_buffer_depth=depth)


def test_native_allocation_rejects_a_voyager_temporary_pool():
    from voyager_compiler.trainium.execution import TrainiumTuning

    with pytest.raises(ValueError, match="strict ISA"):
        TrainiumTuning(strict_realization=False, temporary_buffer_depth=2)
