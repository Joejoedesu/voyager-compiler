# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import operator
import tempfile
import unittest
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule

from voyager_compiler.codegen.artifacts import dump_storage_plan
from voyager_compiler.codegen.execution_order import execution_positions
from voyager_compiler.codegen.hardware import HardwareConfig, TensorRole
from voyager_compiler.codegen.mapping import gen_code, run_memory_mapping
from voyager_compiler.codegen.mapping_utils import (
    TensorWriteSession,
    convert_arg,
    is_addressing_op,
    is_nop,
    map_node,
    save_tensor,
)
from voyager_compiler.codegen.memory import (
    MemoryAllocator,
    Segment,
    physical_access_requirements,
)
from voyager_compiler.codegen.storage import (
    PhysicalPayloadRef,
    PhysicalTensorId,
    SerializedTensorRef,
    StorageGraph,
    StoragePlane,
    StorageRelationKind,
    materialize_aggregate_copies,
)


def _set_value(node, value):
    node.value = value
    node.meta["val"] = value
    if isinstance(value, torch.Tensor):
        node.shape = tuple(value.shape)
        node.meta["dtype"] = str(value.dtype).removeprefix("torch.")


def _independent_pair(value):
    return value + 1, value + 2


def _view_and_copy_graph():
    graph = Graph()
    input_node = graph.placeholder("input")
    view = graph.call_function(
        torch.ops.aten.view.default,
        (input_node, (1, 4)),
    )
    clone = graph.call_function(
        torch.ops.aten.clone.default,
        (view,),
    )
    graph.output(clone)
    model = GraphModule(torch.nn.Module(), graph)

    value = torch.zeros(1, 4)
    _set_value(input_node, value)
    _set_value(view, value.view(1, 4))
    _set_value(clone, value.clone())
    return model, input_node, view, clone


def _split_graph():
    graph = Graph()
    input_node = graph.placeholder("input")
    split = graph.call_function(
        torch.ops.aten.split.Tensor,
        (input_node, 2, 1),
    )
    first = graph.call_function(operator.getitem, (split, 0))
    second = graph.call_function(operator.getitem, (split, 1))
    graph.output((first, second))
    model = GraphModule(torch.nn.Module(), graph)

    value = torch.zeros(1, 4)
    outputs = torch.split(value, 2, 1)
    _set_value(input_node, value)
    split.value = outputs
    split.meta["val"] = outputs
    split.meta["dtype"] = tuple(item.dtype for item in outputs)
    _set_value(first, outputs[0])
    _set_value(second, outputs[1])
    return model, input_node, split, first, second


def _aggregate_graph():
    graph = Graph()
    first_input = graph.placeholder("first_input")
    second_input = graph.placeholder("second_input")
    first = graph.call_function(
        torch.ops.aten.relu.default,
        (first_input,),
    )
    second = graph.call_function(
        torch.ops.aten.relu.default,
        (second_input,),
    )
    cat = graph.call_function(
        torch.ops.aten.cat.default,
        ((first, second), 0),
    )
    output = graph.output(cat)
    model = GraphModule(torch.nn.Module(), graph)

    value = torch.zeros(1, 4)
    _set_value(first_input, value)
    _set_value(second_input, value)
    _set_value(first, value)
    _set_value(second, value)
    _set_value(cat, torch.cat((value, value), 0))
    return model, first, second, cat, output


def _partial_aggregate_graph():
    graph = Graph()
    first_input = graph.placeholder("first_input")
    second_input = graph.placeholder("second_input")
    first = graph.call_function(
        torch.ops.aten.relu.default,
        (first_input,),
    )
    partial = graph.call_function(
        torch.ops.aten.slice.Tensor,
        (first, 0, 0, 1),
    )
    second = graph.call_function(
        torch.ops.aten.relu.default,
        (second_input,),
    )
    cat = graph.call_function(
        torch.ops.aten.cat.default,
        ((partial, second), 0),
    )
    graph.output(cat)
    model = GraphModule(torch.nn.Module(), graph)

    first_value = torch.zeros(2, 4)
    second_value = torch.zeros(1, 4)
    _set_value(first_input, first_value)
    _set_value(second_input, second_value)
    _set_value(first, first_value)
    _set_value(partial, first_value[:1])
    _set_value(second, second_value)
    _set_value(cat, torch.cat((first_value[:1], second_value), 0))
    return model, first, partial, second, cat


class StorageAnalysisTest(unittest.TestCase):
    def test_plan_records_logical_and_explicit_physical_shapes(self):
        model, input_node, _view, _clone = _view_and_copy_graph()
        storage = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        )
        physical_value = torch.zeros(1, 8, dtype=torch.int8)
        storage.register_physical_component(
            input_node,
            physical_value,
            plane=StoragePlane.SCRATCHPAD,
            bits=8,
            backing_size_bytes=8,
            layout="test_padded",
        )

        plan = storage.finalize()
        physical = plan.physical_tensor(
            input_node,
            StoragePlane.SCRATCHPAD,
        )
        serialized = plan.to_dict()["physical_tensors"]["scratchpad"]
        entry = next(
            item for item in serialized if item["name"] == physical.id.name
        )

        logical = plan.logical_tensors[plan.ref(input_node)]
        self.assertEqual(logical.shape, (1, 4))
        self.assertEqual(physical.backing_size_bytes, 8)
        self.assertEqual(physical.layout, "test_padded")
        self.assertEqual(physical.component().shape, (1, 8))
        self.assertEqual(entry["backing_size_bytes"], 8)
        self.assertEqual(entry["components"][0]["shape"], [1, 8])

    def test_plan_rejects_conflicting_shared_consumer_payloads(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        alias = graph.call_function(
            torch.ops.aten.detach.default,
            (input_node,),
        )
        graph.output(alias)
        model = GraphModule(torch.nn.Module(), graph)
        _set_value(input_node, torch.zeros(4, dtype=torch.int8))
        _set_value(alias, torch.zeros(4, dtype=torch.int8))
        storage = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        )
        for logical, payload in (
            (input_node, torch.zeros(4, dtype=torch.int8)),
            (alias, torch.ones(4, dtype=torch.int8)),
        ):
            storage.register_physical_component(
                logical,
                payload,
                plane=StoragePlane.SCRATCHPAD,
                bits=8,
                backing_size_bytes=4,
                layout="test_shared",
            )

        with self.assertRaisesRegex(
            ValueError,
            "conflicting payloads for component 'primary'",
        ):
            storage.finalize()

    def test_named_component_shares_explicit_payload_backing(self):
        graph = Graph()
        payload = graph.placeholder("payload")
        fine_scale = graph.placeholder("fine_scale")
        graph.output((payload, fine_scale))
        model = GraphModule(torch.nn.Module(), graph)
        _set_value(payload, torch.zeros(64, dtype=torch.int8))
        payload.meta["dtype"] = "int4"
        _set_value(fine_scale, torch.zeros(2, dtype=torch.uint8))
        storage = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        )
        storage.set_relation(
            fine_scale,
            StorageRelationKind.COMPONENT,
            plane=StoragePlane.SCRATCHPAD,
            source=payload,
            component="fine_scale",
        )
        storage.register_physical_component(
            payload,
            payload.value,
            plane=StoragePlane.SCRATCHPAD,
            dtype="int4",
            bits=4,
            backing_size_bytes=32,
            layout="test_embedded",
        )
        storage.register_physical_component(
            fine_scale,
            fine_scale.value,
            plane=StoragePlane.SCRATCHPAD,
            dtype="uint8",
            bits=8,
            component="fine_scale",
            backing_size_bytes=32,
            layout="test_embedded",
        )

        plan = storage.finalize()
        physical = plan.physical_tensor(
            payload,
            StoragePlane.SCRATCHPAD,
        )

        self.assertEqual(
            plan.physical_id(fine_scale, StoragePlane.SCRATCHPAD),
            physical.id,
        )
        self.assertEqual(physical.backing_size_bytes, 32)
        self.assertEqual(
            {component.name for component in physical.components},
            {"primary", "fine_scale"},
        )

    def test_payload_serialization_uses_finalized_id_and_segment(self):
        model, input_node, _view, _clone = _view_and_copy_graph()
        storage = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        )
        storage.register_physical_component(
            input_node,
            input_node.value,
            plane=StoragePlane.SCRATCHPAD,
            bits=32,
            backing_size_bytes=16,
            layout="test_dense",
        )
        plan = storage.finalize()
        segment = Segment(
            0,
            16,
            memory_space=1,
            address=0x1234,
            storage_plane=1,
            memory_instance="test",
        )
        payload = plan.payload_ref(
            input_node,
            StoragePlane.SCRATCHPAD,
            "primary",
            segment,
        )

        field = convert_arg(payload).tensor

        self.assertIsInstance(payload, PhysicalPayloadRef)
        self.assertFalse(hasattr(payload, "source"))
        self.assertFalse(hasattr(payload, "value"))
        self.assertEqual(field.node, payload.name)
        self.assertEqual(tuple(field.shape), (1, 4))
        self.assertEqual(field.scratchpad.address, 0x1234)

    def test_nonresident_artifact_serializes_without_memory(self):
        model, input_node, _view, _clone = _view_and_copy_graph()
        storage = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        )
        encoded = torch.zeros(8, dtype=torch.uint8)
        storage.register_serialized_artifact(
            input_node,
            encoded,
            plane=StoragePlane.SCRATCHPAD,
            name="operation.weight_vector_scale_payload",
            dtype="uint8",
            bits=8,
        )
        plan = storage.finalize()
        reference = plan.serialized_ref(
            input_node,
            StoragePlane.SCRATCHPAD,
            "operation.weight_vector_scale_payload",
        )

        field = convert_arg(reference).tensor

        self.assertIsInstance(reference, SerializedTensorRef)
        self.assertEqual(tuple(field.shape), (8,))
        self.assertFalse(field.HasField("memory"))
        self.assertFalse(field.HasField("scratchpad"))
        serialized = plan.to_dict()["physical_tensors"]["scratchpad"]
        entry = next(
            item
            for item in serialized
            if item["name"] == reference.physical_tensor.name
        )
        self.assertEqual(
            entry["serialized_artifacts"][0]["byte_size"],
            8,
        )

    def test_tensor_write_session_deduplicates_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / "payload.bin")
            with TensorWriteSession(max_workers=1):
                save_tensor(torch.tensor([1, 2]), filename)
                save_tensor(torch.tensor([1, 2]), filename)
                with self.assertRaisesRegex(
                    ValueError,
                    "conflicting tensor payloads",
                ):
                    save_tensor(torch.tensor([3, 4]), filename)

    def test_views_share_storage_while_copies_materialize(self):
        model, input_node, view, clone = _view_and_copy_graph()
        plan = StorageGraph(
            model,
            (StoragePlane.DRAM, StoragePlane.SCRATCHPAD),
        ).finalize()

        for plane in (StoragePlane.DRAM, StoragePlane.SCRATCHPAD):
            input_physical = plan.physical_id(input_node, plane)
            self.assertEqual(
                plan.physical_id(view, plane),
                input_physical,
            )
            self.assertNotEqual(
                plan.physical_id(clone, plane),
                input_physical,
            )
            self.assertEqual(
                plan.relations[plane.value][plan.ref(view)].kind,
                StorageRelationKind.VIEW,
            )
            self.assertEqual(
                plan.relations[plane.value][plan.ref(clone)].kind,
                StorageRelationKind.COPY,
            )
            self.assertEqual(
                plan.produced_physical_outputs(view, plane),
                (),
            )
            self.assertEqual(
                plan.produced_physical_outputs(clone, plane),
                (plan.physical_id(clone, plane),),
            )
        self.assertNotEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(input_node, StoragePlane.SCRATCHPAD),
        )

    def test_component_container_records_its_physical_writer(self):
        model, _input_node, split, _first, _second = _split_graph()
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(4, 4),
            cache_size=4096,
            input_buffer_size=64,
            weight_buffer_size=64,
            accum_buffer_size=64,
        )
        plan = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        ).finalize()
        produced = plan.produced_physical_outputs(
            split,
            StoragePlane.SCRATCHPAD,
        )

        self.assertEqual(len(produced), 1)
        self.assertIn(
            (
                TensorRole.OUTPUT,
                hardware.pe.name,
            ),
            physical_access_requirements(
                plan,
                hardware,
                StoragePlane.SCRATCHPAD,
                produced[0],
            ),
        )

    def test_contiguous_with_new_memory_format_materializes(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        channels_last = graph.call_function(
            torch.ops.aten.contiguous.default,
            (input_node,),
            {"memory_format": torch.channels_last},
        )
        graph.output(channels_last)
        model = GraphModule(torch.nn.Module(), graph)

        value = torch.zeros(1, 3, 4, 5)
        _set_value(input_node, value)
        _set_value(
            channels_last,
            value.contiguous(memory_format=torch.channels_last),
        )

        plan = StorageGraph(
            model,
            (StoragePlane.DRAM,),
        ).finalize()
        self.assertEqual(
            plan.relations["memory"][plan.ref(channels_last)].kind,
            StorageRelationKind.COPY,
        )
        self.assertNotEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(channels_last, StoragePlane.DRAM),
        )

    def test_tensor_getitem_requires_normalized_indexing_op(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        indexed = graph.call_function(operator.getitem, (input_node, 0))
        graph.output(indexed)
        model = GraphModule(torch.nn.Module(), graph)

        value = torch.zeros(2, 3)
        _set_value(input_node, value)
        _set_value(indexed, value[0])

        with self.assertRaisesRegex(
            NotImplementedError,
            "normalized to aten.select/slice",
        ):
            StorageGraph(model, (StoragePlane.DRAM,))

    def test_in_place_operator_writes_into_its_logical_input(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        other = graph.placeholder("other")
        add = graph.call_function(
            torch.ops.aten.add_.Tensor,
            (input_node, other),
        )
        graph.output(add)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.zeros(1, 4)
        _set_value(input_node, value)
        _set_value(other, value)
        _set_value(add, value)

        plan = StorageGraph(model, (StoragePlane.DRAM,)).finalize()
        relation = plan.relations["memory"][plan.ref(add)]

        self.assertEqual(relation.kind, StorageRelationKind.WRITE_INTO)
        self.assertEqual(relation.source, plan.ref(input_node))
        self.assertEqual(
            plan.physical_id(add, StoragePlane.DRAM),
            plan.physical_id(input_node, StoragePlane.DRAM),
        )

    def test_clone_has_a_distinct_allocation_and_executable_writer(self):
        model, input_node, _view, clone = _view_and_copy_graph()
        run_memory_mapping(model, MemoryAllocator(1 << 20))

        self.assertNotEqual(
            input_node.meta["memory"].start,
            clone.meta["memory"].start,
        )
        self.assertFalse(is_nop(clone))
        self.assertNotEqual(map_node(clone).op, "nop")

    def test_pad_serializes_as_accelerator_unsupported(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        pad = graph.call_function(
            torch.ops.aten.pad.default,
            (input_node, [1, 1]),
        )
        graph.output(pad)

        value = torch.zeros(1, 2)
        _set_value(input_node, value)
        _set_value(pad, torch.nn.functional.pad(value, (1, 1)))

        self.assertEqual(map_node(pad).op, "unsupported")

    def test_partial_slice_materializes_an_executable_result(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor,
            (input_node, 0, 1, 3),
        )
        graph.output(sliced)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(4, dtype=torch.float32)
        _set_value(input_node, value)
        _set_value(sliced, value[1:3])

        allocator = MemoryAllocator(1 << 20)
        run_memory_mapping(model, allocator)
        plan = model.meta["storage_plan"]

        self.assertEqual(
            plan.relations["memory"][plan.ref(sliced)].kind,
            StorageRelationKind.COPY,
        )
        self.assertNotEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(sliced, StoragePlane.DRAM),
        )
        self.assertNotEqual(
            input_node.meta["memory"].start,
            sliced.meta["memory"].start,
        )
        self.assertFalse(is_nop(sliced))
        self.assertNotEqual(map_node(sliced).op, "nop")

    def test_keyword_partial_slice_materializes(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor,
            (input_node,),
            {
                "dim": 0,
                "start": 1,
                "end": 3,
                "step": 1,
            },
        )
        graph.output(sliced)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(4, dtype=torch.float32)
        _set_value(input_node, value)
        _set_value(sliced, value[1:3])

        plan = StorageGraph(model, (StoragePlane.DRAM,)).finalize()

        self.assertEqual(
            plan.relations["memory"][plan.ref(sliced)].kind,
            StorageRelationKind.COPY,
        )
        self.assertNotEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(sliced, StoragePlane.DRAM),
        )
        self.assertFalse(is_nop(sliced))

    def test_partial_as_strided_materializes_without_strided_segments(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        view = graph.call_function(
            torch.ops.aten.as_strided.default,
            (input_node, (2,), (1,), 1),
        )
        graph.output(view)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(4, dtype=torch.float32)
        _set_value(input_node, value)
        _set_value(view, torch.as_strided(value, (2,), (1,), 1))

        plan = StorageGraph(model, (StoragePlane.DRAM,)).finalize()

        self.assertEqual(
            plan.relations["memory"][plan.ref(view)].kind,
            StorageRelationKind.COPY,
        )
        self.assertNotEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(view, StoragePlane.DRAM),
        )
        self.assertFalse(is_nop(view))

    def test_keyword_negative_select_uses_normalized_byte_offset(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        selected = graph.call_function(
            torch.ops.aten.select.int,
            (input_node,),
            {"dim": 1, "index": -1},
        )
        graph.output(selected)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        _set_value(input_node, value)
        _set_value(selected, value.select(1, -1))

        run_memory_mapping(model, MemoryAllocator(1 << 20))
        plan = model.meta["storage_plan"]

        self.assertEqual(
            plan.relations["memory"][plan.ref(selected)].kind,
            StorageRelationKind.VIEW,
        )
        self.assertEqual(
            selected.meta["memory"].start,
            input_node.meta["memory"].start + 2 * selected.value.numel() * 4,
        )
        self.assertEqual(
            selected.meta["memory"].end - selected.meta["memory"].start,
            selected.value.numel() * 4,
        )
        self.assertFalse(is_addressing_op(selected))
        self.assertTrue(is_nop(selected))

    def test_identity_slice_remains_an_address_only_alias(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        sliced = graph.call_function(
            torch.ops.aten.slice.Tensor,
            (input_node, 0, 0, 4),
        )
        graph.output(sliced)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(4, dtype=torch.float32)
        _set_value(input_node, value)
        _set_value(sliced, value[:])

        run_memory_mapping(model, MemoryAllocator(1 << 20))
        plan = model.meta["storage_plan"]

        self.assertEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(sliced, StoragePlane.DRAM),
        )
        self.assertEqual(input_node.meta["memory"], sliced.meta["memory"])
        self.assertTrue(is_nop(sliced))

    def test_zero_copy_reshape_aliases_and_serializes_as_nop(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            (input_node, (2, 2)),
        )
        graph.output(reshape)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.arange(4)
        _set_value(input_node, value)
        _set_value(reshape, value.reshape(2, 2))

        plan = StorageGraph(model, (StoragePlane.DRAM,)).finalize()
        self.assertEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(reshape, StoragePlane.DRAM),
        )
        self.assertEqual(
            plan.relations["memory"][plan.ref(reshape)].kind,
            StorageRelationKind.VIEW,
        )
        self.assertTrue(is_nop(reshape))
        self.assertEqual(map_node(reshape).op, "nop")

    def test_unchanged_dtype_conversion_aliases_storage(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        convert = graph.call_function(
            torch.ops.aten.to.dtype,
            (input_node, torch.float32),
        )
        graph.output(convert)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.ones(4, dtype=torch.float32)
        _set_value(input_node, value)
        _set_value(convert, value)

        plan = StorageGraph(model, (StoragePlane.DRAM,)).finalize()
        self.assertEqual(
            plan.physical_id(input_node, StoragePlane.DRAM),
            plan.physical_id(convert, StoragePlane.DRAM),
        )
        self.assertEqual(
            plan.relations["memory"][plan.ref(convert)].kind,
            StorageRelationKind.ALIAS,
        )
        self.assertTrue(is_nop(convert))

    def test_bound_allocators_accept_only_planned_physical_identity(self):
        model, input_node, _view, _clone = _view_and_copy_graph()
        plan = StorageGraph(
            model,
            (StoragePlane.DRAM, StoragePlane.SCRATCHPAD),
        ).finalize()
        allocator = MemoryAllocator(1 << 20)
        allocator.bind_storage_plan(plan, StoragePlane.DRAM)

        storage_key = plan.storage_key(input_node, StoragePlane.DRAM)
        self.assertIsInstance(storage_key, PhysicalTensorId)
        allocator.allocate_memory(input_node)
        self.assertIn(storage_key, allocator.memory_map)

        foreign = plan.storage_key(input_node, StoragePlane.SCRATCHPAD)
        with self.assertRaisesRegex(ValueError, "crossed planes"):
            plan.storage_key(foreign, StoragePlane.DRAM)

        graph = Graph()
        unknown = graph.placeholder("unknown")
        _set_value(unknown, torch.zeros(1))
        with self.assertRaisesRegex(KeyError, "not a logical tensor"):
            allocator.allocate_memory(unknown)

    def test_shared_allocator_rejects_storage_plan_bound_after_allocation(
        self,
    ):
        model, input_node, _view, _clone = _view_and_copy_graph()
        plan = StorageGraph(
            model,
            (StoragePlane.DRAM,),
        ).finalize()
        allocator = MemoryAllocator(1 << 20)

        allocator.allocate_memory(input_node)

        with self.assertRaisesRegex(
            RuntimeError,
            "must be bound before memory allocation",
        ):
            allocator.bind_storage_plan(plan, StoragePlane.DRAM)

    def test_multi_output_split_uses_an_explicit_physical_container(self):
        model, input_node, split, first, second = _split_graph()
        graph = StorageGraph(
            model,
            (StoragePlane.DRAM, StoragePlane.SCRATCHPAD),
        )
        plan = graph.finalize()

        split_outputs = plan.output_refs(split)
        self.assertEqual(
            tuple(output.path for output in split_outputs),
            ((0,), (1,)),
        )
        self.assertFalse(plan.has_logical_value(split))
        for plane in (StoragePlane.DRAM, StoragePlane.SCRATCHPAD):
            input_backing = plan.physical_id(input_node, plane)
            output_container = plan.physical_id(split_outputs[0], plane)
            self.assertNotEqual(output_container, input_backing)
            self.assertEqual(output_container.component, "tuple_outputs")
            for logical in (
                *split_outputs,
                plan.ref(first),
                plan.ref(second),
            ):
                self.assertEqual(
                    plan.physical_id(logical, plane),
                    output_container,
                )
            for output in split_outputs:
                self.assertEqual(
                    plan.relations[plane.value][output].kind,
                    StorageRelationKind.COMPONENT,
                )

        debug = plan.to_dict()
        self.assertEqual(len(debug["logical_tensors"]), 5)
        self.assertEqual(len(debug["physical_tensors"]["memory"]), 2)
        self.assertEqual(len(debug["physical_tensors"]["scratchpad"]), 2)

        model.meta["storage_plan"] = plan
        with tempfile.TemporaryDirectory() as directory:
            dump_storage_plan(model, directory)
            dumped = json.loads(
                (Path(directory) / "storage_plan.json").read_text()
            )
        self.assertEqual(dumped["schema_version"], 1)
        self.assertEqual(
            dumped["logical_tensors"],
            debug["logical_tensors"],
        )
        self.assertEqual(
            dumped["physical_tensors"],
            debug["physical_tensors"],
        )

    def test_nested_getitem_preserves_exact_logical_output_identity(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        outer_first = graph.call_function(
            operator.getitem,
            (input_node, 0),
        )
        nested_second = graph.call_function(
            operator.getitem,
            (outer_first, 1),
        )
        graph.output(nested_second)
        model = GraphModule(torch.nn.Module(), graph)

        first = torch.zeros(2)
        second = torch.ones(3)
        third = torch.full((4,), 2.0)
        input_value = ((first, second), third)
        _set_value(input_node, input_value)
        _set_value(outer_first, input_value[0])
        _set_value(nested_second, second)

        plan = StorageGraph(
            model,
            (StoragePlane.DRAM,),
        ).finalize()

        self.assertEqual(
            plan.physical_id(nested_second, StoragePlane.DRAM),
            plan.physical_id(
                plan.ref(input_node, (0, 1)),
                StoragePlane.DRAM,
            ),
        )
        self.assertNotEqual(
            plan.physical_id(nested_second, StoragePlane.DRAM),
            plan.physical_id(
                plan.ref(input_node, (0, 0)),
                StoragePlane.DRAM,
            ),
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "does not yet serialize nested tensor output containers",
        ):
            run_memory_mapping(model, MemoryAllocator(1 << 20))

    def test_negative_getitem_selects_last_logical_and_physical_output(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        pair = graph.call_function(_independent_pair, (input_node,))
        last = graph.call_function(operator.getitem, (pair, -1))
        graph.output(last)
        model = GraphModule(torch.nn.Module(), graph)

        value = torch.arange(4, dtype=torch.float32)
        pair_value = _independent_pair(value)
        _set_value(input_node, value)
        _set_value(pair, pair_value)
        pair.meta["dtype"] = ("float32", "float32")
        _set_value(last, pair_value[-1])

        run_memory_mapping(model, MemoryAllocator(1 << 20))

        plan = model.meta["storage_plan"]
        outputs = plan.output_refs(pair)
        self.assertEqual(
            plan.physical_id(last, StoragePlane.DRAM),
            plan.physical_id(outputs[-1], StoragePlane.DRAM),
        )
        self.assertNotEqual(
            plan.physical_id(last, StoragePlane.DRAM),
            plan.physical_id(outputs[0], StoragePlane.DRAM),
        )
        self.assertEqual(
            last.meta["memory"],
            pair.meta["memory_output_segments"][-1],
        )

    def test_mapping_allocates_independent_multi_outputs_separately(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        pair = graph.call_function(_independent_pair, (input_node,))
        first = graph.call_function(operator.getitem, (pair, 0))
        second = graph.call_function(operator.getitem, (pair, 1))
        graph.output((first, second))
        model = GraphModule(torch.nn.Module(), graph)

        value = torch.arange(4, dtype=torch.float32)
        pair_value = _independent_pair(value)
        _set_value(input_node, value)
        _set_value(pair, pair_value)
        pair.meta["dtype"] = ("float32", "float32")
        _set_value(first, pair_value[0])
        _set_value(second, pair_value[1])

        allocator = MemoryAllocator(1 << 20)
        run_memory_mapping(model, allocator)

        plan = model.meta["storage_plan"]
        outputs = plan.output_refs(pair)
        physical = tuple(
            plan.physical_id(output, StoragePlane.DRAM) for output in outputs
        )
        self.assertNotEqual(physical[0], physical[1])
        segments = pair.meta["memory_output_segments"]
        self.assertEqual(len(segments), 2)
        self.assertNotEqual(segments[0].start, segments[1].start)
        self.assertEqual(first.meta["memory"], segments[0])
        self.assertEqual(second.meta["memory"], segments[1])

        params = gen_code(model, (value,))
        operation = next(
            item
            for item in params.ops
            if item.WhichOneof("op_type") == "op" and item.op.name == pair.name
        )
        self.assertEqual(
            [tensor.memory.address for tensor in operation.outputs.tensors],
            [segment.start for segment in segments],
        )

    def test_hidden_access_extends_both_ends_of_an_auxiliary_interval(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        early = graph.call_function(
            torch.ops.aten.relu.default,
            (input_node,),
        )
        late = graph.call_function(
            torch.ops.aten.relu.default,
            (early,),
        )
        graph.output(late)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.zeros(4)
        _set_value(input_node, value)
        _set_value(early, value)
        _set_value(late, value)

        storage = StorageGraph(model, (StoragePlane.SCRATCHPAD,))
        auxiliary = storage.add_auxiliary(
            late,
            "hidden",
            plane=StoragePlane.SCRATCHPAD,
        )
        storage.add_access(
            auxiliary,
            early,
            plane=StoragePlane.SCRATCHPAD,
            role=TensorRole.ACTIVATION,
            kind="read",
        )
        plan = storage.finalize()

        interval = plan.intervals["scratchpad"][auxiliary]
        positions = execution_positions(plan.execution_order)
        self.assertEqual(interval.birth, positions[early])
        self.assertEqual(interval.death, positions[early] + 1)

    def test_mapping_does_not_allocate_an_unused_placeholder(self):
        graph = Graph()
        used = graph.placeholder("used")
        unused = graph.placeholder("unused")
        result = graph.call_function(
            torch.ops.aten.relu.default,
            (used,),
        )
        graph.output(result)
        model = GraphModule(torch.nn.Module(), graph)
        value = torch.zeros(4)
        _set_value(used, value)
        _set_value(unused, value)
        _set_value(result, value)
        allocator = MemoryAllocator(1 << 20)

        run_memory_mapping(model, allocator)

        plan = model.meta["storage_plan"]
        self.assertFalse(plan.logical_tensors[plan.ref(unused)].users)
        self.assertNotIn("memory", unused.meta)
        self.assertNotIn(
            plan.physical_id(unused, StoragePlane.DRAM),
            allocator.memory_map,
        )

    def test_mapping_allocates_split_container_by_physical_identity(self):
        model, input_node, split, first, second = _split_graph()
        allocator = MemoryAllocator(1 << 20)

        run_memory_mapping(model, allocator)

        plan = model.meta["storage_plan"]
        output_container = plan.physical_id(
            plan.output_refs(split)[0],
            StoragePlane.DRAM,
        )
        self.assertNotEqual(
            output_container,
            plan.physical_id(input_node, StoragePlane.DRAM),
        )
        self.assertIn(output_container, allocator.memory_map)
        container_segment = allocator.allocation_segment(output_container)
        self.assertEqual(container_segment.start, first.meta["memory"].start)
        self.assertEqual(container_segment.end, second.meta["memory"].end)
        self.assertTrue(
            all(
                isinstance(key, PhysicalTensorId)
                for key in allocator.memory_map
            )
        )
        self.assertEqual(first.meta["memory"].start, split.meta["memory"].start)
        self.assertEqual(first.meta["memory"].end, second.meta["memory"].start)

    def test_auxiliary_physical_tensor_can_back_a_logical_result(self):
        model, _input_node, _view, clone = _view_and_copy_graph()
        graph = StorageGraph(model, (StoragePlane.SCRATCHPAD,))
        auxiliary = graph.add_auxiliary(
            clone,
            "secondary",
            plane=StoragePlane.SCRATCHPAD,
        )
        graph.bind_physical(
            clone,
            auxiliary,
            plane=StoragePlane.SCRATCHPAD,
        )
        plan = graph.finalize()

        self.assertEqual(
            plan.physical_id(clone, StoragePlane.SCRATCHPAD),
            auxiliary,
        )
        self.assertEqual(
            plan.physical_tensor(
                auxiliary,
                StoragePlane.SCRATCHPAD,
            ).logical_tensors,
            (plan.ref(clone),),
        )

    def test_physical_binding_requires_declared_auxiliary(self):
        model, _input_node, _view, clone = _view_and_copy_graph()
        graph = StorageGraph(model, (StoragePlane.SCRATCHPAD,))
        undeclared = PhysicalTensorId(
            plane=StoragePlane.SCRATCHPAD.value,
            owner=graph.ref(clone),
            component="undeclared",
        )

        with self.assertRaisesRegex(
            ValueError,
            "declared with add_auxiliary",
        ):
            graph.bind_physical(
                clone,
                undeclared,
                plane=StoragePlane.SCRATCHPAD,
            )

    def test_auxiliary_physical_tensor_owns_explicit_representation(self):
        model, _input_node, _view, clone = _view_and_copy_graph()
        graph = StorageGraph(model, (StoragePlane.SCRATCHPAD,))
        auxiliary = graph.add_auxiliary(
            clone,
            "secondary",
            plane=StoragePlane.SCRATCHPAD,
        )
        graph.register_physical_component(
            auxiliary,
            clone.value,
            plane=StoragePlane.SCRATCHPAD,
            bits=32,
            backing_size_bytes=32,
            layout="test_auxiliary",
        )

        plan = graph.finalize()
        physical = plan.physical_tensor(
            auxiliary,
            StoragePlane.SCRATCHPAD,
        )

        self.assertEqual(physical.layout, "test_auxiliary")
        self.assertEqual(physical.backing_size_bytes, 32)
        self.assertEqual(physical.component().shape, (1, 4))
        self.assertEqual(
            physical.component().logical_tensor,
            plan.ref(clone),
        )

    def test_simple_aggregate_writes_share_one_physical_backing(self):
        model, first, second, cat, _output = _aggregate_graph()
        plan = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        ).finalize()

        backing = plan.physical_id(cat, StoragePlane.SCRATCHPAD)
        self.assertEqual(
            plan.physical_id(first, StoragePlane.SCRATCHPAD),
            backing,
        )
        self.assertEqual(
            plan.physical_id(second, StoragePlane.SCRATCHPAD),
            backing,
        )
        self.assertEqual(
            plan.relations["scratchpad"][plan.ref(first)].kind,
            StorageRelationKind.WRITE_INTO,
        )

    def test_mapping_accepts_implicit_zero_cat_dimension(self):
        model, first, second, cat, _output = _aggregate_graph()
        cat.args = (cat.args[0],)
        model.recompile()

        run_memory_mapping(model, MemoryAllocator(1 << 20))

        self.assertIsNotNone(cat.meta["memory"])
        self.assertEqual(first.meta["memory"].end, second.meta["memory"].start)

    def test_keyword_nonzero_cat_dimension_is_not_address_only(self):
        model, _first, _second, cat, _output = _aggregate_graph()
        cat.args = (cat.args[0],)
        cat.kwargs = {"dim": 1}
        _set_value(
            cat,
            torch.cat(tuple(node.value for node in cat.args[0]), dim=1),
        )
        model.recompile()

        plan = StorageGraph(
            model,
            (StoragePlane.DRAM,),
        ).finalize()

        self.assertFalse(is_addressing_op(cat))
        self.assertEqual(
            plan.relations["memory"][plan.ref(cat)].kind,
            StorageRelationKind.MATERIALIZE,
        )

    def test_executable_partial_slice_writes_directly_into_aggregate(self):
        model, first, partial, second, cat = _partial_aggregate_graph()
        self.assertEqual(materialize_aggregate_copies(model), 0)

        self.assertIs(cat.args[0][0], partial)
        plan = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        ).finalize()
        backing = plan.physical_id(cat, StoragePlane.SCRATCHPAD)
        self.assertEqual(
            plan.physical_id(partial, StoragePlane.SCRATCHPAD),
            backing,
        )
        self.assertEqual(
            plan.physical_id(second, StoragePlane.SCRATCHPAD),
            backing,
        )
        self.assertNotEqual(
            plan.physical_id(first, StoragePlane.SCRATCHPAD),
            backing,
        )
        self.assertEqual(
            plan.relations["scratchpad"][plan.ref(partial)].kind,
            StorageRelationKind.WRITE_INTO,
        )

    def test_repeated_aggregate_input_gets_distinct_copies(self):
        model, first, _second, cat, _output = _aggregate_graph()
        cat.update_arg(0, (first, first))
        _set_value(cat, torch.cat((first.value, first.value), 0))
        model.graph.eliminate_dead_code()
        model.recompile()

        self.assertEqual(materialize_aggregate_copies(model), 2)
        copies = cat.args[0]
        self.assertIsNot(copies[0], copies[1])
        self.assertTrue(all(node.meta["physical_copy"] for node in copies))
        plan = StorageGraph(
            model,
            (StoragePlane.SCRATCHPAD,),
        ).finalize()
        backing = plan.physical_id(cat, StoragePlane.SCRATCHPAD)
        self.assertTrue(
            all(
                plan.physical_id(node, StoragePlane.SCRATCHPAD) == backing
                for node in copies
            )
        )
        self.assertNotEqual(
            plan.physical_id(first, StoragePlane.SCRATCHPAD),
            backing,
        )

    def test_generic_mapping_lowers_executable_slice_into_aggregate(self):
        model, first, partial, second, cat = _partial_aggregate_graph()

        run_memory_mapping(model, MemoryAllocator(1 << 20))

        plan = model.meta["storage_plan"]
        backing = plan.physical_id(cat, StoragePlane.DRAM)
        self.assertIs(cat.args[0][0], partial)
        self.assertEqual(
            plan.physical_id(partial, StoragePlane.DRAM),
            backing,
        )
        self.assertEqual(
            plan.physical_id(second, StoragePlane.DRAM),
            backing,
        )
        self.assertNotEqual(
            plan.physical_id(first, StoragePlane.DRAM),
            backing,
        )
        self.assertNotEqual(
            first.meta["memory"].start,
            partial.meta["memory"].start,
        )

    def test_hidden_physical_access_extends_auxiliary_lifetime(self):
        model, _first, _second, cat, output = _aggregate_graph()
        graph = StorageGraph(model, (StoragePlane.SCRATCHPAD,))
        auxiliary = graph.add_auxiliary(
            cat,
            "sidecar",
            plane=StoragePlane.SCRATCHPAD,
        )
        graph.add_access(
            auxiliary,
            output,
            plane=StoragePlane.SCRATCHPAD,
            role=TensorRole.ACTIVATION,
            kind="read",
        )
        plan = graph.finalize()

        self.assertEqual(
            tuple(
                (
                    access.operation,
                    access.role,
                    access.kind,
                )
                for access in plan.accesses(
                    auxiliary,
                    StoragePlane.SCRATCHPAD,
                )
            ),
            ((output, TensorRole.ACTIVATION, "read"),),
        )
        positions = execution_positions(plan.execution_order)
        self.assertEqual(
            plan.intervals["scratchpad"][auxiliary].death,
            positions[output] + 1,
        )


if __name__ == "__main__":
    unittest.main()
