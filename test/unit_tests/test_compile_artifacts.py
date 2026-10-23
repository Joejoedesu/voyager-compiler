# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule

from scheduler.dispatch import INTERSTELLAR_SCHEDULER
from voyager_compiler import compile as compile_model
from voyager_compiler.codegen.artifacts import (
    discard_supplemental_artifact,
    dump_compile_artifacts,
    prepare_output_directory,
    set_supplemental_artifact,
)
from voyager_compiler.codegen.execution_order import (
    DP_EXECUTION_ORDER_MODE,
)
from voyager_compiler.codegen.hardware import HardwareConfig
from voyager_compiler.codegen.lowering.codegen import gen_code_bufferized
from voyager_compiler.codegen.mapping import gen_code, run_memory_mapping
from voyager_compiler.codegen.mapping_utils import (
    declare_unsupported_operation,
    set_tensor_field,
)
from voyager_compiler.codegen.memory import (
    END_TO_END_SCRATCHPAD_MAPPING,
    MemoryAllocator,
    MemorySpace,
    Segment,
)
from voyager_compiler.codegen.param_pb2 import Tensor
from voyager_compiler.codegen.targets import (
    COMPILATION_TARGET_META,
)

_NON_DEFAULT_TARGET = "unit_test_target"


def _identity_graph():
    graph = Graph()
    value = graph.placeholder("value")
    graph.output(value)
    return GraphModule(torch.nn.Module(), graph)


def _unsupported_relu_graph():
    graph = Graph()
    value = graph.placeholder("value")
    result = graph.call_function(torch.ops.aten.relu.default, (value,))
    graph.output(result)
    model = GraphModule(torch.nn.Module(), graph)
    declare_unsupported_operation(
        result,
        reason_code="unit_test_boundary",
        detail="unit-test unsupported boundary",
        stage="unit_test",
    )
    return model


def _generic_hardware():
    return HardwareConfig.from_legacy(
        unroll_dims=(4, 4),
        cache_size=4096,
        input_buffer_size=64,
        weight_buffer_size=64,
        accum_buffer_size=64,
        num_banks=4,
        bank_width=16,
    )


class CompileArtifactDirectoryTest(unittest.TestCase):
    def test_per_operation_codegen_serializes_unsupported_boundary(self):
        params = gen_code(
            _unsupported_relu_graph(),
            (torch.zeros(1),),
        )

        self.assertEqual(len(params.ops), 1)
        self.assertEqual(params.ops[0].op.op, "unsupported")

    def test_end_to_end_mapping_rejects_unsupported_boundary(self):
        with self.assertRaisesRegex(
            ValueError,
            "cannot execute explicit unsupported boundaries",
        ):
            run_memory_mapping(
                _unsupported_relu_graph(),
                MemoryAllocator(),
                hardware_config=_generic_hardware(),
                scratchpad_mapping=END_TO_END_SCRATCHPAD_MAPPING,
            )

    def test_fresh_directory_accepts_missing_or_empty_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            self.assertEqual(prepare_output_directory(missing), missing)
            self.assertTrue(missing.is_dir())
            self.assertEqual(prepare_output_directory(missing), missing)

    def test_regular_codegen_rejects_stale_tensor_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tensor_files"
            output.mkdir()
            (output / "stale.bin").write_bytes(b"stale")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                gen_code(
                    _identity_graph(),
                    (torch.zeros(1),),
                    output,
                )

    def test_supplemental_manifest_sets_replaces_and_discards_payloads(self):
        model = _identity_graph()
        set_supplemental_artifact(
            model,
            "zeta",
            {"schema_version": 1, "value": 2},
        )
        set_supplemental_artifact(
            model,
            "alpha",
            {"schema_version": 1, "value": 0},
        )
        set_supplemental_artifact(
            model,
            "alpha",
            {"schema_version": 1, "value": 1},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "supplemental_artifacts.json"
            dump_compile_artifacts(model, None, output)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(list(payload["artifacts"]), ["alpha", "zeta"])
            self.assertEqual(payload["artifacts"]["alpha"]["value"], 1)

            discard_supplemental_artifact(model, "alpha")
            discard_supplemental_artifact(model, "zeta")
            dump_compile_artifacts(model, None, output)
            self.assertFalse(path.exists())

    def test_supplemental_manifest_requires_versioned_named_payload(self):
        model = _identity_graph()
        with self.assertRaisesRegex(ValueError, "name"):
            set_supplemental_artifact(
                model,
                "",
                {"schema_version": 1},
            )
        with self.assertRaisesRegex(ValueError, "schema_version"):
            set_supplemental_artifact(model, "evidence", {})

    def test_bufferized_codegen_rejects_stale_tensor_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tensor_files"
            output.mkdir()
            (output / "stale.bin").write_bytes(b"stale")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                gen_code_bufferized(
                    _identity_graph(),
                    (torch.zeros(1),),
                    output,
                )

    def test_low_level_mapping_rejects_a_different_bound_target(self):
        model = _identity_graph()
        model.meta[COMPILATION_TARGET_META] = _NON_DEFAULT_TARGET
        with self.assertRaisesRegex(
            ValueError,
            "bound to compilation target",
        ):
            run_memory_mapping(model)

    def test_low_level_dp_rejects_per_operation_before_target_binding(self):
        model = _identity_graph()
        original_graph = str(model.graph)

        with self.assertRaisesRegex(
            ValueError,
            "operator_graph \\+ end_to_end",
        ):
            run_memory_mapping(
                model,
                execution_order=DP_EXECUTION_ORDER_MODE,
            )

        self.assertEqual(str(model.graph), original_graph)
        self.assertNotIn(COMPILATION_TARGET_META, model.meta)

    def test_bufferized_compile_fails_closed_on_local_scheduler_inputs(self):
        cases = {
            "typed hardware": {
                "hardware_config": _generic_hardware(),
            },
            "explicit backend": {
                "scheduler_backend": INTERSTELLAR_SCHEDULER,
            },
            "scheduler options": {
                "scheduler_options": object(),
            },
            "legacy scheduler": {
                "cache_size": 4096,
                "unroll_dims": (4, 4),
                "input_buffer_size": 64,
                "weight_buffer_size": 64,
                "accum_buffer_size": 64,
            },
            "scheduler double buffering": {
                "double_buffered_accum_buffer": True,
            },
        }
        for name, compile_kwargs in cases.items():
            with self.subTest(name=name):
                model = _identity_graph()
                original_graph = str(model.graph)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "compile-output"
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "cannot preserve local scheduler results",
                    ):
                        compile_model(
                            model,
                            (torch.zeros(1),),
                            output_dir=output,
                            dump_tensors=False,
                            bufferize=True,
                            **compile_kwargs,
                        )

                    self.assertFalse(output.exists())
                self.assertEqual(str(model.graph), original_graph)
                self.assertNotIn(COMPILATION_TARGET_META, model.meta)

    def test_dp_compile_rejects_unsupported_modes_before_output_mutation(
        self,
    ):
        cases = {
            "per-operation": {},
            "bufferized": {"bufferize": True},
        }
        for name, compile_kwargs in cases.items():
            with self.subTest(name=name):
                model = _identity_graph()
                original_graph = str(model.graph)
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "compile-output"
                    with self.assertRaisesRegex(
                        ValueError,
                        "operator_graph \\+ end_to_end",
                    ):
                        compile_model(
                            model,
                            (torch.zeros(1),),
                            output_dir=output,
                            dump_tensors=False,
                            execution_order=DP_EXECUTION_ORDER_MODE,
                            **compile_kwargs,
                        )
                    self.assertFalse(output.exists())
                self.assertEqual(str(model.graph), original_graph)
                self.assertNotIn(COMPILATION_TARGET_META, model.meta)

    def test_serialized_memory_names_the_physical_instance(self):
        model = _identity_graph()
        value = next(iter(model.graph.nodes))
        value.value = torch.zeros(1)
        value.shape = (1,)
        value.meta["dtype"] = "float32"
        value.meta["scratchpad"] = Segment(
            16,
            20,
            memory_space=MemorySpace.SCRATCHPAD,
            memory_instance="GlobalActivation",
        )

        tensor = Tensor()
        set_tensor_field(tensor, value)

        self.assertEqual(tensor.scratchpad.partition, MemorySpace.SCRATCHPAD)
        self.assertEqual(tensor.scratchpad.address, 16)
        self.assertEqual(
            tensor.scratchpad.instance,
            "GlobalActivation",
        )


if __name__ == "__main__":
    unittest.main()
