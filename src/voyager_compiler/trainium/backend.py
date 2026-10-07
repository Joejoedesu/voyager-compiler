"""Reuse Voyager's graph passes, search, bufferization and collateral emitter."""

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path


class TrainiumBackend:
    uses_bufferized_flow = True
    skip_rgb_padding = False

    def prepare_graph(self, model, config):
        from voyager_compiler.codegen.transform.rewrites import (
            normalize_global_average_pool,
        )

        normalize_global_average_pool(model)
        from .lowering import prepare_graph
        prepare_graph(model, config)

    def restore_mapping_policy(self, config, options):
        from .mapping import TrainiumMappingPolicy
        from .execution import TrainiumTuning

        restored = dict(options)
        restored.setdefault("isa_lowering", False)
        restored.setdefault("buffer_allocation", "legacy_logical")
        # The first experimental ISA artifacts used ScalarE eviction before
        # copy routing became an explicit policy field.
        restored.setdefault(
            "copy_policy",
            "scalar" if restored["isa_lowering"] else "balanced",
        )
        policy = TrainiumMappingPolicy(config, TrainiumTuning(**restored))
        policy._recorded_options = dict(options)
        return policy

    def realize(self, root, context, **options):
        from .converter import convert

        return convert(root, context=context, **options)

    def mapping_policy(self, config):
        from .mapping import TrainiumMappingPolicy

        return TrainiumMappingPolicy(config)

    def validate(self, config):
        from .hardware import TrainiumConfig

        if not isinstance(config, TrainiumConfig):
            raise ValueError("Trainium requires its physical hardware graph")

    def fusion_patterns(self, config):
        from voyager_compiler import OpMatcher

        self.validate(config)
        return [
            [OpMatcher("matmul", "linear", "conv2d"), OpMatcher("relu")],
            [OpMatcher("add"), OpMatcher("relu")],
            [OpMatcher("maximum"), OpMatcher("maximum")],
        ]

    def transform(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _transform_voyager

        options["patterns"] = self.fusion_patterns(options["config"])
        options["layout_policy"] = "systolic"
        return _transform_voyager(
            model, example_args, example_kwargs, **options
        )

    def compile(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _compile_voyager

        buffers = options.get("bufferization_options")
        if buffers is not None and buffers.flow != "per_kernel":
            raise NotImplementedError(
                "Trainium currently supports the shared per_kernel flow"
            )
        result = _compile_voyager(
            model, example_args, example_kwargs, **options
        )
        config = options["config"]
        # Bufferization stamps metadata at multiple nesting levels. In the
        # per-kernel flow each top-level region owns one kernel; cached selected
        # mappings may be shared across *different* regions with equal shapes.
        # Deduplicate within the owning region, never across the whole model.
        record = {
            "hardware": asdict(config),
            "placement": model.meta.get("trainium_placement"),
            "mapping_constraints": {
                "instruction_M": 512,
                "instruction_N": 128,
                "instruction_K": 128,
                "software_tile": "partition-aware SBUF fit and whole-bank PSUM reservation",
            },
            "cost_model": "Shared instruction panels and dependency graph, characterized primitive completion and DMA issue/payload, runtime-only selection; explicit ISA placement follows shared bufferization; backend retains engine scheduling",
            "execution_plans": list(
                {
                    (module_name.split(".")[0], id(selected)): asdict(
                        selected.evaluation.execution_plan
                    )
                    for module_name, module in model.named_modules()
                    if hasattr(module, "graph")
                    for n in module.graph.nodes
                    for selected in [n.meta.get("selected_mapping")]
                    if selected is not None
                    and selected.evaluation.execution_plan is not None
                }.values()
            ),
            "estimates": list(
                {
                    (module_name.split(".")[0], id(selected)): dict(
                        selected.evaluation.diagnostics
                    )["estimate"]
                    for module_name, module in model.named_modules()
                    if hasattr(module, "graph")
                    for n in module.graph.nodes
                    for selected in [n.meta.get("selected_mapping")]
                    if selected is not None
                    and "estimate" in dict(selected.evaluation.diagnostics)
                }.values()
            ),
        }

        def encode(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, (set, frozenset)):
                return sorted(value)
            raise TypeError(type(value).__name__)

        (Path(options["output_dir"]) / "hardware.json").write_text(
            json.dumps(record, default=encode, indent=2) + "\n"
        )

        from voyager_compiler.compilation import CompilerContext
        context = options.get("context") or CompilerContext.from_artifacts(options["output_dir"], config)
        if context.policy.tuning.isa_lowering:
            from .planning import select_plan
            select_plan(options["output_dir"], context=context)
        return result

    @staticmethod
    def interstellar_memory(config):
        # Four mapping levels, not four invented physical memories. L0/L1
        # describe an instruction's traversal; L2 is the explicit SBUF arena.
        return dict(
            buf_capacity_list=[
                [1, 1, 1],
                [1 << 40, 1 << 40, 1 << 40],
                [config.usable_scratchpad_size],
                [1 << 40],
            ],
            buf_access_cost_list=[[0, 0, 0], [0, 0, 0], [1], [10]],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            para_count_list=[16384, 1, 1, 1],
            bank_size_list=[None] * 4,
        )
