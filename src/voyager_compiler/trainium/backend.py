"""Reuse Voyager's graph passes, search, bufferization and collateral emitter."""

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path


class TrainiumBackend:
    uses_bufferized_flow = True

    def mapping_policy(self, config):
        from .mapping import TrainiumMappingPolicy

        return TrainiumMappingPolicy(config)

    @staticmethod
    def tile_search(config, node, tile_sizes, shapes, tiling):
        from .cost import vector_candidate

        return vector_candidate(config, node, tile_sizes, shapes, tiling)

    @staticmethod
    def place_local_buffers(model, bufs, config):
        from .constraints import place_local_buffers

        return place_local_buffers(model, bufs, config)

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

        from .converter import convert

        buffers = options.get("bufferization_options")
        if buffers is not None and buffers.flow != "per_kernel":
            raise NotImplementedError(
                "Trainium currently supports the shared per_kernel flow"
            )
        result = _compile_voyager(
            model, example_args, example_kwargs, **options
        )
        config = options["config"]
        record = {
            "hardware": asdict(config),
            "placement": model.meta.get("trainium_placement"),
            "mapping_constraints": {
                "instruction_M": 512,
                "instruction_N": 128,
                "instruction_K": 128,
                "software_tile": "partition-aware SBUF fit; output free dimension <=4096",
            },
            "cost_model": "Trainium instruction/DMA service, dependency prologue/drain, runtime-only selection; analytical, not hardware calibrated",
            "estimates": list(
                {
                    id(rc): rc.estimate
                    for module in model.modules()
                    if hasattr(module, "graph")
                    for n in module.graph.nodes
                    for rc in [n.meta.get("runtime_calculator")]
                    if hasattr(rc, "estimate")
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
        convert(options["output_dir"], target=options["config"].name)
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
