"""Use the default Voyager passes, with Lean hardware constraints."""


class GemminiBackend:
    uses_bufferized_flow = True

    def mapping_policy(self, config):
        from .mapping import GemminiMappingPolicy

        return GemminiMappingPolicy(config)

    def restore_mapping_policy(self, config, options):
        from .mapping import GemminiMappingPolicy, GemminiTuning
        from .scheduling import SubmissionPolicy

        options = dict(options)
        options["submission"] = SubmissionPolicy(
            **options.get("submission", {})
        )
        return GemminiMappingPolicy(config, GemminiTuning(**options))

    def realize(self, root, context, **options):
        from .collateral import convert

        return convert(root, context=context, **options)

    def interstellar_memory(self, config):
        from .hardware import interstellar_memory

        return interstellar_memory(config)

    skip_rgb_padding = False

    def prepare_graph(self, model, config):
        from voyager_compiler.codegen.transform.rewrites import (
            normalize_global_average_pool,
        )

        normalize_global_average_pool(model)

    def validate(self, config):
        from .hardware import LeanConfig, lean_config

        if not isinstance(config, LeanConfig) or config != lean_config():
            raise ValueError(
                "Gemmini requires the pinned Lean hardware configuration"
            )

    def fusion_patterns(self, config):
        from voyager_compiler import OpMatcher

        self.validate(config)
        # Accumulator scaling and ReLU are legal on mvout. The existing fusion
        # rewriter finds these groups; no separate model frontend is used.
        return [
            [
                OpMatcher("dequantize", allow_input_dequantize=True),
                OpMatcher("max_pool2d", "avg_pool2d", "adaptive_avg_pool2d"),
                OpMatcher("quantize"),
            ],
            [
                OpMatcher(
                    "conv2d", "linear", "matmul", allow_input_dequantize=True
                ),
                OpMatcher("dequantize"),
                OpMatcher("relu"),
                OpMatcher("quantize"),
            ],
            [
                OpMatcher("add", allow_input_dequantize=True),
                OpMatcher("relu"),
                OpMatcher("quantize"),
            ],
            [
                OpMatcher("dequantize", allow_input_dequantize=True),
                OpMatcher("relu"),
                OpMatcher("quantize"),
            ],
        ]

    def transform(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _transform_voyager

        return _transform_voyager(
            model, example_args, example_kwargs, **options
        )

    def compile(self, model, example_args, example_kwargs=None, **options):
        from voyager_compiler import _compile_voyager
        from voyager_compiler.codegen.transform.bufferize.bufferization import (
            propagate_logical_dtypes,
        )

        buffer_options = options.get("bufferization_options")
        if buffer_options is not None and buffer_options.flow != "per_kernel":
            raise NotImplementedError(
                "Lean currently requires per-kernel bufferization"
            )

        # Run the existing propagation before tiling too: layout views must
        # carry storage dtypes after PT2E folding and final fusion.
        propagate_logical_dtypes(model)
        # ISA conversion is an explicit consumer of the standard collaterals.
        return _compile_voyager(model, example_args, example_kwargs, **options)
