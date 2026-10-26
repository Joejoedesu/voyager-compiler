"""Static preparation preserves values, state and the stored byte budget."""

import tempfile
import unittest

import torch
from torch.fx import Graph, GraphModule

import voyager_compiler as vc
from voyager_compiler.codegen.transform.data_layout import (
    normalize_gemm_weight_layout,
)
from voyager_compiler.codegen.transform.rewrites import (
    constant_storage_bytes,
    deduplicate_nodes,
    fold_constant_generators,
)
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.shape_prop import ShapeProp, fake_like, written_buffers

aten = torch.ops.aten


def make_graph(build, **buffers):
    root, graph = torch.nn.Module(), Graph()
    graph.placeholder("unused_input")
    attrs = {}
    for name, value in buffers.items():
        root.register_buffer(name, value)
        attrs[name] = graph.get_attr(name)
    graph.output(build(graph.call_function, attrs))
    gm = GraphModule(root, graph)
    ShapeProp(gm).propagate(fake_like(torch.ones(1)))
    return gm


def stored_bytes(gm):
    attrs = {
        n.target: n for n in gm.graph.nodes if n.op == "get_attr" and n.users
    }
    return sum(constant_storage_bytes(n) for n in attrs.values())


class ConstantFoldingTests(unittest.TestCase):
    def verify(self, gm):
        expected = gm(torch.ones(1))
        before = stored_bytes(gm)
        fold_constant_generators(gm)
        torch.testing.assert_close(gm(torch.ones(1)), expected, rtol=0, atol=0)
        self.assertLessEqual(stored_bytes(gm), before)
        return [n for n in gm.graph.nodes if n.op == "call_function"]

    def test_weight_scale_reshape_transpose(self):
        def build(op, a):
            scaled = op(aten.mul.Tensor, (a["w"], a["scale"]))
            shaped = op(aten.reshape.default, (scaled, [8, 16]))
            return op(aten.transpose.int, (shaped, 0, 1))

        gm = make_graph(
            build,
            w=torch.arange(128.0).reshape(8, 16),
            scale=torch.arange(1.0, 9.0).reshape(8, 1),
        )
        self.assertEqual(self.verify(gm), [])
        self.assertEqual(len(gm.state_dict()), 1)
        self.assertTrue(next(gm.buffers()).is_contiguous())

    def test_shared_original_prevents_duplicate_transpose(self):
        gm = make_graph(
            lambda op, a: (a["w"], op(aten.transpose.int, (a["w"], 0, 1))),
            w=torch.arange(32.0).reshape(4, 8),
        )
        self.assertEqual(len(self.verify(gm)), 1)
        self.assertFalse(
            gm.meta["constant_folding"][-1]["components"][0]["folded"]
        )

    def test_split_frontier_replaces_original(self):
        gm = make_graph(
            lambda op, a: (
                op(aten.slice.Tensor, (a["w"], 0, 0, 4)),
                op(aten.slice.Tensor, (a["w"], 0, 4, 8)),
            ),
            w=torch.arange(128.0).reshape(8, 16),
        )
        self.assertEqual(self.verify(gm), [])
        self.assertEqual(
            sum(b.untyped_storage().nbytes() for b in gm.buffers()), 512
        )

    def test_different_getattrs_do_not_hide_live_original(self):
        gm = make_graph(
            lambda op, a: op(aten.neg.default, (a["w"],)), w=torch.arange(32.0)
        )
        output = next(n for n in gm.graph.nodes if n.op == "output")
        with gm.graph.inserting_before(output):
            second = gm.graph.get_attr("w")
        output.args = ((output.args[0], second),)
        gm.recompile()
        ShapeProp(gm).propagate(fake_like(torch.ones(1)))
        self.assertEqual(len(self.verify(gm)), 1)

    def test_per_op_growth_is_rejected(self):
        for target, args, kwargs in (
            (aten.repeat.default, ([2, 1],), {}),
            (aten.expand.default, ([8, 8],), {}),
            (aten.pad.default, ([1, 1],), {}),
            (aten._to_copy.default, (), {"dtype": torch.float64}),
        ):
            with self.subTest(op=target):
                gm = make_graph(
                    lambda op, a: op(target, (a["w"], *args), kwargs),
                    w=torch.arange(8.0).reshape(1, 8),
                )
                self.assertEqual(len(self.verify(gm)), 1)

    def test_narrowing_cast_and_reduction(self):
        gm = make_graph(
            lambda op, a: op(
                aten.sum.dim_IntList,
                (
                    op(
                        aten._to_copy.default,
                        (a["w"],),
                        {"dtype": torch.bfloat16},
                    ),
                    [0],
                ),
            ),
            w=torch.arange(128.0).reshape(8, 16),
        )
        self.assertEqual(self.verify(gm), [])

    def test_broadcast_product_cannot_expand(self):
        gm = make_graph(
            lambda op, a: op(aten.mul.Tensor, (a["w"], a["scale"])),
            w=torch.arange(8.0).reshape(8, 1),
            scale=torch.arange(8.0).reshape(1, 8),
        )
        self.assertEqual(len(self.verify(gm)), 1)

    def test_packed_dtype_is_charged(self):
        gm = make_graph(
            lambda op, a: op(aten.neg.default, (a["w"],)), w=torch.arange(32.0)
        )
        next(n for n in gm.graph.nodes if n.op == "get_attr").meta[
            "dtype"
        ] = "int4"
        self.assertEqual(len(self.verify(gm)), 1)

    def test_write_through_view_and_out_keyword(self):
        for out in (False, True):

            def build(op, a):
                view = op(aten.view.default, (a["w"], [2, 4]))
                if out:
                    op(aten.add.out, (view, view), {"out": view})
                else:
                    op(aten.add_.Scalar, (view, 1))
                return op(aten.mul.Scalar, (a["w"], 2))

            with self.subTest(out=out):
                gm = make_graph(build, w=torch.arange(8.0))
                self.assertIn("w", written_buffers(gm.graph))
                fold_constant_generators(gm)
                self.assertTrue(
                    any(n.target is aten.mul.Scalar for n in gm.graph.nodes)
                )

    def test_random_and_custom_functions_are_not_frozen(self):
        for target in (aten.rand_like.default, lambda w: w + 1):
            with self.subTest(target=target):
                gm = make_graph(
                    lambda op, a: op(target, (a["w"],)), w=torch.ones(8)
                )
                fold_constant_generators(gm)
                self.assertTrue(any(n.target is target for n in gm.graph.nodes))

    def test_fresh_derived_tensor_mutated_each_invocation(self):
        def build(op, a):
            value = op(aten.mul.Scalar, (a["w"], 2))
            return op(aten.add_.Scalar, (value, 1))

        gm = make_graph(build, w=torch.arange(8.0))
        self.assertEqual(written_buffers(gm.graph), set())
        fold_constant_generators(gm)
        self.assertTrue(
            any(n.target is aten.mul.Scalar for n in gm.graph.nodes)
        )
        for _ in range(2):
            torch.testing.assert_close(
                gm(torch.ones(1)), torch.arange(8.0) * 2 + 1, rtol=0, atol=0
            )

    def test_fresh_generated_tensor_mutated_each_invocation(self):
        for full_like in (False, True):

            def build(op, a):
                if full_like:
                    value = op(aten.full_like.default, (a["w"], 1))
                else:
                    value = op(aten.ones.default, ([8],))
                return op(aten.add_.Scalar, (value, 1))

            with self.subTest(full_like=full_like):
                gm = make_graph(build, w=torch.zeros(8))
                fold_constant_generators(gm)
                self.assertEqual(
                    gm.meta["constant_folding"][0]["generator_bytes"], 0
                )
                for _ in range(2):
                    torch.testing.assert_close(
                        gm(torch.ones(1)), torch.full((8,), 2.0), rtol=0, atol=0
                    )

    def test_deferred_deterministic_generator(self):
        from voyager_compiler.shape_prop import constant_value

        gm = make_graph(
            lambda op, a: op(aten.cos.default, (op(aten.arange.default, (8,)),))
        )
        fold_constant_generators(gm)
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                constant_value(gm, node)
        torch.testing.assert_close(gm(torch.ones(1)), torch.arange(8).cos())
        self.assertEqual(gm.meta["constant_folding"][0]["generator_bytes"], 32)

    def test_generated_mask_remains_precomputed(self):
        from voyager_compiler.shape_prop import constant_value

        def build(op, a):
            positions = op(aten.arange.default, (8,))
            row = op(aten.unsqueeze.default, (positions, 1))
            col = op(aten.unsqueeze.default, (positions, 0))
            mask = op(aten.lt.Tensor, (row, col))
            return op(aten._to_copy.default, (mask,), {"dtype": torch.float32})

        gm = make_graph(build)
        fold_constant_generators(gm)
        self.assertFalse(any(n.op == "call_function" for n in gm.graph.nodes))
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                constant_value(gm, node)
        expected = (torch.arange(8)[:, None] < torch.arange(8)[None, :]).float()
        torch.testing.assert_close(gm(torch.ones(1)), expected, rtol=0, atol=0)
        self.assertEqual(gm.meta["constant_folding"][0]["generator_bytes"], 256)

    def test_generator_position_table_with_static_frequency(self):
        from voyager_compiler.shape_prop import constant_value

        def build(op, a):
            positions = op(aten.arange.default, (16,), {"dtype": torch.float32})
            positions = op(aten.unsqueeze.default, (positions, 0))
            frequency = op(aten.unsqueeze.default, (a["frequency"], 1))
            table = op(aten.mm.default, (frequency, positions))
            return op(aten.cos.default, (table,))

        frequency = torch.arange(1.0, 9.0)
        gm = make_graph(build, frequency=frequency)
        fold_constant_generators(gm)
        self.assertFalse(any(n.op == "call_function" for n in gm.graph.nodes))
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                constant_value(gm, node)
        expected = (frequency[:, None] @ torch.arange(16.0)[None, :]).cos()
        torch.testing.assert_close(gm(torch.ones(1)), expected, rtol=0, atol=0)
        self.assertEqual(gm.meta["constant_folding"][0]["generator_bytes"], 512)

    def test_fresh_non_grad_literal_detach_can_fold(self):
        def build(op, a):
            copied = op(aten.lift_fresh_copy.default, (a["w"],))
            detached = op(aten.detach_.default, (copied,))
            return op(aten.neg.default, (detached,))

        gm = make_graph(build, w=torch.arange(8.0))
        self.assertEqual(self.verify(gm), [])

    def test_trainium_weight_layout_uses_same_storage_policy(self):
        for shared in (False, True):

            def build(op, a):
                result = op(aten.linear.default, (a["x"], a["w"]))
                return (result, a["w"]) if shared else result

            with self.subTest(shared=shared):
                gm = make_graph(
                    build,
                    x=torch.ones(2, 16),
                    w=torch.arange(128.0).reshape(8, 16),
                )
                expected = gm(torch.ones(1))
                before = stored_bytes(gm)
                normalize_gemm_weight_layout(gm, "ck", "ck")
                deduplicate_nodes(gm)
                fold_constant_generators(gm, phase="layout")
                self.assertLessEqual(stored_bytes(gm), before)
                self.assertEqual(
                    any(n.target is aten.transpose.int for n in gm.graph.nodes),
                    shared,
                )
                torch.testing.assert_close(
                    gm(torch.ones(1)), expected, rtol=0, atol=0
                )


class LayoutPreparationTests(unittest.TestCase):
    def normalize(self, model, args, **layout):
        expected = model(*args)
        gm = vc.export_model(model, args)
        ShapeProp(gm).propagate(*map(fake_like, args))
        before = stored_bytes(gm)
        vc.transform(gm, args, skip_op_fusion=True, **layout)
        gm.graph.lint()
        torch.testing.assert_close(gm(*args), expected, rtol=0, atol=0)
        self.assertLessEqual(stored_bytes(gm), before)
        return gm

    def test_nhwc_shared_and_private_broadcast_constants(self):
        # A channel scale is view-only; spatial masks require reordering.
        for shape in ((4, 1, 1), (1, 1, 2, 3), (4, 2, 3)):
            for shared in (False, True):
                with self.subTest(shape=shape, shared=shared):

                    class Model(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.conv = torch.nn.Conv2d(1, 4, 1, bias=False)
                            self.conv.weight.data.fill_(1)
                            count = torch.Size(shape).numel()
                            self.register_buffer(
                                "scale",
                                torch.arange(1.0, count + 1).reshape(shape),
                            )

                        def forward(self, x):
                            result = self.conv(x) * self.scale
                            return (result, self.scale) if shared else result

                    gm = self.normalize(
                        Model().eval(),
                        (torch.arange(6.0).reshape(1, 1, 2, 3),),
                        tensor_layout="nhwc",
                    )
                    self.assertEqual(
                        stored_bytes(gm), 16 + 4 * torch.Size(shape).numel()
                    )
                    self.assertEqual(hasattr(gm, "scale"), shared)
                    self.assertEqual(hasattr(gm, "scale_nhwc"), not shared)

    def test_nhwc_dynamic_rank_three_operand(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 4, 1, bias=False)
                self.conv.weight.data.fill_(1)

            def forward(self, x, scale):
                return self.conv(x) * scale, scale

        self.normalize(
            Model().eval(),
            (
                torch.arange(6.0).reshape(1, 1, 2, 3),
                torch.arange(1.0, 25.0).reshape(4, 2, 3),
            ),
            tensor_layout="nhwc",
        )

    @torch.no_grad()
    def test_nhwc_shared_scale_bufferized_replay(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(4, 4, 1, bias=False)
                self.conv.weight.data.fill_(1)
                self.register_buffer(
                    "scale", torch.arange(1.0, 5.0).reshape(4, 1, 1)
                )

            def forward(self, x):
                return self.conv(x) * self.scale, self.scale

        model = Model().eval()
        args = (torch.arange(24.0).reshape(1, 4, 2, 3),)
        expected = model(*args)
        config = AcceleratorConfig(
            pe_array_size=(4, 4),
            scratchpad_size=4096,
            num_banks=4,
            bank_width=4,
            vector_unit_width=4,
        )
        gm = self.normalize(model, args, tensor_layout="nhwc", config=config)
        with tempfile.TemporaryDirectory(
            prefix="voyager-constant-layout-"
        ) as output:
            vc.compile(
                gm, args, config=config, output_dir=output, dump_tensors=False
            )
            actual = ShapeProp(gm).propagate(*args)
            # Exported/bufferized outputs are flattened.
            torch.testing.assert_close(
                list(actual), list(expected), rtol=0, atol=0
            )

    def test_trainium_broadcast_relayout_obeys_storage_budget(self):
        for shape in ((16,), (16, 16)):
            for shared in (False, True):
                with self.subTest(shape=shape, shared=shared):

                    class Model(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            count = torch.Size(shape).numel()
                            self.register_buffer(
                                "scale",
                                torch.arange(1.0, count + 1).reshape(shape),
                            )

                        def forward(self, x, y):
                            result = ((x @ y) * self.scale) @ y
                            return (result, self.scale) if shared else result

                    gm = self.normalize(
                        Model().eval(),
                        (
                            (torch.arange(256.0) % 5 - 2).reshape(16, 16),
                            (torch.arange(256.0) % 7 - 3).reshape(16, 16),
                        ),
                        gemm_layout="cx,ck",
                    )
                    self.assertEqual(hasattr(gm, "scale"), shared)
                    self.assertEqual(hasattr(gm, "scale_kmajor"), not shared)

    def test_trainium_rms_scale_obeys_storage_budget(self):
        for shared in (False, True):
            with self.subTest(shared=shared):

                class Model(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.register_buffer("scale", torch.arange(1.0, 17.0))

                    def forward(self, x, y):
                        result = (
                            torch.nn.functional.rms_norm(
                                x @ y, (16,), self.scale
                            )
                            @ y
                        )
                        return (result, self.scale) if shared else result

                gm = self.normalize(
                    Model().eval(),
                    (
                        (torch.arange(256.0) % 5 - 2).reshape(16, 16),
                        (torch.arange(256.0) % 7 - 3).reshape(16, 16),
                    ),
                    gemm_layout="cx,ck",
                )
                self.assertTrue(
                    any(
                        n.target
                        is torch.ops.quantized_ops.rms_norm_kmajor.default
                        for n in gm.graph.nodes
                    )
                )
                self.assertEqual(hasattr(gm, "scale"), shared)
                self.assertEqual(hasattr(gm, "scale_kmajor"), not shared)


if __name__ == "__main__":
    unittest.main()
