import unittest
from voyager_compiler.trainium.physical_context import transform
from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency as D,
    OperationEvent as E,
    RepeatedGraph as G,
    evaluate_graph,
)


class PhysicalContextTests(unittest.TestCase):
    def test_order_is_issue_not_completion(self):
        g = G((E("a", "TensorE", 10, 10, 100), E("b", "TensorE", 10, 10, 20)))
        ordered, c = transform(g, "ordered")
        self.assertEqual(evaluate_graph(ordered).duration_ns, 100)
        self.assertEqual(ordered.nodes[1].dependencies[-1].milestone, "issue")

    def test_prefix_waits_for_older_result(self):
        g = G(
            (
                E("a", "TensorE", 1, 1, 100),
                E("b", "TensorE", 1, 1, 5),
                E("c", "ScalarE", 1, 1, 10, dependencies=(D(1),)),
            )
        )
        self.assertEqual(
            evaluate_graph(transform(g, "ordered")[0]).duration_ns, 100
        )
        self.assertEqual(
            evaluate_graph(transform(g, "prefix")[0]).duration_ns, 110
        )

    def test_forwarding_is_preserved(self):
        g = G(
            (
                E("a", "TensorE", 5, 5, 100, forward_ns=5),
                E(
                    "b",
                    "TensorE",
                    5,
                    5,
                    100,
                    dependencies=(D(0, milestone="forward"),),
                ),
            )
        )
        self.assertEqual(
            evaluate_graph(transform(g, "prefix")[0]).duration_ns, 105
        )

    def test_independent_engine_overlap(self):
        g = G((E("a", "TensorE", 10, 10, 100), E("b", "VectorE", 10, 10, 100)))
        self.assertEqual(
            evaluate_graph(transform(g, "context-ready")[0]).duration_ns, 100
        )

    def test_gate_anchored_to_dependency(self):
        g = G(
            (
                E("a", "TensorE", 1, 1, 100),
                E(
                    "b",
                    "ScalarE",
                    1,
                    1,
                    10,
                    dependencies=(D(0),),
                    implementation="nki.copy.PSUM.float32.ScalarE",
                ),
            )
        )
        self.assertEqual(
            evaluate_graph(transform(g, "context-ready")[0]).duration_ns, 147
        )

    def test_internal_dma_no_launch_gate(self):
        g = G(
            (
                E("a", "DMAIssue", 1, 1, 5),
                E("b", "DMA", 1, 1, 10, dependencies=(D(0),)),
            )
        )
        _, counts = transform(g, "context-ready")
        self.assertFalse(any(k.startswith("readiness") for k in counts))

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            transform(G(()), "typo")


if __name__ == "__main__":
    unittest.main()
