"""Selection must compare costs and reject invalid refinements visibly."""

import unittest
from types import SimpleNamespace

from voyager_compiler.compiler.search.planner import select_implementation
from voyager_compiler.extensions import (
    CandidateRejected,
    ImplementationSpec,
    Registry,
    WorkloadSpec,
)
from voyager_compiler.ir.plan import AllocationPlan, CompilationPlan, Cost


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.node = SimpleNamespace(name="matmul", meta={"sentinel": 42})
        self.context = SimpleNamespace(
            tiler=SimpleNamespace(runtime_tolerance=0.02)
        )

    def implementation(self, name, latency, traffic, reject=False):
        def build(node, context, parameters):
            node.meta["candidate"] = name
            if reject:
                raise CandidateRejected("private accumulator capacity exceeded")
            return name

        return ImplementationSpec(
            name,
            lambda n, a: True,
            build,
            estimate=lambda *args: Cost(latency, traffic),
        )

    def test_cost_wins_over_registration_order(self):
        explicit = self.implementation("explicit", 100, 1000)
        loop = self.implementation("hardware_loop", 60, 800)
        for specs in ((explicit, loop), (loop, explicit)):
            graph, choice, alternatives, _ = select_implementation(
                self.node, self.node, self.context, specs
            )
            self.assertEqual(
                (graph, choice.implementation),
                ("hardware_loop", "hardware_loop"),
            )
            self.assertEqual(len(alternatives), 2)
            self.assertEqual(self.node.meta["candidate"], "hardware_loop")

    def test_refinement_rejection_and_near_tie_traffic(self):
        specs = (
            self.implementation("invalid", 1, 1, True),
            self.implementation("fast", 100, 1000),
            self.implementation("less_traffic", 101, 700),
        )
        _, choice, _, rejected = select_implementation(
            self.node, self.node, self.context, specs
        )
        self.assertEqual(choice.implementation, "less_traffic")
        self.assertIn("private accumulator", rejected[0])
        self.assertEqual(self.node.meta["sentinel"], 42)

    def test_missing_cost_is_not_a_free_candidate(self):
        specs = (
            self.implementation("priced", 100, 100),
            ImplementationSpec(
                "unpriced", lambda *args: True, lambda *args: "graph"
            ),
        )
        with self.assertRaisesRegex(ValueError, "require modeled costs"):
            select_implementation(self.node, self.node, self.context, specs)

    def test_rejected_candidate_cannot_change_anchor_metadata(self):
        anchor = SimpleNamespace(meta={"layout": {"axis": 1}})

        def invalid(node, context, parameters):
            anchor.meta["layout"]["axis"] = 99
            raise CandidateRejected("invalid ISA")

        def valid(node, context, parameters):
            self.assertEqual(anchor.meta["layout"]["axis"], 1)
            return "valid"

        specs = (
            ImplementationSpec("invalid", lambda *a: True, invalid),
            ImplementationSpec("valid", lambda *a: True, valid),
        )
        select_implementation(self.node, anchor, self.context, specs)
        self.assertEqual(anchor.meta["layout"]["axis"], 1)

    def test_frozen_plan_rejects_selection(self):
        plan = CompilationPlan("test")
        plan.freeze(AllocationPlan(0, 0))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            plan.select("op", None)

    def test_duplicate_registration_is_an_error(self):
        registry = Registry()
        workload = WorkloadSpec(
            "sample", lambda _: None, lambda *args: ((), {})
        )
        registry.register(workload)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(workload)


if __name__ == "__main__":
    unittest.main()
