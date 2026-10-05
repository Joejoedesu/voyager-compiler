"""Selection policy: runtime tolerance, exact ties, and CLI propagation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from interstellar import mapping_point_generator as mpg
from interstellar.mapping_point import MappingPoint
from compilation.context import parse_args
from test_codegen import build_parser
from utils.models.utils import get_compile_args


class SelectionTest(unittest.TestCase):
    def select(self, candidates, **options):
        points = []
        for runtime, cost in candidates:
            p = MappingPoint([[0]] * 7, [[1]] * 7, [[1]] * 7)
            p.runtime, p.cost = runtime, cost
            points.append(p)
        with (
            patch.object(
                mpg,
                "blocking_partitioning_generator_function",
                return_value=[([[1]] * 7, [[1]] * 7, None)],
            ),
            patch.object(
                mpg, "opt_get_loop_order_generator", return_value=points
            ),
            patch.object(
                mpg.cost_model, "get_ideal_performance", return_value=1
            ),
            patch.object(
                mpg, "partitioned_loop_string", return_value=("", None)
            ),
            patch.object(mpg, "get_utilization", return_value=1),
        ):
            result = mpg.opt_mapping_point_generator_function(
                SimpleNamespace(para_index=[]),
                None,
                runtime_calc_func=lambda a, l, p: p.runtime,
                cost_calc_func=lambda a, l, p: p.cost,
                **options,
            )
        return points.index(result[-1])

    def test_default_trades_at_most_tolerance_for_cost(self):
        self.assertEqual(
            self.select(
                [(100, 100), (101, 50), (103, 1)], runtime_tolerance=0.02
            ),
            1,
        )

    def test_off_ignores_cost_and_tolerance(self):
        self.assertEqual(
            self.select(
                [(100, 100), (101, 50), (103, 1)],
                runtime_tolerance=0.5,
                cost_tradeoff=False,
            ),
            0,
        )

    def test_zero_tolerance_still_cost_tiebreaks_but_off_does_not(self):
        candidates = [(100, 100), (100, 1)]
        self.assertEqual(self.select(candidates, runtime_tolerance=0), 1)
        self.assertEqual(self.select(candidates, cost_tradeoff=False), 0)

    def test_off_still_finds_later_faster_candidate(self):
        self.assertEqual(
            self.select([(110, 1), (100, 100)], cost_tradeoff=False), 1
        )

    def test_cli_reaches_compile_options(self):
        parser = build_parser()
        for flag, expected in [
            ([], True),
            (["--no-interstellar-cost-tradeoff"], False),
            (["--interstellar-cost-tradeoff"], True),
        ]:
            args = parse_args(
                parser, ["resnet18", "--model_output_dir", "/tmp/unused", *flag]
            )
            self.assertEqual(
                get_compile_args(args)["interstellar_cost_tradeoff"], expected
            )


if __name__ == "__main__":
    unittest.main()
