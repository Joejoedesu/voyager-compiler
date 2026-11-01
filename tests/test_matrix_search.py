"""Target matrix constraints must be independent of the shared grid builder."""

import math
import unittest
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import interstellar
from voyager_compiler.compiler.search.interstellar import (
    _require_shared_matrix_schedule,
    get_tiling,
    memory_level,
    search_matrix,
    search_tilings,
    validate_mapping,
)
from voyager_compiler.compiler.search.planner import select_implementation
from voyager_compiler.compiler.search.tiler import (
    DEFAULT_RUNTIME_TOLERANCE,
    pick,
)
from voyager_compiler.extensions import ImplementationSpec, NoLegalMapping
from voyager_compiler.ir.plan import (
    Cost,
    GridSpace,
    MatrixSpace,
    Placement,
    TileResult,
)

le = interstellar.le


def rows(mapping):
    return math.prod(mapping.loop_blockings[le.OY][:3])


def one_row(mapping):
    return rows(mapping) == 1


def reject_all(mapping):
    return False


class Model:
    def __init__(self, row_limit=None, mapping_filter=None):
        resource = interstellar.Resource(
            buf_capacity_list=[[1, 1, 1], [64, 64, 64], [4096], [1 << 20]],
            buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100], [1000]],
            buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
            para_count_list=[4, 1, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
            mac_capacity=0,
            partition_mode=[0] * 4,
            invalid_underutilized=False,
            bank_size_list=[None] * 4,
        )
        hints = {}
        for i, name in enumerate(("FX", "FY", "OX", "OY", "OC", "IC", "ON")):
            hints[name] = {
                "level0": dict(
                    order=None, blocking_size=1, partitioning_size=1
                ),
                "level1": dict(
                    order=None, blocking_size=1, partitioning_size=1
                ),
                "level2": dict(order=None, partitioning_size=1),
                # Deliberately NOT reduction innermost: a custom builder contract.
                "level3": dict(order=None, partitioning_size=1),
            }
        hints["IC"]["level3"]["order"] = -1
        hints["IC"]["level0"].pop("blocking_size")
        hints["OC"]["level0"].pop("blocking_size")
        hints["IC"]["level0"].update(order=1, partitioning_size=2)
        hints["OC"]["level0"].update(order=0, partitioning_size=2)
        if row_limit is not None:
            hints["OY"]["level2"]["blocking_size"] = row_limit
        data = interstellar.extract_input.extract_schedule_info(
            {"schedule_hint": hints}, 4
        )
        schedule = interstellar.Schedule(
            data["schedule_hint"], data["partition_loops"]
        )
        self.space = MatrixSpace(resource, schedule, mapping_filter)
        self.key = ("test", row_limit, mapping_filter)
        self.seen = []

    def fit(self, mapping):
        return Placement()

    def cost(self, mapping, placement):
        self.seen.append(mapping)
        launches = math.prod(b[3] for b in mapping.loop_blockings)
        return Cost(100 * launches, 1000 * launches)


def layer():
    return interstellar.Layer(nifm=2, nofm=2, wofm=2, hofm=2, wfil=1, hfil=1)


class MatrixSearchTests(unittest.TestCase):
    def test_three_level_native_space_and_order_priority(self):
        model = Model()
        resource = interstellar.Resource(
            [[4, 4, 4], [128], [1024]],
            [[0, 0, 0], [0], [1]],
            [[0, 0, 0], [0], [0]],
            [4, 1, 1],
            memory_partitions=[[0, 1, 2], [0, 0, 0], [0, 0, 0]],
            invalid_underutilized=False,
        )
        hints = {
            loop: [
                [
                    None,
                    None if loop in (le.IC, le.OC) else 1,
                    2 if loop in (le.IC, le.OC) else 1,
                ],
                [None, None, 1],
                [None, None, 1],
            ]
            for loop in range(le.NUM)
        }
        priority = (le.IC, le.OC, le.OX, le.OY, le.ON, le.FX, le.FY)
        schedule = interstellar.Schedule(
            hints, order_priority={i: priority for i in range(3)}
        )
        model.space = MatrixSpace(resource, schedule, fit_level=1)
        model.cost = lambda m, p: Cost(
            100 * math.prod(b[2] for b in m.loop_blockings)
        )
        best = search_matrix(model, layer())
        validate_mapping(best.tiling, layer(), 3)
        self.assertEqual(best.cost.latency, 100)
        self.assertEqual(best.tiling.loop_orders[le.OX][1], 0)
        self.assertEqual(best.tiling.loop_orders[le.OY][1], 1)
        model.fit = lambda m: None
        with self.assertRaises(NoLegalMapping):
            search_matrix(model, layer())
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            interstellar.Schedule(hints, order_priority={3: priority})

    def test_selected_mapping_is_validated(self):
        from copy import deepcopy

        result = search_matrix(Model(), layer())
        broken = deepcopy(result.tiling)
        broken.loop_blockings = [list(row) for row in broken.loop_blockings]
        broken.loop_blockings[le.IC][0] *= 2
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_mapping(broken, layer(), 4)

    def test_named_memory_translation_checks_units_and_sharing(self):
        from voyager_compiler.hardware import CapacityUnit, MemorySize

        memories = {
            "sram": SimpleNamespace(size=MemorySize(1024, CapacityUnit.BYTES)),
            "acc": SimpleNamespace(size=MemorySize(512, CapacityUnit.BYTES)),
        }
        config = SimpleNamespace(memory_instance=memories.__getitem__)
        self.assertEqual(
            memory_level(
                config,
                ("sram", "acc", "sram"),
                bytes_per_element=(1, 4, 1),
                slots=2,
            ),
            ([512, 64], [0, 1, 0]),
        )
        with self.assertRaisesRegex(ValueError, "Mixed-width"):
            memory_level(config, ("sram",) * 3, bytes_per_element=(1, 4, 1))
        memories["sram"].size = MemorySize(1024, CapacityUnit.ROWS)
        with self.assertRaisesRegex(ValueError, "byte capacity"):
            memory_level(config, ("sram",) * 3)

    def test_target_schedule_can_search_both_spatial_axes(self):
        broad, narrow = Model(), Model(row_limit=1)
        best = search_matrix(broad, layer())
        limited = search_matrix(narrow, layer())
        self.assertEqual(rows(best.tiling), 2)
        self.assertEqual(rows(limited.tiling), 1)
        self.assertLess(best.cost.latency, limited.cost.latency)
        with self.assertRaisesRegex(ValueError, "shared matrix builder"):
            _require_shared_matrix_schedule(broad.space, "test")

    def test_full_mapping_filter_runs_before_cost(self):
        model = Model(mapping_filter=one_row)
        best = search_matrix(model, layer())
        self.assertEqual(rows(best.tiling), 1)
        self.assertTrue(all(one_row(m) for m in model.seen))
        self.assertTrue(all(m.loop_orders is not None for m in model.seen))

    def test_search_does_not_replace_callers_layer_callback(self):
        geometry = layer()
        callback = lambda *args: (_ for _ in ()).throw(
            AssertionError("caller callback used")
        )
        geometry.size_fn = callback
        original = dict(vars(geometry))
        search_matrix(Model(), geometry)
        self.assertEqual(vars(geometry), original)
        with self.assertRaises(NoLegalMapping):
            search_matrix(Model(mapping_filter=reject_all), geometry)
        self.assertEqual(vars(geometry), original)

    def test_rejection_is_not_selected_as_infinite_cost(self):
        with self.assertRaises(NoLegalMapping):
            search_matrix(Model(mapping_filter=reject_all), layer())

    def test_gemv_honors_compilation_tolerance_in_both_search_routes(self):
        class Node:
            name = "gemv"
            kwargs = {}
            args = (
                SimpleNamespace(shape=(1, 4)),
                SimpleNamespace(shape=(4, 4)),
            )
            value = SimpleNamespace(shape=(1, 4), ndim=2)

        node = Node()
        op = SimpleNamespace(
            space=GridSpace(lanes=1, ranked=True),
            fit=lambda mapping: Placement(),
            cost=lambda mapping, placement: (
                Cost(100, 1000)
                if mapping.tile_sizes == (1, 4, 4)
                else Cost(101, 1)
            ),
        )
        paths = "voyager_compiler.compiler.search."
        with ExitStack() as stack:
            for module, symbol, value in [
                ("interstellar", "get_anchor_node", node),
                ("interstellar", "is_gemm_op", True),
                ("interstellar", "is_conv2d", False),
                ("interstellar", "_prepare_search", (op, None)),
                ("tile_shapes", "get_anchor_node", node),
                ("tile_shapes", "is_bmm", False),
                ("tile_shapes", "weight_is_ck", True),
                ("tile_shapes", "mha_head", None),
            ]:
                stack.enter_context(
                    patch(paths + module + "." + symbol, return_value=value)
                )
            for tolerance, latency in [(0, 100), (0.02, 101)]:
                tiler = SimpleNamespace(
                    runtime_tolerance=tolerance, constraints={}
                )
                for route in ("direct", "batch"):
                    node.meta = {}
                    if route == "direct":
                        get_tiling(node, tiler)
                    else:
                        search_tilings([(node, None)], tiler)
                    self.assertEqual(
                        node.meta["tile_result"].cost.latency, latency
                    )
            op.fit = lambda mapping: None
            node.all_input_nodes = []
            with self.assertRaises(NoLegalMapping):
                get_tiling(node, tiler)
            self.assertIsNone(
                search_tilings([(node, None)], tiler)[node, None]
            )

    def test_default_prefers_latency_over_one_percent_traffic_saving(self):
        fast = TileResult(None, Placement(), Cost(100, 1000))
        lean = replace(fast, cost=Cost(101, 1))
        self.assertEqual(DEFAULT_RUNTIME_TOLERANCE, 0)
        self.assertIs(pick([lean, fast], DEFAULT_RUNTIME_TOLERANCE), fast)
        self.assertIs(pick([lean, fast], 0.02), lean)
        node = SimpleNamespace(name="op", meta={})
        context = SimpleNamespace(
            tiler=SimpleNamespace(runtime_tolerance=DEFAULT_RUNTIME_TOLERANCE)
        )
        specs = tuple(
            ImplementationSpec(
                name,
                lambda *a: True,
                lambda *a: "built",
                estimate=lambda *a, cost=cost: cost,
            )
            for name, cost in [("lean", lean.cost), ("fast", fast.cost)]
        )
        _, selected, _, _ = select_implementation(node, node, context, specs)
        self.assertEqual(selected.implementation, "fast")


if __name__ == "__main__":
    unittest.main()
