"""
Top level function of optimization framework
"""

import logging

from . import mapping_point_generator
from . import cost_model

from . import loop_enum as le
from . import buffer_enum as be

logger = logging.getLogger(__name__)


def opt_optimizer(
    resource,
    layer,
    hint=None,
    runtime_calc_func=None,
    verbose=False,
    runtime_tolerance=0.0,
    cost_calc_func=None,
    cost_tradeoff=True,
):
    """
    Evaluate the cost of each mapping point,
    record the mapping_point with the smallest cost

    Args:
        runtime_tolerance: How much longer than the best runtime a mapping may
            take and still be considered, as a fraction; see
            ``opt_mapping_point_generator_function``.
        cost_calc_func: Optional target cost ``(resource, layer, mapping)``.
            Defaults to the existing energy model. Search and Pareto selection
            are shared regardless of the metric.
        cost_tradeoff: False selects runtime only, with first-seen exact ties.
    """

    smallest_cost, smallest_runtime, perf, best_mapping_point = (
        mapping_point_generator.opt_mapping_point_generator_function(
            resource, layer, hint, runtime_calc_func, verbose, runtime_tolerance,
            cost_calc_func, cost_tradeoff,
        )
    )
    access_list, array_cost = cost_model.get_access(
        best_mapping_point, layer, resource
    )
    logger.info("Access_list: %s", access_list)
    logger.info("Array_cost: %s", array_cost)

    if cost_calc_func is None:
        logger.debug("Optimal_Energy_(pJ): %.2e", smallest_cost)
    else:
        logger.debug("Optimal_Target_Cost: %.2e", smallest_cost)
    logger.debug("Runtime_(cycles): %s", perf)

    return [smallest_cost, smallest_runtime, best_mapping_point, perf]


def optimizer(resource, layer, hint=None):
    smallest_cost = float("inf")
    mp_generator = mapping_point_generator.mapping_point_generator_function(
        resource, layer, hint
    )

    for mapping_point in mp_generator:
        cost = cost_model.get_cost(resource, mapping_point, layer)

        if cost < smallest_cost:
            smallest_cost = cost
            best_mapping_point = mapping_point
            logger.debug("Current smallest cost: %s", smallest_cost)
            logger.debug(
                "Current best mapping_point: %s %s",
                mapping_point.loop_blockings,
                mapping_point.loop_orders,
            )

    logger.debug("Smallest cost: %s", smallest_cost)

    return [smallest_cost, best_mapping_point]
