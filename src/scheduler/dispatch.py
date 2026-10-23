# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Backend-neutral entry points for selecting a scheduler provider."""

from __future__ import annotations

from collections.abc import MutableMapping

from scheduler.interstellar import INTERSTELLAR_SCHEDULER
from scheduler.interstellar.voyager import run_interstellar_scheduler
from scheduler.magnet_alpha_scheduler.scheduler import (
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
)
from scheduler.types import (
    LocalScheduleResult,
    OperationWorkload,
    SchedulerProviderSet,
)

GENERIC_SCHEDULER_PROVIDERS = SchedulerProviderSet(
    default_backend=INTERSTELLAR_SCHEDULER,
    gemm_providers={
        INTERSTELLAR_SCHEDULER: run_interstellar_scheduler,
    },
)
LOCAL_SCHEDULERS = (
    INTERSTELLAR_SCHEDULER,
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
)


def resolve_local_scheduler(
    backend,
    *,
    providers: SchedulerProviderSet = GENERIC_SCHEDULER_PROVIDERS,
):
    """Resolve an optional engine inside a compiler-selected provider set."""
    if not isinstance(providers, SchedulerProviderSet):
        raise TypeError("providers must be a SchedulerProviderSet")
    selected = providers.default_backend if backend is None else backend
    if selected not in providers.gemm_providers:
        raise ValueError(
            f"Scheduler {selected!r} is not registered in the selected "
            "provider set"
        )
    return selected


def schedule_operation(
    workload: OperationWorkload,
    hardware_config,
    provider_cache,
    scheduler_options=None,
    result_cache: MutableMapping | None = None,
    *,
    providers: SchedulerProviderSet = GENERIC_SCHEDULER_PROVIDERS,
    backend=None,
) -> LocalScheduleResult | None:
    """Make one operation-or-GEMM scheduling decision.

    The compiler chooses ``providers`` from its compilation target. Target
    operation policy gets the first chance to claim the operation; only an
    unclaimed operation with a canonical GEMM projection reaches the selected
    local GEMM engine.
    """
    if not isinstance(workload, OperationWorkload):
        raise TypeError("scheduler requires an OperationWorkload")
    if not isinstance(providers, SchedulerProviderSet):
        raise TypeError("providers must be a SchedulerProviderSet")

    operation_provider = providers.operation_provider
    if operation_provider is not None:
        result = operation_provider(workload, hardware_config)
        if result is not None:
            return result

    if workload.gemm is None:
        return None

    backend = resolve_local_scheduler(backend, providers=providers)
    provider = providers.gemm_providers[backend]

    cache_key = (backend, workload)
    if result_cache is not None and cache_key in result_cache:
        return result_cache[cache_key]

    result = provider(
        workload,
        hardware_config,
        provider_cache,
        scheduler_options,
    )
    if result_cache is not None:
        result_cache[cache_key] = result
    return result


__all__ = [
    "INTERSTELLAR_SCHEDULER",
    "GENERIC_SCHEDULER_PROVIDERS",
    "LOCAL_SCHEDULERS",
    "MAGNET_ALPHA_RULE_BASED_SCHEDULER",
    "resolve_local_scheduler",
    "schedule_operation",
]
