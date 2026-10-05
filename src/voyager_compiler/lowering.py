"""Admission and memory adapters for the shared bufferized pipeline."""


def validate_bufferized_target(config):
    from voyager_compiler.targets import get_backend

    backend = get_backend(config.backend)
    if not getattr(backend, "uses_bufferized_flow", False):
        raise NotImplementedError(f"{config.backend}: no bufferized lowering")
    backend.validate(config)


def interstellar_memory(config):
    from voyager_compiler.targets import get_backend

    adapter = getattr(get_backend(config.backend), "interstellar_memory", None)
    if adapter is None:
        from voyager_compiler.voyager_adapter import (
            interstellar_memory as adapter,
        )
    return adapter(config)
