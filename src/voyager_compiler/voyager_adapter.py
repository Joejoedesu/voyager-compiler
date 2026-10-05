"""Adapt hardware IR to the currently implemented Voyager lowering.

The IR can describe other accelerators. This adapter explicitly bounds what
the existing four-level Interstellar search and SRAM allocator can lower.
"""

import math

from voyager_compiler.hardware_config import (
    VOYAGER,
    CapacityUnit,
    StorageTarget,
)


def interstellar_memory(config):
    config.require_backend("voyager")
    # These declarations are available to target-specific adapters. The current
    # Voyager algorithms must not silently ignore constraints they do not use.
    if any(unit.modes for unit in config.computation_units):
        raise NotImplementedError(
            "Voyager lowering does not select compute modes yet"
        )
    for mem in config.memory.instances:
        if (
            mem.partitions != 1
            or mem.row_bytes is not None
            or mem.allocation_alignment_bytes != 1
            or mem.partition_start_alignment != 1
            or mem.single_bank_allocation
            or mem.ports
        ):
            raise NotImplementedError(
                "Voyager lowering does not support extended memory contracts yet"
            )
    if any(
        edge.shared_with is not None
        or edge.source_port is not None
        or edge.target_port is not None
        or edge.startup_ns != 0
        or edge.transfer_geometry is not None
        or edge.service_resource is not None
        or edge.service_bandwidth is not None
        for edge in config.connections
    ):
        raise NotImplementedError(
            "Voyager lowering does not model shared ports/bandwidth or transfer startup yet"
        )
    expected_links = {edge.name: edge for edge in VOYAGER.connections}
    if {edge.name for edge in config.connections} != expected_links.keys():
        raise NotImplementedError(
            "Voyager lowering requires its declared connection topology"
        )
    variable_rates = {
        "dram_sram",
        "sram_vector",
        "sram_matrix_vector",
        "matrix_vector_stream",
        "sram_input",
        "sram_weight",
        "sram_accum",
        "sram_sparse",
        "sram_scales",
    }
    for edge in config.connections:
        expected = expected_links[edge.name]
        if (edge.source, edge.target, edge.bidirectional) != (
            expected.source,
            expected.target,
            expected.bidirectional,
        ):
            raise NotImplementedError(
                f"Unsupported Voyager endpoints/direction on {edge.name}"
            )
        if (
            edge.name not in variable_rates
            and edge.bandwidth != expected.bandwidth
        ):
            raise NotImplementedError(
                f"Voyager's register path rate is fixed: {edge.name}"
            )
        if edge.name != "dram_sram" and edge.latency_ns != expected.latency_ns:
            raise NotImplementedError(
                f"Voyager does not model additional latency on {edge.name}"
            )
        if (
            edge.name != "matrix_vector_stream"
            and edge.buffer_depth != expected.buffer_depth
        ):
            raise NotImplementedError(
                f"Voyager does not model additional buffering on {edge.name}"
            )
    levels = config.memory.levels
    if tuple(level.name for level in levels) != ("PE", "L1", "L2", "DRAM"):
        raise NotImplementedError(
            "Voyager lowering requires PE/L1/L2/DRAM levels"
        )
    for name in ("scratchpad", "accum_buffer"):
        if config.memory_instance(name).buffering not in (1, 2):
            raise NotImplementedError(
                "Voyager lowering supports one or two buffer slots"
            )
    scales = config.memory_instance("spmm_scales")
    if scales.size.unit != CapacityUnit.ROWS or scales.size.value is None:
        raise ValueError(
            "Voyager's SpMM scale capacity must be specified in rows"
        )
    for mem in config.memory.instances:
        if mem.name != "scratchpad" and (
            mem.banks is not None or mem.reserved_bytes
        ):
            raise NotImplementedError(
                "Voyager supports bank geometry and reservations on scratchpad only"
            )
        if mem.name not in ("scratchpad", "accum_buffer"):
            if mem.buffering != VOYAGER.memory_instance(mem.name).buffering:
                raise NotImplementedError(
                    f"Voyager does not model variable buffer slots for {mem.name}"
                )
    if len(levels[2].instances) != 1 or len(levels[3].instances) != 1:
        raise NotImplementedError(
            "Voyager lowering requires a unified scratchpad and DRAM"
        )
    for name in ("sram_weight", "sram_accum", "sram_sparse", "sram_scales"):
        if (
            config.connection(name).bandwidth
            != config.connection("sram_input").bandwidth
        ):
            raise NotImplementedError(
                "Voyager matrix timing requires equal L1 port bandwidths"
            )

    roles = (StorageTarget.ACTIVATION, StorageTarget.PSUM, StorageTarget.WEIGHT)
    capacities, access, static, partitions, parallel, banks = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, level in enumerate(levels):
        memories = []
        partition = []
        for role in roles:
            matches = [m for m in level.instances if role in m.targets]
            if len(matches) != 1:
                raise NotImplementedError(
                    f"Voyager needs exactly one {role.value} store at {level.name}"
                )
            mem = matches[0]
            expected_unit = (
                CapacityUnit.ELEMENTS if index < 2 else CapacityUnit.BYTES
            )
            if mem.size.unit != expected_unit:
                raise ValueError(
                    f"{level.name} requires capacities in {expected_unit.value}"
                )
            if mem not in memories:
                memories.append(mem)
            partition.append(memories.index(mem))
        capacities.append([config.capacity(m) for m in memories])
        access.append([m.access_cost for m in memories])
        static.append([m.static_cost for m in memories])
        partitions.append(partition)
        parallel.append(
            math.prod(config.unroll(r) for r in level.spatial_scope)
        )
        banks.append(memories[0].bank_size if index == 2 else None)
    return dict(
        buf_capacity_list=capacities,
        buf_access_cost_list=access,
        buf_unit_static_cost_list=static,
        memory_partitions=partitions,
        para_count_list=parallel,
        bank_size_list=banks,
    )


def fusion_patterns(config):
    """Translate explicit ISA fusion contracts into the existing matchers.

    Physical compute connections are not sufficient to authorize a single
    fused ISA call. Future topology-based discovery must intersect its
    candidates with target ISA legality before supplying these contracts.
    Callers may still supply their own vector_pipeline to transform/fuse.
    """
    import torch

    from voyager_compiler import OpMatcher
    from voyager_compiler.codegen.node_info import is_fully_connected

    config.require_backend("voyager")

    def matrix_tail(node):
        if hasattr(node, "value") and is_fully_connected(node):
            return node.args[0].meta.get("dtype") is not None
        return True

    def constant_divisor(node):
        if node.target != torch.ops.aten.div.Tensor:
            return True
        divisor = node.args[1]
        return (
            not isinstance(divisor, torch.fx.Node) or divisor.value.numel() == 1
        )

    predicates = {
        None: None,
        "matrix_tail": matrix_tail,
        "constant_divisor": constant_divisor,
    }
    patterns = []
    for pipeline in config.isa_pipelines:
        stages = []
        for stage in pipeline.stages:
            if stage.predicate not in predicates:
                raise NotImplementedError(
                    f"Unknown Voyager ISA predicate: {stage.predicate}"
                )
            stages.append(
                OpMatcher(
                    *stage.operations, predicate=predicates[stage.predicate]
                )
            )
        patterns.append(stages)
    return patterns
