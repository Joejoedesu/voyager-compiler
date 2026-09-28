"""Hardware graph validation, unit conversion, and backend legality contracts."""

import unittest
from dataclasses import replace

from voyager_compiler.hardware_config import (
    VOYAGER,
    AcceleratorConfig,
    AccessMode,
    Bandwidth,
    BandwidthUnit,
    CapacityUnit,
    ComputationUnit,
    ComputeMode,
    Connection,
    DataType,
    MemoryHierarchy,
    MemoryInstance,
    MemoryLevel,
    MemoryPort,
    MemorySize,
    OperandCapability,
    OperationCapability,
    OperationTiming,
    ParameterProvenance,
    SpatialUnrolling,
    StorageTarget,
    UnrollingRef,
    voyager_config,
)
from voyager_compiler.voyager_adapter import (
    fusion_patterns,
    interstellar_memory,
)


def update_memory(config, name, **changes):
    levels = tuple(
        replace(
            level,
            instances=tuple(
                replace(mem, **changes) if mem.name == name else mem
                for mem in level.instances
            ),
        )
        for level in config.memory.levels
    )
    return replace(config, memory=replace(config.memory, levels=levels))


def update_unit(config, name, **changes):
    return replace(
        config,
        computation_units=tuple(
            replace(unit, **changes) if unit.name == name else unit
            for unit in config.computation_units
        ),
    )


class HardwareIRTest(unittest.TestCase):

    def test_graph_is_the_source_of_flat_views(self):
        c = voyager_config(scratchpad_size=1 << 20, num_banks=16)
        c = update_memory(
            c,
            "scratchpad",
            word_bytes=24,
            reserved_bytes=65536,
        )
        c = update_unit(
            c,
            "matrix",
            spatial_unrolling=(
                SpatialUnrolling("IC", 8),
                SpatialUnrolling("OC", 48),
            ),
        )
        self.assertEqual(c.pe_array_size, (8, 48))
        self.assertEqual(c.usable_banks, 15)
        self.assertEqual(c.usable_scratchpad_size, 983040)
        self.assertEqual(c.sram_bandwidth_bits(4), 192)
        resource = interstellar_memory(c)
        self.assertEqual(
            resource["buf_capacity_list"],
            [[1, 1, 1], [8192, 49152, 49152], [983040], [16 * 1024**3]],
        )
        self.assertEqual(resource["para_count_list"], [384, 1, 1, 1])
        self.assertEqual(
            resource["memory_partitions"],
            [[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        )

    def test_link_units_and_dtype_dependent_port(self):
        c = voyager_config(pe_array_size=(8, 16), frequency=2)
        self.assertEqual(c.bytes_per_cycle, 32)
        self.assertEqual(c.access_latency_cycles, 200)
        self.assertEqual(c.sram_bandwidth_bits(4), 32)
        self.assertIsInstance(c.sram_bandwidth_bits(4), int)
        self.assertEqual(c.sram_bandwidth_bits(8), 64)
        edge = replace(
            c.connection("dram_sram"),
            bandwidth=Bandwidth(12, BandwidthUnit.BYTES_PER_CYCLE),
        )
        c = replace(
            c,
            connections=tuple(
                edge if e.name == edge.name else e for e in c.connections
            ),
        )
        self.assertEqual(c.bytes_per_cycle, 12)
        self.assertEqual(c.dram_bandwidth, 24)
        # Compute SRAM bandwidth is a separate physical connection.
        self.assertEqual(c.compute_bandwidth(), 32)

    def test_explicit_isa_contracts_authorize_fusion(self):
        vector = VOYAGER.compute_unit("vector")
        self.assertTrue(vector.supports("relu", DataType("bfloat16", 16)))
        self.assertFalse(vector.supports("relu", DataType("float64", 64)))
        # Removing an operation cannot silently rewrite an ISA contract.
        with self.assertRaises(ValueError):
            update_unit(
                VOYAGER,
                "vector",
                supported_operations=vector.supported_operations - {"relu"},
            )
        c = replace(VOYAGER, isa_pipelines=())
        self.assertEqual(c.connections, VOYAGER.connections)
        self.assertEqual(
            fusion_patterns(c), []
        )  # links do not authorize ISA fusion
        c = replace(VOYAGER, isa_pipelines=(VOYAGER.isa_pipelines[-1],))
        self.assertEqual(len(fusion_patterns(c)), 1)
        self.assertEqual(len(fusion_patterns(VOYAGER)), 3)

    def test_other_accelerator_can_describe_its_own_graph(self):
        dtype = DataType("int8", 8)
        units = (
            ComputationUnit(
                "tensor",
                (dtype,),
                frozenset({"matmul"}),
                (SpatialUnrolling("M", 16), SpatialUnrolling("N", 16)),
            ),
            ComputationUnit(
                "alu",
                (dtype,),
                frozenset({"add"}),
                (SpatialUnrolling("lanes", 4),),
            ),
        )
        local = MemoryLevel(
            "local",
            (
                MemoryInstance(
                    "weights",
                    MemorySize(4096, CapacityUnit.BYTES),
                    frozenset({StorageTarget.WEIGHT}),
                ),
                MemoryInstance(
                    "partial_sums",
                    MemorySize(4096, CapacityUnit.BYTES),
                    frozenset({StorageTarget.PSUM}),
                ),
            ),
        )
        outer = MemoryLevel(
            "shared",
            (
                MemoryInstance(
                    "unified",
                    MemorySize(1 << 20, CapacityUnit.BYTES),
                    frozenset({StorageTarget.WEIGHT, StorageTarget.ACTIVATION}),
                ),
            ),
        )
        rate = Bandwidth(8, BandwidthUnit.BYTES_PER_CYCLE)
        connections = (
            Connection("tensor_alu", "tensor", "alu", rate),
            Connection("local_compute", "weights", "tensor", rate),
            Connection("local_outer", "unified", "weights", rate),
        )
        c = AcceleratorConfig(
            name="example-accelerator",
            computation_units=units,
            memory=MemoryHierarchy((local, outer)),
            connections=connections,
            frequency=0.5,
        )
        self.assertEqual(len(c.memory.levels[0].instances), 2)
        self.assertEqual(
            c.connection("tensor_alu").bandwidth.bytes_per_cycle(c), 8
        )
        with self.assertRaises(NotImplementedError):
            interstellar_memory(c)

    def test_compute_modes_share_one_physical_unit(self):
        int8, int32 = DataType("int8", 8), DataType("int32", 32)
        bf16, fp32 = DataType("bf16", 16), DataType("fp32", 32)

        def mode(name, input_type, accumulator_type, lanes):
            return ComputeMode(
                name,
                (
                    OperationCapability(
                        "matmul",
                        (
                            OperandCapability(
                                "lhs",
                                input_type,
                                AccessMode.READ,
                                ("scratchpad",),
                            ),
                            OperandCapability(
                                "rhs",
                                input_type,
                                AccessMode.READ,
                                ("scratchpad",),
                            ),
                            OperandCapability(
                                "accumulator",
                                accumulator_type,
                                AccessMode.INTERNAL,
                            ),
                            OperandCapability(
                                "result",
                                accumulator_type,
                                AccessMode.WRITE,
                                ("accumulator",),
                            ),
                        ),
                    ),
                ),
                (SpatialUnrolling("lanes", lanes),),
                OperationTiming(operations_per_cycle=lanes, latency_cycles=8),
            )

        unit = ComputationUnit(
            "tensor",
            modes=(mode("int8", int8, int32, 32), mode("bf16", bf16, fp32, 16)),
        )
        config = AcceleratorConfig(
            name="two-mode-unit",
            backend="test",
            frequency=1,
            computation_units=(unit,),
            memory=MemoryHierarchy(
                (
                    MemoryLevel(
                        "local",
                        (
                            MemoryInstance(
                                "scratchpad",
                                MemorySize(4096, CapacityUnit.BYTES),
                                frozenset(
                                    {
                                        StorageTarget.ACTIVATION,
                                        StorageTarget.WEIGHT,
                                    }
                                ),
                            ),
                            MemoryInstance(
                                "accumulator",
                                MemorySize(4096, CapacityUnit.BYTES),
                                frozenset({StorageTarget.PSUM}),
                            ),
                        ),
                    ),
                )
            ),
            connections=(
                Connection(
                    "read",
                    "scratchpad",
                    "tensor",
                    Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
                    bidirectional=False,
                ),
                Connection(
                    "write",
                    "tensor",
                    "accumulator",
                    Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
                    bidirectional=False,
                ),
            ),
        )
        self.assertEqual(len(config.computation_units), 1)
        self.assertEqual(unit.mode_concurrency, "exclusive")
        signature = dict(lhs=int8, rhs=int8, accumulator=int32, result=int32)
        self.assertTrue(
            unit.supports_signature(
                "int8", "matmul", signature, {"lhs": "scratchpad"}
            )
        )
        self.assertFalse(unit.supports_signature("bf16", "matmul", signature))
        self.assertFalse(
            unit.supports_signature(
                "int8", "matmul", {**signature, "accumulator": int8}
            )
        )
        self.assertFalse(
            unit.supports_signature(
                "int8", "matmul", signature, {"lhs": "accumulator"}
            )
        )
        self.assertEqual(unit.unrolling("lanes", mode="bf16").factor, 16)
        capability = unit.mode("int8").operations[0]
        self.assertEqual(
            unit.operation_timing("int8", capability).operations_per_cycle, 32
        )
        independent = replace(
            config,
            computation_units=(unit, replace(unit, name="tensor2")),
            connections=config.connections
            + (
                Connection(
                    "read2",
                    "scratchpad",
                    "tensor2",
                    Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
                    bidirectional=False,
                ),
                Connection(
                    "write2",
                    "tensor2",
                    "accumulator",
                    Bandwidth(16, BandwidthUnit.BYTES_PER_CYCLE),
                    bidirectional=False,
                ),
            ),
        )
        self.assertEqual(len(independent.computation_units), 2)
        with self.assertRaises(ValueError):
            replace(unit, mode_concurrency="parallel")
        with self.assertRaises(ValueError):
            replace(config, connections=config.connections[:1])
        with self.assertRaises(ValueError):
            replace(
                config,
                connections=(
                    replace(config.connections[0], source="accumulator"),
                    config.connections[1],
                ),
            )

    def test_partitioned_memory_ports_and_shared_bandwidth(self):
        memory = MemoryInstance(
            "psum",
            MemorySize(2 * 1024**2, CapacityUnit.BYTES),
            frozenset({StorageTarget.PSUM}),
            banks=8,
            partitions=128,
            row_bytes=4,
            partition_start_alignment=32,
            allocation_alignment_bytes=16,
            single_bank_allocation=True,
            ports=(MemoryPort("rw", AccessMode.READ_WRITE, scope="bank"),),
        )
        config = AcceleratorConfig(
            name="shared-port",
            backend="test",
            frequency=2,
            computation_units=(
                ComputationUnit("vector"),
                ComputationUnit("scalar"),
            ),
            memory=MemoryHierarchy((MemoryLevel("local", (memory,)),)),
            connections=(
                Connection(
                    "vector_psum",
                    "psum",
                    "vector",
                    Bandwidth(64, BandwidthUnit.BYTES_PER_CYCLE),
                    source_port="rw",
                    startup_ns=5,
                ),
                Connection(
                    "scalar_psum",
                    "psum",
                    "scalar",
                    shared_with="vector_psum",
                    source_port="rw",
                ),
            ),
            provenance=(
                ParameterProvenance(
                    "connections.vector_psum.bandwidth.value",
                    "assumed",
                    notes="Synthetic test bandwidth",
                ),
            ),
        )
        self.assertEqual(memory.partition_size, 16384)
        self.assertEqual(memory.bank_size, 262144)
        self.assertEqual(memory.partition_bank_size, 2048)
        self.assertIs(
            config.bandwidth_connection("scalar_psum"),
            config.connection("vector_psum"),
        )
        self.assertEqual(config.connection_bytes_per_cycle("scalar_psum"), 64)
        self.assertEqual(
            config.parameter_value(config.provenance[0].parameter), 64
        )
        with self.assertRaises(ValueError):
            replace(memory, partitions=3)
        with self.assertRaises(ValueError):
            replace(memory, ports=(MemoryPort("rw", AccessMode.INTERNAL),))
        with self.assertRaises(ValueError):
            replace(
                config,
                memory=MemoryHierarchy(
                    (
                        MemoryLevel(
                            "local",
                            (
                                replace(
                                    memory,
                                    ports=(MemoryPort("rw", AccessMode.READ),),
                                ),
                            ),
                        ),
                    )
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                config,
                connections=(
                    replace(
                        config.connections[0],
                        bandwidth=None,
                        shared_with="scalar_psum",
                    ),
                    config.connections[1],
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                config,
                connections=(
                    config.connections[0],
                    replace(config.connections[1], source_port="missing"),
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                config,
                provenance=(
                    ParameterProvenance(
                        "connections.missing.bandwidth.value", "assumed"
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                config,
                provenance=(ParameterProvenance("frequency", "measured"),),
            )
        # A shared budget has one source of truth; aliases cannot invent a rate.
        with self.assertRaises(ValueError):
            replace(
                config.connections[1],
                bandwidth=Bandwidth(128, BandwidthUnit.BYTES_PER_CYCLE),
            )

    def test_invalid_graphs_fail_early(self):
        with self.assertRaises(ValueError):
            replace(
                VOYAGER,
                connections=(
                    replace(VOYAGER.connections[0], source="missing"),
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                VOYAGER,
                computation_units=VOYAGER.computation_units
                + (VOYAGER.computation_units[0],),
            )
        with self.assertRaises(ValueError):
            update_unit(
                VOYAGER,
                "vector",
                spatial_unrolling=(
                    SpatialUnrolling(
                        "lanes", fallback=UnrollingRef("vector", "lanes")
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            MemoryLevel("empty", ())
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Bandwidth(value, BandwidthUnit.GB_PER_SECOND)
        for offset in (-1, 1, 1 << 20):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                voyager_config(
                    scratchpad_size=1 << 20,
                    num_banks=16,
                    scratchpad_offset=offset,
                )
        with self.assertRaises(TypeError):
            replace(VOYAGER, pe_array_size=(4, 4))

    def test_backend_does_not_silently_flatten_new_topologies(self):
        levels = VOYAGER.memory.levels
        extra = MemoryLevel(
            "extra",
            (
                MemoryInstance(
                    "extra_store",
                    MemorySize(1024, CapacityUnit.BYTES),
                    frozenset({StorageTarget.ACTIVATION}),
                ),
            ),
        )
        c = replace(VOYAGER, memory=MemoryHierarchy((*levels, extra)))
        with self.assertRaises(NotImplementedError):
            interstellar_memory(c)
        c = update_memory(VOYAGER, "scratchpad", buffering=3)
        with self.assertRaises(NotImplementedError):
            interstellar_memory(c)

    def test_voyager_rejects_unimplemented_hardware_contracts(self):
        for changes in (
            dict(partitions=2),
            dict(allocation_alignment_bytes=16),
            dict(ports=(MemoryPort("rw", AccessMode.READ_WRITE),)),
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(NotImplementedError),
            ):
                interstellar_memory(
                    update_memory(VOYAGER, "scratchpad", **changes)
                )
        edge = replace(
            VOYAGER.connection("sram_vector"),
            bandwidth=None,
            shared_with="sram_input",
        )
        config = replace(
            VOYAGER,
            connections=tuple(
                edge if c.name == edge.name else c for c in VOYAGER.connections
            ),
        )
        with self.assertRaises(NotImplementedError):
            interstellar_memory(config)


if __name__ == "__main__":
    unittest.main()
