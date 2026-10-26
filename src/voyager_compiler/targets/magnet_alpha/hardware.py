# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Alpha's physical memories and execution modes in the common hardware IR.

The source profile does not establish a clock or off-chip bandwidth. They stay
unknown; a device performance profile must supply evidence before time estimates.
"""

from dataclasses import dataclass
from types import SimpleNamespace

from voyager_compiler.hardware import (
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
    MemorySize,
    OperandCapability,
    OperationCapability,
    ParameterProvenance,
    SpatialUnrolling,
    StorageTarget,
)
from voyager_compiler.ir.plan import ExecutionOptions

ALPHA_INPUT_WORD_BITS = 128
ALPHA_OUTPUT_LANES = 32
ALPHA_VSQ_BLOCK_VALUES = 32
ALPHA_COLLECTOR_DEPTH = 32
ALPHA_ACCUMULATOR_BITS = 32
ALPHA_ACTIVATION_READ_BITS_PER_CYCLE = 128
ALPHA_ACTIVATION_WRITE_BITS_PER_CYCLE = 256
ALPHA_WEIGHT_READ_BITS_PER_CYCLE = 512
ALPHA_NMP_READ_BITS_PER_CYCLE = 256
ALPHA_NMP_WRITE_BITS_PER_CYCLE = 256
ALPHA_GLOBAL_MEMORY_KIB = 1024
ALPHA_GLOBAL_DEPTH = 1024


def _mode(bits, lanes):
    dtype = DataType(f"int{bits}", bits)
    operands = (
        OperandCapability("input", dtype, AccessMode.READ, ("input_buffer",)),
        OperandCapability("weight", dtype, AccessMode.READ, ("weight_buffer",)),
        OperandCapability(
            "accumulator",
            DataType("int32", 32),
            AccessMode.READ_WRITE,
            ("output_collector",),
        ),
    )
    return ComputeMode(
        f"w{bits}a{bits}",
        (OperationCapability("gemm", operands),),
        (
            SpatialUnrolling("IC", lanes),
            SpatialUnrolling("OC", ALPHA_OUTPUT_LANES),
        ),
    )


@dataclass(frozen=True, init=False)
class MagnetAlphaHardwareConfig(AcceleratorConfig):
    def __init__(self, *, frequency=None):
        matrix = ComputationUnit(
            "matrix",
            modes=(_mode(4, 32), _mode(8, 16)),
            spatial_unrolling=(
                SpatialUnrolling("IC", 32),
                SpatialUnrolling("OC", 32),
            ),
        )
        nmp = ComputationUnit(
            "nmp",
            (DataType("int8", 8),),
            frozenset(
                (
                    "depthwise_conv2d",
                    "channel_shuffle",
                    "copy",
                    "pool2d",
                    "upsample_nearest2d",
                )
            ),
            (SpatialUnrolling("CHANNEL", 32),),
        )
        shared = frozenset(StorageTarget)

        def memory(name, size, roles, banks=1, word=16, alignment=1):
            return MemoryInstance(
                name,
                MemorySize(size, CapacityUnit.BYTES),
                frozenset(roles),
                banks=banks,
                word_bytes=word,
                allocation_alignment_bytes=alignment,
            )

        hierarchy = MemoryHierarchy(
            (
                MemoryLevel(
                    "local",
                    (
                        memory(
                            "input_buffer", 4096, (StorageTarget.ACTIVATION,)
                        ),
                        memory(
                            "output_collector",
                            4096,
                            (StorageTarget.PSUM,),
                            word=128,
                        ),
                        memory(
                            "weight_buffer",
                            32768,
                            (StorageTarget.WEIGHT,),
                            banks=32,
                        ),
                    ),
                ),
                MemoryLevel(
                    "global",
                    (
                        memory(
                            "global_activation",
                            1 << 20,
                            shared,
                            banks=64,
                            alignment=64,
                        ),
                        memory(
                            "global_weight",
                            1 << 20,
                            (StorageTarget.WEIGHT,),
                            banks=16,
                            word=64,
                            alignment=64,
                        ),
                    ),
                ),
                MemoryLevel(
                    "off_chip",
                    (memory("dram", None, shared, banks=None, word=None),),
                ),
            )
        )

        def link(name, source, destination, width):
            return Connection(
                name,
                source,
                destination,
                Bandwidth(width, BandwidthUnit.BYTES_PER_CYCLE),
                bidirectional=False,
                latency_ns=None,
            )

        links = (
            link("input_to_matrix", "input_buffer", "matrix", None),
            link("weight_to_matrix", "weight_buffer", "matrix", None),
            link("collector_to_matrix", "output_collector", "matrix", None),
            link("matrix_to_collector", "matrix", "output_collector", None),
            link(
                "activation_to_pe_local",
                "global_activation",
                "input_buffer",
                16,
            ),
            link("weight_to_pe_local", "global_weight", "weight_buffer", 64),
            link(
                "pe_local_to_activation",
                "output_collector",
                "global_activation",
                32,
            ),
            link("activation_to_nmp", "global_activation", "nmp", 32),
            link("nmp_to_activation", "nmp", "global_activation", 32),
            link("dram_activation", "dram", "global_activation", None),
            link("dram_weight", "dram", "global_weight", None),
        )
        super().__init__(
            name="magnet-alpha",
            target_name="magnet-alpha",
            frequency=frequency,
            computation_units=(matrix, nmp),
            memory=hierarchy,
            connections=links,
            execution=ExecutionOptions(1, False),
            provenance=(
                ParameterProvenance(
                    "memory",
                    "documented",
                    "nv-integration:497bfbd:magnet_alpha/hardware.py",
                    "Ported physical capacities and bank geometry; no measured clock supplied.",
                ),
            ),
        )

    def memory_instance(self, name):
        aliases = {
            "scratchpad": "global_activation",
            "accum_buffer": "output_collector",
        }
        return super().memory_instance(aliases.get(name, name))

    def connection(self, name):
        return super().connection(
            "dram_activation" if name == "dram_sram" else name
        )

    @property
    def target(self):
        return self.target_name

    @property
    def vector_lanes(self):
        return self.nmp_lanes()

    @property
    def vector_unit_width(self):
        return self.nmp_lanes()

    @property
    def accumulator_lanes(self):
        return self.nmp_lanes()

    def nmp_lanes(self, input_bits=8):
        if input_bits != 8:
            raise ValueError("Alpha NMP requires 8-bit operands")
        return self.compute_unit("nmp").unrolling("CHANNEL").factor

    def input_lanes(self, operand_bits=4):
        return next(
            d.factor
            for d in self._mode(operand_bits).spatial_unrolling
            if d.dimension == "IC"
        )

    def output_lanes(self, operand_bits=4):
        return next(
            d.factor
            for d in self._mode(operand_bits).spatial_unrolling
            if d.dimension == "OC"
        )

    def _mode(self, bits):
        for mode in self.compute_unit("matrix").modes:
            if mode.name == f"w{bits}a{bits}":
                return mode
        raise ValueError(
            f"Alpha matrix unit does not support {bits}-bit operands"
        )

    def dtype_mode(self, operand_bits):
        self._mode(operand_bits)
        return SimpleNamespace(
            operand_bits=operand_bits,
            banks_per_value=operand_bits // 4,
            output_channel_alignment=self.output_lanes(operand_bits)
            * 4
            // operand_bits,
        )

    def matrix_input_resident_channels(self, channels, bits):
        self._mode(bits)
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("matrix input channels must be positive")
        return 8 if bits == 8 and channels == 3 else channels

    pad_depthwise_channels = False

    def supports_partial_matrix_input(self, channels):
        return channels in (1, 3)

    def matrix_output_resident_channels(self, channels, bits):
        alignment = self.dtype_mode(bits).output_channel_alignment
        aligned = ((channels + alignment - 1) // alignment) * alignment
        lanes = self.output_lanes(bits)
        return (
            aligned
            if aligned < lanes
            else ((aligned + lanes - 1) // lanes) * lanes
        )

    def matrix_weight_physical_channels(self, channels, bits):
        channels = self.matrix_input_resident_channels(channels, bits)
        return ((channels + 3) // 4) * 4 if bits == 8 else channels

    def matrix_input_q_alignment(self, channels, bits):
        resident, lanes = self.matrix_input_resident_channels(
            channels, bits
        ), self.input_lanes(bits)
        if resident >= lanes and resident % lanes == 0:
            return 1
        if resident < lanes and lanes % resident == 0:
            return lanes // resident
        raise ValueError(
            "resident channels must divide, or be a multiple of, the input lanes"
        )


def create_alpha_hardware_config():
    return MagnetAlphaHardwareConfig()


ALPHA_HARDWARE_CONFIG = create_alpha_hardware_config()
AOV_COMPAT_HARDWARE_CONFIG = ALPHA_HARDWARE_CONFIG
