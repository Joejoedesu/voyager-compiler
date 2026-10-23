# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha hardware profile owned by the Magnet Alpha target backend."""

from dataclasses import dataclass, replace

from voyager_compiler.codegen.hardware import (
    COMMON_IMAGE_INPUT_CHANNELS,
    CollectorConfig,
    ComputeUnit,
    DTypeMode,
    HardwareConfig,
    HierarchyBandwidth,
    MemoryInstance,
    MemoryLevelConfig,
    TensorRole,
    UnrollDimension,
)
from voyager_compiler.codegen.targets import MAGNET_ALPHA_COMPILATION_TARGET


@dataclass(frozen=True)
class MagnetAlphaHardwareConfig(HardwareConfig):
    """Typed hardware configuration for the Magnet Alpha compilation target."""

    target: str = MAGNET_ALPHA_COMPILATION_TARGET
    nmp_compute_unit: str = "nmp"

    @classmethod
    def _legacy_compute_units(
        cls,
        matrix: ComputeUnit,
    ) -> tuple[ComputeUnit, ...]:
        """Extend a legacy matrix profile with one matching NMP unit."""
        nmp = replace(
            ALPHA_NMP,
            unroll_dimensions=(
                UnrollDimension("CHANNEL", matrix.output_lanes),
            ),
        )
        return matrix, nmp

    def __post_init__(self):
        super().__post_init__()
        if self.target != MAGNET_ALPHA_COMPILATION_TARGET:
            raise ValueError(
                "MagnetAlphaHardwareConfig requires target='magnet-alpha'"
            )
        try:
            unit = self.compute_unit(self.nmp_compute_unit)
        except ValueError as error:
            raise ValueError(
                "MagnetAlphaHardwareConfig requires a configured NMP compute unit"
            ) from error
        unit.unroll_dimension("CHANNEL")

    def nmp_lanes(self, input_bits: int = 8) -> int:
        """Return NMP channel parallelism from its compute resource."""
        unit = self.compute_unit(self.nmp_compute_unit)
        return unit.unroll_extent("CHANNEL", input_bits)

    def compute_unit_bindings(self) -> dict[str, str]:
        """Add the Magnet Alpha non-matrix execution role."""
        return {
            **super().compute_unit_bindings(),
            "nmp": self.nmp_compute_unit,
        }

    def dtype_mode(self, operand_bits: int) -> DTypeMode:
        for mode in self.dtype_modes:
            if mode.operand_bits == operand_bits:
                return mode
        if not self.dtype_modes:
            return super().dtype_mode(operand_bits)
        raise ValueError(
            f"Magnet Alpha hardware does not support {operand_bits}-bit operands"
        )

    def matrix_input_resident_channels(
        self,
        valid_channels: int,
        operand_bits: int,
    ) -> int:
        """Return Alpha's compact resident input-channel representation."""
        if isinstance(valid_channels, bool):
            raise ValueError("matrix input channels must be positive")
        normalized_channels = int(valid_channels)
        if normalized_channels <= 0 or normalized_channels != valid_channels:
            raise ValueError("matrix input channels must be positive")
        if isinstance(operand_bits, bool):
            raise ValueError("matrix input operand bits must be positive")
        normalized_bits = int(operand_bits)
        if normalized_bits <= 0 or normalized_bits != operand_bits:
            raise ValueError("matrix input operand bits must be positive")
        self.dtype_mode(normalized_bits)
        if normalized_bits == 8 and normalized_channels == 3:
            return 8
        return normalized_channels

    def matrix_weight_physical_channels(
        self,
        resident_channels: int,
        operand_bits: int,
    ) -> int:
        """Return Alpha's physical matrix-weight input-channel stride."""
        physical_channels = self.matrix_input_resident_channels(
            resident_channels,
            operand_bits,
        )
        if physical_channels != resident_channels:
            return physical_channels
        if operand_bits == 8:
            return ((resident_channels + 3) // 4) * 4
        return resident_channels

    def matrix_input_q_alignment(
        self,
        valid_channels: int,
        operand_bits: int,
    ) -> int:
        """Return the Q traversal required by one compact Alpha GB line."""
        resident = self.matrix_input_resident_channels(
            valid_channels,
            operand_bits,
        )
        lanes = self.input_lanes(operand_bits)
        if resident >= lanes:
            if resident % lanes:
                raise ValueError(
                    "Alpha matrix input resident channels must be lane "
                    f"aligned: resident={resident}, lanes={lanes}"
                )
            return 1
        if lanes % resident:
            raise ValueError(
                "Alpha matrix input resident channels must divide its "
                f"physical lanes: resident={resident}, lanes={lanes}"
            )
        return lanes // resident


ALPHA_INPUT_WORD_BITS = 128
ALPHA_OUTPUT_LANES = 32
ALPHA_VSQ_BLOCK_VALUES = 32
ALPHA_COLLECTOR_DEPTH = 32
ALPHA_ACCUMULATOR_BITS = 32
ALPHA_ACTIVATION_READ_BITS_PER_CYCLE = 16 * 8
ALPHA_ACTIVATION_WRITE_BITS_PER_CYCLE = 32 * 8
ALPHA_WEIGHT_READ_BITS_PER_CYCLE = 64 * 8
ALPHA_NMP_READ_BITS_PER_CYCLE = ALPHA_OUTPUT_LANES * 8
ALPHA_NMP_WRITE_BITS_PER_CYCLE = ALPHA_OUTPUT_LANES * 8
ALPHA_GLOBAL_MEMORY_KIB = 1024
ALPHA_GLOBAL_DEPTH = (
    ALPHA_GLOBAL_MEMORY_KIB * 1024 * 8 // (64 * ALPHA_INPUT_WORD_BITS)
)

ALPHA_VECTOR_MAC = ComputeUnit(
    name="vector_mac",
    unroll_dimensions=(
        UnrollDimension(
            "IC",
            32,
            dtype_extents=((8, 16),),
        ),
        UnrollDimension("OC", ALPHA_OUTPUT_LANES),
    ),
    accepted_input_bits=frozenset({4, 8}),
    accepted_operations=frozenset({"gemm"}),
    input_word_bits=ALPHA_INPUT_WORD_BITS,
    scalar_bits=8,
)

ALPHA_NMP = ComputeUnit(
    name="nmp",
    unroll_dimensions=(UnrollDimension("CHANNEL", ALPHA_OUTPUT_LANES),),
    accepted_input_bits=frozenset({8}),
    accepted_operations=frozenset(
        {
            "channel_shuffle",
            "channel_shuffle_copy",
            "depthwise_conv2d",
            "group_norm",
            "pool2d",
            "softmax",
            "upsample_nearest2d",
        }
    ),
)


def create_alpha_hardware_config() -> MagnetAlphaHardwareConfig:
    """Return the production Alpha hardware configuration."""
    return MagnetAlphaHardwareConfig(
        pe=ALPHA_VECTOR_MAC,
        compute_units=(ALPHA_VECTOR_MAC, ALPHA_NMP),
        hierarchy_bandwidths=(
            HierarchyBandwidth(
                name="activation_to_pe_local",
                source="global",
                destination="pe_local",
                bandwidth_bits_per_cycle=(ALPHA_ACTIVATION_READ_BITS_PER_CYCLE),
                roles=frozenset({TensorRole.ACTIVATION}),
            ),
            HierarchyBandwidth(
                name="weight_to_pe_local",
                source="global",
                destination="pe_local",
                bandwidth_bits_per_cycle=(ALPHA_WEIGHT_READ_BITS_PER_CYCLE),
                roles=frozenset({TensorRole.WEIGHT}),
            ),
            HierarchyBandwidth(
                name="pe_local_to_activation",
                source="pe_local",
                destination="global",
                bandwidth_bits_per_cycle=(
                    ALPHA_ACTIVATION_WRITE_BITS_PER_CYCLE
                ),
                roles=frozenset({TensorRole.OUTPUT}),
            ),
            HierarchyBandwidth(
                name="activation_to_nmp",
                source="global",
                destination=ALPHA_NMP.name,
                bandwidth_bits_per_cycle=(ALPHA_NMP_READ_BITS_PER_CYCLE),
                roles=frozenset({TensorRole.ACTIVATION}),
            ),
            HierarchyBandwidth(
                name="weight_to_nmp",
                source="global",
                destination=ALPHA_NMP.name,
                bandwidth_bits_per_cycle=(ALPHA_NMP_READ_BITS_PER_CYCLE),
                roles=frozenset({TensorRole.WEIGHT}),
            ),
            HierarchyBandwidth(
                name="nmp_to_activation",
                source=ALPHA_NMP.name,
                destination="global",
                bandwidth_bits_per_cycle=(ALPHA_NMP_WRITE_BITS_PER_CYCLE),
                roles=frozenset({TensorRole.OUTPUT}),
            ),
        ),
        dtype_modes=(
            DTypeMode(
                operand_bits=4,
                banks_per_value=1,
                output_channel_alignment=32,
            ),
            DTypeMode(
                operand_bits=8,
                banks_per_value=2,
                output_channel_alignment=16,
            ),
        ),
        local_memory=MemoryLevelConfig(
            name="pe_local",
            instances=(
                MemoryInstance(
                    name="input_buffer",
                    bank_count=1,
                    depth_per_bank=256,
                    bits_per_bank_entry=ALPHA_INPUT_WORD_BITS,
                    accepted_accesses=frozenset(
                        {
                            (
                                TensorRole.ACTIVATION,
                                ALPHA_VECTOR_MAC.name,
                            )
                        }
                    ),
                    access_cost=10,
                ),
                MemoryInstance(
                    name="output_collector",
                    bank_count=1,
                    depth_per_bank=ALPHA_COLLECTOR_DEPTH,
                    bits_per_bank_entry=(
                        ALPHA_OUTPUT_LANES * ALPHA_ACCUMULATOR_BITS
                    ),
                    accepted_accesses=frozenset(
                        {
                            (
                                TensorRole.OUTPUT,
                                ALPHA_VECTOR_MAC.name,
                            ),
                            (
                                TensorRole.PARTIAL_SUM,
                                ALPHA_VECTOR_MAC.name,
                            ),
                        }
                    ),
                    access_cost=10,
                ),
                MemoryInstance(
                    name="weight_buffer",
                    bank_count=ALPHA_OUTPUT_LANES,
                    depth_per_bank=64,
                    bits_per_bank_entry=ALPHA_INPUT_WORD_BITS,
                    accepted_accesses=frozenset(
                        {
                            (
                                TensorRole.WEIGHT,
                                ALPHA_VECTOR_MAC.name,
                            )
                        }
                    ),
                    access_cost=10,
                ),
            ),
            input_memory="input_buffer",
            output_memory="output_collector",
            weight_memory="weight_buffer",
        ),
        global_memory=MemoryLevelConfig(
            name="global",
            instances=(
                MemoryInstance(
                    name="global_activation",
                    bank_count=64,
                    depth_per_bank=ALPHA_GLOBAL_DEPTH,
                    bits_per_bank_entry=ALPHA_INPUT_WORD_BITS,
                    accepted_accesses=frozenset(
                        {
                            (
                                TensorRole.ACTIVATION,
                                ALPHA_VECTOR_MAC.name,
                            ),
                            (
                                TensorRole.ACTIVATION,
                                ALPHA_NMP.name,
                            ),
                            (TensorRole.BIAS, None),
                            (TensorRole.OUTPUT, None),
                            (TensorRole.SCALE, None),
                            (
                                TensorRole.WEIGHT,
                                ALPHA_VECTOR_MAC.name,
                            ),
                            (
                                TensorRole.WEIGHT,
                                ALPHA_NMP.name,
                            ),
                        }
                    ),
                    access_cost=100,
                    access_granularity_bits=512,
                    read_bandwidth=(ALPHA_ACTIVATION_READ_BITS_PER_CYCLE / 8),
                    write_bandwidth=(ALPHA_ACTIVATION_WRITE_BITS_PER_CYCLE / 8),
                ),
                MemoryInstance(
                    name="global_weight",
                    bank_count=16,
                    depth_per_bank=ALPHA_GLOBAL_DEPTH,
                    bits_per_bank_entry=(4 * ALPHA_INPUT_WORD_BITS),
                    accepted_accesses=frozenset(
                        {
                            (
                                TensorRole.WEIGHT,
                                ALPHA_VECTOR_MAC.name,
                            ),
                            (TensorRole.BIAS, None),
                        }
                    ),
                    access_cost=100,
                    access_granularity_bits=512,
                    read_bandwidth=(ALPHA_WEIGHT_READ_BITS_PER_CYCLE / 8),
                    write_bandwidth=0,
                ),
            ),
            input_memory="global_activation",
            output_memory="global_activation",
            weight_memory="global_weight",
        ),
        output_collector=CollectorConfig(
            name="output_collector",
            depth_words=ALPHA_COLLECTOR_DEPTH,
            lanes_per_word=ALPHA_OUTPUT_LANES,
            accumulator_bits=ALPHA_ACCUMULATOR_BITS,
        ),
        nmp_compute_unit=ALPHA_NMP.name,
        technology_nm=5,
        # Alpha NMP executes one full 32-channel physical row. Preserve the
        # logical channel count with the padding pass's trailing slice, but
        # make DWC storage and execution lane-complete.
        pad_depthwise_channels=True,
        partial_matrix_input_channels=(COMMON_IMAGE_INPUT_CHANNELS),
    )


ALPHA_HARDWARE_CONFIG = create_alpha_hardware_config()
# Compatibility name used by existing AoV-oriented callers.
AOV_COMPAT_HARDWARE_CONFIG = ALPHA_HARDWARE_CONFIG


__all__ = [
    "ALPHA_HARDWARE_CONFIG",
    "ALPHA_VSQ_BLOCK_VALUES",
    "AOV_COMPAT_HARDWARE_CONFIG",
    "MagnetAlphaHardwareConfig",
    "create_alpha_hardware_config",
]
