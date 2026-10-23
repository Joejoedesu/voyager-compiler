# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Typed facts for the pinned, default-derived Gemmini target."""

from dataclasses import dataclass
from types import MappingProxyType

from voyager_compiler.codegen.hardware import (
    CollectorConfig,
    ComputeUnit,
    DTypeMode,
    HardwareConfig,
    MemoryInstance,
    MemoryLevelConfig,
    TensorRole,
    UnrollDimension,
)

# Backend assembly must register this same identity in the central compilation
# target registry. Keeping the identity here makes this owned profile
# independently constructible before that assembly step.
GEMMINI_COMPILATION_TARGET = "gemmini"

GEMMINI_CONFIGURATION = "GemminiConfigs.defaultConfig"
GEMMINI_DERIVED_CONFIGURATION = GEMMINI_CONFIGURATION
GEMMINI_PERMITTED_OVERRIDES: tuple[str, ...] = ()
GEMMINI_DEFAULT_TO_DERIVED_DIFF: tuple[tuple[str, object, object], ...] = ()

GEMMINI_EXTERNAL_REVISION = "8c3f9923a44a2fe2c7930587be297d6d4f8c09ca"
GEMMINI_HELPER_REVISION = "7c540b3adf1b86ad93d07f893abe3a73489b568e"
GEMMINI_CONFIGS_SHA256 = (
    "178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b"
)
GEMMINI_CONFIG_SCHEMA_SHA256 = (
    "7294a082a1df4ae7aa7d81adc5bde75b7d5118683c083f6a4b998c9ce64d102e"
)
GEMMINI_PARAMS_SHA256 = (
    "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
)
GEMMINI_HELPER_API_SHA256 = (
    "18801f4eab0cd2e81e25a6c2aa97c7ed644b693e8d2c24cdeb41ab3a7208336d"
)

GEMMINI_DIM = 16
GEMMINI_INPUT_BITS = 8
GEMMINI_SPATIAL_OUTPUT_BITS = 20
GEMMINI_ACCUMULATOR_BITS = 32
GEMMINI_DATAFLOW = "BOTH"
GEMMINI_OPERANDS_SIGNED = True
GEMMINI_HAS_TRAINING_CONVS = True
GEMMINI_HAS_MAX_POOL = True
GEMMINI_HAS_NONLINEAR_ACTIVATIONS = True
GEMMINI_HAS_DEPTHWISE_CONVS = True
GEMMINI_HAS_LOOP_CONV = True

GEMMINI_MVIN_SCALE_AVAILABLE = True
GEMMINI_ACC_SCALE_AVAILABLE = True
GEMMINI_MVIN_SCALE_ACC_AVAILABLE = False
GEMMINI_MVIN_SCALE_SHARED = False
GEMMINI_SCALE_TYPE = "Float(8, 24)"
GEMMINI_SCALE_HELPER_GRANULARITY = "scalar"
GEMMINI_SCALE_ROUNDING = "round_near_even"
GEMMINI_SCALE_SATURATING = True
GEMMINI_SCALE_IDENTITY = "1.0"

GEMMINI_SCRATCHPAD_BANKS = 4
GEMMINI_SCRATCHPAD_KIB = 256
GEMMINI_SCRATCHPAD_SINGLE_PORTED = True
GEMMINI_SCRATCHPAD_ROW_BITS = GEMMINI_DIM * GEMMINI_INPUT_BITS
GEMMINI_SCRATCHPAD_ROWS_PER_BANK = (
    GEMMINI_SCRATCHPAD_KIB
    * 1024
    * 8
    // (GEMMINI_SCRATCHPAD_BANKS * GEMMINI_SCRATCHPAD_ROW_BITS)
)

GEMMINI_ACCUMULATOR_BANKS = 2
GEMMINI_ACCUMULATOR_KIB = 64
GEMMINI_ACCUMULATOR_SINGLE_PORTED = False
GEMMINI_ACCUMULATOR_ROW_BITS = GEMMINI_DIM * GEMMINI_ACCUMULATOR_BITS
GEMMINI_ACCUMULATOR_ROWS = (
    GEMMINI_ACCUMULATOR_KIB * 1024 * 8 // GEMMINI_ACCUMULATOR_ROW_BITS
)
GEMMINI_ACCUMULATOR_ROWS_PER_BANK = (
    GEMMINI_ACCUMULATOR_ROWS // GEMMINI_ACCUMULATOR_BANKS
)

GEMMINI_DMA_MAX_BYTES = 64
GEMMINI_DMA_BUS_BITS = 128
GEMMINI_ELEM_ALIGNMENT_BYTES = GEMMINI_SCRATCHPAD_ROW_BITS // 8
GEMMINI_ACC_ALIGNMENT_BYTES = GEMMINI_ACCUMULATOR_ROW_BITS // 8

# The shared schema requires a numeric access cost. Zero is an explicit
# unmodeled sentinel in this structural profile, not an energy measurement.
GEMMINI_UNMODELED_ACCESS_COST = 0.0

# These classify native entrypoint families present in the pinned helper. They
# do not by themselves establish semantic support or operation execution.
GEMMINI_NATIVE_OPERATION_FAMILIES = frozenset(
    {
        "add",
        "conv2d",
        "gemm",
    }
)

GEMMINI_FACT_SOURCES = MappingProxyType(
    {
        "configuration": GEMMINI_CONFIGURATION,
        "derived_configuration": GEMMINI_DERIVED_CONFIGURATION,
        "permitted_overrides": GEMMINI_PERMITTED_OVERRIDES,
        "default_to_derived_diff": GEMMINI_DEFAULT_TO_DERIVED_DIFF,
        "target_revision": GEMMINI_EXTERNAL_REVISION,
        "default_config_path": "src/main/scala/gemmini/Configs.scala",
        "default_config_sha256": GEMMINI_CONFIGS_SHA256,
        "inherited_defaults_path": (
            "src/main/scala/gemmini/GemminiConfigs.scala"
        ),
        "inherited_defaults_sha256": GEMMINI_CONFIG_SCHEMA_SHA256,
        "generated_header_path": (
            "software/gemmini-rocc-tests/include/gemmini_params.h"
        ),
        "generated_header_revision": GEMMINI_HELPER_REVISION,
        "generated_header_sha256": GEMMINI_PARAMS_SHA256,
        "generated_header_status": "pinned-pre-generated-collateral",
        "native_helper_path": ("software/gemmini-rocc-tests/include/gemmini.h"),
        "native_helper_sha256": GEMMINI_HELPER_API_SHA256,
    }
)

# These are elaboration and native-API availability facts, not semantic
# support claims for any Voyager operator or quantization scheme.
GEMMINI_DEFAULT_FACTS = MappingProxyType(
    {
        "input_bits": GEMMINI_INPUT_BITS,
        "weight_bits": GEMMINI_INPUT_BITS,
        "accumulator_bits": GEMMINI_ACCUMULATOR_BITS,
        "spatial_output_bits": GEMMINI_SPATIAL_OUTPUT_BITS,
        "operands_signed": GEMMINI_OPERANDS_SIGNED,
        "tile_rows": 1,
        "tile_columns": 1,
        "mesh_rows": GEMMINI_DIM,
        "mesh_columns": GEMMINI_DIM,
        "dataflow": GEMMINI_DATAFLOW,
        "has_training_convs": GEMMINI_HAS_TRAINING_CONVS,
        "has_max_pool": GEMMINI_HAS_MAX_POOL,
        "has_nonlinear_activations": GEMMINI_HAS_NONLINEAR_ACTIVATIONS,
        "has_depthwise_convs": GEMMINI_HAS_DEPTHWISE_CONVS,
        "has_loop_conv": GEMMINI_HAS_LOOP_CONV,
        "scratchpad_kib": GEMMINI_SCRATCHPAD_KIB,
        "scratchpad_banks": GEMMINI_SCRATCHPAD_BANKS,
        "scratchpad_single_ported": GEMMINI_SCRATCHPAD_SINGLE_PORTED,
        "accumulator_kib": GEMMINI_ACCUMULATOR_KIB,
        "accumulator_banks": GEMMINI_ACCUMULATOR_BANKS,
        "accumulator_single_ported": GEMMINI_ACCUMULATOR_SINGLE_PORTED,
        "dma_max_bytes": GEMMINI_DMA_MAX_BYTES,
        "dma_bus_bits": GEMMINI_DMA_BUS_BITS,
        "mvin_scale_available": GEMMINI_MVIN_SCALE_AVAILABLE,
        "acc_scale_available": GEMMINI_ACC_SCALE_AVAILABLE,
        "mvin_scale_acc_available": GEMMINI_MVIN_SCALE_ACC_AVAILABLE,
        "mvin_scale_shared": GEMMINI_MVIN_SCALE_SHARED,
        "scale_type": GEMMINI_SCALE_TYPE,
        "scale_helper_granularity": GEMMINI_SCALE_HELPER_GRANULARITY,
        "scale_rounding": GEMMINI_SCALE_ROUNDING,
        "scale_saturating": GEMMINI_SCALE_SATURATING,
        "scale_identity": GEMMINI_SCALE_IDENTITY,
    }
)

GEMMINI_SYSTOLIC_ARRAY = ComputeUnit(
    name="systolic_array",
    unroll_dimensions=(
        UnrollDimension("IC", GEMMINI_DIM),
        UnrollDimension("OC", GEMMINI_DIM),
    ),
    accepted_input_bits=frozenset({GEMMINI_INPUT_BITS}),
    accepted_operations=GEMMINI_NATIVE_OPERATION_FAMILIES,
    input_word_bits=GEMMINI_SCRATCHPAD_ROW_BITS,
    scalar_bits=GEMMINI_INPUT_BITS,
)

GEMMINI_INT8_MODE = DTypeMode(
    operand_bits=GEMMINI_INPUT_BITS,
    banks_per_value=1,
    output_channel_alignment=GEMMINI_DIM,
)

GEMMINI_OUTPUT_COLLECTOR = CollectorConfig(
    name="accumulator",
    depth_words=GEMMINI_ACCUMULATOR_ROWS,
    lanes_per_word=GEMMINI_DIM,
    accumulator_bits=GEMMINI_ACCUMULATOR_BITS,
)

GEMMINI_ARRAY_LOCAL_MEMORY = MemoryLevelConfig(
    name="array_local",
    instances=(
        MemoryInstance(
            name="activation_row",
            bank_count=1,
            depth_per_bank=1,
            bits_per_bank_entry=GEMMINI_SCRATCHPAD_ROW_BITS,
            accepted_accesses=frozenset(
                {(TensorRole.ACTIVATION, GEMMINI_SYSTOLIC_ARRAY.name)}
            ),
            access_cost=GEMMINI_UNMODELED_ACCESS_COST,
            alignment_bytes=GEMMINI_ELEM_ALIGNMENT_BYTES,
        ),
        MemoryInstance(
            name="weight_row",
            bank_count=1,
            depth_per_bank=1,
            bits_per_bank_entry=GEMMINI_SCRATCHPAD_ROW_BITS,
            accepted_accesses=frozenset(
                {(TensorRole.WEIGHT, GEMMINI_SYSTOLIC_ARRAY.name)}
            ),
            access_cost=GEMMINI_UNMODELED_ACCESS_COST,
            alignment_bytes=GEMMINI_ELEM_ALIGNMENT_BYTES,
        ),
        MemoryInstance(
            name="accumulator_row",
            bank_count=1,
            depth_per_bank=1,
            bits_per_bank_entry=GEMMINI_ACCUMULATOR_ROW_BITS,
            accepted_accesses=frozenset(
                {
                    (TensorRole.OUTPUT, GEMMINI_SYSTOLIC_ARRAY.name),
                    (TensorRole.PARTIAL_SUM, GEMMINI_SYSTOLIC_ARRAY.name),
                }
            ),
            access_cost=GEMMINI_UNMODELED_ACCESS_COST,
            alignment_bytes=GEMMINI_ACC_ALIGNMENT_BYTES,
        ),
    ),
    input_memory="activation_row",
    output_memory="accumulator_row",
    weight_memory="weight_row",
)

GEMMINI_ONCHIP_MEMORY = MemoryLevelConfig(
    name="onchip",
    instances=(
        MemoryInstance(
            name="scratchpad",
            bank_count=GEMMINI_SCRATCHPAD_BANKS,
            depth_per_bank=GEMMINI_SCRATCHPAD_ROWS_PER_BANK,
            bits_per_bank_entry=GEMMINI_SCRATCHPAD_ROW_BITS,
            accepted_accesses=frozenset(
                {
                    (TensorRole.ACTIVATION, GEMMINI_SYSTOLIC_ARRAY.name),
                    (TensorRole.WEIGHT, GEMMINI_SYSTOLIC_ARRAY.name),
                }
            ),
            access_cost=GEMMINI_UNMODELED_ACCESS_COST,
            alignment_bytes=GEMMINI_ELEM_ALIGNMENT_BYTES,
            access_granularity_bits=GEMMINI_DMA_BUS_BITS,
            # Native helpers may partition this storage dynamically. That
            # endpoint choice is not a hardware-fixed ping-pong capacity.
            double_buffered=False,
        ),
        MemoryInstance(
            name="accumulator",
            bank_count=GEMMINI_ACCUMULATOR_BANKS,
            depth_per_bank=GEMMINI_ACCUMULATOR_ROWS_PER_BANK,
            bits_per_bank_entry=GEMMINI_ACCUMULATOR_ROW_BITS,
            accepted_accesses=frozenset(
                {
                    (TensorRole.OUTPUT, GEMMINI_SYSTOLIC_ARRAY.name),
                    (TensorRole.PARTIAL_SUM, GEMMINI_SYSTOLIC_ARRAY.name),
                }
            ),
            access_cost=GEMMINI_UNMODELED_ACCESS_COST,
            alignment_bytes=GEMMINI_ACC_ALIGNMENT_BYTES,
            access_granularity_bits=GEMMINI_DMA_BUS_BITS,
            # See the scratchpad note above; helper-owned tiling is not a
            # Voyager-owned or hardware-fixed double-buffering contract.
            double_buffered=False,
        ),
    ),
    input_memory="scratchpad",
    output_memory="accumulator",
    weight_memory="scratchpad",
)


@dataclass(frozen=True)
class GemminiHardwareConfig(HardwareConfig):
    """Exact typed profile for the pinned signed-INT8 default target."""

    target: str = GEMMINI_COMPILATION_TARGET

    def __post_init__(self):
        super().__post_init__()
        if self.target != GEMMINI_COMPILATION_TARGET:
            raise ValueError("GemminiHardwareConfig requires target='gemmini'")
        if self.pe != GEMMINI_SYSTOLIC_ARRAY:
            raise ValueError("Gemmini requires its default 16-by-16 array")
        if self.compute_units != (GEMMINI_SYSTOLIC_ARRAY,):
            raise ValueError("Gemmini requires its default compute unit")
        if self.dtype_modes != (GEMMINI_INT8_MODE,):
            raise ValueError("Gemmini accepts only its signed INT8 mode")
        if self.local_memory != GEMMINI_ARRAY_LOCAL_MEMORY:
            raise ValueError("Gemmini requires its default array interface")
        if self.global_memory != GEMMINI_ONCHIP_MEMORY:
            raise ValueError("Gemmini requires its default on-chip memories")
        if self.output_collector != GEMMINI_OUTPUT_COLLECTOR:
            raise ValueError("Gemmini requires its default accumulator")
        if self.external_memory is not None:
            raise ValueError("Gemmini external-memory capacity is unspecified")
        if self.hierarchy_bandwidths:
            raise ValueError("Gemmini bandwidth observations are unspecified")
        if self.mapping_num_banks is not None:
            raise ValueError("Gemmini has no Voyager-owned mapping bank count")
        if self.mapping_bank_width is not None:
            raise ValueError("Gemmini has no Voyager-owned mapping bank width")
        if self.technology_nm is not None:
            raise ValueError("Gemmini technology is unspecified")
        if self.pad_depthwise_channels:
            raise ValueError("Gemmini has no profile-owned depthwise padding")
        if self.partial_matrix_input_channels:
            raise ValueError("Gemmini has no profile-owned partial input lanes")

    @property
    def configuration(self) -> str:
        return GEMMINI_CONFIGURATION

    @property
    def external_revision(self) -> str:
        return GEMMINI_EXTERNAL_REVISION

    @property
    def dataflow(self) -> str:
        return GEMMINI_DATAFLOW

    @property
    def operands_signed(self) -> bool:
        return GEMMINI_OPERANDS_SIGNED

    @property
    def spatial_output_bits(self) -> int:
        return GEMMINI_SPATIAL_OUTPUT_BITS

    @property
    def dma_max_bytes(self) -> int:
        return GEMMINI_DMA_MAX_BYTES

    @property
    def dma_bus_bits(self) -> int:
        return GEMMINI_DMA_BUS_BITS

    def dtype_mode(self, operand_bits: int) -> DTypeMode:
        if operand_bits != GEMMINI_INPUT_BITS:
            raise ValueError(
                f"Gemmini does not support {operand_bits}-bit operands"
            )
        return super().dtype_mode(operand_bits)


def create_gemmini_hardware_config() -> GemminiHardwareConfig:
    """Construct the deterministic default-derived Gemmini profile."""
    return GemminiHardwareConfig(
        pe=GEMMINI_SYSTOLIC_ARRAY,
        compute_units=(GEMMINI_SYSTOLIC_ARRAY,),
        dtype_modes=(GEMMINI_INT8_MODE,),
        local_memory=GEMMINI_ARRAY_LOCAL_MEMORY,
        global_memory=GEMMINI_ONCHIP_MEMORY,
        output_collector=GEMMINI_OUTPUT_COLLECTOR,
        pad_depthwise_channels=False,
    )


GEMMINI_HARDWARE_CONFIG = create_gemmini_hardware_config()

__all__ = [
    "GEMMINI_ACC_ALIGNMENT_BYTES",
    "GEMMINI_ACCUMULATOR_BITS",
    "GEMMINI_ACCUMULATOR_KIB",
    "GEMMINI_COMPILATION_TARGET",
    "GEMMINI_CONFIGURATION",
    "GEMMINI_DEFAULT_FACTS",
    "GEMMINI_DEFAULT_TO_DERIVED_DIFF",
    "GEMMINI_DERIVED_CONFIGURATION",
    "GEMMINI_DIM",
    "GEMMINI_DMA_BUS_BITS",
    "GEMMINI_DMA_MAX_BYTES",
    "GEMMINI_ELEM_ALIGNMENT_BYTES",
    "GEMMINI_EXTERNAL_REVISION",
    "GEMMINI_FACT_SOURCES",
    "GEMMINI_HARDWARE_CONFIG",
    "GEMMINI_HAS_DEPTHWISE_CONVS",
    "GEMMINI_HAS_LOOP_CONV",
    "GEMMINI_HAS_MAX_POOL",
    "GEMMINI_HAS_NONLINEAR_ACTIVATIONS",
    "GEMMINI_HAS_TRAINING_CONVS",
    "GEMMINI_INPUT_BITS",
    "GEMMINI_NATIVE_OPERATION_FAMILIES",
    "GEMMINI_PERMITTED_OVERRIDES",
    "GEMMINI_SCRATCHPAD_KIB",
    "GEMMINI_SYSTOLIC_ARRAY",
    "GemminiHardwareConfig",
    "create_gemmini_hardware_config",
]
