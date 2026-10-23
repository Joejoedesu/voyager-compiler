# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha global-memory placement and address encoding."""

import copy
import logging
import math
import operator
import os
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.execution_order import execution_positions
from voyager_compiler.codegen.hardware import (
    TensorRole,
    get_dtype_width,
    get_node_dtype_width,
)
from voyager_compiler.codegen.mapping_utils import (
    TARGET_CODEGEN_KWARGS,
    TARGET_INSTRUCTION_NOP,
    declare_unsupported_operation,
    is_addressing_op,
    is_conv2d,
    is_declared_unsupported_operation,
    is_nop,
)
from voyager_compiler.codegen.memory import (
    AllocationRecord,
    MemoryAllocator,
    MemorySpace,
    Segment,
    allocation_record_label,
    compatible_memory_names,
    index_allocation_records,
    is_tensor_split,
    live_allocation_records,
    logical_tensor_segment,
    payload_size_bytes,
    physical_access_requirements,
    validate_allocation_lifecycles,
    validate_execution_liveness,
)
from voyager_compiler.codegen.operands import (
    classify_storage_role,
    operand_descriptors,
    require_allocation,
    submodule_operand_descriptors,
)
from voyager_compiler.codegen.passes.utils import (
    get_hardware_logical_shape,
    get_leading_channel_slice_source,
    get_named_arg,
)
from voyager_compiler.codegen.storage import (
    PhysicalPayloadRef,
    PhysicalTensorId,
    StorageGraph,
    StoragePlan,
    StorageRelationKind,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.storage_predicates import is_address_only_view
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    ALPHA_SHUFFLE_COPY_KIND,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    is_shuffle_copy as _is_shuffle_copy,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    magnet_alpha_meta as _magnet_alpha_meta,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    shuffle_copy_movements as _shuffle_copy_movements,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering import (
    bound_operation_classification,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    AlphaMemoryPlanError,
    AlphaPreparedMemoryPlan,
    freeze_prepared_memory_plan,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    ConcatLayout as _ConcatLayout,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    ConcatPlan as _ConcatPlan,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    MaterializedInputPadding as _MaterializedInputPadding,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    NmpInputView as _NmpInputView,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    OperationConstraint as _OperationConstraint,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    OperationEgressBoundary as _OperationEgressBoundary,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    PhysicalPayloadUse as _PhysicalPayloadUse,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    ResidentWrite as _ResidentWrite,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory_plan import (
    ShufflePlan as _ShufflePlan,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    CHANNEL_SHUFFLE_OPERATOR,
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.traits import (
    MAGNET_ALPHA_OPERATOR_TRAITS,
)
from voyager_compiler.quantization_contracts import (
    PhysicalQuantizationContract,
    get_physical_dtype,
    get_physical_quantization_contract,
    set_physical_quantization_contract,
)

logger = logging.getLogger(__name__)


ALPHA_GB_SCRATCH_BASE = 0x1000000
ALPHA_GB_WEIGHT_BASE = 0x800000
ALPHA_RESERVED_GB_SCRATCH_BANKS = frozenset({0, 1})
ALPHA_ENTRY_BITS = 10
ALPHA_BANK_BITS = 6
ALPHA_PRECISION_SHIFT = 19
ALPHA_CLUSTER_SHIFT = 16
ALPHA_DENSE_LAYOUT = "alpha_dense"
ALPHA_PADDED_INPUT_LAYOUT = "alpha_padded_input"
ALPHA_SHUFFLE_SECONDARY_LAYOUT = "alpha_shuffle_secondary"
ALPHA_VSQ_LAYOUT = "alpha_vsq_embedded"
PHYSICAL_SCRATCHPAD_MAP = "physical_scratchpad_map"


def _mark_alpha_address_only(node: Node) -> None:
    """Suppress emission for an Alpha address-only graph operation."""
    node.meta[TARGET_INSTRUCTION_NOP] = True


def _is_alpha_vsq_storage_pair(
    fine_scale,
    payload,
    fine_dtype=None,
    payload_dtype=None,
) -> bool:
    """Return whether two tensors have Alpha's packed VSQ storage layout."""
    if not all(
        isinstance(value, torch.Tensor) for value in (fine_scale, payload)
    ):
        return False
    try:
        fine_bits = get_dtype_width(fine_dtype or fine_scale.dtype)
        payload_bits = get_dtype_width(payload_dtype or payload.dtype)
    except (TypeError, ValueError):
        return False
    return (
        fine_bits == 8
        and payload_bits == 4
        and payload.numel() == fine_scale.numel() * ALPHA_VSQ_BLOCK_VALUES
    )


@dataclass(frozen=True)
class AlphaAddress:
    """Target-owned fields represented by one Alpha starting address."""

    base: int
    operand_bits: int
    cluster: int
    virtual_bank: int
    entry: int

    def __post_init__(self):
        if self.operand_bits not in (4, 8):
            raise ValueError("Alpha addresses support only 4- or 8-bit modes")
        if self.cluster not in (1, 2, 4):
            raise ValueError(
                "Alpha bank clusters must contain 1, 2, or 4 banks"
            )
        if not 0 <= self.virtual_bank < (1 << ALPHA_BANK_BITS):
            raise ValueError(
                "Alpha virtual bank does not fit its address field"
            )
        if not 0 <= self.entry < (1 << ALPHA_ENTRY_BITS):
            raise ValueError("Alpha entry does not fit its address field")

    def encode(self) -> int:
        operand_bit = int(self.operand_bits == 8)
        return (
            self.base
            | self.entry
            | (self.virtual_bank << ALPHA_ENTRY_BITS)
            | (operand_bit << ALPHA_PRECISION_SHIFT)
            | (self.cluster << ALPHA_CLUSTER_SHIFT)
        )

    @classmethod
    def decode(cls, address: int):
        lower_mask = (1 << 20) - 1
        return cls(
            base=address & ~lower_mask,
            operand_bits=(8 if (address >> ALPHA_PRECISION_SHIFT) & 1 else 4),
            cluster=(address >> ALPHA_CLUSTER_SHIFT) & 0b111,
            virtual_bank=(address >> ALPHA_ENTRY_BITS) & 0b111111,
            entry=address & 0b1111111111,
        )


@dataclass(frozen=True)
class AlphaAllocationRecord(AllocationRecord):
    """Alpha allocation history including target physical-bank cells."""

    cells: tuple[tuple[int, int], ...]


@dataclass
class _Allocation:
    allocation_id: int
    physical_tensor: PhysicalTensorId
    role: TensorRole
    memory_name: str
    cells: tuple[tuple[int, int], ...]
    banks: frozenset[int]
    segment: Segment
    primary: bool


@dataclass(frozen=True)
class _ResidentPlacement:
    physical_tensor: PhysicalTensorId
    slot_index: int
    channel_offset: int
    slot_channels: int
    resident_channels: int
    concat_channel_index: int | None


@dataclass(frozen=True)
class _ShuffleAccessPattern:
    positions: int
    active_entries: int
    pitch_entries: int

    @property
    def span_entries(self):
        return (self.positions - 1) * self.pitch_entries + self.active_entries

    def offsets(self):
        for position in range(self.positions):
            base = position * self.pitch_entries
            for entry in range(self.active_entries):
                yield base + entry


@dataclass(frozen=True)
class _ShuffleStorage:
    primary_base: Segment
    primary: Segment
    secondary: Segment

    @property
    def outputs(self):
        return self.primary, self.secondary

    @property
    def output_addresses(self):
        return self.primary_base.address, self.secondary.address


def _normalize_operand_bits(width: int) -> int:
    return 4 if width <= 4 else 8


def _node_operand_bits(node) -> int:
    return _normalize_operand_bits(
        get_node_dtype_width(node, get_physical_dtype(node))
    )


def _pad_tensor_axis(value, axis, extent):
    """Return ``value`` zero-padded to ``extent`` along one axis."""
    if extent < value.shape[axis]:
        raise ValueError("physical tensor extent cannot shrink")
    if extent == value.shape[axis]:
        return value
    shape = list(value.shape)
    shape[axis] = extent
    result = value.new_zeros(shape)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, value.shape[axis])
    result[tuple(slices)] = value
    return result


class AlphaScratchpadAllocator:
    """Allocate live tensors over Alpha physical banks and entries."""

    def __init__(self, hardware: MagnetAlphaHardwareConfig):
        if not isinstance(hardware, MagnetAlphaHardwareConfig):
            raise TypeError(
                "AlphaScratchpadAllocator requires MagnetAlphaHardwareConfig"
            )
        self.hardware = hardware
        self._operator_traits = MAGNET_ALPHA_OPERATOR_TRAITS
        self.memory_level = hardware.global_memory
        self.total_memory = sum(
            (
                (memory.bank_count - len(self._reserved_banks(memory.name)))
                * self._depth_per_bank(memory)
                * memory.bits_per_bank_entry
                // 8
            )
            for memory in self.memory_level.instances
        )
        self._base_by_name = {
            self.memory_level.input_memory: ALPHA_GB_SCRATCH_BASE,
            self.memory_level.weight_memory: ALPHA_GB_WEIGHT_BASE,
        }
        if len(self._base_by_name) != 2:
            raise ValueError(
                "Alpha requires separate scratch and weight memories"
            )

        self._owners = {
            memory.name: [
                [None] * self._depth_per_bank(memory)
                for _ in range(memory.bank_count)
            ]
            for memory in self.memory_level.instances
        }
        self._allocations = {}
        self._allocation_records = []
        self._resident_writes = {}
        self._resident_write_paths = {}
        self._shuffle_plans = {}
        self._copy_plans = {}
        self._shuffle_storage = {}
        self._operation_operands = {}
        self._operation_eadd_inputs = {}
        self._materialized_input_padding = {}
        self._physical_payload_uses = []
        self._payload_uses_by_physical = {}
        self._bank_conflicts = {}
        self._cluster_constraints = {}
        self._snapshots = {
            memory.name: [] for memory in self.memory_level.instances
        }
        self._history = None
        self._sealed = False
        self._storage_plan = None
        self._storage_plane = None
        self._model = None
        self._operation_scope_histories = {}
        self._active_operation_scope = None
        self._prepared_plan = None

    @staticmethod
    def _is_batch_select(node):
        if node.target != torch.ops.aten.select.int:
            return False
        source = node.args[0]
        dim = node.args[1]
        index = node.args[2]
        shape = getattr(source, "shape", None)
        return (
            shape is not None
            and dim == 0
            and index == 0
            and len(shape) == 4
            and shape[0] == 1
        )

    @staticmethod
    def _analyzed_storage_source(
        storage_graph,
        storage_plane,
        node,
        source=None,
    ):
        outputs = storage_graph.index.node_outputs.get(node, ())
        if len(outputs) != 1:
            return None
        relation = storage_graph.relation(outputs[0], storage_plane)
        if (
            relation.kind
            in {
                StorageRelationKind.ALIAS,
                StorageRelationKind.VIEW,
            }
            and relation.source is not None
            and (source is None or relation.source.node is source)
        ):
            return relation.source
        return None

    @staticmethod
    def _tuple_component_root(node, component):
        if (
            isinstance(node, Node)
            and node.target == operator.getitem
            and node.args[1] == component
            and isinstance(node.args[0], Node)
        ):
            return node.args[0]
        return None

    @staticmethod
    def _vsq_domain(node):
        values = getattr(node, "value", None)
        dtypes = node.meta.get("dtype")
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            return None
        if not isinstance(dtypes, (tuple, list)) or len(dtypes) != 3:
            return None
        if not _is_alpha_vsq_storage_pair(
            values[1],
            values[2],
            dtypes[1],
            dtypes[2],
        ):
            return None
        return PhysicalQuantizationContract(
            dtypes[2] or values[2].dtype,
            (dtypes[1] or values[1].dtype,),
            ALPHA_VSQ_BLOCK_VALUES,
        )

    @staticmethod
    def _zero_channel_tail_pad(node):
        if node.target != torch.ops.aten.pad.default:
            return False
        pad = node.kwargs.get("pad")
        if pad is None and len(node.args) > 1:
            pad = node.args[1]
        mode = node.kwargs.get("mode")
        if mode is None and len(node.args) > 2:
            mode = node.args[2]
        value = node.kwargs.get("value")
        if value is None and len(node.args) > 3:
            value = node.args[3]
        return (
            isinstance(pad, (tuple, list))
            and len(pad) == 6
            and tuple(pad[:5]) == (0, 0, 0, 0, 0)
            and isinstance(pad[5], int)
            and pad[5] > 0
            and (mode is None or mode == "constant")
            and (value is None or value == 0)
        )

    @staticmethod
    def _hardware_slice_source(node):
        if not node.meta.get("hardware_padding_slice", False):
            return None
        if node.target != torch.ops.aten.slice.Tensor:
            return None
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        step = node.args[4] if len(node.args) > 4 else 1
        if dim not in (1, -3) or start not in (None, 0) or step != 1:
            return None
        return node.args[0]

    def _hardware_repad_source(self, node):
        """Return a padded source only when a crop restores its zero tail."""
        if not self._zero_channel_tail_pad(node):
            return None
        cropped = node.args[0]
        source = self._hardware_slice_source(cropped)
        if not isinstance(source, Node):
            return None
        source_shape = tuple(getattr(source, "shape", ()))
        cropped_shape = tuple(getattr(cropped, "shape", ()))
        padded_shape = tuple(getattr(node, "shape", ()))
        if not all(
            len(shape) == 4
            for shape in (source_shape, cropped_shape, padded_shape)
        ):
            return None
        end = cropped.kwargs.get("end")
        if end is None and len(cropped.args) > 3:
            end = cropped.args[3]
        pad = node.kwargs.get("pad")
        if pad is None and len(node.args) > 1:
            pad = node.args[1]
        if not (
            0 < cropped_shape[1] < source_shape[1]
            and source_shape[0] == cropped_shape[0]
            and source_shape[2:] == cropped_shape[2:]
            and padded_shape == source_shape
            and end == cropped_shape[1]
            and pad[-1] == source_shape[1] - cropped_shape[1]
        ):
            return None
        return source

    def _classify_operation(self, node):
        """Consume the final FX binding without recognizing semantics."""
        binding = bound_operation_classification(node)
        if binding is None:
            return None
        unit, classification = binding
        if not self.hardware.compute_unit(unit).supports_operation(
            classification.kind
        ):
            raise AlphaMemoryPlanError(
                f"{node.name}: {unit} does not support "
                f"{classification.kind}"
            )
        return binding

    def _validate_operation_capability(self, constraint):
        """Reject operand modes unsupported by the selected compute unit."""
        if constraint.unit != self.hardware.nmp_compute_unit:
            return
        nmp = self.hardware.compute_unit(self.hardware.nmp_compute_unit)
        accepted = nmp.accepted_input_bits
        descriptors = tuple(
            descriptor
            for descriptor in constraint.operand_descriptors
            if (
                descriptor.role is TensorRole.ACTIVATION
                or (
                    constraint.kind == "depthwise_conv2d"
                    and descriptor.role is TensorRole.WEIGHT
                )
            )
        )
        activations = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.role is TensorRole.ACTIVATION
        )
        weights = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.role is TensorRole.WEIGHT
        )
        if not activations:
            raise AlphaMemoryPlanError(
                f"Alpha {constraint.kind} has no tensor activation operand"
            )
        if constraint.kind == "depthwise_conv2d" and not weights:
            raise AlphaMemoryPlanError(
                "Alpha depthwise_conv2d has no tensor weight operand"
            )
        for descriptor in descriptors:
            operand = descriptor.tensor
            label = descriptor.key or descriptor.role.value
            if descriptor.occurrence:
                label = f"{label}{descriptor.occurrence!r}"
            try:
                bits = _node_operand_bits(operand)
            except ValueError as error:
                raise AlphaMemoryPlanError(str(error)) from error
            if accepted and bits not in accepted:
                raise AlphaMemoryPlanError(
                    f"Alpha {constraint.kind} {label} uses {bits}-bit data; "
                    f"{nmp.name} accepts {sorted(accepted)}"
                )

    def _make_operation_constraint(
        self,
        owner,
        node,
        unit,
        classification,
        descriptors,
        implementation_descriptors=None,
    ):
        descriptors = tuple(descriptors)
        implementation_descriptors = tuple(
            descriptors
            if implementation_descriptors is None
            else implementation_descriptors
        )
        operands = tuple(
            descriptor
            for descriptor in descriptors
            if require_allocation(
                descriptor.tensor,
                self._operator_traits,
            )
        )
        constraint = _OperationConstraint(
            owner=owner,
            implementation=node,
            unit=unit,
            classification=classification,
            operand_descriptors=operands,
            implementation_descriptors=implementation_descriptors,
        )
        self._validate_operation_capability(constraint)
        return constraint

    def _operation_info(self, model, operation):
        if is_declared_unsupported_operation(operation):
            return None
        if operation.op == "call_function":
            if operation.target == operator.getitem:
                return None
            descriptors = operand_descriptors(
                operation,
                operator_traits=self._operator_traits,
            )
            classification = self._classify_operation(operation)
            if classification is None:
                return None
            unit, classification = classification
            return self._make_operation_constraint(
                operation,
                operation,
                unit,
                classification,
                descriptors=descriptors,
            )

        if operation.op != "call_module":
            return None
        submodule = model.get_submodule(str(operation.target))
        implementations = []
        for inner in submodule.graph.nodes:
            if inner.op != "call_function":
                continue
            descriptors = submodule_operand_descriptors(
                operation,
                inner,
                model,
                operator_traits=self._operator_traits,
            )
            classification = self._classify_operation(inner)
            if classification is None:
                continue
            unit, classification = classification
            implementations.append((inner, unit, classification, descriptors))
        if len(implementations) > 1:
            names = ", ".join(
                implementation.name
                for implementation, _unit, _classification, _descriptors in implementations
            )
            raise AlphaMemoryPlanError(
                f"{operation.name}: fused record has multiple "
                f"Alpha-executable implementations: {names}"
            )
        if not implementations:
            return None
        implementation, unit, classification, descriptors = implementations[0]
        execution_descriptors = operand_descriptors(
            operation,
            model,
            operator_traits=self._operator_traits,
        )
        return self._make_operation_constraint(
            operation,
            implementation,
            unit,
            classification,
            descriptors=execution_descriptors,
            implementation_descriptors=descriptors,
        )

    def _set_cluster_constraint(self, node, cluster):
        previous = self._cluster_constraints.get(node)
        if previous is not None and previous != cluster:
            raise AlphaMemoryPlanError(
                f"{node.name}: conflicting Alpha clusters "
                f"{previous} and {cluster}"
            )
        self._cluster_constraints[node] = cluster

    def _matrix_output_write(self, operation):
        """Return the INT8 tensor physically written by one PE execution."""
        try:
            if (
                get_node_dtype_width(
                    operation,
                    get_physical_dtype(operation),
                )
                == 8
            ):
                return operation
        except ValueError:
            return None

        if len(operation.users) != 1:
            return None
        ppu = next(iter(operation.users))
        if (
            ppu.target != PPU_FIXED_RELU_OPERATOR
            or get_named_arg(ppu, "input") is not operation
        ):
            return None
        if any(user.target == PVS_QUANTIZE_OPERATOR for user in ppu.users):
            return None
        try:
            return (
                ppu
                if get_node_dtype_width(
                    ppu,
                    get_physical_dtype(ppu),
                )
                == 8
                else None
            )
        except ValueError:
            return None

    def _compact_partial_matrix_output(self, operation, operands):
        """Describe one compact, single-vector INT8 PE output."""
        output = self._matrix_output_write(operation)
        if output is None or not operands:
            return None
        shape = tuple(getattr(output, "shape", ()))
        if len(shape) != 4:
            return None

        input_bits = _node_operand_bits(operands[0])
        output_bits = get_node_dtype_width(
            output,
            get_physical_dtype(output),
        )
        resident_channels = int(shape[1])
        output_lanes = self.hardware.output_lanes(input_bits)
        if (
            resident_channels <= 0
            or resident_channels >= output_lanes
            or not self.hardware.supports_partial_matrix_output(
                resident_channels,
                input_bits,
            )
        ):
            return None
        memory = self.memory_level.memory(self.memory_level.output_memory)
        resident_bits = resident_channels * output_bits
        if resident_bits % memory.bits_per_bank_entry:
            return None
        cluster = resident_bits // memory.bits_per_bank_entry
        if cluster not in (1, 2, 4):
            return None
        return output, cluster

    def _prepare_matrix_output_constraint(self, operation, operands):
        compact = self._compact_partial_matrix_output(
            operation,
            operands,
        )
        if compact is None:
            return
        output, cluster = compact
        self._set_cluster_constraint(output, cluster)

    @staticmethod
    def _pair(value, default):
        if value is None:
            return (default, default)
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return tuple(int(item) for item in value)
        raise AlphaMemoryPlanError(
            f"expected an integer or pair, got {value!r}"
        )

    @staticmethod
    def _payload_shape(payload):
        if isinstance(payload, ValueRef):
            value = getattr(payload.node, "value", None)
            try:
                for index in payload.path:
                    value = value[index]
            except (IndexError, TypeError):
                return ()
        elif isinstance(payload, Node):
            value = getattr(payload, "value", None)
        else:
            value = None
        return tuple(getattr(value, "shape", ()))

    @staticmethod
    def _physical_component_name(
        storage_graph,
        logical_tensor,
        storage_plane,
    ):
        relation = storage_graph.relation(
            logical_tensor,
            storage_plane,
        )
        return relation.component or "primary"

    def _register_payload_use(
        self,
        storage_graph,
        storage_plane,
        operation,
        field,
        logical_tensor,
        component,
    ):
        use = _PhysicalPayloadUse(
            operation=operation,
            field=field,
            logical_tensor=storage_graph.ref(logical_tensor),
            component=component,
        )
        if use not in self._physical_payload_uses:
            self._physical_payload_uses.append(use)

    def _prepare_gemm_payload_fields(
        self,
        operation,
        info,
        storage_graph,
        storage_plane,
    ):
        """Serialize exact Alpha PE weight and vector-scale payloads."""
        weight = info.operand("weight")
        input_node = info.operand("input")
        if not all(isinstance(node, Node) for node in (weight, input_node)):
            raise AlphaMemoryPlanError(
                f"{operation.name}: GEMM operands did not resolve"
            )
        value = getattr(weight, "value", None)
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise AlphaMemoryPlanError(
                f"{operation.name}: GEMM weight must be a rank-4 tensor"
            )

        weight_bits = _node_operand_bits(weight)
        logical_weight_shape = get_hardware_logical_shape(weight)
        if len(logical_weight_shape) != 4:
            raise AlphaMemoryPlanError(
                f"{operation.name}: logical GEMM weight must be rank 4"
            )
        valid_c = int(logical_weight_shape[1])
        valid_k = int(logical_weight_shape[0])
        resident_c = int(value.shape[1])
        resident_k = int(value.shape[0])
        physical_c = self.hardware.matrix_weight_physical_channels(
            resident_c,
            weight_bits,
        )
        physical_weight = _pad_tensor_axis(value, 1, physical_c)
        physical_k = int(physical_weight.shape[0])
        if not (
            0 < valid_c <= resident_c <= physical_c
            and 0 < valid_k <= resident_k <= physical_k
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: invalid logical/resident/physical "
                "GEMM weight extents: "
                f"C={valid_c}/{resident_c}/{physical_c}, "
                f"K={valid_k}/{resident_k}/{physical_k}"
            )

        target_kwargs = operation.meta.setdefault(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        target_kwargs.update(
            {
                "weight_valid_input_channels": valid_c,
                "weight_valid_output_channels": valid_k,
            }
        )

        scale = info.implementation_operand("weight_scale")
        scale_value = getattr(scale, "value", None)
        valid_scale_k = None
        encoded_scale_k = None
        if scale is not None:
            logical_scale_shape = get_hardware_logical_shape(scale)
            if (
                not isinstance(scale, Node)
                or not isinstance(scale_value, torch.Tensor)
                or scale_value.ndim != 4
                or len(logical_scale_shape) != 4
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: GEMM weight scale must be rank 4"
                )
            valid_scale_k = int(logical_scale_shape[0])
            input_bits = _node_operand_bits(input_node)
            scale_k = int(scale_value.shape[0])
            encoded_scale_k = math.ceil(
                scale_k / self.hardware.output_lanes(input_bits)
            ) * self.hardware.output_lanes(input_bits)

        weight_component = self._physical_component_name(
            storage_graph,
            weight,
            storage_plane,
        )
        scale_component = (
            None
            if scale is None
            else self._physical_component_name(
                storage_graph,
                scale,
                storage_plane,
            )
        )
        shares_vsq_backing = (
            scale_component == "fine_scale"
            and storage_graph.trace_storage_chain(
                scale,
                storage_plane,
            )[-1]
            == storage_graph.trace_storage_chain(
                weight,
                storage_plane,
            )[-1]
        )
        weight_backing_size = payload_size_bytes(
            physical_weight,
            weight_bits,
        )
        weight_layout = (
            ALPHA_VSQ_LAYOUT if shares_vsq_backing else ALPHA_DENSE_LAYOUT
        )
        storage_graph.register_physical_component(
            weight,
            physical_weight,
            plane=storage_plane,
            component=weight_component,
            dtype=weight.meta.get("dtype", physical_weight.dtype),
            bits=weight_bits,
            backing_size_bytes=weight_backing_size,
            layout=weight_layout,
        )
        self._register_payload_use(
            storage_graph,
            storage_plane,
            operation,
            "weight_payload",
            weight,
            weight_component,
        )

        if scale is None:
            return
        scale_bits = _node_operand_bits(scale)
        if shares_vsq_backing:
            storage_graph.register_physical_component(
                scale,
                scale_value,
                plane=storage_plane,
                component=scale_component,
                dtype=scale.meta.get("dtype", scale_value.dtype),
                bits=scale_bits,
                backing_size_bytes=weight_backing_size,
                layout=ALPHA_VSQ_LAYOUT,
            )
        target_kwargs.update(
            {
                "weight_scale_valid_output_channels": valid_scale_k,
            }
        )
        operation.meta.setdefault("magnet_alpha", {})[
            "weight_scale_encoded_output_channels"
        ] = encoded_scale_k

    def _prepare_convolution_padding(
        self,
        model,
        operation,
        info,
        output_payload,
    ):
        """Project logical convolution geometry onto Alpha runtime padding."""
        implementation = self._convolution_implementation(
            model,
            operation,
        )
        input_node = info.operand("input")
        weight = info.operand("weight")
        if (
            implementation is None
            or not isinstance(input_node, Node)
            or not isinstance(weight, Node)
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: convolution operands did not resolve"
            )
        input_shape = tuple(getattr(input_node, "shape", ()))
        weight_shape = tuple(getattr(weight, "shape", ()))
        output_shape = self._payload_shape(output_payload)
        if any(
            len(shape) != 4
            for shape in (input_shape, weight_shape, output_shape)
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: convolution geometry must be rank 4"
            )

        physical_input_shape = input_shape
        materialized = self._materialized_input_padding.get(
            (input_node, operation)
        )
        if materialized is not None:
            physical_input_shape = materialized.physical_shape

        kernel = tuple(int(value) for value in weight_shape[-2:])
        stride = self._pair(
            get_named_arg(implementation, "stride"),
            1,
        )
        dilation = self._pair(
            get_named_arg(implementation, "dilation"),
            1,
        )
        torch_padding = self._pair(
            get_named_arg(implementation, "padding"),
            0,
        )
        pads = []
        for axis in range(2):
            effective = dilation[axis] * (kernel[axis] - 1) + 1
            needed = max(
                (int(output_shape[axis + 2]) - 1) * stride[axis]
                + effective
                - int(physical_input_shape[axis + 2]),
                0,
            )
            before = min(torch_padding[axis], needed)
            pads.append((before, needed - before))

        target_kwargs = operation.meta.setdefault(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        target_kwargs.update(
            {
                "pad_h_up": pads[0][0],
                "pad_h_down": pads[0][1],
                "pad_w_left": pads[1][0],
                "pad_w_right": pads[1][1],
            }
        )
        if info.kind != "gemm":
            return
        input_bits = _node_operand_bits(input_node)
        valid_c = int(weight_shape[1])
        resident_c = (
            materialized.resident_channels
            if materialized is not None
            else valid_c
        )
        valid_k = int(weight_shape[0])
        target_kwargs["padding_c"] = (-resident_c) % self.hardware.input_lanes(
            input_bits
        )
        target_kwargs["padding_k"] = (-valid_k) % self.hardware.output_lanes(
            input_bits
        )

    def _convolution_implementation(self, model, operation):
        if operation.op == "call_function":
            return (
                operation
                if is_conv2d(operation, self._operator_traits)
                else None
            )
        if operation.op != "call_module":
            return None
        submodule = model.get_submodule(str(operation.target))
        return next(
            (
                inner
                for inner in submodule.graph.nodes
                if (
                    inner.op == "call_function"
                    and is_conv2d(inner, self._operator_traits)
                )
            ),
            None,
        )

    def _is_host_materialized_input(self, model, input_node, operation):
        users = set(input_node.users)
        if operation not in users:
            return False
        for user in users:
            info = self._operation_info(model, user)
            if (
                info is None
                or info.kind != "gemm"
                or info.operand("input") is not input_node
            ):
                return False

        pending = [input_node]
        visited = set()
        reaches_placeholder = False
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node.op == "placeholder":
                reaches_placeholder = True
                continue
            if node.op == "get_attr":
                continue
            if node.op == "call_module":
                if self._operation_info(model, node) is not None:
                    return False
            elif (
                node.op == "call_function"
                and self._operation_info(model, node) is not None
            ):
                return False
            pending.extend(node.all_input_nodes)
        return reaches_placeholder

    def _register_materialized_input(
        self,
        operation,
        implementation,
        input_node,
        physical_value,
        padding,
        operand_bits,
        valid_channels,
        resident_channels,
        storage_graph,
        storage_plane,
        *,
        serialize_channels,
    ):
        """Register one exact host-prepared Alpha input representation."""
        physical_shape = tuple(int(dim) for dim in physical_value.shape)
        plan = _MaterializedInputPadding(
            operation=operation,
            implementation=implementation,
            input=input_node,
            padding=tuple(int(value) for value in padding),
            physical_shape=physical_shape,
            operand_bits=operand_bits,
            valid_channels=int(valid_channels),
            resident_channels=int(resident_channels),
            serialize_channels=bool(serialize_channels),
        )
        key = (input_node, operation)
        previous = self._materialized_input_padding.get(key)
        if previous is not None and previous != plan:
            raise AlphaMemoryPlanError(
                f"{input_node.name}: conflicting materialized input padding"
            )
        self._materialized_input_padding[key] = plan
        self._set_cluster_constraint(input_node, 1)

        component = self._physical_component_name(
            storage_graph,
            input_node,
            storage_plane,
        )
        backing_size = payload_size_bytes(physical_value, operand_bits)
        storage_graph.register_physical_component(
            input_node,
            physical_value,
            plane=storage_plane,
            component=component,
            dtype=input_node.meta.get("dtype", physical_value.dtype),
            bits=operand_bits,
            backing_size_bytes=backing_size,
            layout=ALPHA_PADDED_INPUT_LAYOUT,
        )
        self._register_payload_use(
            storage_graph,
            storage_plane,
            implementation,
            "materialized_input_payload",
            input_node,
            component,
        )

        target_kwargs = implementation.meta.setdefault(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        existing = target_kwargs.get("materialized_input_padding")
        if existing is not None and tuple(existing) != plan.padding:
            raise AlphaMemoryPlanError(
                f"{implementation.name}: conflicting serialized input "
                "padding"
            )
        target_kwargs["materialized_input_padding"] = plan.padding
        if serialize_channels:
            fields = {
                "materialized_input_valid_channels": plan.valid_channels,
                "materialized_input_resident_channels": (
                    plan.resident_channels
                ),
            }
            for field, value in fields.items():
                existing = target_kwargs.get(field)
                if existing is not None and int(existing) != value:
                    raise AlphaMemoryPlanError(
                        f"{implementation.name}: conflicting {field}"
                    )
                target_kwargs[field] = value

    def _prepare_host_input_storage(
        self,
        model,
        operation,
        info,
        storage_graph,
        storage_plane,
    ):
        """Declare the encoded Alpha backing for host-prepared PE input."""
        implementation = self._convolution_implementation(
            model,
            operation,
        )
        input_node = info.operand("input")
        weight = info.operand("weight")
        if implementation is None or input_node is None or weight is None:
            return

        input_shape = tuple(getattr(input_node, "shape", ()))
        output_shape = tuple(getattr(operation, "shape", ()))
        weight_shape = tuple(getattr(weight, "shape", ()))
        if (
            len(input_shape) != 4
            or len(output_shape) != 4
            or len(weight_shape) != 4
        ):
            return

        operand_bits = _node_operand_bits(input_node)
        valid_channels = int(input_shape[1])
        if operand_bits != 8 or not self._is_host_materialized_input(
            model,
            input_node,
            operation,
        ):
            return

        kernel = tuple(int(value) for value in weight_shape[-2:])
        stride = self._pair(
            get_named_arg(implementation, "stride"),
            1,
        )
        dilation = self._pair(
            get_named_arg(implementation, "dilation"),
            1,
        )
        torch_padding = self._pair(
            get_named_arg(implementation, "padding"),
            0,
        )
        if (
            valid_channels == self.hardware.input_lanes(operand_bits)
            and int(input_shape[2]) == 1
            and int(output_shape[2]) == 1
            and kernel == (1, 1)
            and stride == (1, 1)
            and dilation == (1, 1)
            and torch_padding == (0, 0)
        ):
            value = getattr(input_node, "value", None)
            if not isinstance(value, torch.Tensor):
                raise AlphaMemoryPlanError(
                    f"{input_node.name}: host input has no tensor value"
                )
            self._set_cluster_constraint(input_node, 1)
            component = self._physical_component_name(
                storage_graph,
                input_node,
                storage_plane,
            )
            storage_graph.register_physical_component(
                input_node,
                value,
                plane=storage_plane,
                component=component,
                dtype=get_physical_dtype(input_node) or value.dtype,
                bits=operand_bits,
                backing_size_bytes=payload_size_bytes(value, operand_bits),
                layout=ALPHA_DENSE_LAYOUT,
            )
            return

        resident_channels = self.hardware.matrix_input_resident_channels(
            valid_channels,
            operand_bits,
        )
        if resident_channels != valid_channels:
            if (
                not self.hardware.supports_partial_matrix_input(valid_channels)
                or int(weight_shape[1]) != valid_channels
                or int(get_named_arg(implementation, "groups") or 1) != 1
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: invalid compact Alpha matrix input "
                    f"contract for C={valid_channels}"
                )
            value = getattr(input_node, "value", None)
            if not isinstance(value, torch.Tensor):
                raise AlphaMemoryPlanError(
                    f"{input_node.name}: host input has no tensor value"
                )
            physical_value = _pad_tensor_axis(
                value,
                1,
                resident_channels,
            )
            self._register_materialized_input(
                operation,
                implementation,
                input_node,
                physical_value,
                (0, 0, 0, 0),
                operand_bits,
                valid_channels,
                resident_channels,
                storage_graph,
                storage_plane,
                serialize_channels=True,
            )
            return

        if (
            valid_channels != 1
            or not self.hardware.supports_partial_matrix_input(valid_channels)
        ):
            return

        if kernel != (3, 3) or stride != (2, 2) or dilation != (1, 1):
            return

        effective_width = dilation[1] * (kernel[1] - 1) + 1
        needed_width = max(
            (int(output_shape[3]) - 1) * stride[1]
            + effective_width
            - int(input_shape[3]),
            0,
        )
        pad_left = min(torch_padding[1], needed_width)
        pad_right = needed_width - pad_left
        if pad_left != 1 or pad_right != 0:
            return

        value = getattr(input_node, "value", None)
        if not isinstance(value, torch.Tensor):
            raise AlphaMemoryPlanError(
                f"{input_node.name}: host input has no tensor value"
            )
        padding = (pad_left, 0, 0, 0)
        physical_shape = (
            *input_shape[:3],
            input_shape[3] + pad_left,
        )
        physical_value = torch.nn.functional.pad(
            value,
            padding,
            mode="constant",
            value=0,
        )
        if tuple(physical_value.shape) != physical_shape:
            raise AlphaMemoryPlanError(
                f"{input_node.name}: padded input shape "
                f"{tuple(physical_value.shape)} does not match "
                f"{physical_shape}"
            )
        self._register_materialized_input(
            operation,
            implementation,
            input_node,
            physical_value,
            padding,
            operand_bits,
            valid_channels,
            valid_channels,
            storage_graph,
            storage_plane,
            serialize_channels=False,
        )

    def _pool_uses_one_bank(self, pool, operands):
        if not operands:
            return False
        input_node = operands[0]
        input_shape = tuple(getattr(input_node, "shape", ()))
        output_shape = tuple(getattr(pool, "shape", ()))
        if len(input_shape) != 4 or len(output_shape) != 4:
            return False
        try:
            input_bits = _node_operand_bits(input_node)
            output_bits = _node_operand_bits(pool)
        except ValueError:
            return False
        if input_bits != 8 or output_bits != 8:
            return False
        memory = self.memory_level.memory(self.memory_level.input_memory)
        channels_per_bank = memory.bits_per_bank_entry // input_bits
        return (
            input_shape[1] <= channels_per_bank
            and output_shape[1] <= channels_per_bank
        )

    def _prepare_execution_constraints(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        for operation in model.graph.nodes:
            if (
                operation.op == "call_function"
                and operation.target == SATURATING_ADD_OPERATOR
            ):
                declare_unsupported_operation(
                    operation,
                    reason_code=("no_magnet_alpha_fused_eadd_execution_owner"),
                    detail=(
                        "final standalone EADD is outside one supported "
                        "fused Alpha execution owner"
                    ),
                    stage="operation_graph_preparation",
                    facts={
                        "output_shape": tuple(operation.shape),
                        "target": str(operation.target),
                    },
                )
                continue
            info = self._operation_info(model, operation)
            if info is None:
                continue
            if (
                info.kind == "depthwise_conv2d"
                and operation.op == "call_function"
                and operation.target == INT8_CONV2D_ACCUMULATOR_OPERATOR
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: exact Alpha DWC execution region was "
                    "not fused before memory planning"
                )
            self._operation_operands[operation] = info
            operation.meta["compute_unit"] = info.unit
            target_kwargs = operation.meta.setdefault(
                TARGET_CODEGEN_KWARGS,
                {},
            )
            output_payload = target_kwargs.get("output_payload")
            if output_payload is None:
                output_payload = self._single_storage_ref(
                    storage_graph,
                    operation,
                )
                target_kwargs["output_payload"] = output_payload
            if info.kind == "gemm":
                self._prepare_matrix_output_constraint(
                    operation,
                    info.operands,
                )
                self._prepare_host_input_storage(
                    model,
                    operation,
                    info,
                    storage_graph,
                    storage_plane,
                )
                self._prepare_gemm_payload_fields(
                    operation,
                    info,
                    storage_graph,
                    storage_plane,
                )
            if info.kind in {"gemm", "depthwise_conv2d"}:
                self._prepare_convolution_padding(
                    model,
                    operation,
                    info,
                    output_payload,
                )
            if info.kind != "pool2d":
                continue
            one_bank = self._pool_uses_one_bank(
                operation,
                info.operands,
            )
            target_kwargs["is_pool_1bank"] = one_bank
            if not one_bank:
                continue
            self._set_cluster_constraint(operation, 1)
            for source in info.operands[:1]:
                try:
                    if _node_operand_bits(source) == 8:
                        self._set_cluster_constraint(source, 1)
                except ValueError:
                    continue

        constrained = set(self._cluster_constraints)
        for node in model.graph.nodes:
            layout = self._concat_layout(
                node,
                storage_graph,
                storage_plane,
            )
            if layout is None:
                continue
            writes = [
                self._trace_write_owner(
                    tensor,
                    storage_graph,
                    storage_plane,
                )
                for tensor in layout.inputs
            ]
            if not any(owner in constrained for owner, _path in writes):
                continue
            if any(_node_operand_bits(owner) != 8 for owner, _path in writes):
                continue
            self._set_cluster_constraint(node, 1)
            for owner, path in writes:
                self._set_cluster_constraint(owner, 1)
                for path_node in path:
                    self._set_cluster_constraint(path_node, 1)

    @staticmethod
    def _ingress_placeholder(quantizer):
        """Trace one public input through shared address-only operations."""
        current = get_named_arg(quantizer, "input")
        visited = set()
        while isinstance(current, Node):
            if current.op == "placeholder":
                return current
            if current in visited or not is_nop(current):
                return None
            visited.add(current)
            inputs = tuple(current.all_input_nodes)
            if len(inputs) != 1:
                return None
            current = inputs[0]
        return None

    def _ingress_quantizer_owner(self, quantizer):
        """Prove one quantizer is absorbed by exact Alpha call owners."""
        if (
            quantizer.op != "call_function"
            or quantizer.target != torch.ops.quantized_ops.quantize.default
            or self._ingress_placeholder(quantizer) is None
        ):
            return None

        owners = set()
        pending = [(user, quantizer) for user in quantizer.users]
        visited = set()
        while pending:
            user, predecessor = pending.pop()
            edge = (predecessor, user)
            if edge in visited:
                continue
            visited.add(edge)
            if user in self._operation_operands:
                constraint = self._operation_operands[user]
                if constraint.operand("input") is not predecessor:
                    return None
                owners.add(user)
                continue
            if user.op not in {"call_function", "call_module"}:
                return None
            if not is_nop(user):
                return None
            pending.extend((next_user, user) for next_user in user.users)

        if not owners:
            return None
        return frozenset(owners)

    @staticmethod
    def _single_user(node):
        if len(node.users) != 1:
            return None
        return next(iter(node.users))

    @staticmethod
    def _serialized_nop(node):
        return (
            node.target == operator.getitem
            or is_nop(node)
            or node.meta.get(TARGET_INSTRUCTION_NOP, False)
        )

    def _terminal_egress_owners(self, source):
        """Resolve every output owner through complete serialized-NOP paths."""
        if not isinstance(source, Node):
            return None
        pending = [source]
        visited = set()
        owners = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in self._operation_operands:
                owners.add(node)
                continue
            if node.op not in {
                "call_function",
                "call_module",
            } or not self._serialized_nop(node):
                return None
            inputs = tuple(node.all_input_nodes)
            if not inputs:
                return None
            pending.extend(inputs)
        if not owners:
            return None
        return frozenset(owners)

    @staticmethod
    def _node_shape(node):
        shape = getattr(node, "shape", None)
        if shape is None:
            shape = getattr(node.meta.get("tensor_meta"), "shape", None)
        if shape is None:
            shape = getattr(node.meta.get("val"), "shape", None)
        try:
            return tuple(int(size) for size in shape)
        except (TypeError, ValueError):
            return ()

    @staticmethod
    def _node_width_is(node, width):
        try:
            return get_node_dtype_width(node) == width
        except (TypeError, ValueError):
            return False

    def _terminal_padding_slice(self, node, dequantize):
        """Prove one target-inserted crop of padded output channels."""
        if (
            node.target != torch.ops.aten.slice.Tensor
            or not node.meta.get("hardware_padding_slice", False)
            or not node.args
            or node.args[0] is not dequantize
        ):
            return False
        source_shape = self._node_shape(dequantize)
        result_shape = self._node_shape(node)
        if (
            not source_shape
            or len(source_shape) != len(result_shape)
            or len(source_shape) < 2
        ):
            return False
        dim = get_named_arg(node, "dim")
        start = get_named_arg(node, "start")
        end = get_named_arg(node, "end")
        step = get_named_arg(node, "step")
        if dim is None:
            dim = 0
        if start is None:
            start = 0
        if step is None:
            step = 1
        normalized_dim = dim + len(source_shape) if dim < 0 else dim
        if (
            dim not in (1, -3)
            or normalized_dim != 1
            or start != 0
            or step != 1
            or not isinstance(end, int)
            or end <= 0
            or end >= source_shape[1]
            or result_shape[1] != end
            or source_shape[:1] != result_shape[:1]
            or source_shape[2:] != result_shape[2:]
            or not self._node_width_is(node, 32)
        ):
            return False
        return True

    def _terminal_yolo_egress(
        self,
        owners,
        source,
        dequantize,
        padding_slice,
    ):
        view = self._single_user(padding_slice)
        if (
            not isinstance(view, Node)
            or view.target != torch.ops.aten.view.default
            or not view.args
            or view.args[0] is not padding_slice
            or not is_nop(view)
        ):
            return None
        slice_shape = self._node_shape(padding_slice)
        view_shape = self._node_shape(view)
        declared_shape = get_named_arg(view, "size")
        try:
            declared_shape = tuple(int(size) for size in declared_shape)
        except (TypeError, ValueError):
            return None
        if (
            len(slice_shape) != 4
            or slice_shape[1] != 48
            or view_shape
            != (
                slice_shape[0],
                3,
                16,
                slice_shape[2],
                slice_shape[3],
            )
            or declared_shape != view_shape
            or not self._node_width_is(view, 32)
        ):
            return None

        permute = self._single_user(view)
        if (
            not isinstance(permute, Node)
            or permute.target != torch.ops.aten.permute.default
            or not permute.args
            or permute.args[0] is not view
            or is_nop(permute)
        ):
            return None
        dims = get_named_arg(permute, "dims")
        try:
            dims = tuple(int(dim) for dim in dims)
        except (TypeError, ValueError):
            return None
        expected_shape = (
            view_shape[0],
            view_shape[1],
            view_shape[3],
            view_shape[4],
            view_shape[2],
        )
        if (
            dims != (0, 1, 3, 4, 2)
            or self._node_shape(permute) != expected_shape
            or not self._node_width_is(permute, 32)
        ):
            return None

        contiguous = self._single_user(permute)
        if (
            not isinstance(contiguous, Node)
            or contiguous.target != torch.ops.aten.contiguous.default
            or not contiguous.args
            or contiguous.args[0] is not permute
            or is_nop(contiguous)
        ):
            return None
        memory_format = get_named_arg(contiguous, "memory_format")
        if (
            memory_format not in (None, torch.contiguous_format)
            or self._node_shape(contiguous) != expected_shape
            or not self._node_width_is(contiguous, 32)
        ):
            return None
        output = self._single_user(contiguous)
        if not isinstance(output, Node) or output.op != "output":
            return None
        return _OperationEgressBoundary(
            owners=owners,
            source=source,
            dequantize=dequantize,
            padding_slice=padding_slice,
            shared_views=(view,),
            host_layouts=(permute, contiguous),
            output=output,
        )

    def _terminal_flatten_egress(
        self,
        owners,
        source,
        dequantize,
        padding_slice,
    ):
        flatten = self._single_user(padding_slice)
        if (
            not isinstance(flatten, Node)
            or flatten.target != torch.ops.aten.flatten.using_ints
            or not flatten.args
            or flatten.args[0] is not padding_slice
            or not is_nop(flatten)
        ):
            return None
        start_dim = get_named_arg(flatten, "start_dim")
        end_dim = get_named_arg(flatten, "end_dim")
        if start_dim is None:
            start_dim = 0
        if end_dim is None:
            end_dim = -1
        slice_shape = self._node_shape(padding_slice)
        if len(slice_shape) < 2:
            return None
        expected_shape = (slice_shape[0], math.prod(slice_shape[1:]))
        if (
            start_dim != 1
            or end_dim != -1
            or self._node_shape(flatten) != expected_shape
            or not self._node_width_is(flatten, 32)
        ):
            return None
        output = self._single_user(flatten)
        if not isinstance(output, Node) or output.op != "output":
            return None
        return _OperationEgressBoundary(
            owners=owners,
            source=source,
            dequantize=dequantize,
            padding_slice=padding_slice,
            shared_views=(flatten,),
            host_layouts=(),
            output=output,
        )

    def _terminal_egress_boundary(self, dequantize):
        """Match one complete compiler-supported terminal suffix.

        A plural owner set remains explicit here.  The collateral converter's
        scalar fusion-owner contract is separate and stays fail-closed.
        """
        if (
            dequantize.op != "call_function"
            or dequantize.target != torch.ops.quantized_ops.dequantize.default
        ):
            return None
        source = get_named_arg(dequantize, "input")
        scale = get_named_arg(dequantize, "scale")
        owners = self._terminal_egress_owners(source)
        if (
            owners is None
            or not isinstance(source, Node)
            or not isinstance(scale, Node)
            or not self._node_width_is(source, 8)
            or not self._node_width_is(dequantize, 32)
            or self._node_shape(source) != self._node_shape(dequantize)
        ):
            return None
        padding_slice = self._single_user(dequantize)
        if not isinstance(padding_slice, Node):
            return None
        if not self._terminal_padding_slice(
            padding_slice,
            dequantize,
        ):
            return None
        closure = self._terminal_yolo_egress(
            owners,
            source,
            dequantize,
            padding_slice,
        )
        if closure is not None:
            return closure
        return self._terminal_flatten_egress(
            owners,
            source,
            dequantize,
            padding_slice,
        )

    def _trace_write_owner(
        self,
        input_node,
        storage_graph,
        storage_plane,
    ):
        """Resolve a resident writer through shared storage relations."""
        start = self._address_source_ref(storage_graph, input_node)
        refs = storage_graph.trace_storage_chain(
            start,
            storage_plane,
            follow_kinds={
                StorageRelationKind.ALIAS,
                StorageRelationKind.VIEW,
                StorageRelationKind.COMPONENT,
            },
            follow_copy=lambda ref: _is_shuffle_copy(ref.node),
            follow_write_input=lambda ref: (
                is_nop(ref.node)
                or ref.node.meta.get(TARGET_INSTRUCTION_NOP, False)
                or _is_shuffle_copy(ref.node)
                or get_leading_channel_slice_source(ref.node) is not None
            ),
        )
        nodes = tuple(dict.fromkeys(ref.node for ref in refs))
        return nodes[-1], nodes[:-1]

    def _prepare_physical_quantization_contracts(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        for node in model.graph.nodes:
            domain = self._vsq_domain(node)
            if domain is not None:
                set_physical_quantization_contract(node, domain)

        for node in model.graph.nodes:
            if node.target == torch.ops.quantized_ops.dequantize_vsq.default:
                payload = get_named_arg(node, "input")
                fine_scale = get_named_arg(node, "fine_scale")
                coarse_scale = get_named_arg(node, "coarse_scale")
                roots = (
                    self._tuple_component_root(payload, 2),
                    self._tuple_component_root(fine_scale, 1),
                    self._tuple_component_root(coarse_scale, 0),
                )
                if roots[0] is not None and roots.count(roots[0]) == 3:
                    domain = get_physical_quantization_contract(roots[0])
                    if domain is not None:
                        self._mark_address_alias(
                            node,
                            roots[0],
                            storage_graph,
                            storage_plane,
                            domain,
                        )
                continue

            inputs = node.all_input_nodes
            if not inputs:
                continue
            analyzed_source = self._analyzed_storage_source(
                storage_graph,
                storage_plane,
                node,
                inputs[0],
            )
            if analyzed_source is not None:
                domain = get_physical_quantization_contract(inputs[0])
                if domain is not None:
                    set_physical_quantization_contract(node, domain)
                    _mark_alpha_address_only(node)
                continue

            if self._is_batch_select(node):
                domain = get_physical_quantization_contract(inputs[0])
                if domain is not None:
                    self._mark_address_alias(
                        node,
                        inputs[0],
                        storage_graph,
                        storage_plane,
                        domain,
                    )
                continue

            if node.target not in (
                torch.ops.aten.cat.default,
                torch.ops.aten.stack.default,
            ):
                continue
            domains = [
                get_physical_quantization_contract(input_node)
                for input_node in inputs
            ]
            if (
                domains
                and domains[0] is not None
                and all(domain == domains[0] for domain in domains)
            ):
                set_physical_quantization_contract(node, domains[0])

    def _single_storage_ref(self, storage_graph, node):
        outputs = storage_graph.index.node_outputs.get(node, ())
        if len(outputs) != 1:
            raise AlphaMemoryPlanError(
                f"{node.name}: expected one logical tensor result, "
                f"found {len(outputs)}"
            )
        return outputs[0]

    @staticmethod
    def _vsq_storage_refs(storage_graph, node):
        """Return Alpha coarse, fine, and payload logical results."""
        by_path = {
            output.path: output
            for output in storage_graph.index.node_outputs.get(node, ())
        }
        paths = ((0,), (1,), (2,))
        if any(path not in by_path for path in paths):
            raise AlphaMemoryPlanError(
                f"{node.name}: Alpha VSQ storage requires logical outputs "
                "at output[0], output[1], and output[2]"
            )
        return tuple(by_path[path] for path in paths)

    def _address_source_ref(self, storage_graph, source):
        if self._vsq_tuple_payload(source) is not None:
            return self._vsq_storage_refs(storage_graph, source)[2]
        return self._single_storage_ref(storage_graph, source)

    @staticmethod
    def _is_alpha_declared_relation(relation):
        return relation.component == "fine_scale" or (
            relation.transform is not None
            and relation.transform.kind.startswith("alpha_")
        )

    def _set_alpha_relation(
        self,
        storage_graph,
        storage_plane,
        result,
        kind,
        *,
        source,
        transform=None,
        component=None,
        allow_copy_override=False,
    ):
        """Set one target relation, rejecting contradictory target rules."""
        result_ref = (
            self._single_storage_ref(storage_graph, result)
            if isinstance(result, Node)
            else storage_graph.ref(result)
        )
        source_ref = (
            source
            if not isinstance(source, Node)
            else self._address_source_ref(storage_graph, source)
        )
        previous = storage_graph.relation(result_ref, storage_plane)
        if (
            previous.kind is StorageRelationKind.COPY
            and not allow_copy_override
            and kind
            in {
                StorageRelationKind.ALIAS,
                StorageRelationKind.VIEW,
                StorageRelationKind.COMPONENT,
            }
        ):
            return previous
        if self._is_alpha_declared_relation(previous):
            same_relation = (
                previous.kind is kind
                and previous.source == source_ref
                and previous.physical_source is None
                and previous.transform == transform
                and previous.component == component
            )
            compatible_resident_chain = (
                previous.kind is StorageRelationKind.VIEW
                and previous.source is not None
                and previous.transform is not None
                and previous.transform.kind == "alpha_address_only"
                and kind is StorageRelationKind.VIEW
                and transform is not None
                and transform.kind == "alpha_resident_view"
                and source_ref
                in storage_graph.trace_storage_chain(
                    result_ref,
                    storage_plane,
                )[1:]
            )
            # Keep the more precise logical view chain when its producer has
            # just been rebound to the requested resident backing.  Replacing
            # it with a direct view would duplicate the same physical
            # identity and can incorrectly look like a conflicting source.
            if compatible_resident_chain:
                return previous
            if not same_relation:
                previous_source = (
                    previous.source.name
                    if previous.source is not None
                    else previous.physical_source.name
                )
                raise AlphaMemoryPlanError(
                    f"{result_ref.name}: conflicting Alpha storage sources "
                    f"{previous_source} and {source_ref.name}"
                )
            return previous
        return storage_graph.set_relation(
            result_ref,
            kind,
            plane=storage_plane,
            source=source_ref,
            transform=transform,
            component=component,
        )

    def _bind_alpha_physical(
        self,
        storage_graph,
        storage_plane,
        result,
        physical,
        *,
        transform,
    ):
        """Bind target storage without replacing another target rule."""
        result_ref = storage_graph.ref(result)
        previous = storage_graph.relation(result_ref, storage_plane)
        if previous.kind is StorageRelationKind.COPY:
            return previous
        if self._is_alpha_declared_relation(previous):
            same_relation = (
                previous.kind is StorageRelationKind.VIEW
                and previous.source is None
                and previous.physical_source == physical
                and previous.transform == transform
            )
            split_secondary_override = (
                previous.kind is StorageRelationKind.VIEW
                and previous.source == physical.owner
                and previous.transform is not None
                and previous.transform.kind == "alpha_channel_split"
                and result_ref.path == (1,)
                and physical.component == "output[1]"
                and transform.kind == "alpha_shuffle_output"
            )
            if not same_relation and not split_secondary_override:
                raise AlphaMemoryPlanError(
                    f"{result_ref.name}: conflicting Alpha physical "
                    "storage bindings"
                )
            if same_relation:
                return previous
        return storage_graph.bind_physical(
            result_ref,
            physical,
            plane=storage_plane,
            transform=transform,
        )

    def _mark_address_alias(
        self,
        node,
        source,
        storage_graph,
        storage_plane,
        domain=None,
        allow_copy_override=False,
    ):
        """Mark a graph value as an address-only view before allocation."""
        previous = storage_graph.relation(node, storage_plane)
        if (
            previous.kind is StorageRelationKind.COPY
            and not allow_copy_override
        ):
            # A target may suppress an already analyzed view, but it must not
            # erase a real writer selected by shared storage analysis.
            return False
        self._set_alpha_relation(
            storage_graph,
            storage_plane,
            node,
            StorageRelationKind.VIEW,
            source=source,
            transform=ViewTransform("alpha_address_only"),
            allow_copy_override=allow_copy_override,
        )
        _mark_alpha_address_only(node)
        if domain is not None:
            set_physical_quantization_contract(node, domain)
        return True

    def _prepare_padding_aliases(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        for slice_node in model.graph.nodes:
            source = self._hardware_slice_source(slice_node)
            if not isinstance(source, Node):
                continue
            domain = get_physical_quantization_contract(source)
            if domain is not None:
                self._mark_address_alias(
                    slice_node,
                    source,
                    storage_graph,
                    storage_plane,
                    domain,
                    allow_copy_override=True,
                )
            else:
                self._mark_address_alias(
                    slice_node,
                    source,
                    storage_graph,
                    storage_plane,
                    allow_copy_override=True,
                )

        for pad in model.graph.nodes:
            source = self._hardware_repad_source(pad)
            if not isinstance(source, Node):
                continue
            input_node = pad.args[0]
            for alias in (input_node, pad):
                self._mark_address_alias(
                    alias,
                    source,
                    storage_graph,
                    storage_plane,
                    allow_copy_override=bool(
                        alias.meta.get(
                            "hardware_padding_slice",
                            False,
                        )
                    ),
                )
            domain = get_physical_quantization_contract(source)
            if domain is not None:
                set_physical_quantization_contract(pad, domain)

    def _concat_downstream_aliases(
        self,
        cat,
        storage_graph,
        storage_plane,
    ):
        def successor(current):
            if len(current.users) != 1:
                return None
            user = next(iter(current.users))
            if (
                self._analyzed_storage_source(
                    storage_graph,
                    storage_plane,
                    user,
                    current,
                )
                is not None
            ):
                return user
            if user.target == torch.ops.aten.stack.default:
                tensors = user.args[0]
                dim = user.args[1] if len(user.args) > 1 else 0
                if len(tensors) == 1 and tensors[0] is current and dim == 0:
                    return user
            return None

        chain = []
        visited = set()
        current = cat
        while current is not None:
            if current in visited:
                names = " -> ".join(node.name for node in (*chain, current))
                raise AlphaMemoryPlanError(f"Value-relation cycle: {names}")
            visited.add(current)
            chain.append(current)
            current = successor(current)
        return tuple(chain)

    def _concat_channel_axis(self, node, tensors, shapes, aliases):
        """Recognize channel concat, including its lowered stack form."""
        rank = len(shapes[0])
        dim = node.args[1] if len(node.args) > 1 else 0
        if node.target == torch.ops.aten.cat.default:
            if dim < 0:
                dim += rank
            if rank == 4 and dim == 1:
                return 1 if all(shape[0] == 1 for shape in shapes) else None
            if (
                rank == 3
                and dim == 0
                and all(self._is_batch_select(item) for item in tensors)
            ):
                return 0
            return None

        if dim < 0:
            dim += rank + 1
        if (
            rank != 4
            or dim != 0
            or shapes[0][0] != 1
            or any(shape != shapes[0] for shape in shapes[1:])
            or len(aliases) == 1
        ):
            return None

        batch, channels, height, width = shapes[0]
        expected = (batch, len(shapes) * channels, height, width)
        return (
            1 if tuple(getattr(aliases[-1], "shape", ())) == expected else None
        )

    def _concat_layout(self, node, storage_graph, storage_plane):
        if node.target not in (
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        ):
            return None
        tensors = node.args[0]
        if not isinstance(tensors, (tuple, list)) or len(tensors) < 2:
            return None
        if not all(isinstance(input_node, Node) for input_node in tensors):
            return None
        tensors = tuple(tensors)

        shapes = tuple(
            tuple(getattr(input_node, "shape", ())) for input_node in tensors
        )
        if not shapes or any(not shape for shape in shapes):
            return None
        if any(len(shape) != len(shapes[0]) for shape in shapes):
            return None

        aliases = self._concat_downstream_aliases(
            node,
            storage_graph,
            storage_plane,
        )
        channel_axis = self._concat_channel_axis(
            node,
            tensors,
            shapes,
            aliases,
        )
        if channel_axis is None:
            return None

        reference = shapes[0]
        if any(
            extent != reference[index]
            for shape in shapes[1:]
            for index, extent in enumerate(shape)
            if index != channel_axis
        ):
            return None
        return _ConcatLayout(
            inputs=tensors,
            shapes=shapes,
            channel_axis=channel_axis,
            aliases=aliases,
        )

    def _concat_channel_granularity(self, node, operand_bits):
        domain = get_physical_quantization_contract(node)
        if domain is not None and domain.companion_dtypes:
            # Alpha's O4 concat mode moves each payload block together with
            # its fine scale, even when a bank-cluster entry is wider.
            if domain.block_size is not None:
                return domain.block_size

        memory_name = self.memory_level.output_memory
        memory = self.memory_level.memory(memory_name)
        cluster = self._cluster_for(
            node,
            TensorRole.OUTPUT,
            memory_name,
            operand_bits,
        )
        logical_entry_bits = cluster * memory.bits_per_bank_entry
        if logical_entry_bits % operand_bits:
            return None
        entry_channels = logical_entry_bits // operand_bits
        if operand_bits == 8 and cluster == 1:
            return max(entry_channels, self.hardware.nmp_lanes(8))
        return entry_channels

    def _concat_writes_for(
        self,
        cat,
        storage_graph,
        storage_plane,
    ):
        layout = self._concat_layout(
            cat,
            storage_graph,
            storage_plane,
        )
        if layout is None:
            return None

        operand_bits = _node_operand_bits(cat)
        channel_granularity = self._concat_channel_granularity(
            cat,
            operand_bits,
        )
        if channel_granularity is None:
            return None

        writes = []
        logical_channel_sizes = [
            shape[layout.channel_axis] for shape in layout.shapes
        ]
        physical_channel_sizes = [
            math.ceil(size / channel_granularity) * channel_granularity
            for size in logical_channel_sizes
        ]
        logical_total_channels = sum(logical_channel_sizes)
        total_channels = sum(physical_channel_sizes)
        channel_offset = 0
        for input_index, (input_node, physical_size) in enumerate(
            zip(layout.inputs, physical_channel_sizes)
        ):
            owner, path = self._trace_write_owner(
                input_node,
                storage_graph,
                storage_plane,
            )
            if not isinstance(owner, Node):
                return None
            if owner.op not in ("call_function", "call_module"):
                return None
            owner_shape = tuple(getattr(owner, "shape", ()))
            if len(owner_shape) == 4 and owner_shape[1] > physical_size:
                return None
            writes.append(
                _ResidentWrite(
                    backing=cat,
                    owner=owner,
                    path=path,
                    slot_index=input_index,
                    channel_offset=channel_offset,
                    slot_channels=physical_size,
                    logical_backing_channels=logical_total_channels,
                    resident_channels=total_channels,
                    concat_channel_index=1,
                )
            )
            channel_offset += physical_size

        owners = [write.owner for write in writes]
        if len(set(owners)) != len(owners):
            return None
        if any(_node_operand_bits(owner) != operand_bits for owner in owners):
            return None
        return _ConcatPlan(layout, tuple(writes))

    def _register_resident_write(
        self,
        write,
        backing_aliases,
    ):
        existing = self._resident_writes.get(write.owner)
        if existing is not None and existing != write:
            raise AlphaMemoryPlanError(
                f"{write.owner.name}: conflicting resident writes"
            )
        aliases = tuple(backing_aliases)

        self._resident_writes[write.owner] = write
        self._resident_write_paths[write.owner] = write.path
        for alias in aliases:
            if alias is not write.owner:
                _mark_alpha_address_only(alias)
        for path_node in write.path:
            if not _is_shuffle_copy(path_node):
                _mark_alpha_address_only(path_node)

    def _prepare_concat_writes(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        for node in model.graph.nodes:
            outputs = storage_graph.index.node_outputs.get(node, ())
            if (
                len(outputs) == 1
                and storage_graph.relation(
                    outputs[0],
                    storage_plane,
                ).kind
                is StorageRelationKind.COMPONENT
            ):
                continue
            plan = self._concat_writes_for(
                node,
                storage_graph,
                storage_plane,
            )
            if plan is None:
                continue
            writes = plan.writes
            if any(write.owner in self._resident_writes for write in writes):
                logger.warning(
                    "%s: producer has conflicting Alpha concat writes",
                    node.name,
                )
                continue

            for write in writes:
                self._register_resident_write(
                    write,
                    plan.layout.aliases,
                )

    def _prepare_shuffle_input_writes(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        """Give Alpha's dense shuffle branch its resident input pitch."""
        for shuffle in model.graph.nodes:
            if shuffle.target != CHANNEL_SHUFFLE_OPERATOR:
                continue

            input1 = get_named_arg(shuffle, "input1")
            input2 = get_named_arg(shuffle, "input2")
            if not all(isinstance(node, Node) for node in (input1, input2)):
                continue
            shapes = tuple(
                tuple(getattr(node, "shape", ()))
                for node in (input1, input2, shuffle)
            )
            if any(len(shape) != 4 for shape in shapes):
                continue
            if any(shape[0] != 1 for shape in shapes):
                continue
            if shapes[0][2:] != shapes[1][2:]:
                continue
            if shapes[2][1] != shapes[0][1] + shapes[1][1]:
                continue

            owner, path = self._trace_write_owner(
                input2,
                storage_graph,
                storage_plane,
            )
            if not isinstance(owner, Node):
                continue
            if owner not in self._operation_operands:
                continue

            operand_bits = _node_operand_bits(owner)
            granularity = self._concat_channel_granularity(
                owner,
                operand_bits,
            )
            if granularity is None:
                continue
            physical_sizes = tuple(
                math.ceil(shape[1] / granularity) * granularity
                for shape in shapes[:2]
            )
            total_channels = sum(physical_sizes)
            dense_channels = physical_sizes[1]
            if total_channels <= dense_channels:
                continue

            try:
                input1_chain = storage_graph.trace_storage_chain(
                    input1,
                    storage_plane,
                )
            except (KeyError, ValueError):
                input1_chain = ()
            completes_split = any(
                (
                    relation.transform is not None
                    and relation.transform.kind == "alpha_channel_split"
                )
                for ref in input1_chain
                for relation in (storage_graph.relation(ref, storage_plane),)
            )

            existing = self._resident_writes.get(owner)
            if existing is not None:
                if (
                    existing.slot_index == 0
                    and existing.channel_offset == 0
                    and existing.slot_channels == dense_channels
                    and existing.resident_channels == total_channels
                    and existing.concat_channel_index
                    == (0 if completes_split else None)
                ):
                    continue
                raise AlphaMemoryPlanError(
                    f"{owner.name}: shuffle input pitch conflicts with "
                    "an existing concat write"
                )

            write = _ResidentWrite(
                backing=owner,
                owner=owner,
                path=path,
                slot_index=0,
                channel_offset=0,
                slot_channels=dense_channels,
                logical_backing_channels=shapes[1][1],
                resident_channels=total_channels,
                concat_channel_index=0 if completes_split else None,
            )
            self._register_resident_write(
                write,
                (owner,),
            )

    def _prepare_shuffle_plans(self, model):
        block_channels = self.hardware.nmp_lanes(8)
        copies_by_parent = {}
        for node in model.graph.nodes:
            if not _is_shuffle_copy(node):
                continue
            parent = _magnet_alpha_meta(node).get("logical_parent")
            copies_by_parent.setdefault(parent, []).append(node)

        for shuffle in model.graph.nodes:
            if shuffle.target != CHANNEL_SHUFFLE_OPERATOR:
                continue
            channels = shuffle.shape[1]
            channel_blocks = channels // block_channels
            movements = _shuffle_copy_movements(channel_blocks)
            copies = tuple(
                sorted(
                    copies_by_parent.get(shuffle.name, ()),
                    key=lambda node: _magnet_alpha_meta(node)["copy_index"],
                )
            )
            if len(copies) != len(movements):
                raise ValueError(
                    f"{shuffle.name}: expected {len(movements)} Alpha "
                    f"shuffle copies, found {len(copies)}"
                )
            if any(
                _magnet_alpha_meta(copy).get("block_channels") != block_channels
                for copy in copies
            ):
                raise ValueError(
                    f"{shuffle.name}: shuffle copies do not match the "
                    f"configured {block_channels}-channel NMP width"
                )
            plan = _ShufflePlan(
                shuffle,
                copies,
                movements,
                block_channels,
            )
            self._shuffle_plans[shuffle] = plan
            for copy_node, movement in zip(copies, movements):
                self._copy_plans[copy_node] = (plan, movement)

    @staticmethod
    def _channel_split_layout(split_node):
        if not is_tensor_split(split_node):
            return None
        source = split_node.args[0]
        source_value = getattr(source, "value", None)
        outputs = getattr(split_node, "value", None)
        if not isinstance(source_value, torch.Tensor):
            return None
        if not isinstance(outputs, (tuple, list)) or not outputs:
            return None
        if not all(isinstance(output, torch.Tensor) for output in outputs):
            return None

        dim = split_node.kwargs.get("dim")
        if dim is None:
            dim = split_node.args[2] if len(split_node.args) > 2 else 0
        if not isinstance(dim, int):
            return None

        source_shape = tuple(source_value.shape)
        if len(source_shape) != 4:
            return None
        dim %= len(source_shape)
        if source_shape[0] != 1 or dim != 1:
            return None
        if any(
            tuple(output.shape[:dim] + output.shape[dim + 1 :])
            != source_shape[:dim] + source_shape[dim + 1 :]
            for output in outputs
        ):
            return None

        channel_sizes = tuple(int(output.shape[dim]) for output in outputs)
        if any(size <= 0 for size in channel_sizes):
            return None
        if sum(channel_sizes) != source_shape[dim]:
            return None
        return source, tuple(outputs), channel_sizes

    def _prepare_split_relations(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        """Share only channel splits statically encodable by Alpha."""
        memory = self.memory_level.memory(self.memory_level.output_memory)
        for split_node in model.graph.nodes:
            layout = self._channel_split_layout(split_node)
            if layout is None:
                continue
            source, outputs, channel_sizes = layout
            # Static parameters do not distinguish a get_attr weight from a
            # scale. Leave those values in the generic split container.
            if source.op == "get_attr":
                continue

            operand_bits = _node_operand_bits(source)
            overrides = {
                value
                for value in (self._cluster_constraints.get(source),)
                if value is not None
            }
            if len(overrides) > 1:
                raise AlphaMemoryPlanError(
                    f"{source.name}: conflicting Alpha cluster overrides "
                    f"{sorted(overrides)}"
                )
            cluster = (
                int(next(iter(overrides)))
                if overrides
                else self.hardware.dtype_mode(operand_bits).banks_per_value
            )
            logical_entry_bits = cluster * memory.bits_per_bank_entry
            if logical_entry_bits % operand_bits:
                continue
            channels_per_entry = logical_entry_bits // operand_bits
            if any(size % channels_per_entry for size in channel_sizes):
                continue

            split_outputs = storage_graph.index.node_outputs.get(
                split_node,
                (),
            )
            if len(split_outputs) != len(outputs):
                continue
            source_ref = self._single_storage_ref(
                storage_graph,
                source,
            )
            channel_offset = 0
            for output_ref, channel_size in zip(
                split_outputs,
                channel_sizes,
            ):
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    output_ref,
                    StorageRelationKind.VIEW,
                    source=source_ref,
                    transform=ViewTransform(
                        "alpha_channel_split",
                        parameters=(
                            ("channel_offset", channel_offset),
                            ("channel_size", channel_size),
                        ),
                    ),
                )
                channel_offset += channel_size
            _mark_alpha_address_only(split_node)

    def _physical_row_pitch_channels(
        self,
        operand,
        storage_plan,
        storage_plane,
    ):
        logical = storage_plan.ref(operand)
        shape = tuple(storage_plan.logical_tensors[logical].shape)
        if len(shape) != 4:
            raise AlphaMemoryPlanError(
                f"{operand.name}: NMP physical pitch requires rank four"
            )
        physical = storage_plan.physical_tensor(
            logical,
            storage_plane,
        )
        relation = storage_plan.relation(logical, storage_plane)
        component_name = relation.component or "primary"
        dense_primary_layout = (
            physical.layout == ALPHA_VSQ_LAYOUT and component_name == "primary"
        )
        if (
            physical.layout not in {"dense", ALPHA_DENSE_LAYOUT}
            and not dense_primary_layout
        ):
            raise AlphaMemoryPlanError(
                f"{operand.name}: NMP pitch cannot be inferred from "
                f"physical layout {physical.layout!r}"
            )
        rows = int(shape[0]) * int(shape[2]) * int(shape[3])
        component = physical.component(component_name)
        operand_bits = (
            component.bits
            if isinstance(operand, ValueRef)
            else _node_operand_bits(operand)
        )
        if component.bits != operand_bits:
            raise AlphaMemoryPlanError(
                f"{operand.name}: physical component width "
                f"{component.bits} does not match {operand_bits}-bit NMP"
            )
        row_divisor = rows * operand_bits
        backing_bits = (
            storage_plan.backing_size_bytes(
                operand,
                storage_plane,
            )
            * 8
        )
        if rows <= 0 or backing_bits % row_divisor:
            raise AlphaMemoryPlanError(
                f"{operand.name}: physical backing is not an integral "
                "NCHW channel pitch"
            )
        return backing_bits // row_divisor

    def _nmp_input_view_fact(
        self,
        operand,
        storage_plan,
        storage_plane,
        *,
        include_resident,
    ):
        """Describe one strided NMP input view from finalized storage."""
        try:
            chain = storage_plan.trace_storage_chain(
                operand,
                storage_plane,
            )
        except (KeyError, ValueError):
            return None

        split_ref = None
        split_relation = None
        shuffle_output_relation = None
        resident_parameters = None
        for ref in chain:
            relation = storage_plan.relation(ref, storage_plane)
            transform = relation.transform
            if (
                transform is not None
                and transform.kind == "alpha_resident_write"
            ):
                resident_parameters = dict(transform.parameters)
            if (
                transform is not None
                and transform.kind == "alpha_channel_split"
                and relation.source is not None
            ):
                split_ref = ref
                split_relation = relation
                break
            if (
                transform is not None
                and transform.kind == "alpha_shuffle_output"
            ):
                shuffle_output_relation = relation

        logical = storage_plan.ref(operand)
        shape = tuple(storage_plan.logical_tensors[logical].shape)
        if len(shape) != 4:
            return None
        logical_channels = int(shape[1])
        resident_group_count = None
        uniform_group_channels = None

        if split_relation is not None:
            physical_channels = self._physical_row_pitch_channels(
                operand,
                storage_plan,
                storage_plane,
            )
            parameters = dict(split_relation.transform.parameters)
            channel_size = int(parameters["channel_size"])
            if channel_size != logical_channels:
                raise AlphaMemoryPlanError(
                    f"{operand.name}: split view shape does not match its "
                    "storage relation"
                )
            channel_offset = int(parameters["channel_offset"])
            split_index = (
                int(split_ref.path[0])
                if split_ref.path
                else channel_offset // channel_size
            )
            source_kind = "channel_split"
        elif shuffle_output_relation is not None:
            physical = storage_plan.physical_tensor(
                logical,
                storage_plane,
            )
            relation = storage_plan.relation(logical, storage_plane)
            component = physical.component(relation.component or "primary")
            component_shape = tuple(component.shape)
            if len(component_shape) != 4:
                return None
            physical_channels = int(component_shape[1])
            split_index = 0
            source_kind = "shuffle_output"
        elif include_resident and resident_parameters is not None:
            physical_channels = self._physical_row_pitch_channels(
                operand,
                storage_plan,
                storage_plane,
            )
            if physical_channels <= logical_channels:
                return None
            split_index = int(resident_parameters["slot_index"])
            source_kind = "resident"
            physical = storage_plan.physical_tensor(
                logical,
                storage_plane,
            )
            slots = {}
            for resident_logical in physical.logical_tensors:
                relation = storage_plan.relation(
                    resident_logical,
                    storage_plane,
                )
                transform = relation.transform
                if (
                    transform is None
                    or transform.kind != "alpha_resident_write"
                ):
                    continue
                parameters = dict(transform.parameters)
                slots[int(parameters["slot_index"])] = int(
                    parameters["slot_channels"]
                )
            if slots:
                resident_group_count = len(slots)
                widths = set(slots.values())
                if len(widths) == 1:
                    uniform_group_channels = next(iter(widths))
        else:
            return None

        lanes = self.hardware.nmp_lanes(_node_operand_bits(operand))
        pitch_gap_channels = physical_channels - logical_channels
        if pitch_gap_channels < 0 or pitch_gap_channels % lanes:
            raise AlphaMemoryPlanError(
                f"{operand.name}: split row-pitch gap "
                f"{pitch_gap_channels} is not representable in "
                f"{lanes}-channel NMP words"
            )
        return _NmpInputView(
            source_kind=source_kind,
            channel_index=1,
            split_index=split_index,
            current_channels=logical_channels,
            total_channels=physical_channels,
            pitch_gap_words=pitch_gap_channels // lanes,
            resident_group_count=resident_group_count,
            uniform_group_channels=uniform_group_channels,
        )

    def _prepare_nmp_storage_fields(
        self,
        storage_plan,
        storage_plane,
    ):
        """Serialize NMP address-generator facts from the physical plan."""
        for operation, info in self._operation_operands.items():
            kind = getattr(info, "kind", None)
            if kind is None:
                continue
            target_kwargs = operation.meta.setdefault(
                TARGET_CODEGEN_KWARGS,
                {},
            )
            for field in (
                "is_from_split",
                "split_channel_idx",
                "split_cur_size",
                "split_idx",
                "split_scale",
                "split_total_size",
                "nmp_c2",
                "nmp_k2",
                "nmp_groups",
                "nmp_channels_per_group",
            ):
                target_kwargs.pop(field, None)

            nmp_c2 = 0
            nmp_k2 = 0
            copy_plan = self._copy_plans.get(operation)
            if copy_plan is not None:
                plan, _movement = copy_plan
                pitch_entries = self._shuffle_access_pattern(
                    plan.shuffle
                ).pitch_entries
                nmp_c2 = pitch_entries - 1
                nmp_k2 = pitch_entries - 1
            elif kind == "pool2d":
                nmp_k2 = self._nmp_output_pitch_gap_words(
                    operation,
                    storage_plan,
                    storage_plane,
                )
            elif kind == "upsample_nearest2d":
                input_node = info.operand("input")
                output_payload = target_kwargs["output_payload"]
                output_node = output_payload.node
                nmp_c2 = self._nmp_pitch_gap_words(
                    input_node,
                    storage_plan,
                    storage_plane,
                )
                nmp_k2 = self._nmp_pitch_gap_words(
                    output_node,
                    storage_plan,
                    storage_plane,
                )
            elif kind == "channel_shuffle":
                output_payload = target_kwargs["output_payload"]
                nmp_k2 = self._nmp_pitch_gap_words(
                    output_payload.node,
                    storage_plan,
                    storage_plane,
                )

            include_resident = kind in {"gemm", "depthwise_conv2d"}
            split_facts = [
                fact
                for operand in info.operands
                if (
                    fact := self._nmp_input_view_fact(
                        operand,
                        storage_plan,
                        storage_plane,
                        include_resident=include_resident,
                    )
                )
                is not None
            ]
            split_facts = list(dict.fromkeys(split_facts))
            if len(split_facts) > 1:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: multiple split storage views "
                    "cannot share one NMP descriptor"
                )
            if split_facts:
                split = split_facts[0]
                split_index = split.split_index
                if split.source_kind == "channel_split":
                    if kind == "gemm":
                        split_index = 0
                    elif kind == "channel_shuffle":
                        split_index = 1
                target_kwargs.update(
                    {
                        "is_from_split": True,
                        "split_channel_idx": split.channel_index,
                        "split_cur_size": split.current_channels,
                        "split_idx": split_index,
                        "split_scale": split.scale,
                        "split_total_size": split.total_channels,
                    }
                )
                if kind != "gemm":
                    nmp_c2 = split.pitch_gap_words
                if (
                    split.source_kind == "resident"
                    and split.resident_group_count is not None
                    and split.resident_group_count > 1
                ):
                    target_kwargs["nmp_groups"] = split.resident_group_count
                    if split.uniform_group_channels is not None:
                        target_kwargs["nmp_channels_per_group"] = (
                            split.uniform_group_channels
                        )

            if kind == "depthwise_conv2d" and "nmp_groups" not in (
                target_kwargs
            ):
                weight = info.operand("weight")
                logical_weight_shape = get_hardware_logical_shape(weight)
                if logical_weight_shape:
                    target_kwargs["nmp_groups"] = int(logical_weight_shape[0])

            if nmp_c2:
                target_kwargs["nmp_c2"] = nmp_c2
            if nmp_k2:
                target_kwargs["nmp_k2"] = nmp_k2
            if kind in {
                "depthwise_conv2d",
                "channel_shuffle",
                "upsample_nearest2d",
            }:
                target_kwargs["padding_c"] = 0
                target_kwargs["padding_k"] = 0

    def _nmp_output_pitch_gap_words(
        self,
        node,
        storage_plan,
        storage_plane,
    ):
        """Return physical minus logical output pitch in NMP entries."""
        logical = storage_plan.ref(node)
        logical_channels = int(storage_plan.logical_tensors[logical].shape[1])
        physical_channels = self._physical_row_pitch_channels(
            node,
            storage_plan,
            storage_plane,
        )
        lanes = self.hardware.nmp_lanes(_node_operand_bits(node))
        if logical_channels <= 0 or physical_channels < logical_channels:
            raise AlphaMemoryPlanError(
                f"{node.name}: NMP output pitch {physical_channels} "
                f"cannot contain logical C={logical_channels}"
            )
        logical_entries = (logical_channels + lanes - 1) // lanes
        physical_entries = (physical_channels + lanes - 1) // lanes
        return physical_entries - logical_entries

    def _validate_pool_output_pitch(self, operation):
        expected_gap = self._nmp_output_pitch_gap_words(
            operation,
            self._storage_plan,
            self._storage_plane,
        )
        serialized_gap = operation.meta.get(
            TARGET_CODEGEN_KWARGS,
            {},
        ).get("nmp_k2", 0)
        if type(serialized_gap) is not int or serialized_gap != expected_gap:
            raise AlphaMemoryPlanError(
                f"{operation.name}: serialized MaxPool output pitch gap "
                f"{serialized_gap!r} does not match physical gap "
                f"{expected_gap}"
            )

    def _nmp_pitch_gap_words(
        self,
        node,
        storage_plan,
        storage_plane,
    ):
        """Return physical row pitch minus logical channels in NMP words."""
        logical = storage_plan.ref(node)
        logical_channels = int(storage_plan.logical_tensors[logical].shape[1])
        resident_channels = self._physical_row_pitch_channels(
            node,
            storage_plan,
            storage_plane,
        )
        gap = resident_channels - logical_channels
        lanes = self.hardware.nmp_lanes(_node_operand_bits(node))
        if gap < 0 or gap % lanes:
            raise AlphaMemoryPlanError(
                f"{node.name}: NMP physical pitch {resident_channels} "
                f"cannot represent logical C={logical_channels} in "
                f"{lanes}-channel words"
            )
        return gap // lanes

    @staticmethod
    def _is_embedded_fine_scale(scale, payload):
        if not all(isinstance(node, Node) for node in (scale, payload)):
            return False
        if (scale.op == "get_attr") != (payload.op == "get_attr"):
            return False
        if not all(
            isinstance(getattr(node, "value", None), torch.Tensor)
            for node in (scale, payload)
        ):
            return False
        return _is_alpha_vsq_storage_pair(
            scale.value,
            payload.value,
            scale.meta.get("dtype"),
            payload.meta.get("dtype"),
        )

    def _record_embedded_fine_scale(
        self,
        scale,
        payload,
        storage_graph,
        storage_plane,
    ):
        if not self._is_embedded_fine_scale(scale, payload):
            return

        pending = [(scale, payload)]
        seen = set()
        while pending:
            scale_path, payload_path = pending.pop()
            pair = (scale_path, payload_path)
            if pair in seen or not self._is_embedded_fine_scale(*pair):
                continue
            seen.add(pair)

            relation = self._set_alpha_relation(
                storage_graph,
                storage_plane,
                scale_path,
                StorageRelationKind.COMPONENT,
                source=payload_path,
                component="fine_scale",
            )
            if relation.kind is not StorageRelationKind.COMPONENT:
                continue
            _mark_alpha_address_only(scale_path)

            if is_address_only_view(scale_path) and is_address_only_view(
                payload_path
            ):
                pending.append((scale_path.args[0], payload_path.args[0]))
                continue

            if scale_path.target != payload_path.target:
                continue
            if scale_path.target not in (
                torch.ops.aten.cat.default,
                torch.ops.aten.stack.default,
            ):
                continue
            scale_inputs = scale_path.args[0]
            payload_inputs = payload_path.args[0]
            if len(scale_inputs) != len(payload_inputs):
                continue
            pending.extend(zip(scale_inputs, payload_inputs))

    def _register_vsq_backing(
        self,
        fine_scale,
        payload,
        storage_graph,
        storage_plane,
    ):
        """Declare Alpha's fine scale as a component of payload storage."""
        payload_ref = storage_graph.ref(payload)
        fine_ref = storage_graph.ref(fine_scale)
        payload_relation = storage_graph.relation(
            payload_ref,
            storage_plane,
        )
        if payload_relation.kind not in {
            StorageRelationKind.MATERIALIZE,
            StorageRelationKind.COPY,
        }:
            return False

        def logical_value(ref):
            value = getattr(ref.node, "value", None)
            try:
                for index in ref.path:
                    value = value[index]
            except (IndexError, TypeError):
                return None
            return value

        payload_value = logical_value(payload_ref)
        fine_value = logical_value(fine_ref)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (payload_value, fine_value)
        ):
            raise AlphaMemoryPlanError(
                f"{payload_ref.name}: VSQ backing has no concrete payload"
            )
        payload_dtype = storage_graph.index.logical_tensors[payload_ref].dtype
        fine_dtype = storage_graph.index.logical_tensors[fine_ref].dtype
        payload_bits = _normalize_operand_bits(get_dtype_width(payload_dtype))
        fine_bits = _normalize_operand_bits(get_dtype_width(fine_dtype))
        backing_size = payload_size_bytes(payload_value, payload_bits)
        payload_component = self._physical_component_name(
            storage_graph,
            payload_ref,
            storage_plane,
        )
        fine_component = self._physical_component_name(
            storage_graph,
            fine_ref,
            storage_plane,
        )
        storage_graph.register_physical_component(
            payload_ref,
            payload_value,
            plane=storage_plane,
            component=payload_component,
            dtype=payload_dtype,
            bits=payload_bits,
            backing_size_bytes=backing_size,
            layout=ALPHA_VSQ_LAYOUT,
        )
        storage_graph.register_physical_component(
            fine_ref,
            fine_value,
            plane=storage_plane,
            component=fine_component,
            dtype=fine_dtype,
            bits=fine_bits,
            backing_size_bytes=backing_size,
            layout=ALPHA_VSQ_LAYOUT,
        )
        return True

    def _register_vsq_backings(
        self,
        storage_graph,
        storage_plane,
    ):
        """Register VSQ components after all physical relations are known."""
        for fine_scale in storage_graph.index.logical_tensors:
            relation = storage_graph.relation(
                fine_scale,
                storage_plane,
            )
            if (
                relation.kind is not StorageRelationKind.COMPONENT
                or relation.component != "fine_scale"
                or relation.source is None
            ):
                continue
            self._register_vsq_backing(
                fine_scale,
                relation.source,
                storage_graph,
                storage_plane,
            )

    def _prepare_vsq_storage(
        self,
        model,
        storage_graph,
        storage_plane,
    ):
        for node in model.graph.nodes:
            if self._vsq_tuple_payload(node) is None:
                continue
            coarse_scale, fine_scale, payload = self._vsq_storage_refs(
                storage_graph,
                node,
            )
            target_kwargs = node.meta.setdefault(
                TARGET_CODEGEN_KWARGS,
                {},
            )
            target_kwargs["output_payload"] = payload
            target_kwargs["output_vector_scale"] = fine_scale
            # The payload is the physical result produced by PVS. Its fine
            # scale is embedded in that allocation, while the returned coarse
            # scale is the unchanged logical input and therefore aliases its
            # source when that source is visible at this graph level.
            storage_graph.set_relation(
                payload,
                StorageRelationKind.MATERIALIZE,
                plane=storage_plane,
            )
            coarse_source = None
            contains_pvs = node.target == PVS_QUANTIZE_OPERATOR
            if node.target == PVS_QUANTIZE_OPERATOR:
                coarse_source = get_named_arg(node, "coarse_scale")
            elif node.op == "call_module":
                submodule = model.get_submodule(node.target)
                inner_pvs = [
                    inner
                    for inner in submodule.graph.nodes
                    if inner.target == PVS_QUANTIZE_OPERATOR
                ]
                contains_pvs = bool(inner_pvs)
                inner_sources = [
                    descriptor.tensor
                    for inner in inner_pvs
                    for descriptor in submodule_operand_descriptors(
                        node,
                        inner,
                        model,
                        operator_traits=self._operator_traits,
                    )
                    if descriptor.key == "coarse_scale"
                ]
                visible_sources = [
                    source
                    for source in inner_sources
                    if isinstance(source, Node)
                ]
                if len(visible_sources) == 1:
                    coarse_source = visible_sources[0]
            if contains_pvs and not isinstance(coarse_source, Node):
                raise AlphaMemoryPlanError(
                    f"{node.name}: PVS coarse scale must resolve to one "
                    "visible graph tensor so output[0] can alias it"
                )
            coarse_source_outputs = (
                storage_graph.index.node_outputs.get(
                    coarse_source,
                    (),
                )
                if isinstance(coarse_source, Node)
                else ()
            )
            if contains_pvs and len(coarse_source_outputs) != 1:
                raise AlphaMemoryPlanError(
                    f"{node.name}: PVS coarse-scale source must have one "
                    "logical tensor result"
                )
            if (
                isinstance(coarse_source, Node)
                and len(coarse_source_outputs) == 1
            ):
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    coarse_scale,
                    StorageRelationKind.ALIAS,
                    source=coarse_source,
                )
            self._set_alpha_relation(
                storage_graph,
                storage_plane,
                fine_scale,
                StorageRelationKind.COMPONENT,
                source=payload,
                component="fine_scale",
            )

        target = VSQ_CONV2D_OPERATOR
        for operation in model.graph.nodes:
            if operation.target == target:
                self._record_embedded_fine_scale(
                    get_named_arg(operation, "input_scale"),
                    get_named_arg(operation, "input"),
                    storage_graph,
                    storage_plane,
                )
                self._record_embedded_fine_scale(
                    get_named_arg(operation, "weight_scale"),
                    get_named_arg(operation, "weight"),
                    storage_graph,
                    storage_plane,
                )
                continue
            if operation.op != "call_module":
                continue

            submodule = model.get_submodule(operation.target)
            for inner in submodule.graph.nodes:
                if inner.target != target:
                    continue
                descriptors = submodule_operand_descriptors(
                    operation,
                    inner,
                    model,
                    operator_traits=self._operator_traits,
                )
                operands = {
                    descriptor.key: descriptor.tensor
                    for descriptor in descriptors
                }
                scale = operands.get("weight_scale")
                payload = operands.get("weight")
                input_scale = operands.get("input_scale")
                input_payload = operands.get("input")
                self._record_embedded_fine_scale(
                    input_scale,
                    input_payload,
                    storage_graph,
                    storage_plane,
                )
                self._record_embedded_fine_scale(
                    scale,
                    payload,
                    storage_graph,
                    storage_plane,
                )

    @staticmethod
    def _vsq_tuple_payload(node):
        values = getattr(node, "value", None)
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            return None
        dtypes = node.meta.get("dtype")
        if not isinstance(dtypes, (tuple, list)) or len(dtypes) != 3:
            return None
        if not _is_alpha_vsq_storage_pair(
            values[1],
            values[2],
            dtypes[1],
            dtypes[2],
        ):
            return None
        return values[2]

    @staticmethod
    def _resident_backing_numel(write):
        value = getattr(write.backing, "value", None)
        if not isinstance(value, torch.Tensor):
            raise AlphaMemoryPlanError(
                f"{write.backing.name}: resident backing has no tensor value"
            )
        logical_numel = value.numel()
        if logical_numel % write.logical_backing_channels:
            raise AlphaMemoryPlanError(
                f"{write.backing.name}: resident backing shape is not "
                "divisible by its logical channel extent"
            )
        return (
            logical_numel
            // write.logical_backing_channels
            * write.resident_channels
        )

    def _register_resident_backings(
        self,
        storage_graph,
        storage_plane,
    ):
        """Make resident channel pitch part of the physical tensor plan."""
        declarations = {}
        for write in self._resident_writes.values():
            backing = self._address_source_ref(
                storage_graph,
                write.backing,
            )
            operand_bits = _node_operand_bits(write.backing)
            backing_size = payload_size_bytes(
                self._resident_backing_numel(write),
                operand_bits,
            )
            declaration = (
                write.logical_backing_channels,
                write.resident_channels,
                operand_bits,
                backing_size,
            )
            previous = declarations.get(backing)
            if previous is not None and previous != declaration:
                raise AlphaMemoryPlanError(
                    f"{write.backing.name}: inconsistent resident backing "
                    "declarations"
                )
            declarations[backing] = declaration

        for backing, (
            logical_channels,
            resident_channels,
            operand_bits,
            backing_size,
        ) in declarations.items():
            if logical_channels == resident_channels:
                continue
            domain = get_physical_quantization_contract(backing.node)
            if domain is not None and domain.companion_dtypes:
                raise AlphaMemoryPlanError(
                    f"{backing.name}: padded resident storage with "
                    "companion components is unsupported"
                )
            value = getattr(backing.node, "value", None)
            if not isinstance(value, torch.Tensor):
                raise AlphaMemoryPlanError(
                    f"{backing.name}: resident backing has no tensor value"
                )
            component = self._physical_component_name(
                storage_graph,
                backing,
                storage_plane,
            )
            storage_graph.register_physical_component(
                backing,
                value,
                plane=storage_plane,
                component=component,
                dtype=backing.node.meta.get("dtype", value.dtype),
                bits=operand_bits,
                backing_size_bytes=backing_size,
                layout=ALPHA_DENSE_LAYOUT,
            )

    def _declare_storage_relations(
        self,
        model: GraphModule,
        storage_graph: StorageGraph,
        storage_plane: str,
    ) -> None:
        """Project late Alpha resident-write and shuffle relations."""
        for write in self._resident_writes.values():
            owner_ref = self._address_source_ref(
                storage_graph,
                write.owner,
            )
            backing_ref = self._address_source_ref(
                storage_graph,
                write.backing,
            )
            resident_transform = ViewTransform(
                "alpha_resident_write",
                parameters=(
                    ("channel_offset", write.channel_offset),
                    ("slot_index", write.slot_index),
                    ("slot_channels", write.slot_channels),
                    ("resident_channels", write.resident_channels),
                )
                + (
                    ()
                    if write.concat_channel_index is None
                    else (
                        (
                            "concat_channel_index",
                            write.concat_channel_index,
                        ),
                    )
                ),
            )
            if owner_ref == backing_ref:
                previous = storage_graph.relation(
                    owner_ref,
                    storage_plane,
                )
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    owner_ref,
                    previous.kind,
                    source=previous.source,
                    transform=resident_transform,
                    component=previous.component,
                )
            else:
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    owner_ref,
                    StorageRelationKind.WRITE_INTO,
                    source=backing_ref,
                    transform=resident_transform,
                )
            for path_node in write.path:
                if _is_shuffle_copy(path_node):
                    continue
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    path_node,
                    StorageRelationKind.VIEW,
                    source=backing_ref,
                    transform=ViewTransform("alpha_resident_view"),
                    allow_copy_override=True,
                )

        self._register_resident_backings(
            storage_graph,
            storage_plane,
        )

        for copy_node in model.graph.nodes:
            if (
                not _is_shuffle_copy(copy_node)
                or copy_node in self._copy_plans
                or not copy_node.all_input_nodes
            ):
                continue
            self._set_alpha_relation(
                storage_graph,
                storage_plane,
                copy_node,
                StorageRelationKind.VIEW,
                source=copy_node.all_input_nodes[0],
                transform=ViewTransform("alpha_orphan_shuffle_forwarding"),
            )

        for shuffle, plan in self._shuffle_plans.items():
            secondary = storage_graph.add_auxiliary(
                shuffle,
                "output[1]",
                plane=storage_plane,
            )
            value = getattr(shuffle, "value", None)
            if not isinstance(value, torch.Tensor):
                raise AlphaMemoryPlanError(
                    f"{shuffle.name}: shuffle has no concrete payload"
                )
            operand_bits = _node_operand_bits(shuffle)
            memory = self.memory_level.memory(self.memory_level.output_memory)
            cluster = self._cluster_for(
                shuffle,
                TensorRole.OUTPUT,
                memory.name,
                operand_bits,
            )
            entry_bytes = cluster * memory.bits_per_bank_entry // 8
            resident_write = self._resident_writes.get(shuffle)
            resident_channels = (
                resident_write.resident_channels
                if resident_write is not None
                else int(shuffle.shape[1])
            )
            backing_size = (
                self._shuffle_access_pattern(
                    shuffle,
                    resident_channels=resident_channels,
                ).span_entries
                * entry_bytes
            )
            storage_graph.register_physical_component(
                secondary,
                value,
                plane=storage_plane,
                component="primary",
                dtype=shuffle.meta.get("dtype", value.dtype),
                bits=operand_bits,
                backing_size_bytes=backing_size,
                layout=ALPHA_SHUFFLE_SECONDARY_LAYOUT,
            )
            storage_graph.add_access(
                secondary,
                shuffle,
                plane=storage_plane,
                role=TensorRole.OUTPUT,
                kind="write",
            )
            for copy_node, movement in zip(
                plan.copies,
                plan.movements,
            ):
                self._set_alpha_relation(
                    storage_graph,
                    storage_plane,
                    copy_node,
                    StorageRelationKind.WRITE_INTO,
                    source=shuffle,
                    transform=ViewTransform(
                        "alpha_shuffle_copy",
                        parameters=(("copy_index", movement.index),),
                    ),
                )
                if movement.source_output == 1:
                    storage_graph.add_access(
                        secondary,
                        copy_node,
                        plane=storage_plane,
                        role=TensorRole.ACTIVATION,
                        kind="read",
                    )

            if plan.movements:
                continue
            for user in shuffle.users:
                if not is_tensor_split(user):
                    continue
                outputs = storage_graph.index.node_outputs.get(user, ())
                if len(outputs) != 2:
                    continue
                self._bind_alpha_physical(
                    storage_graph,
                    storage_plane,
                    outputs[1],
                    secondary,
                    transform=ViewTransform(
                        "alpha_shuffle_output",
                        parameters=(("output", 1),),
                    ),
                )

    def _detach_operation_outputs_from_aggregates(
        self,
        storage_graph,
        storage_plane,
    ):
        """Materialize call outputs that drain between operation scopes."""
        for operation in self._operation_operands:
            for output in storage_graph.index.node_outputs.get(
                operation,
                (),
            ):
                relation = storage_graph.relation(
                    output,
                    storage_plane,
                )
                transform = relation.transform
                if transform is None or transform.kind != "aggregate_slot":
                    continue
                storage_graph.set_relation(
                    output,
                    StorageRelationKind.MATERIALIZE,
                    plane=storage_plane,
                )

    @staticmethod
    def _returns_mutable_second_operand(node):
        """Return whether an operator writes and returns its second input."""
        schema = getattr(node.target, "_schema", None)
        if schema is None or len(schema.returns) != 1:
            return False
        input2 = next(
            (
                argument
                for argument in schema.arguments
                if argument.name == "input2"
            ),
            None,
        )
        if input2 is None:
            return False
        input_alias = input2.alias_info
        output_alias = schema.returns[0].alias_info
        if input_alias is None or output_alias is None:
            return False
        return (
            input_alias.is_write
            and output_alias.is_write
            and bool(
                input_alias.after_set.intersection(
                    output_alias.after_set,
                )
            )
        )

    def _prepare_eadd_output_alias(
        self,
        model,
        operation,
        info,
        storage_graph,
        storage_plane,
    ):
        """Bind one fused EADD output to its mutable second operand."""
        if operation.op != "call_module":
            return
        submodule = model.get_submodule(str(operation.target))
        eadds = tuple(
            node
            for node in submodule.graph.nodes
            if node.target == SATURATING_ADD_OPERATOR
        )
        if not eadds:
            return
        if len(eadds) != 1:
            raise AlphaMemoryPlanError(
                f"{operation.name}: one Alpha call contains "
                f"{len(eadds)} EADD operations"
            )
        eadd = eadds[0]
        input1 = get_named_arg(eadd, "input1")
        input2 = get_named_arg(eadd, "input2")
        if not all(isinstance(node, Node) for node in (input1, input2)):
            raise AlphaMemoryPlanError(
                f"{operation.name}: Alpha EADD operands did not resolve"
            )
        ppu = input1
        accumulator = get_named_arg(ppu, "input")
        if (
            ppu.target != PPU_FIXED_RELU_OPERATOR
            or accumulator is not info.implementation
            or get_named_arg(ppu, "relu") is not False
            or get_named_arg(ppu, "output_precision") != 8
            or get_named_arg(ppu, "integer_accumulator") is not True
            or set(accumulator.users) != {ppu}
            or set(ppu.users) != {eadd}
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: EADD must follow one INT8 PE "
                "accumulator and a non-ReLU O8 integer PPU"
            )
        if not self._returns_mutable_second_operand(eadd):
            raise AlphaMemoryPlanError(
                f"{operation.name}: EADD must return mutable input2"
            )

        output = next(
            node for node in submodule.graph.nodes if node.op == "output"
        )
        if output.args[0] is not eadd or set(eadd.users) != {output}:
            raise AlphaMemoryPlanError(
                f"{operation.name}: fused EADD must be the call output"
            )
        descriptors = submodule_operand_descriptors(
            operation,
            eadd,
            model,
            operator_traits=self._operator_traits,
        )
        outer_inputs = tuple(
            dict.fromkeys(
                descriptor.tensor
                for descriptor in descriptors
                if descriptor.key == "input2"
            )
        )
        if len(outer_inputs) != 1:
            raise AlphaMemoryPlanError(
                f"{operation.name}: EADD input2 does not resolve to one "
                "outer tensor"
            )
        outer_input2 = outer_inputs[0]
        if tuple(operation.shape) != tuple(outer_input2.shape):
            raise AlphaMemoryPlanError(
                f"{operation.name}: EADD output and input2 shapes differ"
            )
        output_ref = self._single_storage_ref(storage_graph, operation)
        input2_ref = self._single_storage_ref(
            storage_graph,
            outer_input2,
        )
        storage_graph.set_relation(
            output_ref,
            StorageRelationKind.WRITE_INTO,
            plane=storage_plane,
            source=input2_ref,
            transform=ViewTransform("alpha_eadd_operand2"),
        )
        self._operation_eadd_inputs[operation] = input2_ref

    def prepare_graph(
        self,
        model: GraphModule,
        *,
        storage_graph: StorageGraph | None = None,
        storage_plane: str = "scratchpad",
        persistent_storage: bool = True,
    ):
        """Recognize Alpha constraints and declare physical relations."""
        self._ensure_mutable()
        if self._model is not None and self._model is not model:
            raise AlphaMemoryPlanError(
                "Alpha allocator is already prepared for another model"
            )
        self._model = model
        owns_storage_graph = storage_graph is None
        if storage_graph is None:
            storage_graph = StorageGraph(model, (storage_plane,))
        self._storage_plane = storage_plane
        self._prepare_physical_quantization_contracts(
            model,
            storage_graph,
            storage_plane,
        )
        self._prepare_vsq_storage(
            model,
            storage_graph,
            storage_plane,
        )
        self._prepare_padding_aliases(
            model,
            storage_graph,
            storage_plane,
        )
        self._prepare_execution_constraints(
            model,
            storage_graph,
            storage_plane,
        )
        for operation, info in self._operation_operands.items():
            self._prepare_eadd_output_alias(
                model,
                operation,
                info,
                storage_graph,
                storage_plane,
            )
        if not persistent_storage:
            self._detach_operation_outputs_from_aggregates(
                storage_graph,
                storage_plane,
            )
        self._prepare_split_relations(
            model,
            storage_graph,
            storage_plane,
        )
        if persistent_storage:
            self._prepare_concat_writes(
                model,
                storage_graph,
                storage_plane,
            )
            self._prepare_shuffle_input_writes(
                model,
                storage_graph,
                storage_plane,
            )
            self._prepare_shuffle_plans(model)
        self._declare_storage_relations(
            model,
            storage_graph,
            storage_plane,
        )
        self._register_vsq_backings(
            storage_graph,
            storage_plane,
        )
        self._prepared_plan = freeze_prepared_memory_plan(
            persistent_storage=persistent_storage,
            operation_constraints=self._operation_operands,
            eadd_inputs=self._operation_eadd_inputs,
            materialized_inputs=self._materialized_input_padding,
            payload_uses=self._physical_payload_uses,
            resident_writes=self._resident_writes,
            shuffle_plans=self._shuffle_plans,
            copy_plans=self._copy_plans,
            cluster_constraints=self._cluster_constraints,
            bank_conflicts=self._bank_conflicts,
        )
        if owns_storage_graph:
            self.bind_storage_plan(
                storage_graph.finalize(),
                storage_plane,
            )

    def validate_end_to_end_operations(self, model: GraphModule) -> None:
        """Reject unsupported Alpha calls before model-wide allocation."""
        planned_operations = (
            self._prepared_plan.operation_nodes
            if self._prepared_plan is not None
            else frozenset(self._operation_operands)
        )
        boundaries = set()
        for operation in model.graph.nodes:
            if self._ingress_quantizer_owner(operation) is not None:
                boundaries.add(operation)
            closure = self._terminal_egress_boundary(operation)
            if closure is not None:
                boundaries.update(closure.boundary_nodes)
        unsupported = tuple(
            operation
            for operation in model.graph.nodes
            if operation.op in {"call_function", "call_module"}
            and operation.target != operator.getitem
            and not is_nop(operation)
            and not is_addressing_op(operation)
            and not operation.meta.get(TARGET_INSTRUCTION_NOP, False)
            and operation not in planned_operations
            and operation not in boundaries
        )
        if unsupported:
            raise AlphaMemoryPlanError(
                "Alpha end-to-end scratchpad mapping does not support: "
                + ", ".join(
                    f"{operation.name} ({operation.target})"
                    for operation in unsupported
                )
            )

    def operation_constraints(self):
        """Return the selected execution constraints consumed by codegen."""
        if self._prepared_plan is not None:
            return self._prepared_plan.operation_constraints
        return tuple(self._operation_operands.items())

    @property
    def prepared_memory_plan(self) -> AlphaPreparedMemoryPlan:
        """Return immutable target planning facts before bank allocation."""
        if self._prepared_plan is None:
            raise RuntimeError("Alpha memory graph has not been prepared")
        return self._prepared_plan

    def bind_storage_plan(
        self,
        plan: StoragePlan,
        plane: str,
    ) -> None:
        """Bind allocation and validation to shared physical identities."""
        plane = getattr(plane, "value", plane)
        if plane not in plan.relations:
            raise ValueError(f"Storage plan has no {plane} plane")
        if self._storage_plan is not None:
            if self._storage_plan is plan and self._storage_plane == plane:
                return
            raise AlphaMemoryPlanError(
                "Alpha scratchpad storage plan is already bound"
            )
        if self._allocations:
            raise AlphaMemoryPlanError(
                "Alpha storage plan must be bound before allocation"
            )
        self._storage_plan = plan
        self._storage_plane = plane
        payload_uses = {}
        for use in self._physical_payload_uses:
            physical = plan.storage_key(use.logical_tensor, plane)
            plan.physical_component(
                physical,
                plane,
                use.component,
            )
            payload_uses.setdefault(physical, []).append(use)
        self._payload_uses_by_physical = {
            physical: tuple(uses) for physical, uses in payload_uses.items()
        }
        self._physical_payload_uses.clear()
        self._resident_writes.clear()
        self._prepare_nmp_storage_fields(plan, plane)
        self._bind_physical_constraints()
        for operation, input2 in self._operation_eadd_inputs.items():
            output_physical = plan.physical_id(operation, plane)
            input2_physical = plan.physical_id(input2, plane)
            if output_physical != input2_physical:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: EADD output is not backed by "
                    "its mutable input2"
                )

    def has_operation_scope(self, operation: Node) -> bool:
        """Return whether one final FX node owns an Alpha execution call."""
        if self._prepared_plan is not None:
            return operation in self._prepared_plan.operation_nodes
        return operation in self._operation_operands

    def _operation_logical_ref(self, value):
        """Resolve one call operand to an exact logical tensor identity."""
        self._require_storage_plan()
        if isinstance(value, ValueRef):
            if not self._storage_plan.has_logical_value(value):
                raise AlphaMemoryPlanError(
                    f"{value.name}: operation operand is not in the plan"
                )
            return value
        if not isinstance(value, Node):
            raise AlphaMemoryPlanError(
                f"operation operand has no logical identity: {value!r}"
            )
        root = ValueRef(value)
        if self._storage_plan.has_logical_value(root):
            return root
        outputs = self._storage_plan.output_refs(value)
        if len(outputs) != 1:
            raise AlphaMemoryPlanError(
                f"{value.name}: operation operand has {len(outputs)} "
                "logical results; an exact ValueRef is required"
            )
        return outputs[0]

    def _operation_payload_uses(self, operation):
        if operation.op == "call_module":
            submodule = self._model.get_submodule(str(operation.target))
            inner_graph = submodule.graph
        else:
            inner_graph = None
        return tuple(
            use
            for uses in self._payload_uses_by_physical.values()
            for use in uses
            if (
                use.operation is operation
                or (
                    inner_graph is not None
                    and use.operation.graph is inner_graph
                )
            )
        )

    def _operation_physical_roles(self, operation):
        """Collect exact call-local physical identities and access roles."""
        info = self._operation_operands.get(operation)
        if info is None or info.owner is not operation:
            raise AlphaMemoryPlanError(
                f"{operation.name}: no Alpha executable call is bound"
            )
        roles = {}
        logicals = {}

        def add_physical(physical, role, logical=None):
            physical = self._storage_plan.storage_key(
                physical,
                self._storage_plane,
            )
            roles.setdefault(physical, set()).add(TensorRole(role))
            values = logicals.setdefault(physical, [])
            if logical is not None and logical not in values:
                values.append(logical)

        def add(logical, role):
            add_physical(logical, role, logical)

        for descriptor in info.operand_descriptors:
            add(
                self._operation_logical_ref(descriptor.tensor),
                descriptor.role,
            )

        produced = set(
            self._storage_plan.produced_physical_outputs(
                operation,
                self._storage_plane,
            )
        )
        if not produced:
            raise AlphaMemoryPlanError(
                f"{operation.name}: Alpha call has no physical output"
            )
        for output in self._storage_plan.output_refs(operation):
            physical = self._storage_plan.storage_key(
                output,
                self._storage_plane,
            )
            if physical in produced:
                add(output, TensorRole.OUTPUT)

        for physical, tensor in self._storage_plan.physical_tensors[
            self._storage_plane
        ].items():
            for access in tensor.accesses:
                if access.operation is not operation:
                    continue
                if tensor.logical_tensors:
                    for logical in tensor.logical_tensors:
                        add_physical(physical, access.role, logical)
                else:
                    add_physical(physical, access.role)

        for use in self._operation_payload_uses(operation):
            physical = self._storage_plan.storage_key(
                use.logical_tensor,
                self._storage_plane,
            )
            matched_roles = roles.get(physical, ())
            if not matched_roles:
                matched_roles = (
                    classify_storage_role(
                        use.logical_tensor.node,
                        self._operator_traits,
                    ),
                )
            for role in tuple(matched_roles):
                add_physical(physical, role, use.logical_tensor)

        return {
            physical: (frozenset(roles[physical]), tuple(values))
            for physical, values in logicals.items()
        }

    @staticmethod
    def _operation_allocation_role(roles):
        for role in (
            TensorRole.WEIGHT,
            TensorRole.ACTIVATION,
            TensorRole.OUTPUT,
            TensorRole.BIAS,
            TensorRole.SCALE,
            TensorRole.PARTIAL_SUM,
        ):
            if role in roles:
                return role
        raise AlphaMemoryPlanError(
            "operation physical tensor has no allocation role"
        )

    def _reset_operation_arena(self):
        """Release all Alpha banks between independent operation calls."""
        for memory in self.memory_level.instances:
            for bank in self._owners[memory.name]:
                bank[:] = [None] * len(bank)
        self._allocations.clear()
        self._allocation_records.clear()
        self._shuffle_storage.clear()

    def _operation_scope_history(self, operation):
        position = self._execution_position(operation)
        allocation_id_base = sum(
            len(history) for history in self._operation_scope_histories.values()
        )
        records = []
        for allocation in self._allocation_records:
            physical = allocation.physical_tensor
            tensor = self._storage_plan.physical_tensor(
                physical,
                self._storage_plane,
            )
            records.append(
                AlphaAllocationRecord(
                    allocation_id=(
                        allocation_id_base + allocation.allocation_id
                    ),
                    physical_tensor=physical,
                    logical_tensors=tensor.logical_tensors,
                    role=allocation.role,
                    memory_name=allocation.memory_name,
                    cells=allocation.cells,
                    start=allocation.segment.start,
                    end=allocation.segment.end,
                    memory_space=allocation.segment.memory_space,
                    address=allocation.segment.address,
                    birth_position=position,
                    death_position=position + 1,
                    primary=allocation.primary,
                )
            )
        return tuple(records)

    def _validate_operation_scope_history(
        self,
        operation,
        history,
        physical_map=None,
    ):
        self._validate_physical_history(history)
        for record in history:
            physical = record.physical_tensor
            backing_size = self._storage_plan.backing_size_bytes(
                physical,
                self._storage_plane,
            )
            if record.end - record.start < backing_size:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: {physical.name} allocation does "
                    "not cover its physical backing"
                )
            memory = self.memory_level.memory(record.memory_name)
            requirements = self._access_requirements_for(physical)
            if not memory.accepts_accesses(requirements):
                accesses = sorted(
                    (role.value, unit) for role, unit in requirements
                )
                raise AlphaMemoryPlanError(
                    f"{operation.name}: {record.memory_name} does not "
                    f"accept {accesses} for {physical.name}"
                )
            if physical_map is None:
                continue
            segment = physical_map.get(physical)
            if (
                not isinstance(segment, Segment)
                or segment.memory_space != record.memory_space
                or segment.address != record.address
                or segment.start != record.start
                or segment.end != record.end
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: {physical.name} map disagrees "
                    "with its operation-scope allocation"
                )

    def allocate_operation_scope(self, operation: Node):
        """Allocate one Alpha call in a fresh operation-local arena."""
        self._ensure_mutable()
        self._require_storage_plan()
        if not self.has_operation_scope(operation):
            return None
        if operation in self._operation_scope_histories:
            raise AlphaMemoryPlanError(
                f"{operation.name}: operation scope is already allocated"
            )

        physical_roles = self._operation_physical_roles(operation)
        self._reset_operation_arena()
        self._active_operation_scope = operation
        try:
            ordered = sorted(
                physical_roles.items(),
                key=lambda item: (
                    self._operation_allocation_role(item[1][0]).value,
                    item[0].name,
                ),
            )
            for physical, (roles, _logicals) in ordered:
                self._allocate_by_role(
                    physical.owner.node,
                    self._operation_allocation_role(roles),
                    allocation_key=physical,
                )

            history = self._operation_scope_history(operation)
            physical_map = {}
            for physical, (_roles, logicals) in physical_roles.items():
                allocation = self._allocations.get(physical)
                if allocation is None:
                    raise AlphaMemoryPlanError(
                        f"{operation.name}: {physical.name} was not allocated"
                    )
                physical_map[physical] = copy.deepcopy(allocation.segment)
                for logical in logicals:
                    physical_map[logical] = copy.deepcopy(allocation.segment)
                    if not logical.path:
                        physical_map[logical.node] = copy.deepcopy(
                            allocation.segment
                        )

            self._validate_operation_scope_history(
                operation,
                history,
                physical_map,
            )
            operation.meta[PHYSICAL_SCRATCHPAD_MAP] = physical_map
            self._operation_scope_histories[operation] = history
            return physical_map
        finally:
            self._active_operation_scope = None
            self._reset_operation_arena()

    def validate_operation_scopes(self, model: GraphModule):
        """Validate complete fresh-arena placement for every Alpha call."""
        self._ensure_mutable()
        if model is not self._model:
            raise AlphaMemoryPlanError(
                "operation scopes belong to another FX model"
            )
        expected = set(self._operation_operands)
        actual = set(self._operation_scope_histories)
        if expected != actual:
            missing = sorted(node.name for node in expected - actual)
            unexpected = sorted(node.name for node in actual - expected)
            raise AlphaMemoryPlanError(
                "Alpha operation scopes are incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for operation, history in self._operation_scope_histories.items():
            physical_map = operation.meta.get(PHYSICAL_SCRATCHPAD_MAP)
            if not isinstance(physical_map, dict) or not physical_map:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: physical scratchpad map is empty"
                )
            self._validate_operation_scope_history(
                operation,
                history,
                physical_map,
            )
        self._validate_physical_payload_uses()
        self._history = tuple(
            record
            for history in self._operation_scope_histories.values()
            for record in history
        )
        self._sealed = True
        return dict(self._operation_scope_histories)

    def _publish_physical_payloads(self, physical, segment):
        """Attach finalized plan component references to their operations."""
        if self._storage_plan is None or self._storage_plane is None:
            return
        for use in self._payload_uses_by_physical.get(physical, ()):
            if not self._payload_use_is_in_active_scope(use):
                continue
            payload = self._storage_plan.payload_ref(
                physical,
                self._storage_plane,
                use.component,
                segment,
            )
            target_kwargs = use.operation.meta.setdefault(
                TARGET_CODEGEN_KWARGS,
                {},
            )
            previous = target_kwargs.get(use.field)
            if previous is not None and (
                not isinstance(previous, PhysicalPayloadRef)
                or previous.physical_tensor != payload.physical_tensor
                or previous.representation is not payload.representation
                or previous.segment != payload.segment
            ):
                raise AlphaMemoryPlanError(
                    f"{use.operation.name}: conflicting finalized "
                    f"{use.field} payload"
                )
            target_kwargs[use.field] = payload

    def _payload_use_is_in_active_scope(self, use):
        operation = self._active_operation_scope
        if operation is None:
            return True
        if use.operation is operation:
            return True
        if operation.op != "call_module" or self._model is None:
            return False
        submodule = self._model.get_submodule(str(operation.target))
        return use.operation.graph is submodule.graph

    def _allocation_payload(self, physical, allocation):
        """Return the primary serialized payload for one full allocation."""
        tensor = self._storage_plan.physical_tensor(
            physical,
            self._storage_plane,
        )
        components = {
            component.name: component for component in tensor.components
        }
        component = components.get("primary")
        if component is None and len(components) == 1:
            component = next(iter(components.values()))
        if component is None:
            raise AlphaMemoryPlanError(
                f"{physical.name}: no unique primary payload component"
            )
        return self._storage_plan.payload_ref(
            physical,
            self._storage_plane,
            component.name,
            allocation.segment,
        )

    def _constraint_physical_id(self, value):
        """Return an exact physical key when ``value`` stores one tensor."""
        if isinstance(value, PhysicalTensorId):
            return self._storage_plan.storage_key(
                value,
                self._storage_plane,
            )
        if isinstance(value, ValueRef):
            return self._storage_plan.physical_id(
                value,
                self._storage_plane,
            )
        if not isinstance(value, Node):
            return None
        if not self._storage_plan.has_logical_value(value):
            return None
        return self._storage_plan.physical_id(value, self._storage_plane)

    @staticmethod
    def _add_symmetric_conflict(conflicts, left, right):
        if left is None or right is None or left == right:
            return
        conflicts.setdefault(left, set()).add(right)
        conflicts.setdefault(right, set()).add(left)

    def _bind_physical_constraints(self) -> None:
        """Lower target hazards onto the shared physical identity graph."""
        normalized_clusters = {}
        for value, cluster in self._cluster_constraints.items():
            key = self._constraint_physical_id(value) or value
            previous = normalized_clusters.get(key)
            if previous is not None and previous != cluster:
                raise AlphaMemoryPlanError(
                    f"{getattr(key, 'name', key)}: conflicting Alpha "
                    f"clusters {previous} and {cluster}"
                )
            normalized_clusters[key] = cluster
        self._cluster_constraints = normalized_clusters

        hidden_by_operation = {}
        for physical in self._storage_plan.physical_tensors[
            self._storage_plane
        ].values():
            for access in physical.accesses:
                hidden_by_operation.setdefault(
                    access.operation,
                    [],
                ).append(physical.id)

        conflicts = {}
        for operation in self._storage_plan.index.model.graph.nodes:
            if operation.op not in {"call_function", "call_module"}:
                continue
            if (
                operation.target == operator.getitem
                or is_nop(operation)
                or operation.meta.get(TARGET_INSTRUCTION_NOP, False)
            ):
                continue

            physical_operands = [
                self._constraint_physical_id(logical)
                for logical in self._storage_plan.index.node_inputs.get(
                    operation,
                    (),
                )
            ]
            info = self._operation_operands.get(operation)
            if info is not None:
                physical_operands.extend(
                    self._constraint_physical_id(operand)
                    for operand in info.operands
                )
            physical_operands.extend(
                self._storage_plan.produced_physical_outputs(
                    operation,
                    self._storage_plane,
                )
            )
            physical_operands.extend(hidden_by_operation.get(operation, ()))
            physical_operands = tuple(
                dict.fromkeys(
                    physical
                    for physical in physical_operands
                    if physical is not None
                )
            )
            for index, left in enumerate(physical_operands):
                for right in physical_operands[index + 1 :]:
                    self._add_symmetric_conflict(
                        conflicts,
                        left,
                        right,
                    )
        self._bank_conflicts = conflicts

    def _ensure_mutable(self):
        if self._sealed:
            raise RuntimeError("Alpha memory plan has already been finalized")

    def _execution_position(self, node: Node) -> int:
        self._require_storage_plan()
        try:
            return execution_positions(self._storage_plan.execution_order)[node]
        except KeyError as error:
            raise AlphaMemoryPlanError(
                f"{node.name}: node is absent from the storage plan"
            ) from error

    def _require_storage_plan(self) -> None:
        if self._storage_plan is None or self._storage_plane is None:
            raise AlphaMemoryPlanError(
                "Alpha allocation requires a bound storage plan"
            )

    def _physical_id(self, value):
        self._require_storage_plan()
        if isinstance(value, PhysicalTensorId):
            return self._storage_plan.storage_key(
                value,
                self._storage_plane,
            )
        if isinstance(value, Node) and not self._storage_plan.has_logical_value(
            value
        ):
            outputs = self._storage_plan.output_refs(value)
            raise AlphaMemoryPlanError(
                f"{value.name}: raw multi-output producer is not an "
                f"address identity ({len(outputs)} logical results); "
                "use an exact ValueRef or PhysicalTensorId"
            )
        return self._storage_plan.physical_id(
            value,
            self._storage_plane,
        )

    def _storage_key(self, value):
        physical = self._physical_id(value)
        return self._storage_plan.storage_key(
            physical,
            self._storage_plane,
        )

    def _shuffle_secondary_id(self, shuffle: Node) -> PhysicalTensorId:
        """Derive the shuffle side output from the finalized storage plan."""
        if self._storage_plan is None:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: shuffle storage plan is not bound"
            )
        secondary = PhysicalTensorId(
            plane=self._storage_plane,
            owner=self._storage_plan.ref(shuffle),
            component="output[1]",
        )
        return self._storage_plan.storage_key(
            secondary,
            self._storage_plane,
        )

    @staticmethod
    def _depth_per_bank(memory):
        """Return the physical depth representable by an Alpha address."""
        return min(memory.depth_per_bank, 1 << ALPHA_ENTRY_BITS)

    def _reserved_banks(self, memory_name):
        """Return physical banks unavailable to Alpha tensor operands."""
        if memory_name == self.memory_level.input_memory:
            return ALPHA_RESERVED_GB_SCRATCH_BANKS
        return frozenset()

    def _cluster_for(
        self,
        node,
        role,
        memory_name,
        operand_bits,
        *,
        allocation_key=None,
    ):
        if memory_name == self.memory_level.weight_memory:
            return 1

        constraint_key = node
        if self._storage_plan is not None:
            constraint_key = (
                allocation_key
                if allocation_key is not None
                else self._constraint_physical_id(node)
            )
        overrides = {
            value
            for value in (self._cluster_constraints.get(constraint_key),)
            if value is not None
        }
        magnet_alpha = node.meta.get("magnet_alpha", {})
        if magnet_alpha.get("kind") == ALPHA_SHUFFLE_COPY_KIND:
            overrides.add(4)
        if len(overrides) > 1:
            raise AlphaMemoryPlanError(
                f"{node.name}: conflicting Alpha cluster overrides "
                f"{sorted(overrides)}"
            )
        if overrides:
            cluster = int(next(iter(overrides)))
        elif role == TensorRole.WEIGHT:
            consumers = self._consumer_units_for(
                allocation_key if allocation_key is not None else node
            )
            if self.hardware.nmp_compute_unit in consumers:
                cluster = 2
            else:
                cluster = 4
        else:
            cluster = self.hardware.dtype_mode(operand_bits).banks_per_value

        if cluster not in (1, 2, 4):
            raise ValueError(
                f"{node.name}: unsupported Alpha bank cluster {cluster}"
            )
        return cluster

    def _consumer_units_for(self, node):
        return frozenset(
            unit
            for _role, unit in self._access_requirements_for(node)
            if unit is not None
        )

    def _access_requirements_for(self, node):
        if self._storage_plan is None or self._storage_plane is None:
            return frozenset()
        return frozenset(
            physical_access_requirements(
                self._storage_plan,
                self.hardware,
                self._storage_plane,
                self._storage_key(node),
                self._operator_traits,
            )
        )

    def tensor_size_bytes(self, node):
        """Return the target storage size for one logical tensor value."""
        if self._storage_plan is None:
            return None
        try:
            return self._storage_plan.logical_size_bytes(node)
        except (KeyError, ValueError):
            return None

    def _banks_to_avoid(self, node, memory_name):
        banks = set()
        conflict_key = self._constraint_physical_id(node)
        for peer in self._bank_conflicts.get(conflict_key, ()):
            allocation = self._allocation_for(peer)
            if allocation is not None and allocation.memory_name == memory_name:
                banks.update(allocation.banks)
        return banks

    def _allocation_for(self, node):
        """Resolve one logical value through shared physical identity."""
        return self._allocations.get(self._storage_key(node))

    def allocation_segment(self, value):
        """Return the live Alpha segment for one exact physical identity."""
        allocation = self._allocation_for(value)
        if allocation is None:
            return None
        return copy.deepcopy(allocation.segment)

    @staticmethod
    def _entry_cells(memory, linear_entry, cluster):
        depth_per_bank = AlphaScratchpadAllocator._depth_per_bank(memory)
        virtual_bank = linear_entry // depth_per_bank
        entry = linear_entry % depth_per_bank
        first_bank = virtual_bank * cluster
        return tuple((first_bank + lane, entry) for lane in range(cluster))

    def _find_open_block(
        self,
        memory,
        cluster,
        number_of_entries,
        banks_to_avoid,
    ):
        if memory.bank_count % cluster:
            raise ValueError(
                f"{memory.name}: {memory.bank_count} banks cannot form "
                f"clusters of {cluster}"
            )
        total_entries = (
            memory.bank_count // cluster * self._depth_per_bank(memory)
        )
        if number_of_entries > total_entries:
            return None

        owners = self._owners[memory.name]
        run_start = None
        run_length = 0
        for linear_entry in range(total_entries):
            cells = self._entry_cells(memory, linear_entry, cluster)
            is_free = all(
                bank not in banks_to_avoid and owners[bank][entry] is None
                for bank, entry in cells
            )
            if is_free:
                if run_start is None:
                    run_start = linear_entry
                run_length += 1
                if run_length == number_of_entries:
                    return run_start
            else:
                run_start = None
                run_length = 0
        return None

    def _allocate_in_memory(
        self,
        node,
        role,
        memory_name,
        operand_bits,
        *,
        allocation_key=None,
        primary=True,
        banks_to_avoid=None,
        tensor_size=None,
    ):
        key = (
            self._storage_key(node)
            if allocation_key is None
            else self._storage_key(allocation_key)
        )
        existing = self._allocations.get(key)
        if existing is not None:
            self._publish_physical_payloads(key, existing.segment)
            return existing.segment

        memory = self.memory_level.memory(memory_name)
        cluster = self._cluster_for(
            node,
            role,
            memory_name,
            operand_bits,
            allocation_key=key,
        )
        logical_entry_bytes = cluster * memory.bits_per_bank_entry // 8
        planned_size = self._storage_plan.backing_size_bytes(
            key,
            self._storage_plane,
        )
        if tensor_size is not None and tensor_size != planned_size:
            raise AlphaMemoryPlanError(
                f"{key.name}: requested allocation size {tensor_size} "
                f"does not match physical backing size {planned_size}"
            )
        tensor_size = planned_size
        number_of_entries = math.ceil(tensor_size / logical_entry_bytes)
        if banks_to_avoid is None:
            banks_to_avoid = self._banks_to_avoid(key, memory_name)
        else:
            banks_to_avoid = set(banks_to_avoid)
        reserved_banks = self._reserved_banks(memory_name)
        linear_entry = self._find_open_block(
            memory,
            cluster,
            number_of_entries,
            banks_to_avoid | reserved_banks,
        )
        if linear_entry is None:
            return None

        cells = tuple(
            cell
            for index in range(number_of_entries)
            for cell in self._entry_cells(
                memory,
                linear_entry + index,
                cluster,
            )
        )
        owners = self._owners[memory_name]
        for bank, entry in cells:
            if owners[bank][entry] is not None:
                raise RuntimeError("Alpha allocator selected an occupied entry")
            owners[bank][entry] = key

        depth_per_bank = self._depth_per_bank(memory)
        virtual_bank = linear_entry // depth_per_bank
        entry = linear_entry % depth_per_bank
        encoded = AlphaAddress(
            base=self._base_by_name[memory_name],
            operand_bits=operand_bits,
            cluster=cluster,
            virtual_bank=virtual_bank,
            entry=entry,
        ).encode()
        start = linear_entry * logical_entry_bytes
        segment = Segment(
            start=start,
            end=start + number_of_entries * logical_entry_bytes,
            memory_space=self.memory_level.memory_index(memory_name),
            address=encoded,
            storage_plane=MemorySpace.SCRATCHPAD,
            memory_instance=memory_name,
        )
        allocation = _Allocation(
            allocation_id=len(self._allocation_records),
            physical_tensor=self._physical_id(key),
            role=role,
            memory_name=memory_name,
            cells=cells,
            banks=frozenset(bank for bank, _entry in cells),
            segment=segment,
            primary=primary,
        )
        self._allocations[key] = allocation
        self._allocation_records.append(allocation)
        self._publish_physical_payloads(key, segment)
        return segment

    def _allocate_by_role(
        self,
        node,
        role,
        *,
        allocation_key=None,
        primary=True,
        banks_to_avoid=None,
        tensor_size=None,
    ):
        if (
            isinstance(allocation_key, PhysicalTensorId)
            and self._storage_plan is not None
        ):
            if allocation_key.owner.path:
                logical = self._storage_plan.logical_tensors.get(
                    allocation_key.owner
                )
                if logical is None:
                    raise AlphaMemoryPlanError(
                        f"{allocation_key.name}: physical owner is not a "
                        "logical tensor"
                    )
                operand_bits = get_dtype_width(logical.dtype)
            else:
                operand_bits = _node_operand_bits(allocation_key.owner.node)
        else:
            operand_bits = _node_operand_bits(node)
        access_requirements = self._access_requirements_for(
            node if allocation_key is None else allocation_key
        )
        candidates = compatible_memory_names(
            self.memory_level,
            role,
            access_requirements,
        )
        if not candidates:
            serialized_accesses = sorted(
                (item.value, unit) for item, unit in access_requirements
            )
            raise AlphaMemoryPlanError(
                f"{node.name}: no Alpha memory accepts physical accesses "
                f"{serialized_accesses}"
            )
        for memory_name in candidates:
            segment = self._allocate_in_memory(
                node,
                role,
                memory_name,
                operand_bits,
                allocation_key=allocation_key,
                primary=primary,
                banks_to_avoid=banks_to_avoid,
                tensor_size=tensor_size,
            )
            if segment is not None:
                return segment
        usage = []
        for memory in self.memory_level.instances:
            occupied = sum(
                owner is not None
                for bank in self._owners[memory.name]
                for owner in bank
            )
            usage.append(
                f"{memory.name}={occupied}/"
                f"{memory.bank_count * self._depth_per_bank(memory)} entries"
            )
        raise RuntimeError(
            f"Alpha global-memory allocation failed for tensor {node.name}; "
            + ", ".join(usage)
        )

    def _ensure_resident_backing(self, placement):
        allocation = self._allocation_for(placement.physical_tensor)
        if allocation is None:
            backing = placement.physical_tensor.owner.node
            memory_name = self.memory_level.output_memory
            banks_to_avoid = set()
            for owner in self._resident_owners(placement.physical_tensor):
                banks_to_avoid.update(
                    self._banks_to_avoid(
                        owner,
                        memory_name,
                    )
                )
            segment = self._allocate_by_role(
                backing,
                TensorRole.OUTPUT,
                allocation_key=placement.physical_tensor,
                banks_to_avoid=banks_to_avoid,
            )
            allocation = self._allocation_for(placement.physical_tensor)
        return allocation.segment

    def _entry_view(self, segment, entry_offset):
        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        logical_entry_bytes = decoded.cluster * memory.bits_per_bank_entry // 8
        start = segment.start + entry_offset * logical_entry_bytes
        if start + logical_entry_bytes > segment.end:
            raise ValueError("Alpha shuffle copy exceeds its output buffer")
        return Segment(
            start=start,
            end=start + logical_entry_bytes,
            memory_space=segment.memory_space,
            address=self._address_with_entry_offset(
                segment,
                entry_offset,
            ),
            storage_plane=segment.storage_plane,
            memory_instance=segment.memory_instance,
        )

    def _shuffle_access_pattern(
        self,
        shuffle,
        *,
        resident_channels=None,
    ):
        """Describe Alpha's paired shuffle writes without expanding them."""
        shape = tuple(getattr(shuffle, "shape", ()))
        if len(shape) != 4 or shape[0] != 1:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: Alpha shuffle requires batch-one NCHW"
            )

        operand_bits = _node_operand_bits(shuffle)
        lanes = self.hardware.nmp_lanes(operand_bits)
        channels = int(shape[1])
        if channels % lanes:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: {channels} output channels are not "
                f"divisible by {lanes} Alpha NMP lanes"
            )
        active_entries = channels // lanes

        if resident_channels is None:
            placement = self._resident_placement(shuffle)
            resident_channels = (
                placement.resident_channels
                if placement is not None
                else channels
            )
        if resident_channels % lanes:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: resident channel pitch "
                f"{resident_channels} is not divisible by {lanes}"
            )
        pitch_entries = resident_channels // lanes
        if pitch_entries < active_entries:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: resident shuffle pitch is too small"
            )

        positions = math.prod((int(shape[0]), int(shape[2]), int(shape[3])))
        return _ShuffleAccessPattern(
            positions=positions,
            active_entries=active_entries,
            pitch_entries=pitch_entries,
        )

    def _resident_entry_offset(self, node, segment, placement=None):
        """Return one resident writer's entry offset within its backing."""
        if placement is None:
            placement = self._resident_placement(node)
        if placement is None:
            return 0

        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        channels_per_entry = (
            decoded.cluster * memory.bits_per_bank_entry // decoded.operand_bits
        )
        if placement.channel_offset % channels_per_entry:
            raise AlphaMemoryPlanError(
                f"{node.name}: concat slot is not entry aligned"
            )
        return placement.channel_offset // channels_per_entry

    def _resident_output_view(self, node, segment, placement=None):
        entry_offset = self._resident_entry_offset(
            node,
            segment,
            placement,
        )
        if entry_offset == 0:
            return segment
        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        entry_bytes = decoded.cluster * memory.bits_per_bank_entry // 8
        return Segment(
            start=segment.start + entry_offset * entry_bytes,
            end=segment.end,
            memory_space=segment.memory_space,
            address=self._address_with_entry_offset(
                segment,
                entry_offset,
            ),
            storage_plane=segment.storage_plane,
            memory_instance=segment.memory_instance,
        )

    def _bind_resident_consumer_views(self, node, segment, placement):
        view = self._resident_output_view(node, segment, placement)
        if view is segment:
            return
        aliases = {
            node,
            *self._resident_write_paths.get(node, ()),
        }
        for operation, info in self._operation_operands.items():
            matched = aliases.intersection(info.operands)
            if not matched:
                continue
            physical_map = operation.meta.setdefault(
                PHYSICAL_SCRATCHPAD_MAP,
                {},
            )
            physical_map[node] = view
            for operand in matched:
                physical_map[operand] = view

    def _shuffle_banks_at_offset(self, address, offset):
        decoded = AlphaAddress.decode(address)
        memory = self.memory_level.memory(self.memory_level.output_memory)
        depth = self._depth_per_bank(memory)
        linear_entry = decoded.entry + offset
        virtual_bank = decoded.virtual_bank + linear_entry // depth
        first_bank = virtual_bank * decoded.cluster
        last_bank = first_bank + decoded.cluster
        if last_bank > memory.bank_count:
            raise AlphaMemoryPlanError("Alpha shuffle write exceeds GBScratch")
        return frozenset(range(first_bank, last_bank))

    def _bind_shuffle_outputs(self, shuffle, primary):
        if shuffle in self._shuffle_storage:
            return
        plan = self._shuffle_plans.get(shuffle)
        if plan is None:
            return

        primary_allocation = self._allocation_for(shuffle)
        if primary_allocation is None:
            raise RuntimeError(
                f"{shuffle.name}: missing primary Alpha allocation"
            )
        operand_bits = _node_operand_bits(shuffle)
        access = self._shuffle_access_pattern(shuffle)
        memory = self.memory_level.memory(self.memory_level.output_memory)
        cluster = self._cluster_for(
            shuffle,
            TensorRole.OUTPUT,
            memory.name,
            operand_bits,
        )
        entry_bytes = cluster * memory.bits_per_bank_entry // 8
        secondary_size = access.span_entries * entry_bytes
        auxiliary_key = self._shuffle_secondary_id(shuffle)
        banks_to_avoid = set(primary_allocation.banks)
        banks_to_avoid.update(self._banks_to_avoid(shuffle, memory.name))
        secondary = self._allocate_by_role(
            shuffle,
            TensorRole.OUTPUT,
            allocation_key=auxiliary_key,
            primary=False,
            banks_to_avoid=banks_to_avoid,
            tensor_size=secondary_size,
        )
        primary_offset = self._resident_entry_offset(shuffle, primary)
        primary_size = (primary_offset + access.span_entries) * entry_bytes
        if primary_size > primary.end - primary.start:
            raise AlphaMemoryPlanError(
                f"{shuffle.name}: primary shuffle write exceeds its backing"
            )
        primary_view = self._resident_output_view(shuffle, primary)
        storage = _ShuffleStorage(
            primary_base=primary,
            primary=primary_view,
            secondary=secondary,
        )
        self._shuffle_storage[shuffle] = storage
        target_kwargs = shuffle.meta.setdefault(TARGET_CODEGEN_KWARGS, {})
        target_kwargs.update(
            {
                "output_addresses": list(storage.output_addresses),
                "output0_memory_instance": (
                    storage.primary_base.memory_instance
                ),
                "output1_memory_instance": (storage.secondary.memory_instance),
            }
        )

    def _bind_shuffle_copy(self, copy_node):
        plan, movement = self._copy_plans[copy_node]
        storage = self._shuffle_storage.get(plan.shuffle)
        if storage is None:
            raise RuntimeError(
                f"{copy_node.name}: shuffle outputs were not allocated"
            )
        source = self._entry_view(
            storage.outputs[movement.source_output],
            movement.source_entry,
        )
        destination = self._entry_view(
            storage.primary,
            movement.destination_entry,
        )
        if movement.source_output == 0:
            source_allocation = self._allocation_for(plan.shuffle)
        else:
            source_key = self._shuffle_secondary_id(plan.shuffle)
            source_allocation = self._allocations.get(source_key)
        if source_allocation is None:
            raise AlphaMemoryPlanError(
                f"{copy_node.name}: shuffle copy source is not allocated"
            )
        destination_allocation = self._allocation_for(plan.shuffle)
        if destination_allocation is None:
            raise AlphaMemoryPlanError(
                f"{copy_node.name}: shuffle copy destination is not allocated"
            )
        source_payload = self._allocation_payload(
            (
                source_key
                if movement.source_output == 1
                else source_allocation.physical_tensor
            ),
            source_allocation,
        )
        destination_payload = self._allocation_payload(
            destination_allocation.physical_tensor,
            destination_allocation,
        )
        copy_node.meta[PHYSICAL_SCRATCHPAD_MAP] = {
            copy_node.args[0]: source,
            copy_node: destination,
        }
        target_kwargs = copy_node.meta.setdefault(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        pitch_entries = self._shuffle_access_pattern(plan.shuffle).pitch_entries
        target_kwargs.update(
            {
                "block_channels": plan.block_channels,
                "concat_total_size_0": int(plan.shuffle.shape[1]),
                "copy_destination_entry": movement.destination_entry,
                "copy_source_entry": movement.source_entry,
                "copy_source_output": movement.source_output,
                "copy_source_payload": source_payload,
                "copy_destination_payload": destination_payload,
                "is_to_concat": True,
            }
        )
        return destination_allocation.segment

    def _storage_relation(self, node):
        if self._storage_plan is None:
            return None
        try:
            ref = self._storage_plan.ref(node)
        except (KeyError, ValueError):
            return None
        return self._storage_plan.relations[self._storage_plane][ref]

    def _resident_placement(self, node):
        relation = self._storage_relation(node)
        transform = None if relation is None else relation.transform
        if transform is None or transform.kind != "alpha_resident_write":
            return None
        parameters = dict(transform.parameters)
        required = {
            "channel_offset",
            "slot_index",
            "slot_channels",
            "resident_channels",
        }
        if not required.issubset(parameters):
            raise AlphaMemoryPlanError(
                f"{node.name}: resident relation is incomplete"
            )
        physical = self._physical_id(node)
        placement = _ResidentPlacement(
            physical_tensor=physical,
            slot_index=int(parameters["slot_index"]),
            channel_offset=int(parameters["channel_offset"]),
            slot_channels=int(parameters["slot_channels"]),
            resident_channels=int(parameters["resident_channels"]),
            concat_channel_index=(
                None
                if "concat_channel_index" not in parameters
                else int(parameters["concat_channel_index"])
            ),
        )
        if (
            placement.slot_index < 0
            or placement.channel_offset < 0
            or placement.slot_channels <= 0
            or placement.resident_channels <= 0
            or placement.concat_channel_index not in {None, 0, 1}
            or (
                placement.channel_offset + placement.slot_channels
                > placement.resident_channels
            )
        ):
            raise AlphaMemoryPlanError(
                f"{node.name}: resident relation has invalid extents"
            )
        return placement

    def _resident_output_placement(self, node, physical_tensor):
        """Resolve a producer's resident write for one physical output."""
        candidates = []
        for output in self._storage_plan.output_refs(node):
            if (
                physical_tensor is not None
                and self._storage_plan.storage_key(
                    output,
                    self._storage_plane,
                )
                != physical_tensor
            ):
                continue
            placement = self._resident_placement(output)
            if placement is not None and placement not in candidates:
                candidates.append(placement)
        if len(candidates) > 1:
            raise AlphaMemoryPlanError(
                f"{node.name}: one physical output has multiple resident "
                "placements"
            )
        return candidates[0] if candidates else None

    def _resident_owners(self, physical):
        tensor = self._storage_plan.physical_tensor(
            physical,
            self._storage_plane,
        )
        return tuple(
            dict.fromkeys(
                logical.node
                for logical in tensor.logical_tensors
                if self._resident_placement(logical) is not None
            )
        )

    def allocate_memory(
        self,
        node,
        role: TensorRole,
        *,
        physical_tensor=None,
        tensor_size=None,
    ):
        self._ensure_mutable()
        self._require_storage_plan()
        if physical_tensor is not None:
            physical_tensor = self._storage_plan.storage_key(
                physical_tensor,
                self._storage_plane,
            )
            expected = self._storage_plan.produced_physical_outputs(
                node,
                self._storage_plane,
            )
            if physical_tensor not in expected:
                raise AlphaMemoryPlanError(
                    f"{node.name}: selected physical output does not match "
                    "the Alpha storage plan"
                )
        if node in self._copy_plans:
            return self._bind_shuffle_copy(node)

        placement = self._resident_output_placement(
            node,
            physical_tensor,
        )
        if placement is not None:
            segment = self._ensure_resident_backing(placement)
            if placement.concat_channel_index is not None:
                target_kwargs = node.meta.setdefault(
                    TARGET_CODEGEN_KWARGS,
                    {},
                )
                target_kwargs.update(
                    {
                        "concat_channel_index_0": (
                            placement.concat_channel_index
                        ),
                        "concat_idx_0": placement.slot_index,
                        "concat_size_0": placement.slot_channels,
                        "concat_total_size_0": (placement.resident_channels),
                        "is_to_concat": True,
                    }
                )
            if node in self._shuffle_plans:
                self._bind_shuffle_outputs(node, segment)
            self._bind_resident_consumer_views(
                node,
                segment,
                placement,
            )
            return segment

        relation = self._storage_relation(node)
        if relation is not None and relation.kind in {
            StorageRelationKind.ALIAS,
            StorageRelationKind.VIEW,
            StorageRelationKind.COMPONENT,
        }:
            physical = self._physical_id(node)
            allocation = self._allocation_for(physical)
            if allocation is None:
                owner = physical.owner.node
                owner_role = classify_storage_role(
                    owner,
                    self._operator_traits,
                )
                self.allocate_memory(owner, owner_role)
                allocation = self._allocation_for(physical)
            if allocation is None:
                raise AlphaMemoryPlanError(
                    f"{node.name}: analyzed physical tensor "
                    f"{physical.name} is not allocated"
                )
            return allocation.segment

        segment = self._allocate_by_role(
            node,
            role,
            allocation_key=physical_tensor,
            tensor_size=tensor_size,
        )

        if node in self._shuffle_plans:
            self._bind_shuffle_outputs(node, segment)
        return segment

    def _free_allocation(self, key):
        allocation = self._allocations.pop(key, None)
        if allocation is None:
            return
        owners = self._owners[allocation.memory_name]
        for bank, entry in allocation.cells:
            if owners[bank][entry] == key:
                owners[bank][entry] = None

    def free_memory(self, node):
        self._ensure_mutable()
        self._free_allocation(self._storage_key(node))

    def _address_with_entry_offset(self, segment, entry_offset):
        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        depth_per_bank = self._depth_per_bank(memory)
        linear_entry = (
            decoded.virtual_bank * depth_per_bank + decoded.entry + entry_offset
        )
        total_entries = memory.bank_count // decoded.cluster * depth_per_bank
        if not 0 <= linear_entry < total_entries:
            raise ValueError("Alpha view exceeds its physical memory")

        return AlphaAddress(
            base=decoded.base,
            operand_bits=decoded.operand_bits,
            cluster=decoded.cluster,
            virtual_bank=linear_entry // depth_per_bank,
            entry=linear_entry % depth_per_bank,
        ).encode()

    def create_split_views(self, split_node, segment):
        """Lower finalized Alpha split relations to starting addresses."""
        self._require_storage_plan()
        if segment is None:
            return None

        split_outputs = self._storage_plan.output_refs(split_node)
        if not split_outputs:
            return None
        planned = tuple(
            self._storage_plan.relations[self._storage_plane][output]
            for output in split_outputs
        )
        if any(
            relation.transform is None
            or relation.transform.kind
            not in {
                "alpha_channel_split",
                "alpha_shuffle_output",
            }
            for relation in planned
        ):
            return None
        logical_shapes = tuple(
            tuple(self._storage_plan.logical_tensors[output].shape)
            for output in split_outputs
        )
        if any(
            len(shape) != 4
            or shape[0] != 1
            or not isinstance(shape[1], int)
            or shape[1] <= 0
            for shape in logical_shapes
        ):
            raise AlphaMemoryPlanError(
                f"{split_node.name}: finalized Alpha split outputs must be "
                "batch-one rank-four tensors"
            )

        output_physical = tuple(
            self._storage_plan.physical_id(
                output,
                self._storage_plane,
            )
            for output in split_outputs
        )
        shuffle_storage = next(
            (
                self._shuffle_storage[ref.node]
                for output in split_outputs
                for ref in self._storage_plan.trace_storage_chain(
                    output,
                    self._storage_plane,
                )
                if ref.node in self._shuffle_storage
            ),
            None,
        )
        if len(set(output_physical)) > 1:
            if shuffle_storage is None or len(shuffle_storage.outputs) != len(
                split_outputs
            ):
                raise AlphaMemoryPlanError(
                    f"{split_node.name}: finalized split outputs do not "
                    "match allocated shuffle storage"
                )
            views = []
            for output, physical_id, physical in zip(
                split_outputs,
                output_physical,
                shuffle_storage.outputs,
            ):
                allocation = self._allocation_for(physical_id)
                if allocation is None:
                    return None
                if not (
                    allocation.segment.memory_space == physical.memory_space
                    and allocation.segment.start <= physical.start
                    and physical.end <= allocation.segment.end
                ):
                    raise AlphaMemoryPlanError(
                        f"{output.name}: allocated split view is outside "
                        "its physical backing"
                    )
                size_bytes = self._storage_plan.logical_size_bytes(output)
                if physical.start + size_bytes > physical.end:
                    raise AlphaMemoryPlanError(
                        f"{output.name}: logical split output exceeds its "
                        "allocated shuffle view"
                    )
                views.append(
                    Segment(
                        start=physical.start,
                        end=physical.start + size_bytes,
                        memory_space=physical.memory_space,
                        address=physical.address,
                        storage_plane=physical.storage_plane,
                        memory_instance=physical.memory_instance,
                    )
                )
            _mark_alpha_address_only(split_node)
            return tuple(views)

        source_physical = output_physical[0]
        if shuffle_storage is not None:
            segment = shuffle_storage.primary
        else:
            backing_allocation = self._allocation_for(source_physical)
            if backing_allocation is None:
                return None
            segment = backing_allocation.segment
        if segment.address is None:
            return None

        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        logical_entry_bits = decoded.cluster * memory.bits_per_bank_entry
        if logical_entry_bits % decoded.operand_bits:
            raise AlphaMemoryPlanError(
                f"{split_node.name}: physical split entry has fractional "
                "channels"
            )
        channels_per_entry = logical_entry_bits // decoded.operand_bits

        logical_entry_bytes = logical_entry_bits // 8
        views = []
        for output, relation, shape in zip(
            split_outputs,
            planned,
            logical_shapes,
        ):
            if relation.transform.kind != "alpha_channel_split":
                raise AlphaMemoryPlanError(
                    f"{output.name}: shared split backing requires an "
                    "alpha_channel_split relation"
                )
            channel_offset = relation.transform.parameter("channel_offset")
            channel_size = relation.transform.parameter("channel_size")
            if (
                isinstance(channel_offset, bool)
                or not isinstance(channel_offset, int)
                or channel_offset < 0
                or isinstance(channel_size, bool)
                or not isinstance(channel_size, int)
                or channel_size != shape[1]
                or channel_offset % channels_per_entry
                or channel_size % channels_per_entry
            ):
                raise AlphaMemoryPlanError(
                    f"{output.name}: finalized split relation is not "
                    "entry-aligned"
                )
            entry_offset = channel_offset // channels_per_entry
            byte_offset = entry_offset * logical_entry_bytes
            size_bytes = self._storage_plan.logical_size_bytes(output)
            if segment.start + byte_offset + size_bytes > segment.end:
                raise AlphaMemoryPlanError(
                    f"{output.name}: logical split view exceeds its backing"
                )
            views.append(
                Segment(
                    start=segment.start + byte_offset,
                    end=segment.start + byte_offset + size_bytes,
                    memory_space=segment.memory_space,
                    address=self._address_with_entry_offset(
                        segment,
                        entry_offset,
                    ),
                    storage_plane=segment.storage_plane,
                    memory_instance=segment.memory_instance,
                )
            )

        _mark_alpha_address_only(split_node)
        return tuple(views)

    def create_view(self, segment, byte_offset, size):
        if segment.address is None:
            raise ValueError("Alpha view requires an encoded starting address")
        if int(byte_offset) != byte_offset:
            raise ValueError(
                "Alpha starting addresses cannot encode a sub-byte view"
            )
        byte_offset = int(byte_offset)
        if byte_offset < 0 or byte_offset + size > segment.end - segment.start:
            raise ValueError("View exceeds its backing Alpha allocation")

        decoded = AlphaAddress.decode(segment.address)
        memory = self.memory_level.instances[segment.memory_space]
        logical_entry_bytes = decoded.cluster * memory.bits_per_bank_entry // 8
        if byte_offset % logical_entry_bytes:
            raise ValueError(
                "Alpha starting addresses cannot encode a sub-entry view"
            )

        start = segment.start + byte_offset
        return Segment(
            start=start,
            end=start + size,
            memory_space=segment.memory_space,
            address=self._address_with_entry_offset(
                segment,
                byte_offset // logical_entry_bytes,
            ),
            storage_plane=segment.storage_plane,
            memory_instance=segment.memory_instance,
        )

    def _freeze_history(self):
        records = []
        for allocation in self._allocation_records:
            physical = allocation.physical_tensor
            if physical is None or self._storage_plan is None:
                owner = (
                    physical.owner.node.name
                    if physical is not None
                    else "<unknown>"
                )
                raise AlphaMemoryPlanError(
                    f"{owner}: allocation has no shared " "physical tensor"
                )
            interval = self._storage_plan.intervals[self._storage_plane][
                physical
            ]
            tensor = self._storage_plan.physical_tensor(
                physical,
                self._storage_plane,
            )
            records.append(
                AlphaAllocationRecord(
                    allocation_id=allocation.allocation_id,
                    physical_tensor=physical,
                    logical_tensors=tensor.logical_tensors,
                    role=allocation.role,
                    memory_name=allocation.memory_name,
                    cells=allocation.cells,
                    start=allocation.segment.start,
                    end=allocation.segment.end,
                    memory_space=allocation.segment.memory_space,
                    address=allocation.segment.address,
                    birth_position=interval.birth,
                    death_position=interval.death,
                    primary=allocation.primary,
                )
            )
        return tuple(records)

    def _validate_target_record_geometry(self, record):
        """Validate Alpha address encoding and physical-bank cells."""
        label = allocation_record_label(record)
        memory = self.memory_level.memory(record.memory_name)
        if record.address is None:
            raise AlphaMemoryPlanError(
                f"{label}: allocation has no Alpha address"
            )

        try:
            address = AlphaAddress.decode(record.address)
        except ValueError as error:
            raise AlphaMemoryPlanError(
                f"{label}: invalid Alpha address {record.address}"
            ) from error
        if address.base != self._base_by_name[record.memory_name]:
            raise AlphaMemoryPlanError(
                f"{label}: address base {address.base:#x} does not match "
                f"{record.memory_name}"
            )

        entry_bytes = address.cluster * memory.bits_per_bank_entry // 8
        size = record.end - record.start
        if size % entry_bytes:
            raise AlphaMemoryPlanError(
                f"{label}: byte size {size} is not entry aligned"
            )
        entry_count = size // entry_bytes
        depth = self._depth_per_bank(memory)
        linear_entry = address.virtual_bank * depth + address.entry
        if record.start != linear_entry * entry_bytes:
            raise AlphaMemoryPlanError(
                f"{label}: byte offset {record.start} disagrees with "
                f"encoded entry {linear_entry}"
            )

        expected_cells = tuple(
            cell
            for index in range(entry_count)
            for cell in self._entry_cells(
                memory,
                linear_entry + index,
                address.cluster,
            )
        )
        if record.cells != expected_cells:
            raise AlphaMemoryPlanError(
                f"{label}: recorded cells disagree with its address range"
            )
        if len(set(record.cells)) != len(record.cells):
            raise AlphaMemoryPlanError(
                f"{label}: allocation contains duplicate physical cells"
            )
        for bank, entry in record.cells:
            if not 0 <= bank < memory.bank_count:
                raise AlphaMemoryPlanError(
                    f"{label}: bank {bank} is out of range"
                )
            if not 0 <= entry < depth:
                raise AlphaMemoryPlanError(
                    f"{label}: entry {entry} is out of range"
                )
            if bank in self._reserved_banks(record.memory_name):
                raise AlphaMemoryPlanError(
                    f"{label}: bank {bank} is reserved for Alpha runtime"
                )

    def _validate_physical_history(self, history):
        events = []
        for record in history:
            self._validate_target_record_geometry(record)
            events.append(
                (
                    record.birth_position,
                    1,
                    record.allocation_id,
                    record,
                )
            )
            events.append(
                (
                    record.death_position,
                    0,
                    record.allocation_id,
                    record,
                )
            )

        owners = {}
        for _position, is_birth, _record_id, record in sorted(events):
            label = allocation_record_label(record)
            for cell in record.cells:
                key = (record.memory_name, *cell)
                if is_birth:
                    previous = owners.get(key)
                    if previous is not None:
                        raise AlphaMemoryPlanError(
                            f"{label}: cell {cell} in "
                            f"{record.memory_name} overlaps "
                            f"{allocation_record_label(previous)}"
                        )
                    owners[key] = record
                    continue

                previous = owners.get(key)
                if previous is not record:
                    previous_label = (
                        "no allocation"
                        if previous is None
                        else allocation_record_label(previous)
                    )
                    raise AlphaMemoryPlanError(
                        f"{label}: cell {cell} is owned by "
                        f"{previous_label} at release"
                    )
                del owners[key]

        if owners:
            raise AlphaMemoryPlanError(
                "allocation replay ended with live physical cells"
            )

    def validate_allocation_history(self, history):
        """Validate immutable liveness plus Alpha physical-cell ownership."""
        history = tuple(history)
        validate_allocation_lifecycles(
            history,
            memory_level=self.memory_level,
            access_requirements_for=self._access_requirements_for,
            error_type=AlphaMemoryPlanError,
        )
        self._validate_physical_history(history)

    def _validate_execution_liveness(
        self,
        model,
        records_by_tensor,
    ):
        validate_execution_liveness(
            model,
            records_by_tensor,
            requires_allocation=self._requires_execution_allocation,
            storage_plan=self._storage_plan,
            storage_plane=self._storage_plane,
            error_type=AlphaMemoryPlanError,
            allocation_name="Alpha scratchpad",
        )

    def _requires_execution_allocation(self, tensor):
        return require_allocation(tensor, self._operator_traits)

    def _required_live_record(
        self,
        operation,
        tensor,
        position,
        records_by_tensor,
        role,
    ):
        if not self._requires_execution_allocation(tensor):
            return None
        logical = self._storage_plan.ref(tensor)
        segment = logical_tensor_segment(logical, "scratchpad")
        if not isinstance(segment, Segment):
            raise AlphaMemoryPlanError(
                f"{operation.name}: required {role} tensor {tensor.name} "
                "has no Alpha scratchpad allocation"
            )
        candidates = live_allocation_records(
            segment,
            position,
            records_by_tensor.get(logical, ()),
        )
        if not candidates:
            raise AlphaMemoryPlanError(
                f"{operation.name}: required {role} tensor {tensor.name} "
                f"is not live at execution position {position}"
            )
        if len(candidates) > 1:
            labels = ", ".join(
                allocation_record_label(record) for record in candidates
            )
            raise AlphaMemoryPlanError(
                f"{operation.name}: required {role} tensor {tensor.name} "
                f"has ambiguous live allocations: {labels}"
            )
        return candidates[0]

    def _live_record_for(self, tensor, position, records_by_tensor):
        logical = self._storage_plan.ref(tensor)
        segment = logical_tensor_segment(logical, "scratchpad")
        if not isinstance(segment, Segment):
            return None
        candidates = live_allocation_records(
            segment,
            position,
            records_by_tensor.get(logical, ()),
        )
        if len(candidates) != 1:
            return None
        return candidates[0]

    @staticmethod
    def _live_record_for_physical_segment(
        physical,
        segment,
        position,
        records_by_tensor,
    ):
        """Resolve a view through its exact physical identity, not address."""
        candidates = live_allocation_records(
            segment,
            position,
            records_by_tensor.get(physical, ()),
        )
        if len(candidates) != 1:
            return None
        return candidates[0]

    @staticmethod
    def _record_banks(record):
        return frozenset(bank for bank, _entry in record.cells)

    def _validate_nmp_input_view(self, operation, info):
        """Check serialized NMP view fields against finalized storage."""
        include_resident = info.kind in {"gemm", "depthwise_conv2d"}
        split_facts = [
            fact
            for operand in info.operands
            if (
                fact := self._nmp_input_view_fact(
                    operand,
                    self._storage_plan,
                    self._storage_plane,
                    include_resident=include_resident,
                )
            )
            is not None
        ]
        split_facts = list(dict.fromkeys(split_facts))
        serialized = operation.meta.get(TARGET_CODEGEN_KWARGS, {})
        if bool(split_facts) != bool(serialized.get("is_from_split")):
            raise AlphaMemoryPlanError(
                f"{operation.name}: serialized split state does not match "
                "its physical storage relation"
            )
        if not split_facts:
            return
        if len(split_facts) != 1:
            raise AlphaMemoryPlanError(
                f"{operation.name}: multiple physical split inputs cannot "
                "share one NMP descriptor"
            )
        split = split_facts[0]
        split_index = split.split_index
        if split.source_kind == "channel_split":
            if info.kind == "gemm":
                split_index = 0
            elif info.kind == "channel_shuffle":
                split_index = 1
        if (
            serialized.get("split_channel_idx") != split.channel_index
            or serialized.get("split_cur_size") != split.current_channels
            or serialized.get("split_idx") != split_index
            or serialized.get("split_scale") != split.scale
            or serialized.get("split_total_size") != split.total_channels
            or (
                info.kind != "gemm"
                and serialized.get("nmp_c2", 0) != split.pitch_gap_words
            )
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: split input view does not match its "
                "physical backing"
            )
        if (
            split.source_kind == "resident"
            and split.resident_group_count is not None
            and split.resident_group_count > 1
            and (
                serialized.get("nmp_groups") != split.resident_group_count
                or serialized.get("nmp_channels_per_group")
                != split.uniform_group_channels
            )
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: resident group metadata does not "
                "match its physical backing"
            )

    def _validate_resident_placement(self, value):
        """Check concat fields against the finalized resident relation."""
        logical = self._storage_plan.ref(value)
        operation = logical.node
        placement = self._resident_placement(logical)
        if placement is None:
            return
        serialized = operation.meta.get(TARGET_CODEGEN_KWARGS, {})
        if placement.concat_channel_index is None:
            if serialized.get("is_to_concat"):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: pitch-only resident write was "
                    "serialized as a concat"
                )
        elif (
            serialized.get("concat_channel_index_0")
            != placement.concat_channel_index
            or serialized.get("concat_idx_0") != placement.slot_index
            or serialized.get("concat_size_0") != placement.slot_channels
            or serialized.get("concat_total_size_0")
            != placement.resident_channels
            or serialized.get("is_to_concat") is not True
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: resident concat fields do not match "
                "its physical storage relation"
            )
        shape = tuple(self._storage_plan.logical_tensors[logical].shape)
        if (
            len(shape) == 4
            and self._physical_row_pitch_channels(
                logical,
                self._storage_plan,
                self._storage_plane,
            )
            != placement.resident_channels
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: resident concat pitch does not match "
                "its physical backing"
            )

    def _validate_resident_placements(self):
        """Validate every resident writer declared by the finalized plan."""
        for logical, relation in self._storage_plan.relations[
            self._storage_plane
        ].items():
            transform = relation.transform
            if (
                transform is not None
                and transform.kind == "alpha_resident_write"
            ):
                self._validate_resident_placement(logical)

    def _validate_distinct_operand_banks(
        self,
        operation,
        operands,
        records_by_tensor,
    ):
        position = self._execution_position(operation)
        records = []
        for operand in operands:
            record = self._required_live_record(
                operation,
                operand,
                position,
                records_by_tensor,
                "execution",
            )
            if record is not None:
                records.append((operand, record))
        for index, (left_node, left) in enumerate(records):
            for right_node, right in records[index + 1 :]:
                if left.physical_tensor == right.physical_tensor:
                    # Components such as an int4 payload and its embedded
                    # scale are distinct logical operands of one physical
                    # word. Their intentional co-location is not a bank
                    # conflict.
                    continue
                if left.memory_name != right.memory_name:
                    continue
                overlap = self._record_banks(left).intersection(
                    self._record_banks(right)
                )
                if overlap:
                    raise AlphaMemoryPlanError(
                        f"{operation.name}: operands {left_node.name} and "
                        f"{right_node.name} share banks {sorted(overlap)} "
                        f"({allocation_record_label(left)} versus "
                        f"{allocation_record_label(right)})"
                    )

    def _validate_one_bank_pool(
        self,
        operation,
        operands,
        records_by_tensor,
    ):
        position = self._execution_position(operation)
        tensors = [*operands[:1], operation]
        for tensor in tensors:
            record = self._live_record_for(
                tensor,
                position,
                records_by_tensor,
            )
            if record is None:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: missing one-bank storage for "
                    f"{tensor.name}"
                )
            address = AlphaAddress.decode(record.address)
            if (
                record.memory_name != self.memory_level.input_memory
                or address.cluster != 1
                or address.operand_bits != 8
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: {tensor.name} must use INT8 "
                    "GBScratch cluster 1"
                )

    def _validate_nmp_depthwise_weight(
        self,
        operation,
        info,
        records_by_tensor,
    ):
        weights = tuple(
            dict.fromkeys(
                descriptor.tensor
                for descriptor in info.operand_descriptors
                if descriptor.role is TensorRole.WEIGHT
            )
        )
        if not weights:
            raise AlphaMemoryPlanError(
                f"{operation.name}: depthwise NMP has no weight operand"
            )
        position = self._execution_position(operation)
        for weight in weights:
            record = self._live_record_for(
                weight,
                position,
                records_by_tensor,
            )
            if record is None:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: depthwise weight {weight.name} is "
                    "not live"
                )
            address = AlphaAddress.decode(record.address)
            if (
                record.memory_name != self.memory_level.input_memory
                or address.cluster != 2
                or address.operand_bits != 8
            ):
                raise AlphaMemoryPlanError(
                    f"{operation.name}: depthwise weight {weight.name} must "
                    "use INT8 GBScratch cluster 2"
                )

    def _validate_shuffle_copy(
        self,
        operation,
        operands,
        records_by_tensor,
    ):
        """Validate one copy's exact serialized source and destination.

        A shuffle-copy node aliases the whole shuffle backing for liveness,
        while ``physical_scratchpad_map`` narrows execution to one source and
        one destination entry. A move may read and then write different
        entries in the same bank, so the ordinary simultaneous-operand bank
        rule does not apply to this serialized operation.
        """
        if len(operands) != 1:
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy requires one input"
            )

        source_node = operands[0]
        physical_map = operation.meta.get(PHYSICAL_SCRATCHPAD_MAP)
        if not isinstance(physical_map, dict):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy has no physical map"
            )
        source = physical_map.get(source_node)
        destination = physical_map.get(operation)
        if not isinstance(source, Segment) or not isinstance(
            destination,
            Segment,
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy physical map is incomplete"
            )

        target_kwargs = operation.meta.get(TARGET_CODEGEN_KWARGS, {})
        plan, _movement = self._copy_plans[operation]
        pitch_entries = self._shuffle_access_pattern(plan.shuffle).pitch_entries
        if (
            target_kwargs.get("nmp_c2") != pitch_entries - 1
            or target_kwargs.get("nmp_k2") != pitch_entries - 1
            or target_kwargs.get("is_to_concat") is not True
            or target_kwargs.get("concat_total_size_0")
            != int(plan.shuffle.shape[1])
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy pitch does not match "
                "its physical backing"
            )
        source_payload = target_kwargs.get("copy_source_payload")
        destination_payload = target_kwargs.get("copy_destination_payload")
        if not isinstance(source_payload, PhysicalPayloadRef) or not isinstance(
            destination_payload, PhysicalPayloadRef
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy has no physical payload "
                "identities"
            )

        position = self._execution_position(operation)
        source_record = self._live_record_for_physical_segment(
            source_payload.physical_tensor,
            source,
            position,
            records_by_tensor,
        )
        destination_record = self._live_record_for_physical_segment(
            destination_payload.physical_tensor,
            destination,
            position,
            records_by_tensor,
        )
        if source_record is None or destination_record is None:
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy physical view is not "
                "uniquely live"
            )
        if source_record.memory_name != destination_record.memory_name:
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy crosses memory instances"
            )
        if (
            source_payload.physical_tensor != source_record.physical_tensor
            or destination_payload.physical_tensor
            != destination_record.physical_tensor
            or source_payload.segment.address != source_record.address
            or destination_payload.segment.address != destination_record.address
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy payload endpoints do not "
                "match its live physical allocations"
            )

        block_channels = operation.meta.get("magnet_alpha", {}).get(
            "block_channels"
        )
        if not isinstance(block_channels, int) or block_channels <= 0:
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy has invalid block width"
            )
        operand_bits = _node_operand_bits(operation)
        expected_bytes = payload_size_bytes(block_channels, operand_bits)
        if (
            source.end - source.start != expected_bytes
            or destination.end - destination.start != expected_bytes
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy does not describe one "
                f"{block_channels}-channel block"
            )

        memory = self.memory_level.memory(source_record.memory_name)
        depth = self._depth_per_bank(memory)

        def entry_cells(segment):
            address = AlphaAddress.decode(segment.address)
            linear_entry = address.virtual_bank * depth + address.entry
            return frozenset(
                self._entry_cells(memory, linear_entry, address.cluster)
            )

        if entry_cells(source).intersection(entry_cells(destination)):
            raise AlphaMemoryPlanError(
                f"{operation.name}: shuffle copy source and destination "
                "overlap"
            )

    def _validate_shuffle_outputs(
        self,
        operation,
        operands,
        records_by_tensor,
    ):
        storage = self._shuffle_storage.get(operation)
        if storage is None:
            raise AlphaMemoryPlanError(
                f"{operation.name}: Alpha shuffle storage is missing"
            )
        serialized = operation.meta.get(TARGET_CODEGEN_KWARGS, {})
        addresses = serialized.get("output_addresses", ())
        if tuple(addresses) != storage.output_addresses:
            raise AlphaMemoryPlanError(
                f"{operation.name}: serialized shuffle addresses do not "
                "match its storage plan"
            )
        instances = (
            serialized.get("output0_memory_instance"),
            serialized.get("output1_memory_instance"),
        )
        expected_instances = tuple(
            output.memory_instance for output in storage.outputs
        )
        if instances != expected_instances or any(
            not isinstance(instance, str) or not instance
            for instance in instances
        ):
            raise AlphaMemoryPlanError(
                f"{operation.name}: serialized shuffle memory instances "
                "do not match its storage plan"
            )
        position = self._execution_position(operation)
        output_records = []
        output_physical = (
            self._storage_plan.physical_id(
                operation,
                self._storage_plane,
            ),
            self._shuffle_secondary_id(operation),
        )
        for output, physical in zip(storage.outputs, output_physical):
            record = self._live_record_for_physical_segment(
                physical,
                output,
                position,
                records_by_tensor,
            )
            if record is None:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: shuffle output {output.address:#x} "
                    "has no unique live allocation"
                )
            output_records.append(record)

        input_records = []
        for operand in operands:
            record = self._required_live_record(
                operation,
                operand,
                position,
                records_by_tensor,
                "shuffle input",
            )
            if record is not None:
                input_records.append((operand, record))
        for output_record in output_records:
            for operand, input_record in input_records:
                if output_record.memory_name != input_record.memory_name:
                    continue
                overlap = self._record_banks(output_record).intersection(
                    self._record_banks(input_record)
                )
                if overlap:
                    raise AlphaMemoryPlanError(
                        f"{operation.name}: shuffle output and "
                        f"{operand.name} share banks {sorted(overlap)}"
                    )

        access = self._shuffle_access_pattern(operation)
        for offset in access.offsets():
            first = self._shuffle_banks_at_offset(addresses[0], offset)
            second = self._shuffle_banks_at_offset(addresses[1], offset)
            overlap = first.intersection(second)
            if overlap:
                raise AlphaMemoryPlanError(
                    f"{operation.name}: simultaneous shuffle write at "
                    f"offset {offset} shares banks {sorted(overlap)}"
                )

    def _validate_operation_addresses(self, records_by_tensor):
        positions = execution_positions(self._storage_plan.execution_order)
        for operation, info in self._operation_operands.items():
            if operation not in positions:
                continue
            if info.kind == "channel_shuffle_copy":
                self._validate_shuffle_copy(
                    operation,
                    info.operands,
                    records_by_tensor,
                )
            else:
                self._validate_distinct_operand_banks(
                    operation,
                    (*info.operands, operation),
                    records_by_tensor,
                )
            if info.kind != "gemm":
                self._validate_nmp_input_view(
                    operation,
                    info,
                )
            if info.kind == "pool2d":
                self._validate_pool_output_pitch(operation)
                physical = self._constraint_physical_id(operation)
                one_bank = self._cluster_constraints.get(physical) == 1
                serialized = operation.meta.get(
                    TARGET_CODEGEN_KWARGS,
                    {},
                ).get("is_pool_1bank")
                if serialized is not one_bank:
                    raise AlphaMemoryPlanError(
                        f"{operation.name}: serialized one-bank pool state "
                        "does not match its physical placement constraint"
                    )
                if one_bank:
                    self._validate_one_bank_pool(
                        operation,
                        info.operands,
                        records_by_tensor,
                    )
            if info.kind == "depthwise_conv2d":
                self._validate_nmp_depthwise_weight(
                    operation,
                    info,
                    records_by_tensor,
                )
            if info.kind == "channel_shuffle":
                self._validate_shuffle_outputs(
                    operation,
                    info.operands,
                    records_by_tensor,
                )

    def _validate_materialized_input_storage(self, records_by_tensor):
        for plan in self._materialized_input_padding.values():
            position = self._execution_position(plan.operation)
            record = self._required_live_record(
                plan.operation,
                plan.input,
                position,
                records_by_tensor,
                "materialized input",
            )
            required = payload_size_bytes(
                math.prod(plan.physical_shape),
                plan.operand_bits,
            )
            if record is None or record.end - record.start < required:
                raise AlphaMemoryPlanError(
                    f"{plan.operation.name}: materialized input requires "
                    f"{required} bytes"
                )
            address = AlphaAddress.decode(record.address)
            if address.operand_bits != 8 or address.cluster != 1:
                raise AlphaMemoryPlanError(
                    f"{plan.operation.name}: materialized image input must "
                    "use an INT8 one-bank cluster"
                )
            serialized = plan.implementation.meta.get(
                TARGET_CODEGEN_KWARGS,
                {},
            )
            padding = serialized.get("materialized_input_padding")
            payload = serialized.get("materialized_input_payload")
            if tuple(padding or ()) != plan.padding:
                raise AlphaMemoryPlanError(
                    f"{plan.operation.name}: materialized input padding is "
                    "not serialized"
                )
            valid_channels = serialized.get("materialized_input_valid_channels")
            resident_channels = serialized.get(
                "materialized_input_resident_channels"
            )
            if plan.serialize_channels:
                if (
                    valid_channels != plan.valid_channels
                    or resident_channels != plan.resident_channels
                ):
                    raise AlphaMemoryPlanError(
                        f"{plan.operation.name}: materialized input channel "
                        "contract is not serialized"
                    )
            elif valid_channels is not None or resident_channels is not None:
                raise AlphaMemoryPlanError(
                    f"{plan.operation.name}: spatial-only materialized "
                    "input declares a channel contract"
                )
            if (
                not isinstance(payload, PhysicalPayloadRef)
                or payload.physical_tensor
                != self._storage_plan.physical_id(
                    plan.input,
                    self._storage_plane,
                )
                or payload.representation.shape != plan.physical_shape
                or payload.segment.address != record.address
            ):
                raise AlphaMemoryPlanError(
                    f"{plan.operation.name}: materialized input payload is "
                    "not bound to its physical storage"
                )

    def _validate_general_plan(self, model, history, records_by_tensor):
        """Validate target-independent allocation lifetime invariants."""
        validate_allocation_lifecycles(
            history,
            memory_level=self.memory_level,
            access_requirements_for=self._access_requirements_for,
            error_type=AlphaMemoryPlanError,
        )
        self._validate_execution_liveness(model, records_by_tensor)

    def _validate_alpha_rules(self, history, records_by_tensor):
        """Validate Alpha address encoding and execution-unit constraints."""
        self._validate_physical_history(history)
        self._validate_resident_placements()
        self._validate_operation_addresses(records_by_tensor)
        self._validate_materialized_input_storage(records_by_tensor)

    def _validate_physical_payload_uses(self):
        for physical, uses in self._payload_uses_by_physical.items():
            for use in uses:
                payload = use.operation.meta.get(
                    TARGET_CODEGEN_KWARGS,
                    {},
                ).get(use.field)
                if not isinstance(payload, PhysicalPayloadRef):
                    raise AlphaMemoryPlanError(
                        f"{use.operation.name}: {use.field} was not bound "
                        "to finalized physical storage"
                    )
                if (
                    payload.physical_tensor != physical
                    or payload.component != use.component
                ):
                    raise AlphaMemoryPlanError(
                        f"{use.operation.name}: {use.field} references the "
                        "wrong physical component"
                    )

    def validate_plan(self, model: GraphModule):
        """Finalize and validate Alpha storage over the FX execution chain."""
        self._ensure_mutable()
        history = self._freeze_history()
        records_by_tensor = index_allocation_records(history)
        self._validate_general_plan(model, history, records_by_tensor)
        self._validate_alpha_rules(history, records_by_tensor)
        self._validate_physical_payload_uses()
        self._history = history
        self._sealed = True
        return history

    @property
    def allocation_history(self):
        """Return the finalized immutable Alpha allocation history."""
        if self._history is None:
            raise RuntimeError("Alpha memory plan has not been finalized")
        return self._history

    def snapshot(self):
        for memory in self.memory_level.instances:
            active = [
                copy.copy(allocation.segment)
                for allocation in self._allocations.values()
                if allocation.memory_name == memory.name
            ]
            self._snapshots[memory.name].append(active)

    def dump_snapshots(self, filename="dump_snapshot.png"):
        root, extension = os.path.splitext(filename)
        for index, memory in enumerate(self.memory_level.instances):
            snapshots = self._snapshots[memory.name]
            if not snapshots or not any(snapshots):
                continue
            plotter = MemoryAllocator(
                total_memory=(
                    memory.bank_count
                    * self._depth_per_bank(memory)
                    * memory.bits_per_bank_entry
                    // 8
                ),
                memory_space=index,
            )
            plotter.snapshots = snapshots
            plotter.dump_snapshots(f"{root}_{memory.name}{extension}")


__all__ = [
    "ALPHA_GB_SCRATCH_BASE",
    "ALPHA_GB_WEIGHT_BASE",
    "ALPHA_RESERVED_GB_SCRATCH_BANKS",
    "AlphaAllocationRecord",
    "AlphaAddress",
    "AlphaMemoryPlanError",
    "AlphaScratchpadAllocator",
]
