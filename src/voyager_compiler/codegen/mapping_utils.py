# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import contextvars
import copy
import hashlib
import logging
import operator
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

from voyager_compiler.codegen.hardware import COMMON_IMAGE_INPUT_CHANNELS
from voyager_compiler.codegen.param_pb2 import (
    Argument,
    LevelAccessCount,
    LevelTiling,
    LoopBound,
    OpOverload,
    Tensor,
    Tiling,
)
from voyager_compiler.codegen.passes.utils import get_named_arg
from voyager_compiler.codegen.storage.types import (
    PhysicalPayloadRef,
    SerializedTensorRef,
    StoragePlane,
    ValueRef,
)
from voyager_compiler.codegen.storage_predicates import (
    is_address_only_view,
    is_identity_indexing_op,
    is_zero_copy_storage_op,
)

logger = logging.getLogger(__name__)

SOURCE_OP_NAME = "source_op_name"
LOCAL_SCHEDULE = "local_schedule"
LOCAL_SCHEDULER_WORKLOAD = "local_scheduler_workload"
SCHEDULE_FACTOR_PREFIX = "schedule_factor_"
TARGET_INSTRUCTION_NOP = "target_instruction_nop"
TARGET_CODEGEN_KWARGS = "target_codegen_kwargs"
UNSUPPORTED_OPERATION_META = "unsupported_operation"
SERIALIZED_UNSUPPORTED_OPERATION = "unsupported"
DECLARED_UNSUPPORTED_DISPOSITION = "declared_unsupported"
FIRST_CONV_INPUT_CHANNELS = COMMON_IMAGE_INPUT_CHANNELS

_UNSUPPORTED_EXECUTION_META_KEYS = (
    "compute_unit",
    LOCAL_SCHEDULE,
    LOCAL_SCHEDULER_WORKLOAD,
    "physical_scratchpad_map",
    "scratchpad",
    "scratchpad_map",
)


def _serialized_target_name(node: Node) -> str:
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        return schema.name
    return str(node.target)


def _unsupported_source_operation(node: Node) -> dict[str, str]:
    return {
        "name": node.name,
        "op": node.op,
        "target": _serialized_target_name(node),
    }


def _assert_no_unsupported_execution_state(node: Node) -> None:
    present = [
        key
        for key in _UNSUPPORTED_EXECUTION_META_KEYS
        if node.meta.get(key) is not None
    ]
    if present:
        raise ValueError(
            f"{node.name}: declared unsupported operation has target "
            "execution state: " + ", ".join(sorted(present))
        )


def validate_unsupported_operation_declaration(
    node: Node,
) -> dict[str, object]:
    """Validate and normalize one explicit unsupported call boundary."""
    declaration = node.meta.get(UNSUPPORTED_OPERATION_META)
    if not isinstance(declaration, Mapping):
        raise TypeError(
            f"{node.name}: invalid unsupported-operation declaration"
        )
    if declaration.get("schema_version") != 1:
        raise ValueError(
            f"{node.name}: unsupported declaration has an unknown schema"
        )
    if declaration.get("disposition") != DECLARED_UNSUPPORTED_DISPOSITION:
        raise ValueError(
            f"{node.name}: unsupported declaration has invalid disposition"
        )

    normalized = dict(declaration)
    for field in ("reason_code", "detail", "stage"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{node.name}: unsupported declaration needs {field}"
            )

    sources = normalized.get("source_operations")
    if not isinstance(sources, (tuple, list)) or not sources:
        raise TypeError(
            f"{node.name}: unsupported declaration needs source_operations"
        )
    normalized_sources = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError(
                f"{node.name}: unsupported source operation must be a mapping"
            )
        fields = {name: source.get(name) for name in ("name", "op", "target")}
        if any(
            not isinstance(value, str) or not value for value in fields.values()
        ):
            raise ValueError(
                f"{node.name}: unsupported source operation is incomplete"
            )
        normalized_sources.append(fields)
    normalized["source_operations"] = tuple(normalized_sources)

    facts = normalized.get("facts", {})
    if not isinstance(facts, Mapping):
        raise TypeError(f"{node.name}: unsupported facts must be a mapping")
    normalized["facts"] = dict(facts)
    _assert_no_unsupported_execution_state(node)
    return normalized


def declare_unsupported_operation(
    node: Node,
    *,
    reason_code: str,
    detail: str,
    stage: str,
    facts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Mark one final call as an explicit non-accelerator boundary."""
    if node.op not in {"call_function", "call_module"}:
        raise ValueError(
            f"{node.name}: only call owners may be declared unsupported"
        )
    for field, value in (
        ("reason_code", reason_code),
        ("detail", detail),
        ("stage", stage),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"unsupported declaration needs nonempty {field}")
    if facts is not None and not isinstance(facts, Mapping):
        raise TypeError("unsupported declaration facts must be a mapping")

    declaration = {
        "schema_version": 1,
        "disposition": DECLARED_UNSUPPORTED_DISPOSITION,
        "reason_code": reason_code,
        "detail": detail,
        "stage": stage,
        "source_operations": (_unsupported_source_operation(node),),
        "facts": dict(facts or {}),
    }
    _assert_no_unsupported_execution_state(node)
    previous = node.meta.get(UNSUPPORTED_OPERATION_META)
    if previous is not None:
        previous = validate_unsupported_operation_declaration(node)
        if previous != declaration:
            raise ValueError(
                f"{node.name}: conflicting unsupported-operation declaration"
            )
        return previous
    node.meta[UNSUPPORTED_OPERATION_META] = declaration
    return validate_unsupported_operation_declaration(node)


def propagate_unsupported_operation(
    nodes: Sequence[Node],
    owner: Node,
) -> None:
    """Move member declarations to their complete fused call owner."""
    declarations = [
        validate_unsupported_operation_declaration(node)
        for node in nodes
        if UNSUPPORTED_OPERATION_META in node.meta
    ]
    if not declarations:
        return

    reason_codes = tuple(
        dict.fromkeys(item["reason_code"] for item in declarations)
    )
    details = tuple(dict.fromkeys(item["detail"] for item in declarations))
    stages = tuple(dict.fromkeys(item["stage"] for item in declarations))
    sources = []
    facts = []
    for declaration in declarations:
        for source in declaration["source_operations"]:
            if source not in sources:
                sources.append(source)
        facts.append(declaration["facts"])
    merged = {
        "schema_version": 1,
        "disposition": DECLARED_UNSUPPORTED_DISPOSITION,
        "reason_code": "+".join(reason_codes),
        "detail": "; ".join(details),
        "stage": "+".join(stages),
        "source_operations": tuple(sources),
        "facts": {"members": tuple(facts)},
    }
    _assert_no_unsupported_execution_state(owner)
    previous = owner.meta.get(UNSUPPORTED_OPERATION_META)
    if previous is not None:
        previous = validate_unsupported_operation_declaration(owner)
        if previous != merged:
            raise ValueError(
                f"{owner.name}: conflicting fused unsupported declaration"
            )
        return
    owner.meta[UNSUPPORTED_OPERATION_META] = merged
    validate_unsupported_operation_declaration(owner)


def is_declared_unsupported_operation(node: Node) -> bool:
    """Return whether one node has a valid explicit declaration."""
    if UNSUPPORTED_OPERATION_META not in node.meta:
        return False
    validate_unsupported_operation_declaration(node)
    return True


def bind_local_schedule(node, result, workload=None) -> None:
    """Bind one schedule and its immutable named target facts.

    Rebinding is idempotent only for an equal result. Target facts are
    serialized through the ordinary target-codegen kwargs path so fused and
    standalone records use one contract.
    """
    previous_result = node.meta.get(LOCAL_SCHEDULE)
    if previous_result is not None and previous_result != result:
        raise ValueError(f"{node.name}: conflicting local schedules")
    previous_workload = node.meta.get(LOCAL_SCHEDULER_WORKLOAD)
    if (
        previous_workload is not None
        and workload is not None
        and previous_workload != workload
    ):
        raise ValueError(f"{node.name}: conflicting scheduler workloads")

    target_kwargs = dict(node.meta.get(TARGET_CODEGEN_KWARGS, {}))
    named_kwargs = {
        f"{SCHEDULE_FACTOR_PREFIX}{name}": value
        for name, value in result.named_factors
    }
    existing_factor_keys = {
        key for key in target_kwargs if key.startswith(SCHEDULE_FACTOR_PREFIX)
    }
    stale_factor_keys = existing_factor_keys - set(named_kwargs)
    if stale_factor_keys:
        raise ValueError(
            f"{node.name}: conflicting named schedule factors: "
            + ", ".join(sorted(stale_factor_keys))
        )
    for key, value in named_kwargs.items():
        if key in target_kwargs and target_kwargs[key] != value:
            raise ValueError(
                f"{node.name}: conflicting named schedule factor {key}"
            )
        target_kwargs[key] = value
    if named_kwargs:
        node.meta[TARGET_CODEGEN_KWARGS] = target_kwargs
    node.meta[LOCAL_SCHEDULE] = result
    if workload is not None:
        node.meta[LOCAL_SCHEDULER_WORKLOAD] = workload


def _write_numpy(np_array: np.ndarray, filename: str, shape: tuple):
    """Worker function: write the tensor and log when done."""
    try:
        np_array.tofile(filename)
        logger.info(f"✅ Saved tensor {shape} -> {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save tensor {shape} -> {filename}: {e}")
        raise


class TensorWriteSession:
    """Own asynchronous tensor writes for exactly one serialization run."""

    def __init__(self, max_workers=None):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or os.cpu_count() or 4
        )
        self._pending = []
        self._pending_lock = threading.Lock()
        self._payload_digests = {}
        self._token = None
        self._closed = False

    @staticmethod
    def _numpy_payload(tensor):
        value = tensor.detach().cpu().contiguous().to(torch.float32)
        return value.numpy(), tuple(value.shape)

    def submit(self, tensor: torch.Tensor, filename: str) -> None:
        if self._closed:
            raise RuntimeError("Tensor write session is closed")
        np_array, shape = self._numpy_payload(tensor)
        digest = hashlib.sha256(np_array.tobytes()).digest()
        signature = (shape, np_array.dtype.str, digest)
        previous = self._payload_digests.get(filename)
        if previous is not None:
            if previous != signature:
                raise ValueError(
                    f"{filename}: conflicting tensor payloads in one compile"
                )
            return
        self._payload_digests[filename] = signature
        future = self._executor.submit(
            _write_numpy,
            np_array,
            filename,
            shape,
        )
        with self._pending_lock:
            self._pending.append(future)

    def drain(self) -> None:
        while True:
            with self._pending_lock:
                pending = tuple(self._pending)
                self._pending.clear()
            if not pending:
                return
            for future in pending:
                future.result()

    def __enter__(self):
        if _active_tensor_write_session.get() is not None:
            raise RuntimeError("Tensor write sessions cannot be nested")
        self._token = _active_tensor_write_session.set(self)
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True
            if self._token is not None:
                _active_tensor_write_session.reset(self._token)
                self._token = None
        return False


_active_tensor_write_session = contextvars.ContextVar(
    "voyager_tensor_write_session",
    default=None,
)


def _save_tensor(tensor: torch.Tensor, filename: str):
    """Save through the active compile session, or synchronously if absent."""
    session = _active_tensor_write_session.get()
    if session is not None:
        session.submit(tensor, filename)
        return
    np_array, shape = TensorWriteSession._numpy_payload(tensor)
    _write_numpy(np_array, filename, shape)


def save_tensor(tensor, filename):
    """Save one serialized tensor through the compile-scoped writer."""
    _save_tensor(tensor, filename)


def _apply_transform(node, key, field, is_output=False):
    if (fused_op := node.meta.get(key)) is not None:
        fused_op.meta["tiled_shapes"] = node.meta.get("_tiled_shapes")
        fused_op.meta["scratchpad_map"] = node.meta.get("_scratchpad_map")
        field.CopyFrom(map_node(fused_op))
        if key == "reshape":
            field.kwargs["output_shape"].int_list.values.extend(node.shape)
        return fused_op.args[0] if not is_output else node
    return node


def _set_meminfo(field, segment):
    field.partition = int(segment.memory_space)
    field.address = int(
        segment.address if segment.address is not None else segment.start
    )
    if segment.memory_instance is not None:
        field.instance = segment.memory_instance


def set_tensor_field(field, node, output_dir=None, is_output=False):
    if not isinstance(node, Node) or not hasattr(node, "value"):
        raise TypeError(f"Expected node with value attribute, got {node!r}")

    tiled_shapes = node.meta.get("_tiled_shapes")
    tile_strides = node.meta.get("_tile_strides")
    scratchpad_map = node.meta.get("_scratchpad_map")

    # Apply transformations
    if node.op != "call_module" or is_output:
        node = _apply_transform(node, "reshape", field.reshape, is_output)
    node = _apply_transform(node, "dequantize", field.dequant)

    if (source_node := node.meta.get("source_node")) is not None:
        node = source_node

    if tiled_shapes is not None and node in tiled_shapes:
        field.tiled_shape.extend(tiled_shapes[node])

    if tile_strides is not None and node in tile_strides:
        field.tile_strides.extend(tile_strides[node])

    if scratchpad_map is not None and node in scratchpad_map:
        _set_meminfo(field.scratchpad, scratchpad_map[node])

    # Model-wide and bufferized paths carry a scratchpad ``Segment`` directly
    # (symmetric with ``memory`` below), rather than a per-consumer map.
    elif (scratchpad := node.meta.get("scratchpad")) is not None:
        _set_meminfo(field.scratchpad, scratchpad)

    # Tensor properties
    field.node = node.name
    field.shape.extend(node.shape or [1])

    if (dtype := node.meta.get("dtype")) is not None:
        field.dtype = dtype
    else:
        field.dtype = str(node.value.dtype).split(".")[1]

    if (memory := node.meta.get("memory")) is not None:
        _set_meminfo(field.memory, memory)

    if output_dir is not None:
        save_tensor(node.value, os.path.join(output_dir, f"{node.name}.bin"))


def _is_tile_dma(node):
    """True for a side-effecting tile-DMA node (``voyager.copy_tile`` / ``store_tile``):
    returns ``None``, mutating its ``dst`` (``args[1]``) in place, so its emitted output
    is described from that dest buffer."""
    schema = getattr(getattr(node, "target", None), "_schema", None)
    return schema is not None and schema.name in (
        "voyager::copy_tile",
        "voyager::store_tile",
    )


def _set_output_component_field(
    field,
    node,
    index,
    output_dir=None,
):
    """Serialize one explicit component of a tuple-valued FX result."""
    values = node.value
    if (
        not isinstance(values, (tuple, list))
        or index < 0
        or index >= len(values)
        or not isinstance(values[index], torch.Tensor)
    ):
        raise ValueError(f"{node.name}: output[{index}] is not a tensor result")
    value = values[index]
    dtypes = node.meta.get("dtype")
    dtype = dtypes[index] if isinstance(dtypes, (tuple, list)) else None
    field.node = f"{node.name}_{index}"
    field.shape.extend(value.shape)
    field.dtype = dtype or str(value.dtype).split(".")[1]

    for field_name, segment_key, aggregate_key, sizes_key in (
        ("memory", "memory_output_segments", "memory", "output_sizes"),
        (
            "scratchpad",
            "scratchpad_output_segments",
            "scratchpad",
            "output_sizes",
        ),
    ):
        segments = node.meta.get(segment_key)
        segment = None
        if segments is not None:
            if len(segments) != len(values):
                raise ValueError(
                    f"{node.name}: {segment_key} count does not match "
                    "its output count"
                )
            segment = segments[index]
        else:
            aggregate = node.meta.get(aggregate_key)
            sizes = node.meta.get(sizes_key)
            if aggregate is not None and sizes is not None:
                segment = copy.copy(aggregate)
                segment.start += sum(sizes[:index])
        if segment is not None:
            _set_meminfo(getattr(field, field_name), segment)

    if not field.HasField("scratchpad"):
        scratchpad_maps = (
            node.meta.get("_scratchpad_map"),
            node.meta.get("physical_scratchpad_map"),
            node.meta.get("scratchpad_map"),
        )
        output = ValueRef(node, (index,))
        segment = None
        for scratchpad_map in scratchpad_maps:
            if scratchpad_map is None:
                continue
            if output in scratchpad_map:
                segment = scratchpad_map[output]
            elif node in scratchpad_map:
                segment = copy.copy(scratchpad_map[node])
                sizes = node.meta.get("tiled_output_sizes")
                if sizes is not None:
                    segment.start += sum(sizes[:index])
            if segment is not None:
                break
        if segment is not None:
            _set_meminfo(field.scratchpad, segment)

    tiled_shapes = node.meta.get("tiled_shapes")
    if tiled_shapes is not None and node in tiled_shapes:
        field.tiled_shape.extend(tiled_shapes[node][index])
    if output_dir is not None:
        save_tensor(
            value,
            os.path.join(output_dir, f"{node.name}_{index}.bin"),
        )


def _set_physical_payload_field(
    field,
    payload,
    output_dir=None,
):
    """Serialize one finalized plan component at its explicit placement."""
    representation = payload.representation
    field.node = payload.name
    field.shape.extend(representation.shape)
    field.dtype = str(representation.dtype).split(".")[-1]

    if payload.physical_tensor.plane == StoragePlane.SCRATCHPAD.value:
        _set_meminfo(field.scratchpad, payload.segment)
    elif payload.physical_tensor.plane == StoragePlane.DRAM.value:
        _set_meminfo(field.memory, payload.segment)
    else:
        raise ValueError(
            f"{payload.name}: unknown physical storage plane "
            f"{payload.physical_tensor.plane!r}"
        )
    if output_dir is not None:
        if representation.value is None:
            raise ValueError(
                f"{payload.name}: physical component has no serializable value"
            )
        save_tensor(
            representation.value,
            os.path.join(output_dir, f"{payload.name}.bin"),
        )


def _set_serialized_artifact_field(
    field,
    reference,
    output_dir=None,
):
    """Serialize a plan-owned nonresident target encoding."""
    artifact = reference.artifact
    field.node = reference.name
    field.shape.extend(artifact.shape)
    field.dtype = str(artifact.dtype).split(".")[-1]
    if output_dir is not None:
        save_tensor(
            artifact.value,
            os.path.join(output_dir, f"{reference.name}.bin"),
        )


def set_output_field(param, node, output_dir):
    if isinstance(node.value, torch.Tensor):
        node.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        node.meta["_scratchpad_map"] = node.meta.get(
            "physical_scratchpad_map",
            node.meta.get("scratchpad_map"),
        )

        set_tensor_field(param.output, node, output_dir, True)

        node.meta.pop("_tiled_shapes", None)
        node.meta.pop("_scratchpad_map", None)
    elif (
        isinstance(node.value, (tuple, list))
        and node.value
        and all(isinstance(v, int) for v in node.value)
    ):
        # An integer index vector (``increment_indices`` / ``delinearize_index``): a
        # named handle that the per-dimension component getitems index into — a tile
        # block index — not a tensor list.
        param.output.node = node.name
    elif isinstance(node.value, (tuple, list)):
        for i, _value in enumerate(node.value):
            tensor = Tensor()
            _set_output_component_field(
                tensor,
                node,
                i,
                output_dir,
            )
            param.outputs.tensors.append(tensor)
    elif isinstance(node.value, (int, bool)):
        # A scalar-valued op is an integer tile-index computation (a pipelined
        # prefetch's ``j + 1``); its result is a named scalar a load/store references,
        # not a tensor — record the name so the reference resolves.
        param.output.node = node.name
    elif node.value is None and _is_tile_dma(node):
        # A side-effecting tile DMA (``copy_tile`` / ``store_tile``; returns ``None`` — its
        # dest buffer is a closed-over additional input mutated in place).  Its result is
        # the tile it writes, described from the dest buffer (``args[1]``), which carries
        # the planned address; the source tile and block index travel in the op's args.
        set_tensor_field(param.output, node.args[1], output_dir, is_output=True)
    else:
        logger.warning(f"Unsupported output type: {type(node.value)}")


def build_tiling_proto(node):
    """Convert the canonical local schedule to a Tiling proto."""
    result = node.meta[LOCAL_SCHEDULE]
    mapping = result.mapping
    tiling = Tiling(name=node.name)

    # The PE spatial level is implicit in the hardware configuration.
    for level in range(1, mapping.num_levels):
        lt = LevelTiling()
        loop_index = 0
        while loop_index < len(mapping.loop_orders):
            matched = False
            for loop in range(len(mapping.loop_orders)):
                if mapping.loop_orders[loop][level] == loop_index:
                    lt.loop_bounds.append(
                        LoopBound(
                            loop=loop,
                            bound=mapping.loop_blockings[loop][level],
                        )
                    )
                    loop_index += 1
                    matched = True
                    break
            if not matched:
                break
        tiling.level_tilings.append(lt)
        if level < len(result.access_counts):
            access_counts = result.access_counts[level]
            tiling.level_access_counts.append(
                LevelAccessCount(
                    input_access_count=access_counts.input,
                    output_access_count=access_counts.output,
                    weight_access_count=access_counts.weight,
                )
            )

    return tiling


def convert_arg(value, output_dir=None) -> Argument:
    """
    Converts an argument (which could be a Tensor, list, int, float, etc.)
    into an Argument protobuf.
    """
    arg = Argument()

    if isinstance(value, PhysicalPayloadRef):
        _set_physical_payload_field(arg.tensor, value, output_dir)
    elif isinstance(value, SerializedTensorRef):
        _set_serialized_artifact_field(arg.tensor, value, output_dir)
    elif isinstance(value, ValueRef):
        if not value.path:
            set_tensor_field(arg.tensor, value.node, output_dir)
        elif len(value.path) == 1:
            _set_output_component_field(
                arg.tensor,
                value.node,
                value.path[0],
                output_dir,
            )
        else:
            raise ValueError(
                f"{value.name}: nested output paths are not serializable"
            )
    elif isinstance(value, torch.fx.Node):
        val = getattr(value, "value", None)

        if isinstance(val, torch.Tensor):
            set_tensor_field(arg.tensor, value, output_dir)
        elif isinstance(val, (tuple, list)):
            raise ValueError(
                f"{value.name}: tuple-valued operands require explicit "
                "ValueRef components"
            )
        else:
            # Scalar-valued node: a tile block index — the loop induction variable, or
            # a computed index such as a pipelined prefetch's ``j + 1`` (emitted as its
            # own Operation by ``_emit_node``).  Reference it by name so the address is
            # explicit (the loop's ``node`` names the induction variable), rather than
            # an empty positional placeholder that can only mean the bare counter.
            arg.tensor.node = value.name
    elif isinstance(value, bool):
        arg.bool_value = value
    elif isinstance(value, int):
        arg.int_value = value
    elif isinstance(value, float):
        arg.float_value = value
    elif isinstance(value, str):
        arg.str_value = value
    elif isinstance(
        value, (torch.dtype, torch.layout, torch.device, torch.memory_format)
    ):
        arg.str_value = str(value).split(".")[-1]
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, torch.fx.Node) or x is None for x in value):
            arg.tensor_list.tensors.extend(
                [
                    (
                        convert_arg(x).tensor
                        if x is not None
                        else Tensor(is_none=True)
                    )
                    for x in value
                ]
            )
        elif all(isinstance(x, bool) for x in value):
            arg.bool_list.values.extend(value)
        elif all(isinstance(x, int) for x in value):
            arg.int_list.values.extend(value)
        elif all(isinstance(x, (int, float, bool)) for x in value):
            arg.scalar_list.values.extend(value)
        else:
            raise TypeError(f"Unsupported list value: {value}")
    else:
        raise TypeError(f"Unsupported arg type: {type(value)}")

    return arg


def _target_reference_nodes(value):
    if isinstance(value, ValueRef):
        yield value.node
    elif isinstance(value, dict):
        for item in value.values():
            yield from _target_reference_nodes(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _target_reference_nodes(item)


def map_node(
    node: torch.fx.Node,
    output_dir=None,
    *,
    target_output_dir=None,
) -> OpOverload:
    """
    Converts a torch.fx.Node into an OpOverload protobuf message.
    """
    if hasattr(node.target, "_schema"):
        target = node.target._schema.name.split("::")[1]
    else:
        target = str(node.target)

    op_overload = OpOverload(
        name=node.name,
        op=node.op,
        target=target,
    )

    if (
        is_target_instruction_nop(node)
        or is_addressing_op(node)
        or node.target == operator.getitem
    ):
        op_overload.op = "nop"

    if node.target == torch.ops.aten.pad.default and not node.meta.get(
        TARGET_INSTRUCTION_NOP, False
    ):
        # This is an accelerator assignment descriptor, not a CPU lowering.
        # External system integration may assume CPU execution for it.
        op_overload.op = "unsupported"

    new_args_and_kwargs = normalize_function(
        node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
    )

    if new_args_and_kwargs is not None:
        args, kwargs = new_args_and_kwargs.args, new_args_and_kwargs.kwargs
    else:
        args, kwargs = node.args, node.kwargs

    if (
        node.target == operator.getitem
        and len(args) > 1
        and isinstance(args[0], Node)
        and isinstance(getattr(args[0], "value", None), (tuple, list))
        and isinstance(args[1], int)
    ):
        outputs = args[0].value
        index = args[1] % len(outputs)
        args = (ValueRef(args[0], (index,)), *args[1:])

    target_kwargs = node.meta.get(TARGET_CODEGEN_KWARGS, {})
    referenced_nodes = tuple(
        dict.fromkeys(
            reference
            for value in target_kwargs.values()
            for reference in _target_reference_nodes(value)
        )
    )
    context_nodes = tuple(
        dict.fromkeys((*node.all_input_nodes, *referenced_nodes))
    )
    temporary_metadata = {}
    # Pass L2 tiling metadata to both schema operands and explicit physical
    # payload references. Fused target metadata can refer to an outer FX node
    # that is not one of the inner operation's ordinary inputs.
    for n in context_nodes:
        temporary_metadata[n] = {
            key: n.meta.get(key)
            for key in (
                "_tiled_shapes",
                "_tile_strides",
                "_scratchpad_map",
            )
        }
        n.meta["_tiled_shapes"] = node.meta.get("tiled_shapes")
        n.meta["_tile_strides"] = node.meta.get("tile_strides")
        n.meta["_scratchpad_map"] = node.meta.get(
            "physical_scratchpad_map",
            node.meta.get("scratchpad_map"),
        )

    # Convert positional arguments
    for arg in args:
        op_overload.args.append(convert_arg(arg))

    # Convert keyword arguments
    for key, value in kwargs.items():
        if not "qmap" in key and value is not None:
            op_overload.kwargs[key].CopyFrom(convert_arg(value, output_dir))

    for key, value in target_kwargs.items():
        if key in kwargs:
            raise ValueError(
                f"{node.name}: target codegen kwarg conflicts with {key}"
            )
        payload_output_dir = (
            target_output_dir
            if isinstance(
                value,
                (PhysicalPayloadRef, SerializedTensorRef),
            )
            else output_dir
        )
        op_overload.kwargs[key].CopyFrom(convert_arg(value, payload_output_dir))

    for n, previous in temporary_metadata.items():
        for key, value in previous.items():
            if value is None:
                n.meta.pop(key, None)
            else:
                n.meta[key] = value

    if "l2_tiling" in node.meta:
        op_overload.kwargs["l2_tiling"].int_list.values.extend(
            node.meta["l2_tiling"]
        )

    return op_overload


aten = torch.ops.aten

_GENERIC_GEMM_OPERATORS = (
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.matmul.default,
    torch.ops.quantized_ops.conv2d.default,
    torch.ops.quantized_ops.linear.default,
    torch.ops.quantized_ops.matmul.default,
    torch.ops.quantized_ops.conv2d_mx.default,
    torch.ops.quantized_ops.linear_mx.default,
    torch.ops.quantized_ops.matmul_mx.default,
)
_GENERIC_CONV2D_OPERATORS = (
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
    torch.ops.quantized_ops.conv2d_mx.default,
)


def is_gemm_op(node: Node, operator_traits=None) -> bool:
    """Recognize generic GEMMs plus operators claimed by one target."""
    target_operators = (
        () if operator_traits is None else operator_traits.gemm_operators
    )
    return node.target in (*_GENERIC_GEMM_OPERATORS, *target_operators)


def is_conv2d(node: Node, operator_traits=None) -> bool:
    """Recognize generic convolutions plus operators claimed by one target."""
    target_operators = (
        () if operator_traits is None else operator_traits.conv2d_operators
    )
    return node.target in (
        *_GENERIC_CONV2D_OPERATORS,
        *target_operators,
    )


def is_depthwise_conv_shape(input_shape, weight_shape, groups) -> bool:
    """Recognize depthwise convolution from resolved operand geometry."""
    if (
        not isinstance(groups, int)
        or groups == 1
        or input_shape is None
        or weight_shape is None
        or len(input_shape) < 2
        or len(weight_shape) < 2
    ):
        return False
    input_channels = input_shape[1]
    weight_input_channels = weight_shape[1]
    output_channels = weight_shape[0]
    return (
        groups == input_channels
        and weight_input_channels == 1
        and output_channels % input_channels == 0
    )


def is_depthwise_conv(node: Node, operator_traits=None) -> bool:
    if not is_conv2d(node, operator_traits):
        return False
    groups = get_named_arg(node, "groups")
    input_node = get_named_arg(node, "input")
    weight = get_named_arg(node, "weight")
    return is_depthwise_conv_shape(
        getattr(input_node, "shape", None),
        getattr(weight, "shape", None),
        groups,
    )


def is_linear(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.linear.default,
        torch.ops.quantized_ops.linear_mx.default,
    ]


def is_matmul(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.matmul.default,
        torch.ops.quantized_ops.matmul.default,
        torch.ops.quantized_ops.matmul_mx.default,
    ]


def is_bmm(node: Node) -> bool:
    if is_matmul(node):
        input_shape = node.args[0].shape
        other_shape = node.args[1].shape
        return len(input_shape) > 2 or len(other_shape) > 2
    return False


def is_fully_connected(node: Node) -> bool:
    if is_linear(node):
        input_shape = node.args[0].shape
        return all(s == 1 for s in input_shape[:-1])

    if is_matmul(node):
        input_shape = node.args[0].shape
        other_shape = node.args[1].shape

        if is_bmm(node):
            return input_shape[-2] == 1
        else:
            return all(s == 1 for s in input_shape[:-1])

    return False


def is_pooling(node: Node) -> bool:
    return node.target in [
        # Core Aten IR
        aten._adaptive_avg_pool2d,
        aten._adaptive_avg_pool3d,
        aten.adaptive_avg_pool1d,
        aten.avg_pool1d,
        aten.avg_pool2d,
        aten.avg_pool3d,
        aten.max_pool2d_with_indices,
        aten.max_pool3d_with_indices,
        # export_for_training IR
        aten.adaptive_avg_pool2d.default,
        aten.avg_pool2d.default,
        aten.max_pool2d.default,
        # NHWC variants (after the data-layout transform)
        torch.ops.quantized_ops.max_pool2d.default,
        torch.ops.quantized_ops.adaptive_avg_pool2d.default,
    ]


def is_elementwise_op(node: Node) -> bool:
    return node.target in [
        # Core aten ops
        torch.ops.aten.abs.default,
        torch.ops.aten.acos.default,
        torch.ops.aten.acosh.default,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.asin.default,
        torch.ops.aten.asinh.default,
        torch.ops.aten.atan.default,
        torch.ops.aten.atan2.default,
        torch.ops.aten.atan2.out,
        torch.ops.aten.atanh.default,
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_and.Tensor,
        torch.ops.aten.bitwise_not.default,
        torch.ops.aten.bitwise_or.Scalar,
        torch.ops.aten.bitwise_or.Tensor,
        torch.ops.aten.bitwise_xor.Scalar,
        torch.ops.aten.bitwise_xor.Tensor,
        torch.ops.aten.ceil.default,
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp.Tensor,
        torch.ops.aten.cos.default,
        torch.ops.aten.cosh.default,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.div.Scalar_mode,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Tensor_mode,
        torch.ops.aten.eq.Scalar,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.erf.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.expm1.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.fmod.Scalar,
        torch.ops.aten.fmod.Tensor,
        torch.ops.aten.ge.Scalar,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.gelu.default,
        torch.ops.aten.gt.Scalar,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.isinf.default,
        torch.ops.aten.isnan.default,
        torch.ops.aten.le.Scalar,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.log.default,
        torch.ops.aten.log10.default,
        torch.ops.aten.log1p.default,
        torch.ops.aten.log2.default,
        torch.ops.aten.logical_and.default,
        torch.ops.aten.logical_not.default,
        torch.ops.aten.logical_or.default,
        torch.ops.aten.logical_xor.default,
        torch.ops.aten.lt.Scalar,
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.maximum.default,
        torch.ops.aten.minimum.default,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.ne.Scalar,
        torch.ops.aten.ne.Tensor,
        torch.ops.aten.neg.default,
        torch.ops.aten.nonzero.default,
        torch.ops.aten.pow.Scalar,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.pow.Tensor_Tensor,
        torch.ops.aten.reciprocal.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.remainder.Scalar,
        torch.ops.aten.remainder.Tensor,
        torch.ops.aten.round.default,
        torch.ops.aten.rsqrt.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.sign.default,
        torch.ops.aten.sin.default,
        torch.ops.aten.sinh.default,
        torch.ops.aten.sqrt.default,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.tan.default,
        torch.ops.aten.tanh.default,
        torch.ops.aten.trunc.default,
        torch.ops.aten.where.self,
        # Inplace versions of the above operations
        torch.ops.aten.add_.Scalar,
        torch.ops.aten.add_.Tensor,
        torch.ops.aten.mul_.Scalar,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.gelu_.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.relu_.default,
        # Quantization operations
        torch.ops.quantized_ops.dequantize.default,
        torch.ops.quantized_ops.quantize.default,
        torch.ops.quantized_ops.vmap.default,
        # Not in the core aten operator set. Will be removed in the future.
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
    ]


def is_reshape_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
    ]


def is_indexing_or_concatenation_op(node: Node) -> bool:
    return node.target in [
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]


def is_prunable_op(node: Node) -> bool:
    """Operations that can be safely deleted from fx.Graph."""
    return is_identity_indexing_op(node)


def is_nop(node: Node) -> bool:
    """Return whether ``node`` needs no accelerator instruction.

    This is the shared, target-independent emission policy for operations
    represented as NOPs in the serialized program.  It does not describe
    physical storage identity: view, copy, and target-specific forwarding
    relations are modeled by storage analysis.
    """
    if is_prunable_op(node):
        return True

    if node.target in {
        torch.ops.aten.dropout.default,
        torch.ops.aten.to.dtype,
    }:
        return is_zero_copy_storage_op(node)

    return is_address_only_view(node)


def is_target_instruction_nop(node: Node) -> bool:
    """Return whether shared or selected-target policy emits a NOP."""
    return is_nop(node) or bool(node.meta.get(TARGET_INSTRUCTION_NOP, False))


def is_addressing_op(node: Node) -> bool:
    """
    The following operations are handled by the memory placement and
    thus require no additional handling:
    """
    if node.target in [
        torch.ops.aten.stack.default,
        torch.ops.aten.cat.default,
    ]:
        dim = node.kwargs.get(
            "dim",
            node.args[1] if len(node.args) > 1 else 0,
        )
        return dim == 0

    return False
