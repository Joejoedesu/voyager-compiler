"""Encode frozen Alpha calls and the explicit buffer-transfer control program.

CMOD layer files are projections of selected instructions, never another memory
planner. Buffer movement remains visible in program.json and the canonical
model.txt. Host replay is not physical-bank or C-model execution evidence.
"""

import hashlib
import json
import operator
from collections import Counter

import numpy as np
import torch

from voyager_compiler.artifacts import to_data
from voyager_compiler.compiler.allocate import _buffer_identity


def encoded_address(node, config, bits, entry_offset=0):
    memory_name = node.meta.get("memory_instance")
    if memory_name not in ("global_activation", "global_weight"):
        raise ValueError(
            f"{node.name}: native Alpha operand needs a selected global memory"
        )
    memory = config.memory_instance(memory_name)
    layout = node.meta.get("bank_layout")
    if layout is None or layout.cluster not in (1, 2, 4):
        raise ValueError(
            f"{node.name}: native Alpha operand needs a supported bank cluster"
        )
    depth = memory.size.value // (memory.banks * memory.word_bytes)
    if depth != 1024:
        raise ValueError("Alpha address encoding requires 10-bit bank entries")
    entry = node.meta.get("bank_entry")
    if entry is None or entry_offset < 0:
        raise ValueError(f"{node.name}: missing final bank placement")
    linear = entry + entry_offset
    bank, offset = divmod(linear, depth)
    cells = {
        (b, e)
        for b, start, stop in node.meta["bank_ranges"]
        for e in range(start, stop)
    }
    if any(
        (bank * layout.cluster + lane, offset) not in cells
        for lane in range(layout.cluster)
    ):
        raise ValueError(
            f"{node.name}: instruction address is outside its allocated bank cells"
        )
    if bank >= 64 or bits not in (4, 8):
        raise ValueError(f"{node.name}: address encoding exceeds Alpha fields")
    base = 0x1000000 if memory_name == "global_activation" else 0x800000
    return (
        base
        | offset
        | (bank << 10)
        | ((bits == 8) << 19)
        | (layout.cluster << 16)
    )


def payload_array(spec, nodes):
    value = nodes[spec["node"]].value
    for index in spec["path"]:
        value = value[index]
    if not isinstance(value, torch.Tensor) or isinstance(
        value, torch._subclasses.fake_tensor.FakeTensor
    ):
        raise ValueError(
            f"{spec['file']}: concrete compiler replay tensors are required"
        )
    array = value.detach().cpu().numpy()
    if spec["channel_slice"] is not None:
        start, stop = spec["channel_slice"]
        array = array[:, start:stop]
    physical = spec.get("physical_shape")
    if physical is not None and tuple(physical) != array.shape:
        if len(physical) != array.ndim or any(
            a > b for a, b in zip(array.shape, physical)
        ):
            raise ValueError(
                "encoded padding cannot truncate a logical payload"
            )
        array = np.pad(
            array, tuple((0, b - a) for a, b in zip(array.shape, physical))
        )
    encoding = spec["encoding"]
    if encoding == "activation_scale":
        if array.ndim != 4:
            raise ValueError("Alpha activation scale must be rank four")
        array = array.transpose(0, 3, 2, 1)
    elif encoding == "weight_scale":
        k, c, r, s = array.shape
        if k % 32:
            raise ValueError("Alpha weight scale must fill 32 output lanes")
        array = array.reshape(-1, 32, c, r, s).transpose(0, 1, 4, 3, 2)
    elif encoding not in ("plain", "unsigned"):
        raise ValueError(f"unknown Alpha payload encoding {encoding}")
    unsigned = encoding in ("unsigned", "activation_scale", "weight_scale")
    width = spec["bits"]
    lower, upper = (
        (0, (1 << width) - 1)
        if unsigned
        else (-(1 << (width - 1)), (1 << (width - 1)) - 1)
    )
    if (
        not np.isfinite(array).all()
        or not np.equal(array, np.rint(array)).all()
        or (array < lower).any()
        or (array > upper).any()
    ):
        raise ValueError(
            f"{spec['file']}: payload is not exact {width}-bit integer data"
        )
    return array.astype(np.int64)


def _host_output_only(node):
    """Host layout/dequantization is legal only downstream of all device work."""
    allowed = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.voyager.subview.default,
    }
    pending = list(node.users)
    visited = set()
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        if user.op == "output":
            continue
        if user.target not in allowed:
            return False
        pending.extend(user.users)
    return True


def emit(program, encoded, config, output):
    program.assert_frozen()
    nodes = {
        f"{scope}/{node.name}": node
        for scope, module in program.graph.named_modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
    }
    names = {node: name for name, node in nodes.items()}
    identities = _buffer_identity(program.graph)
    root = lambda n: identities.get(n, n)
    staged = []
    seen = set()
    for node in program.graph.graph.nodes:
        selected = node.meta.get("target_schedule")
        if selected is None:
            continue
        if selected["name"] in seen:
            raise ValueError(
                f"duplicate native Alpha layer name: {selected['name']}"
            )
        seen.add(selected["name"])
        fields = dict(
            network="compiler",
            layer=selected["name"],
            test_id=1,
            **selected["fields"],
        )
        for ref in selected["addresses"]:
            buffer = root(nodes[ref["buffer"]])
            if (
                selected["kind"].startswith("nmp_")
                and buffer.meta["memory_instance"] != "global_activation"
            ):
                raise ValueError(
                    f"{node.name}: NMP cannot access the weight arena"
                )
            fields[ref["field"]] = hex(
                encoded_address(
                    buffer, config, ref["bits"], ref["entry_offset"]
                )
            )
        arrays = [
            (spec, payload_array(spec, nodes)) for spec in selected["payloads"]
        ]
        staged.append((node, selected, fields, arrays))
    native_names = {node: selected["name"] for node, selected, _, _ in staged}

    def argument(value):
        if isinstance(value, torch.fx.Node):
            return dict(value=names[value], buffer=names[root(value)])
        if isinstance(value, (tuple, list)):
            return [argument(v) for v in value]
        if isinstance(value, dict):
            return {k: argument(v) for k, v in value.items()}
        if isinstance(value, torch.dtype):
            return str(value)
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        raise TypeError(f"unsupported Alpha control argument: {type(value)}")

    driver = []
    host = []
    aliases = {
        operator.getitem,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.voyager.subview.default,
    }
    movement = {
        torch.ops.voyager.async_copy.default,
        torch.ops.voyager.async_wait.default,
        torch.ops.voyager.insert.default,
        torch.ops.voyager.alloc.default,
        torch.ops.voyager.zeros.default,
        torch.ops.voyager.fill.default,
    }
    host_ops = {
        torch.ops.alpha_quant.dequantize.default,
        torch.ops.aten.permute.default,
    }
    for node in program.graph.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node in native_names:
            kind = "native"
        elif node.target in aliases:
            kind = "view"
        elif node.target in movement:
            kind = "buffer_operation"
        elif node.target in host_ops and _host_output_only(node):
            kind = "host_output"
            host.append(names[node])
        else:
            raise ValueError(
                f"{node.name}: Alpha has no encoded instruction or boundary for {node.target}"
            )
        driver.append(
            dict(
                node=names[node],
                kind=kind,
                operation=str(node.target),
                layer=native_names.get(node),
                args=argument(node.args),
                kwargs=argument(dict(node.kwargs)),
            )
        )
    directory = output / "alpha-cmod"
    directory.mkdir(exist_ok=False)
    results = []
    for node, selected, fields, arrays in staged:
        folder = directory / selected["name"]
        folder.mkdir()
        text = "".join(f"{key} = {value}\n" for key, value in fields.items())
        for index in (0, 1, 10):
            (folder / f"schedule{index}.txt").write_text(text)
        files = []
        for spec, array in arrays:
            target = folder / spec["file"]
            np.savetxt(target, array.reshape(1, -1), delimiter=",", fmt="%d")
            files.append(
                dict(
                    file=spec["file"],
                    shape=list(array.shape),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    source=spec,
                )
            )
        results.append(
            dict(
                name=selected["name"],
                kind=selected["kind"],
                node=names[node],
                reads=to_data(node.meta["target_invoke"].reads),
                writes=to_data(node.meta["target_invoke"].writes),
                schedule=fields,
                payloads=files,
            )
        )
    (directory / "layers.txt").write_text(
        "".join(r["name"] + "\n" for r in results)
    )
    (directory / "program.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                graph_digest=program.frozen_digest,
                plan_digest=program.plan_digest,
                steps=driver,
                allocation=to_data(program.plan.allocation),
                scope="Selected native calls plus explicit transfers, views and host output conversion",
            ),
            indent=2,
        )
        + "\n"
    )
    report = dict(
        schema_version=1,
        status="complete",
        emitted_layers=len(results),
        layer_kinds=dict(Counter(r["kind"] for r in results)),
        layers=results,
        host_output_nodes=host,
        buffer_operations=sum(r["kind"] == "buffer_operation" for r in driver),
        claims=dict(
            all_buffer_program_operations_accounted=True,
            all_native_instructions_emitted=True,
            connected_via_explicit_buffer_program=True,
            cmod_or_device_executed=False,
        ),
        issues=[],
        scope="Compiler collateral and logical host replay. Native layout encoding follows the Alpha reference; no C-model or device execution.",
    )
    (directory / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    program.assert_frozen()
