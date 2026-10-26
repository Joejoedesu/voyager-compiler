"""Encode the frozen selected invocation and its declared transfers as RoCC."""

import json

import torch

from ._encoding import ACC, command, ex, ld, loop_mm, st, tile

DRAM_BASE = 0x80000000


def emit(program, encoded, config, output):
    program.assert_frozen()
    commands, bodies = [], []

    def address(node):
        segment = node.meta.get("scratchpad") or node.meta.get("memory")
        if segment is None and node.target in (
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
        ):
            source = node.args[0]
            if source.value.numel() != node.value.numel():
                raise ValueError("Gemmini view changes allocation extent")
            return address(source)
        if segment is None:
            raise ValueError(f"{node.name}: missing selected allocation")
        return segment.start + (
            DRAM_BASE if node.meta.get("space") == "DRAM" else 0
        )

    for node in program.graph.graph.nodes:
        allowed = (
            torch.ops.voyager.alloc.default,
            torch.ops.voyager.zeros.default,
            torch.ops.voyager.async_copy.default,
            torch.ops.voyager.async_wait.default,
            torch.ops.voyager.insert.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
        )
        if (
            node.op == "call_function"
            and node.target not in allowed
            and "target_invoke" not in node.meta
        ):
            raise NotImplementedError(
                f"Gemmini has no selected encoding for {node.name}: {node.target}"
            )
        if node.target == torch.ops.voyager.async_copy.default:
            source, dest, offsets, sizes, _ = node.args[:5]
            if (
                any(offsets)
                or tuple(sizes) != tuple(source.value.shape)
                or len(sizes) != 2
            ):
                raise NotImplementedError(
                    "Gemmini whole-matrix DMA requires a full 2D window"
                )
            rows, cols = sizes
            if source.meta.get("space") == "DRAM":
                if (
                    source.value.dtype != torch.int8
                    or dest.meta.get("memory_instance") != "scratchpad"
                ):
                    raise ValueError(
                        "Gemmini mvin requires an INT8 scratchpad operand"
                    )
                for col in range(0, cols, 16):
                    for row in range(0, rows, 16):
                        commands += [
                            ld(cols, block_stride=rows),
                            command(
                                2,
                                address(source) + row * cols + col,
                                tile(
                                    address(dest) // 16
                                    + (col // 16) * rows
                                    + row,
                                    16,
                                    16,
                                ),
                            ),
                        ]
            else:
                if (
                    source.meta.get("memory_instance") != "accumulator"
                    or dest.value.dtype != torch.int32
                ):
                    raise ValueError(
                        "Gemmini full-width mvout requires an INT32 accumulator"
                    )
                commands.append(st(cols * 4))
                for col in range(0, cols, 16):
                    for row in range(0, rows, 16):
                        local = (
                            ACC
                            | (1 << 29)
                            | (address(source) // 64 + (col // 16) * rows + row)
                        )
                        commands.append(
                            command(
                                3,
                                address(dest) + (row * cols + col) * 4,
                                tile(local, 16, 16),
                            )
                        )
        invocation = node.meta.get("target_invoke")
        if invocation is None:
            continue
        if invocation.opcode in (
            "matmul",
            "conv2d",
            "requantize",
            "add",
            "average",
            "maxpool",
        ):
            from ..semantics import facts
            from ._commands import expand, explicit_matmul

            kind, params, operands = facts(node)
            choice = program.plan.choices[node.meta["scope"][0]]
            selected = dict(choice.parameters)
            if choice.implementation != invocation.implementation:
                raise ValueError(
                    "Gemmini invocation differs from frozen selection"
                )
            if selected["tile"]:
                params["tile"] = selected["tile"]
            destination = next(
                u.args[1]
                for u in node.users
                if u.target == torch.ops.voyager.insert.default
            )
            addresses = {key: address(value) for key, value in operands.items()}
            addresses["y"] = address(destination)
            start = len(commands)
            part = (
                explicit_matmul(params, addresses)
                if kind == "matmul" and selected["mode"] == "explicit"
                else expand(kind, params, addresses)
            )
            commands.extend(part)
            # TargetInvoke is synchronous at the IR boundary, including output DMA.
            commands.append(dict(type="drain"))
            bodies.append(
                dict(
                    node=node.name,
                    implementation=choice.implementation,
                    command_start=start,
                    command_end=len(commands),
                    parameters=selected,
                    operands={
                        key: dict(
                            node=value.name,
                            address=addresses[key],
                            shape=list(value.value.shape),
                            dtype=str(value.value.dtype),
                        )
                        for key, value in operands.items()
                    },
                    output=dict(
                        node=destination.name,
                        address=addresses["y"],
                        shape=list(destination.value.shape),
                        dtype=str(destination.value.dtype),
                    ),
                )
            )
            continue
        a, b = node.args
        m, k = a.value.shape
        n = b.value.shape[1]
        dest = next(
            u.args[1]
            for u in node.users
            if u.target == torch.ops.voyager.insert.default
        )
        start = len(commands)
        if invocation.opcode == "loop_ws":
            commands += [ex(), ld(k, 0), ld(n, 1), st(n * 4)]
            commands += loop_mm(
                m,
                n,
                k,
                address(a),
                address(b),
                0,
                address(dest),
                k,
                n,
                n,
                full_output=True,
            )
        elif invocation.opcode == "ws_matmul":
            commands.append(ex())
            for col in range(0, n, 16):
                for row in range(0, m, 16):
                    for depth in range(0, k, 16):
                        out = ACC | (
                            address(dest) // 64 + (col // 16) * m + row
                        )
                        if depth:
                            out |= 0x40000000
                        ar = address(a) // 16 + (depth // 16) * m + row
                        br = address(b) // 16 + (col // 16) * k + depth
                        commands += [
                            command(6, tile(br, 16, 16), tile(out, 16, 16)),
                            command(
                                4, tile(ar, 16, 16), tile(0xFFFFFFFF, 16, 16)
                            ),
                        ]
        else:
            raise ValueError(
                f"unimplemented selected Gemmini opcode: {invocation.opcode}"
            )
        bodies.append(
            dict(
                node=node.name,
                implementation=invocation.implementation,
                command_start=start,
                command_end=len(commands),
            )
        )
    if not bodies:
        raise NotImplementedError("no supported Gemmini matrix invocation")
    directory = output / "gemmini"
    directory.mkdir(exist_ok=False)
    (directory / "commands.json").write_text(
        json.dumps(
            dict(
                commands=commands,
                bodies=bodies,
                scope="Emitted RoCC stream; endpoint execution evidence is recorded separately",
            ),
            indent=2,
        )
        + "\n"
    )
