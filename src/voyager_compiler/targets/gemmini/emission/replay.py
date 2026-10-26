"""Replay collateral for the pinned Gemmini endpoint, using frozen addresses.

The initial image contains only actual inputs/parameters and poison. Expected
bytes are in a separate comparison file, never loaded as accelerator inputs.
No command selection, address relocation or per-layer host substitution occurs.
"""

import json
from pathlib import Path

import torch
from torch.fx import Interpreter

from ._encoding import STATUS

BASE = 0x80000000
APERTURE = 16 << 20


def write_replay(result, directory, commands_path):
    result.program.assert_frozen()
    directory = Path(directory)
    program = result.program.graph
    size = (result.program.plan.allocation.dram_bytes + 15) // 16 * 16
    if size > APERTURE:
        raise ValueError(
            "Whole-program replay exceeds the pinned 16 MiB endpoint aperture"
        )
    directory.mkdir(parents=True, exist_ok=False)
    initial = bytearray([0xA5] * size)
    expected = bytearray(initial)

    def store(image, node, value):
        segment = node.meta.get("memory")
        if segment is None:
            raise ValueError(
                f"{node.name}: replay requires a finalized DRAM allocation"
            )
        if value.dtype not in (torch.int8, torch.int32):
            raise ValueError(
                "Gemmini replay accepts signed integer payloads only"
            )
        payload = value.detach().cpu().contiguous().numpy().tobytes()
        if len(payload) > segment.end - segment.start:
            raise ValueError("Replay payload exceeds allocation")
        image[segment.start : segment.start + len(payload)] = payload

    class Replay(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node.op in ("placeholder", "get_attr") and isinstance(
                value, torch.Tensor
            ):
                store(initial, node, value)
                store(expected, node, value)
            if node.target == torch.ops.voyager.insert.default:
                source, destination = node.args
                store(expected, destination, self.env[source])
            return value

    with torch.no_grad():
        Replay(program).run(*result.args)
    command_stream = json.loads(Path(commands_path).read_text())["commands"]
    records = [
        dict(type="defaults", status=STATUS),
        *command_stream,
        dict(type="drain"),
    ]
    (directory / "commands.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (directory / "initial.bin").write_bytes(initial)
    (directory / "expected.bin").write_bytes(expected)
    (directory / "memory.json").write_text(
        json.dumps(
            dict(
                version=1,
                base=hex(BASE),
                size=APERTURE,
                alignment=16,
                regions=[
                    dict(
                        name="program",
                        address=hex(BASE),
                        size=size,
                        input="initial.bin",
                        output="actual.bin",
                        expected="expected.bin",
                    )
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    return directory
