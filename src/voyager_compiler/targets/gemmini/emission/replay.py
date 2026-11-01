"""Replay collateral for the pinned Gemmini endpoint, using frozen addresses.

The initial image contains only actual inputs/parameters and poison. Expected
bytes are in a separate comparison file, never loaded as accelerator inputs.
No command selection, address relocation or per-layer host substitution occurs.
"""

import json
from pathlib import Path

import torch

from voyager_compiler.replay import collect_replay_images

from ._encoding import STATUS

BASE = 0x80000000
APERTURE = 16 << 20


def write_replay(result, directory, commands_path):
    def encode(value):
        if value.dtype not in (torch.int8, torch.int32):
            raise ValueError(
                "Gemmini replay accepts signed integer payloads only"
            )
        return value.detach().cpu().contiguous().numpy().tobytes()

    images = collect_replay_images(
        result, encode, alignment=16, capacity=APERTURE
    )
    initial, expected = images.initial, images.expected
    size = len(initial)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
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
