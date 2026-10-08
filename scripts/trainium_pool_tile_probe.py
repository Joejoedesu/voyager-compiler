"""Diagnostic fixed-row maxpool probe, outside production schedule selection.

Retains the current four-max expansion, two halo slots and explicit addresses.
Only row tiling/tail handling differ. Use trainium_run_hardware.py to measure.
"""

import argparse
import json
import math
from pathlib import Path


def source(tile):
    width, output, height = 4096, 4094, 4094
    tiles = math.ceil(height / tile)
    lines = [
        "import neuronxcc.nki as nki",
        "import neuronxcc.nki.language as nl",
        "import neuronxcc.nki.isa as nisa",
        "import neuronxcc.nki.compiler as ncc",
        "@nki.compiler.skip_middle_end_transformations",
        "@nki.jit",
        "def kernel(a0):",
        "    s = nl.ndarray((128,32768), dtype=nl.float32, buffer=ncc.sbuf.alloc(lambda idx, pdim_size, fdim_size: (0,0)))",
        "    out = nl.ndarray((1,4094,4094,1), dtype=nl.float32, buffer=nl.shared_hbm)",
    ]

    def load(index, count, slot):
        for halo in range(3):
            base = slot * 12288 + halo * width
            lines.append(
                f"    nisa.dma_copy(dst=s[nl.arange({count})[:,None], {base}+nl.arange(4096)[None,:]], src=a0.reshape((16777216,))[(({index})*{tile}+{halo})*4096 + nl.arange({count})[:,None]*4096 + nl.arange(4096)[None,:]])"
            )

    def compute(index, count, slot):
        base = slot * 12288
        lines.extend(
            [
                f"    s[:{count},24576:28672] = nisa.tensor_tensor(s[:{count},{base}:{base+4096}],s[:{count},{base+4096}:{base+8192}],op=nl.maximum,engine=nisa.vector_engine)",
                f"    s[:{count},28672:32768] = nisa.tensor_tensor(s[:{count},24576:28672],s[:{count},{base+8192}:{base+12288}],op=nl.maximum,engine=nisa.vector_engine)",
                f"    s[:{count},{base}:{base+4094}] = nisa.tensor_tensor(s[:{count},28672:32766],s[:{count},28673:32767],op=nl.maximum,engine=nisa.vector_engine)",
                f"    s[:{count},{base+4096}:{base+8190}] = nisa.tensor_tensor(s[:{count},{base}:{base+4094}],s[:{count},28674:32768],op=nl.maximum,engine=nisa.vector_engine)",
                f"    nisa.dma_copy(dst=out.reshape((16760836,))[({index})*{tile}*4094 + nl.arange({count})[:,None]*4094 + nl.arange(4094)[None,:]],src=s[nl.arange({count})[:,None],{base+4096}+nl.arange(4094)[None,:]])",
            ]
        )

    load("0", tile, 0)
    cycles = (tiles - 2) // 2
    lines.append(f"    for q in nl.sequential_range({cycles}):")
    start = len(lines)
    load("2*q+1", tile, 1)
    compute("2*q", tile, 0)
    load("2*q+2", tile, 0)
    compute("2*q+1", tile, 1)
    lines[start:] = ["    " + line for line in lines[start:]]
    for i in range(2 * cycles, tiles):
        count = min(tile, height - i * tile)
        if i + 1 < tiles:
            load(str(i + 1), min(tile, height - (i + 1) * tile), (i + 1) % 2)
        compute(str(i), count, i % 2)
    lines.append("    return (out,)")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    for tile in (89, 128):
        root = a.output / f"rows{tile}"
        (root / "nki").mkdir(parents=True, exist_ok=True)
        (root / "nki/program.py").write_text(source(tile))
        (root / "nki/plan.json").write_text(
            json.dumps(dict(diagnostic=True, tile_rows=tile))
        )
        (root / "model.txt").write_text(
            "Diagnostic hand-instantiated current maxpool template; not a Voyager-selected collateral.\n"
        )
        (root / "instructions.json").write_text(
            json.dumps(dict(diagnostic=True, selected_isa_plan=False))
        )
        (root / "hardware.json").write_text(
            json.dumps(
                dict(
                    estimates=[],
                    hardware="Trainium2 one core",
                    sbuf_bytes=16 << 20,
                )
            )
        )
        (root / "generation.json").write_text(
            json.dumps(
                dict(
                    input_dtype="float32",
                    diagnostic=True,
                    tolerance=dict(atol=0, rtol=0),
                )
            )
        )
        import numpy as np

        with np.load(a.reference) as ref:
            np.savez(
                root / "reference.npz",
                a0=ref["a0"],
                expected=ref["expected"].reshape(1, 4094, 4094, 1),
            )


if __name__ == "__main__":
    main()
