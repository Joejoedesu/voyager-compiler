"""Compare small Trainium mappings with compiler-2, using host replay and NKI.

No Neuron device is required; this does not measure accelerator performance.
Run from voyager: PYTHONPATH=src python validation/compare_trainium_constants.py OUTPUT
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

CASES = (
    "dynamic_matmul",
    "linear",
    "scaled_layout",
    "slice_scale",
    "shared_weight",
    "private_broadcast",
    "shared_broadcast",
    "private_rms",
    "shared_rms",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(checkout, output):
    import torch
    from torch.utils._pytree import tree_flatten

    import voyager_compiler as vc
    from voyager_compiler.shape_prop import ShapeProp

    prefix = (
        "targets"
        if (checkout / "src/voyager_compiler/targets").exists()
        else "backend"
    )
    backend = importlib.import_module(f"voyager_compiler.{prefix}.trainium")
    imported = Path(vc.__file__).resolve()
    assert imported.is_relative_to(checkout / "src"), imported
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output.mkdir(parents=True)
    summaries = {}
    for case in CASES:
        torch.manual_seed(71)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                rows = 256 if case == "slice_scale" else 128
                self.register_buffer(
                    "weight",
                    torch.randint(-2, 3, (rows, 128)).to(torch.bfloat16),
                )
                self.register_buffer(
                    "scale",
                    torch.full(
                        (
                            (128,)
                            if case.endswith(("broadcast", "rms"))
                            else (128, 1)
                        ),
                        0.5,
                        dtype=torch.bfloat16,
                    ),
                )

            def forward(self, x, y):
                if case == "dynamic_matmul":
                    return x @ y
                if case.endswith(("broadcast", "rms")):
                    result = x @ y
                    if case.endswith("rms"):
                        result = torch.nn.functional.rms_norm(
                            result, (128,), self.scale, eps=1e-6
                        )
                    else:
                        result = result * self.scale
                    result = result @ y
                    return (
                        (result, self.scale)
                        if case.startswith("shared_")
                        else result
                    )
                if case == "scaled_layout":
                    w = (
                        (self.weight * self.scale)
                        .reshape(16, 8, 128)
                        .reshape(128, 128)
                    )
                    return x @ w.T
                if case == "slice_scale":
                    return torch.nn.functional.linear(
                        x, self.weight[:128] * self.scale
                    )
                if case == "shared_weight":
                    return (
                        torch.nn.functional.linear(x, self.weight),
                        x @ self.weight,
                    )
                return torch.nn.functional.linear(x, self.weight)

        model = Model().eval()
        args = tuple(
            torch.randint(-2, 3, (128, 128)).to(torch.bfloat16)
            for _ in range(2)
        )
        reference = model(*args)
        gm = vc.export_model(model, args)
        vc.transform(gm, args, config=backend.TRN2, **backend.LAYOUT)
        folder = output / case
        folder.mkdir()
        (folder / "normalized.txt").write_text(str(gm.graph))
        attrs = {
            n.target: n
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(n, "value", None), torch.Tensor)
        }
        stored = sum(
            n.value.numel() * n.value.element_size() for n in attrs.values()
        )
        folding = gm.meta.get("constant_folding", [])
        nodes = backend.kernel_nodes(gm)
        ir = vc.compile(
            gm,
            args,
            config=backend.TRN2,
            output_dir=str(folder),
            dump_tensors=False,
        )
        manifest = backend.translate(
            ir, str(folder / "nki"), backend.TRN2, backend.immediates(gm, ir)
        )
        actual = ShapeProp(gm).propagate(*args)
        flat, _ = tree_flatten(actual)
        torch.testing.assert_close(
            flat, tree_flatten(reference)[0], rtol=0, atol=0
        )
        torch.save(flat, folder / "outputs.pt")
        kernel = (folder / "nki/kernels.py").read_text()
        instructions = Counter(
            n.func.attr
            for n in ast.walk(ast.parse(kernel))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "nisa"
        )
        tiles = []
        from voyager_compiler.codegen.node_info import get_anchor_node

        for n, _ in nodes:
            anchor = get_anchor_node(n) or n
            tile = anchor.meta.get("tile_result") or n.meta.get("tile_result")
            if tile is not None:
                tiles.append(
                    dict(
                        tiling=(
                            vars(tile.tiling)
                            if hasattr(tile.tiling, "loop_orders")
                            else repr(tile.tiling)
                        ),
                        placement=repr(tile.placement),
                        cost=repr(tile.cost),
                    )
                )
        summaries[case] = dict(
            passed=True,
            normalized_constant_bytes=stored,
            folding=folding,
            tiles=tiles,
            predicted_seconds=backend.predicted_seconds(nodes, backend.TRN2),
            nki_instructions=dict(instructions),
            kernel_calls=sum(
                s["kernel"] is not None for s in manifest["steps"]
            ),
            nki_sha256=hashlib.sha256(kernel.encode()).hexdigest(),
        )
    save(
        output / "summary.json",
        dict(compiler=str(imported), torch=torch.__version__, cases=summaries),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--donor",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "voyager-compiler-2",
    )
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve(), args.output.resolve())
        return
    current = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    for name, checkout in (
        ("current", current),
        ("compiler-2", args.donor.resolve()),
    ):
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    str(args.output / name),
                    "--worker",
                    str(checkout),
                ],
                cwd=current,
                env=dict(
                    os.environ,
                    PYTHONPATH=str(checkout / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    import torch

    new, old = [
        json.loads((args.output / name / "summary.json").read_text())
        for name in ("current", "compiler-2")
    ]
    cases = {}
    for case in CASES:
        a, b = [
            torch.load(
                args.output / name / case / "outputs.pt", weights_only=True
            )
            for name in ("current", "compiler-2")
        ]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        n, d = new["cases"][case], old["cases"][case]
        cases[case] = dict(
            outputs_bitwise_equal=True,
            current_constant_bytes=n["normalized_constant_bytes"],
            donor_constant_bytes=d["normalized_constant_bytes"],
            same_mapping=n["tiles"] == d["tiles"],
            same_nki_instructions=n["nki_instructions"]
            == d["nki_instructions"],
            current_estimate_seconds=sum(n["predicted_seconds"]),
            donor_estimate_seconds=sum(d["predicted_seconds"]),
            passed=(
                n["normalized_constant_bytes"] <= d["normalized_constant_bytes"]
                and (
                    case.startswith("shared_")
                    or (
                        n["tiles"] == d["tiles"]
                        and n["nki_instructions"] == d["nki_instructions"]
                        and n["predicted_seconds"] == d["predicted_seconds"]
                    )
                )
            ),
        )
    result = dict(
        passed=all(c["passed"] for c in cases.values()),
        scope="CPU reference and bufferized replay, mapping/model comparison and NKI emission; no Neuron device execution or measured speedup",
        cases=cases,
    )
    save(args.output / "comparison.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
