"""Encode a frozen buffer program; this stage has no planning authority."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from google.protobuf import text_format

from voyager_compiler.artifacts import EncodedIR, SavedBundle, to_data
from voyager_compiler.compiler.bufferize.emit import (
    flush_tensor_files,
    gen_code_bufferized,
    print_layer_table,
)
from voyager_compiler.shape_prop import constant_value, fake_like


def encode(
    program, flat_inputs, output_dir, dump_tensors, *, target=None, config=None
):
    program.assert_frozen()
    output = Path(output_dir)
    for module in program.graph.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.find_nodes(op="get_attr"):
                constant_value(module, node)
    model = gen_code_bufferized(
        program.graph,
        flat_inputs if dump_tensors else list(map(fake_like, flat_inputs)),
        str(output / "tensor_files") if dump_tensors else None,
    )
    encoded = EncodedIR(model)
    if target is not None and target.encode is not None:
        encoded = target.encode(encoded, program, config)
    if not isinstance(encoded, EncodedIR):
        raise TypeError("TargetSpec.encode must return EncodedIR")
    model = encoded.model
    (output / "target_ir.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                target=program.plan.target,
                data=to_data(encoded.target_data),
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    for name, array in encoded.payloads.items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"invalid target payload name: {name}")
        folder = output / "target_payloads"
        folder.mkdir(exist_ok=True)
        np.save(folder / f"{name}.npy", array, allow_pickle=False)
    (output / "model.txt").write_text(text_format.MessageToString(model))
    (output / "layers.txt").write_text(
        print_layer_table(program.graph, model, to_string=True)
    )
    flush_tensor_files()
    program.assert_frozen()
    return model


def emit_bundle(bundle_dir, output_dir, *, target=None):
    """Generate target files using only a saved bundle, including in a new process.

    The input manifest remains immutable. A separate emission.json records the
    source identity and output checksums. Output may be the compilation directory
    (the convenience pipeline default) or a separate fresh directory.
    """
    from voyager_compiler.extensions import registry

    bundle = SavedBundle.load(bundle_dir)
    target = target or registry.target(bundle.manifest["target"])
    if target.name != bundle.manifest["target"]:
        raise ValueError("emitter target does not match bundle target")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if target.emit is not None:
        target.emit(bundle, output)
    bundle.verify()
    inputs = {bundle.path(name) for name in bundle.manifest["files"]}
    inputs.add(bundle.directory / "bundle.json")
    files = {}
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and path.resolve() not in inputs
            and path.name != "emission.json"
        ):
            with path.open("rb") as stream:
                files[str(path.relative_to(output))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    receipt = dict(
        schema_version=1,
        target=target.name,
        source_bundle_sha256=hashlib.sha256(
            (bundle.directory / "bundle.json").read_bytes()
        ).hexdigest(),
        graph_digest=bundle.manifest["graph_digest"],
        plan_digest=bundle.manifest["plan_digest"],
        files=files,
    )
    (output / "emission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit target files from a saved compilation bundle"
    )
    parser.add_argument("bundle")
    parser.add_argument("output")
    arguments = parser.parse_args()
    emit_bundle(arguments.bundle, arguments.output)
