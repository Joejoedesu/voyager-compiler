"""Encode a frozen buffer program; this stage has no planning authority."""

from pathlib import Path

import torch
from google.protobuf import text_format

from voyager_compiler.compiler.bufferize.emit import (
    flush_tensor_files,
    gen_code_bufferized,
    print_layer_table,
)
from voyager_compiler.shape_prop import constant_value, fake_like


def emit(
    program, flat_inputs, output_dir, dump_tensors, *, encode=None, config=None
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
    if encode is not None:
        model = encode(model, program, config)
    (output / "model.txt").write_text(text_format.MessageToString(model))
    (output / "layers.txt").write_text(
        print_layer_table(program.graph, model, to_string=True)
    )
    flush_tensor_files()
    program.assert_frozen()
    return model
