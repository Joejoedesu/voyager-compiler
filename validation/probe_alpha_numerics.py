"""Diagnostic of the Alpha inference recipe on the current Torch installation."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from voyager_compiler.shape_prop import ShapeProp
from voyager_compiler.targets.magnet_alpha.graph import apply_graph_rewrites
from voyager_compiler.targets.magnet_alpha.hardware import ALPHA_HARDWARE_CONFIG
from voyager_compiler.targets.magnet_alpha.quantization import (
    apply_quantization_rewrites,
)
from voyager_compiler.targets.magnet_alpha.target import TARGET
from voyager_compiler.workloads import yolo5_face_nv as workload

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[2]
options = dict(
    model_name_or_path=str(
        root / "voyager-nv-integration/libraries/weights/yolov5n-0.5.pt"
    ),
    seed=0,
)
model = workload.build(options)
args, kwargs = workload.example_inputs(model, options)
settings = workload.recipe_options(options)
quantizer = workload.configure_quantizer(None, settings)
gm = workload.prepare_calibrated_model(
    model, args, quantizer, [], settings, ALPHA_HARDWARE_CONFIG
)
ShapeProp(gm).propagate(*args)
gm.meta["target"] = "magnet-alpha"
apply_graph_rewrites(gm)
apply_quantization_rewrites(gm)
ShapeProp(gm).propagate(*args)
outputs, _ = tree_flatten(gm(*args))
reference = np.load(
    root
    / "reference-point/alpha-yolov5n05-2026-10-23/numerical-reference/post_rewrite_outputs.npz"
)
for i, value in enumerate(outputs):
    if not isinstance(value, torch.Tensor):
        continue
    expected = torch.from_numpy(reference[f"tensor_{i}"])
    print(
        i,
        tuple(value.shape),
        value.min().item(),
        value.max().item(),
        value.float().norm().item(),
        "max_abs_difference",
        (expected - value).abs().max().item(),
        flush=True,
    )
if os.environ.get("ALPHA_BUFFER_PROBE"):
    from collections import Counter

    from voyager_compiler.pipeline import compile_bufferized

    print(
        Counter(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        ),
        flush=True,
    )
    from voyager_compiler.targets.magnet_alpha.graph import (
        apply_memory_rewrites,
    )
    from voyager_compiler.targets.magnet_alpha.implementations import (
        form_execution_regions,
    )

    apply_memory_rewrites(gm, ALPHA_HARDWARE_CONFIG)
    print(
        "movement",
        Counter(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        ),
        flush=True,
    )
    form_execution_regions(gm)
    compile_bufferized(
        gm,
        args,
        config=ALPHA_HARDWARE_CONFIG,
        target=TARGET,
        output_dir=os.environ["ALPHA_BUFFER_PROBE"],
        dump_tensors=False,
        reference_outputs=gm(*args),
    )
