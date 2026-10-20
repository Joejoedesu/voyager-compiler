"""Trace a translation's program, kernels and host ops, into one NEFF.

Runs in the ``trn`` env, by path, on a CPU-only machine:

    python trace.py <dir> --phase {prefill,decode} [--target trn2] \
        [--weights-from <HF model>]

Writes ``<dir>/step/``: ``step.pt``, the traced module a NeuronCore runs
(``run.py``), and its NEFF.  With ``--weights-from <HF model>``, every
weight that is a tensor of that checkpoint, as stored or transposed, is left
out of ``step.pt``: ``weights.json`` names its source for ``run.py`` to
rebuild it from the host's copy, and the other weights go to
``extra.safetensors``.  The kernels become NKI custom calls and the
host ops XLA ops of the same graph, so one launch runs the whole step, as
NxD Inference runs a prefill or a decode step, with its compiler flags and
decoder-layer markers.  Weights are the module's parameters, kept out of the
NEFF so they load once; the state the step writes stays on the device.
"""

import argparse
import importlib.util
import json
import os

import torch
import torch_neuronx
from huggingface_hub import snapshot_download
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)
from safetensors import safe_open
from safetensors.torch import save_file

MARKERS = {"start": ModuleMarkerStartWrapper(), "end": ModuleMarkerEndWrapper()}

# The torch type a kernel takes a weight of each IR type as (``check.py``).
# A boolean stays one: the trace cannot hand an integer parameter to a
# kernel, so the program's cast to uint8 hands over its bytes.
KERNEL_TYPES = {
    "int64": torch.int32,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e4m3_ieee": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}

# NxD Inference's compiler flags (models/model_wrapper.py, with its layer
# boundary markers).  A prefill step compiles in the modular flow, which
# splits the graph at the markers and compiles one decoder layer for all of
# them; a decode step compiles at -O2.  A step computing in Trn2's IEEE E4M3
# has its HLO's float8_e4m3fn compiled as that format.
COMPILER_FLAGS = (
    "--auto-cast=none --model-type=transformer --tensorizer-options="
    "'--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 "
    "--vectorize-strided-dma' --lnc=1 "
)
PHASE_FLAGS = {"prefill": "-O1", "decode": "-O2"}
HLO_FLAGS = {"prefill": "--modular-flow-mac-threshold=10 ", "decode": ""}
FP8_HLO_FLAG = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 "


def load_module(directory, name):
    """``<directory>/<name>.py`` as a module."""
    path = os.path.join(directory, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step(torch.nn.Module):
    """One run of the program: its inputs in, its outputs out.

    Weights are parameters, which the trace keeps out of the graph; the
    state the step writes (a KV cache) is buffers, which aliasing keeps on
    the device and updates in place.
    """

    def __init__(self, program, kernels, params, state, names, scalars):
        super().__init__()
        self.program = program
        self.kernels = kernels
        self.param_names = list(params)
        for i, (name, tensor) in enumerate(params.items()):
            if name in state:
                self.register_buffer(f"param{i}", tensor)
            else:
                parameter = torch.nn.Parameter(tensor, requires_grad=False)
                self.register_parameter(f"param{i}", parameter)
        self.state = state
        self.input_names = names
        self.scalars = scalars

    def forward(self, *tensors):
        params = {
            name: getattr(self, f"param{i}")
            for i, name in enumerate(self.param_names)
        }
        inputs = dict(zip(self.input_names, tensors))
        outputs = self.program.run(
            inputs, params, self.call_kernel, self.mark, self.scalars
        )
        # Reassigning the buffers is what aliasing tracks.
        for i, name in enumerate(self.param_names):
            if name in self.state:
                setattr(self, f"param{i}", params[name])
        return outputs

    def call_kernel(self, name, *args):
        """Kernel ``name``'s results, as a tuple even when there is one.  A
        kernel takes no 0-d tensor: a scalar goes as one element."""
        args = [a.reshape(1) if a.dim() == 0 else a for a in args]
        results = getattr(self.kernels, name)(*args)
        if isinstance(results, (list, tuple)):
            return tuple(results)
        return (results,)

    @staticmethod
    def mark(kind, *values):
        marked = MARKERS[kind](*values)
        return (marked,) if isinstance(marked, torch.Tensor) else tuple(marked)


def stacked_keys(stem, keys):
    """The checkpoint keys whose weights ``stem`` stacks, in row order, or
    ``None``: ``merge_sibling_linears`` names a stacked weight after its
    linears joined by ``_`` under their common module
    (``layers_0_self_attn_q_proj_k_proj_v_proj_weight``)."""
    suffix = "_weight"
    if not stem.endswith(suffix):
        return None
    for cut in (i + 1 for i, c in enumerate(stem) if c == "_"):
        scope, rest = stem[:cut], stem[cut : -len(suffix)]
        names = {
            k[cut : -len(suffix)]: key
            for k, key in keys.items()
            if k.startswith(scope) and k.endswith(suffix)
        }
        parts = []
        while rest:
            name = next((n for n in names if rest.startswith(n + "_")), None)
            name = name or (rest if rest in names else None)
            if name is None:
                break
            parts.append(names[name])
            rest = rest[len(name) + 1 :]
        if not rest and len(parts) > 1:
            return parts
    return None


def checkpoint_sources(params, model):
    """Each of ``params`` that is a tensor of HF checkpoint ``model``, or a
    stack of its tensors along the rows (``stacked_keys``), as ``{name:
    (keys, transposed)}``, checked bit for bit: same shape, dtype and
    values.

    A parameter's name is the checkpoint key with ``.`` as ``_``, behind any
    number of ``model_`` scopes, and a ``_t`` suffix when it is that tensor
    transposed (``model_layers_0_self_attn_q_proj_weight_t`` is
    ``model.layers.0.self_attn.q_proj.weight``, transposed); ``lm_head``
    is the input embedding in a checkpoint storing only that.
    """
    path = snapshot_download(model, allow_patterns=["*.json"])
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = json.load(f)["weight_map"]
    else:
        # An unsharded checkpoint is one file, with no index.
        single = "model.safetensors"
        path = snapshot_download(model, allow_patterns=[single])
        with safe_open(os.path.join(path, single), "pt") as f:
            files = dict.fromkeys(f.keys(), single)
    keys = {key.replace(".", "_"): key for key in files}
    # A checkpoint tying its output projection to the input embedding
    # stores only the embedding.
    embedding = [k for k in files if k.endswith("embed_tokens.weight")]
    if "lm_head_weight" not in keys and embedding:
        keys["lm_head_weight"] = embedding[0]
    sources = {}
    for name, tensor in params.items():
        transposed = name.endswith("_t")
        stem = name[:-2] if transposed else name
        parts = [keys[stem]] if stem in keys else stacked_keys(stem, keys)
        while parts is None and stem.startswith("model_"):
            stem = stem[len("model_") :]
            parts = [keys[stem]] if stem in keys else stacked_keys(stem, keys)
        if parts is None:
            continue
        values = []
        for key in parts:
            file = snapshot_download(model, allow_patterns=[files[key]])
            with safe_open(os.path.join(file, files[key]), "pt") as f:
                values.append(f.get_tensor(key))
        value = torch.cat(values)
        if transposed:
            value = value.T
        if value.dtype == tensor.dtype and torch.equal(value, tensor):
            sources[name] = (parts, transposed)
    return sources


def strip_weights(traced, step, params, model, step_dir):
    """Leave out of ``traced`` every weight of checkpoint ``model``.

    Writes ``weights.json`` -- each weight slot's parameter, and its
    checkpoint key and orientation or else ``extra`` -- and
    ``extra.safetensors``, the slots ``model`` does not hold.  The slots
    keep zero-strided placeholders, so ``step.pt`` holds no weight bytes.
    """
    sources = checkpoint_sources(params, model)
    slots, extra, placeholders = {}, {}, {}
    for slot, value in traced.weights._parameters.items():
        name = step.param_names[int(slot[len("param") :])]
        if name in sources:
            keys, transposed = sources[name]
            slots[slot] = {"name": name, "keys": keys, "transposed": transposed}
        else:
            slots[slot] = {"name": name, "extra": True}
            extra[slot] = value.contiguous()
        placeholders[slot] = torch.zeros(1, dtype=value.dtype).expand(
            value.shape
        )
    torch_neuronx.replace_weights(traced, placeholders)
    revision = os.path.basename(snapshot_download(model, allow_patterns=[]))
    with open(os.path.join(step_dir, "weights.json"), "w") as f:
        json.dump(
            {"model": model, "revision": revision, "slots": slots}, f, indent=1
        )
    save_file(extra, os.path.join(step_dir, "extra.safetensors"))
    print(f"{len(sources)} weights from {model}, {len(extra)} in extra")


def main(directory, phase, target, weights_from):
    with open(os.path.join(directory, "manifest.json")) as f:
        manifest = json.load(f)
    tensors = torch.load(os.path.join(directory, "tensors.pt"))
    # Weights in the type the kernels take them, once, not every step.
    dtypes = manifest["dtypes"]
    params = {
        name: tensors[name].to(
            KERNEL_TYPES.get(dtypes[name], tensors[name].dtype)
        )
        for name in manifest["params"]
    }
    inputs = [tensors[name] for name in manifest["inputs"]]
    step = Step(
        load_module(directory, "program"),
        load_module(directory, "kernels"),
        params,
        set(manifest["state"]),
        manifest["inputs"],
        # The host scalars of the example run: the trace is specialized to
        # them (a decode step to its position).
        {name: tensors[name].item() for name in manifest["scalars"]},
    )
    step_dir = os.path.join(directory, "step")
    os.makedirs(step_dir, exist_ok=True)
    hlo_flags = HLO_FLAGS[phase]
    if "fp8_e4m3_ieee" in dtypes.values():
        # torch carries the IEEE E4M3 as float8_e4m3fn, the same bits up to
        # its 240: NKI's non-OCP mode, and NxD Inference's flag.
        os.environ["NKI_FP8_E4M3_MODE"] = "non_ocp"
        hlo_flags += FP8_HLO_FLAG
    hlo_flags += "--verify-hlo=true --recursive-layer-det=false"
    compiler_args = (
        f"--target={target} {COMPILER_FLAGS}{PHASE_FLAGS[phase]} "
        f"--internal-hlo2tensorizer-options='{hlo_flags}'"
    )
    traced = torch_neuronx.trace(
        step,
        tuple(inputs),
        compiler_workdir=step_dir,
        inline_weights_to_neff=False,
        enable_aliasing=True,
        compiler_args=compiler_args,
    )
    if weights_from is not None:
        strip_weights(traced, step, params, weights_from, step_dir)
    torch.jit.save(traced, os.path.join(step_dir, "step.pt"))
    print(f"traced -> {step_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--phase", choices=tuple(PHASE_FLAGS), required=True)
    parser.add_argument("--target", default="trn2")
    parser.add_argument("--weights-from", default=None)
    args = parser.parse_args()
    main(args.directory, args.phase, args.target, args.weights_from)
