"""One transform/compile/verification path for every model adapter."""

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils._pytree import tree_map

from voyager_compiler import (
    compile,
    extract_input_preprocessor,
    fuse_operator,
    transform,
)


def identity(value):
    return value


@dataclass
class PreparedModel:
    graph: object
    example_args: tuple
    reference: object
    example_kwargs: dict = field(default_factory=dict)
    extract_preprocessor: bool = False
    output_adapter: Callable = identity
    restore_state: Callable | None = None
    capture_state: Callable | None = None


def compile_prepared(prepared, args, vector_stages):
    from utils.models.utils import get_compile_args, get_transform_args

    gm = prepared.graph
    inputs = prepared.example_args
    kwargs = prepared.example_kwargs
    clone = lambda x: x.clone() if isinstance(x, torch.Tensor) else x
    reference = tree_map(clone, prepared.reference)
    if prepared.restore_state is not None:
        prepared.restore_state(gm)
    transform(
        gm,
        inputs,
        kwargs,
        **get_transform_args(args, vector_stages),
        skip_op_fusion=prepared.extract_preprocessor,
    )
    preprocess_fn = None
    if prepared.extract_preprocessor:
        gm, preprocess_fn = extract_input_preprocessor(gm)
        inputs = (preprocess_fn(*inputs),)
        fuse_operator(gm, vector_stages)

    verification_inputs = (
        tree_map(clone, (inputs, kwargs)) if args.debug else None
    )

    restore_state = prepared.restore_state

    def capture_lowered_state(graph):
        nonlocal restore_state
        restore_state = prepared.capture_state(graph)

    # Lowering and optional tensor dumps can mutate captured state. Verification
    # always starts from the adapter's reference state, with fresh input copies.
    options = get_compile_args(args)
    if args.debug and prepared.capture_state is not None:
        options["before_emit"] = capture_lowered_state
    compile(gm, inputs, kwargs, **options)
    gm.graph.print_tabular()
    output = None
    if args.debug:
        if restore_state is not None:
            restore_state(gm)
        copied_inputs, copied_kwargs = verification_inputs
        try:
            output = tree_map(
                clone,
                prepared.output_adapter(gm(*copied_inputs, **copied_kwargs)),
            )
        finally:
            if restore_state is not None:
                restore_state(gm)
    result = (gm, reference, output)
    return (*result, preprocess_fn) if prepared.extract_preprocessor else result
