"""The shared prepare → normalize → search/bufferize → allocate → emit flow."""

import copy
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils._pytree import tree_flatten

from voyager_compiler.artifacts import hardware_digest, save_bundle
from voyager_compiler.compiler.allocate import (
    MemoryCapacityError,
    buffer_bindings,
    plan_memory,
)
from voyager_compiler.compiler.analysis import analyze_values
from voyager_compiler.compiler.bufferize import (
    bufferize_graph,
    print_bufferized_graph,
)
from voyager_compiler.compiler.bufferize.emit import gen_compute_graph
from voyager_compiler.compiler.bufferize.ops import oracle_disabled
from voyager_compiler.compiler.emit import emit
from voyager_compiler.compiler.normalize import transform
from voyager_compiler.compiler.prepare import prepare
from voyager_compiler.compiler.quantize import quantize
from voyager_compiler.compiler.search.planner import candidate_key
from voyager_compiler.compiler.search.tiler import (
    DEFAULT_RUNTIME_TOLERANCE,
    Tiler,
)
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.ir.buffer import BufferProgram
from voyager_compiler.ir.plan import CompilationPlan
from voyager_compiler.shape_prop import ShapeProp, fake_like


@dataclass
class CompilationResult:
    program: BufferProgram
    model: object
    bundle: dict
    reference: object = None
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


def compile_bufferized(
    model,
    example_args,
    example_kwargs=None,
    config=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    runtime_tolerance=None,
    dram_layout=None,
    accumulate_fp32=False,
    bool_mask=True,
    *,
    target=None,
    provenance=None,
    performance_profile=None,
    max_replans=2,
    reference_outputs=None,
):
    if config is None:
        config = target.hardware() if target else AcceleratorConfig()
    target = config.target_spec() if target is None else target
    if output_dir is None:
        raise ValueError("output_dir is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    flat_inputs, _ = tree_flatten((example_args, example_kwargs or {}))
    ShapeProp(model).propagate(*map(fake_like, flat_inputs))
    relations = analyze_values(model)
    gen_compute_graph(model, str(output / output_file))
    tolerance = (
        DEFAULT_RUNTIME_TOLERANCE
        if runtime_tolerance is None
        else runtime_tolerance
    )
    runtime_model = target.model(config, accumulate_fp32)
    if performance_profile is not None:
        from voyager_compiler.performance.profile import CalibratedModel

        if (
            performance_profile.target != target.name
            or performance_profile.hardware_digest != hardware_digest(config)
        ):
            raise ValueError(
                "performance profile does not match the selected target/hardware"
            )
        runtime_model = CalibratedModel(runtime_model, performance_profile)
        provenance = dict(
            provenance or {}, performance_profile=performance_profile.identity
        )
    if max_replans < 0:
        raise ValueError("max_replans must be nonnegative")
    # Retain graph structure for bounded retries without copying model weights.
    memo = {
        id(tensor): tensor for tensor in (*model.parameters(), *model.buffers())
    }
    original = copy.deepcopy(model, memo) if max_replans else None
    excluded, retries = {}, []
    for attempt in range(max_replans + 1):
        tiler = Tiler(runtime_model, tolerance)
        plan = CompilationPlan(target.name)
        bufferize_graph(
            model,
            pipelined=config.double_buffered_l2,
            tiler=tiler,
            async_pipeline=config.async_pipeline,
            bool_mask=bool_mask,
            implementations=target.implementations,
            compilation_plan=plan,
            excluded_choices=excluded,
        )
        program = BufferProgram(model, plan)
        try:
            if target.refine is not None:
                target.refine(program, config)
            allocation = plan_memory(model, config, dram_layout)
            from voyager_compiler.compiler.bufferize.invocation import (
                finalize_invocations,
            )

            allocation = finalize_invocations(program, config, allocation)
        except MemoryCapacityError as error:
            site = next(
                (
                    site
                    for site in (*error.sites, *plan.choices)
                    if site in plan.choices
                    and any(
                        choice != plan.choices[site]
                        and candidate_key(
                            choice.implementation, choice.parameters
                        )
                        not in excluded.get(site, ())
                        for choice in plan.alternatives.get(site, ())
                    )
                ),
                None,
            )
            if attempt == max_replans or site is None:
                raise
            choice = plan.choices[site]
            excluded.setdefault(site, set()).add(
                candidate_key(choice.implementation, choice.parameters)
            )
            retries.append(
                dict(
                    site=site,
                    implementation=choice.implementation,
                    reason=str(error),
                )
            )
            restored = copy.deepcopy(original, memo.copy())
            model.__dict__.update(restored.__dict__)
            model.graph = restored.graph
            model.recompile()
            ShapeProp(model).propagate(*map(fake_like, flat_inputs))
            continue
        break
    program.bindings = buffer_bindings(model)
    # Execute shape/effect contracts on the expanded graph after placement.
    # Real numerical replay is optional here; a runtime/device check is separate.
    for module in model.modules():
        if hasattr(module, "graph"):
            module.graph.lint()
    with oracle_disabled():
        ShapeProp(model, recurse=True).propagate(*map(fake_like, flat_inputs))
    program.validation.update(
        kind="buffer_shape_and_allocation", replans=retries
    )
    if reference_outputs is not None:
        import torch

        with torch.no_grad():
            replay = model(*example_args, **(example_kwargs or {}))
        torch.testing.assert_close(replay, reference_outputs)
        program.validation["numerical_replay"] = "passed"
    program.freeze(allocation)
    model.meta["compilation_plan"] = plan
    model.meta["value_relations_source"] = relations.graph_digest
    print_bufferized_graph(model)
    encoded = emit(
        program,
        flat_inputs,
        output,
        dump_tensors,
        encode=target.encode,
        config=config,
    )
    if target.emit is not None:
        target.emit(program, encoded, config, output)
        program.assert_frozen()
    bundle = save_bundle(program, config, output, provenance)
    return CompilationResult(
        program,
        encoded,
        bundle,
        args=tuple(example_args),
        kwargs=dict(example_kwargs or {}),
    )


def compile_workload(
    workload,
    target,
    output_dir,
    *,
    config=None,
    options=None,
    normalize_options=None,
    **compile_options,
):
    """Compile a registered workload using the same path as the FX entry point."""
    options = {} if options is None else dict(options)
    options.setdefault("target", target.name)
    config = target.hardware() if config is None else config
    prepared = prepare(workload, options)
    if workload.calibration is not None and "calibration" not in options:
        options["calibration"] = workload.calibration(prepared, options)
    prepared = quantize(prepared, target, options, config=config)
    if target.legalize is not None:
        target.legalize(prepared, config, options)
    transform(
        prepared.graph,
        prepared.args,
        prepared.kwargs,
        config=config,
        **(dict(target.normalization) | (normalize_options or {})),
    )
    if target.lower is not None:
        target.lower(prepared, config, options)
    result = compile_bufferized(
        prepared.graph,
        prepared.args,
        prepared.kwargs,
        config=config,
        output_dir=output_dir,
        target=target,
        provenance=prepared.provenance,
        **compile_options,
    )
    result.reference = prepared.reference
    return result
