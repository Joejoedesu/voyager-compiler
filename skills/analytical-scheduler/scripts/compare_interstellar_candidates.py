#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compare exact local Interstellar tiles under Voyager's current model."""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_tile(value):
    """Parse NAME:OYxOX or OYxOX into a candidate tuple."""
    if ":" in value:
        name, shape = value.split(":", 1)
    else:
        name = value
        shape = value
    try:
        oy, ox = (int(part) for part in shape.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "tile must be NAME:OYxOX or OYxOX"
        ) from exc
    if oy <= 0 or ox <= 0:
        raise argparse.ArgumentTypeError("tile extents must be positive")
    return name, oy, ox


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = os.environ.get("VOYAGER_ROOT", "voyager-compiler")
    parser.add_argument("--voyager-root", default=default_root)
    parser.add_argument("--ic", type=int, required=True)
    parser.add_argument("--oc", type=int, required=True)
    parser.add_argument("--oy", type=int, required=True)
    parser.add_argument("--ox", type=int, required=True)
    parser.add_argument("--fy", type=int, default=1)
    parser.add_argument("--fx", type=int, default=1)
    parser.add_argument("--stride-y", type=int, default=1)
    parser.add_argument("--stride-x", type=int, default=1)
    parser.add_argument("--ic0", type=int, required=True)
    parser.add_argument("--oc0", type=int, required=True)
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--input-buffer-size", type=int, required=True)
    parser.add_argument("--weight-buffer-size", type=int, required=True)
    parser.add_argument("--accum-buffer-size", type=int, required=True)
    parser.add_argument("--input-bits", type=int, default=8)
    parser.add_argument("--output-bits", type=int, default=8)
    parser.add_argument(
        "--double-buffered-accum-buffer",
        action="store_true",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_tile,
        required=True,
    )
    return parser


def load_modules(voyager_root):
    # Voyager is an optional, workspace-local dependency of this skill.
    src = Path(voyager_root).resolve() / "src"
    if not src.is_dir():
        raise SystemExit(f"Voyager source directory not found: {src}")
    sys.path.insert(0, str(src))

    import interstellar
    from interstellar.mapping_point import MappingPoint
    from interstellar.mapping_point_generator import (
        blocking_partitioning_generator_function,
        opt_get_loop_order_generator,
    )
    from voyager_compiler.codegen.tiler import (
        RuntimeCalculator,
        build_architecture_and_schedule,
    )

    return (
        interstellar,
        MappingPoint,
        blocking_partitioning_generator_function,
        opt_get_loop_order_generator,
        RuntimeCalculator,
        build_architecture_and_schedule,
    )


def candidate_blocking(args, local_oy, local_ox):
    if args.oy % local_oy or args.ox % local_ox:
        return None
    return [
        (1, 1, args.fx),
        (1, 1, args.fy),
        (1, local_ox, args.ox // local_ox),
        (1, local_oy, args.oy // local_oy),
        (1, 1, args.oc // args.oc0),
        (1, 1, args.ic // args.ic0),
        (1, 1, 1),
    ]


def find_generated_candidate(generator, target):
    for blocking, partitioning, para_dim in generator:
        if list(blocking) == target:
            return blocking, partitioning, para_dim
    return None


def evaluate_candidate(modules, architecture, schedule, layer, runtime, found):
    interstellar, MappingPoint, _, order_generator, _, _ = modules
    blocking, partitioning, para_dim = found
    dummy = MappingPoint(None, blocking, partitioning, para_dim)
    best = None
    for mapping in order_generator(
        architecture,
        layer,
        dummy,
        schedule,
        False,
    ):
        cycles = runtime.calculate_runtime(architecture, layer, mapping)
        energy = sum(
            interstellar.cost_model.get_level_cost(
                architecture,
                mapping,
                layer,
                level,
                False,
            )
            for level in range(architecture.buffer_levels())
        )
        key = cycles, energy
        if best is None or key < best[0]:
            accesses = interstellar.cost_model.get_access(
                mapping,
                layer,
                architecture,
            )[0]
            best = key, mapping, accesses
    return best


def main():
    args = build_parser().parse_args()
    if args.ic % args.ic0 or args.oc % args.oc0:
        raise SystemExit("IC and OC must be divisible by IC0 and OC0")

    modules = load_modules(args.voyager_root)
    interstellar = modules[0]
    generator_fn = modules[2]
    RuntimeCalculator = modules[4]
    build_architecture = modules[5]

    architecture, schedule = build_architecture(
        args.ic0,
        args.oc0,
        args.cache_size,
        args.input_buffer_size,
        args.weight_buffer_size,
        args.accum_buffer_size,
    )
    layer = interstellar.Layer(
        args.ic,
        args.oc,
        args.ox,
        args.oy,
        args.fx,
        args.fy,
        wstd=args.stride_x,
        hstd=args.stride_y,
    )
    runtime = RuntimeCalculator(
        args.input_bits,
        args.output_bits,
        args.double_buffered_accum_buffer,
    )

    results = []
    for name, local_oy, local_ox in args.candidate:
        target = candidate_blocking(args, local_oy, local_ox)
        result = {
            "name": name,
            "local_oy": local_oy,
            "local_ox": local_ox,
        }
        if target is None:
            result["legal"] = False
            result["reason"] = "local tile does not divide output extent"
            results.append(result)
            continue

        generated = find_generated_candidate(
            generator_fn(architecture, layer, schedule),
            target,
        )
        if generated is None:
            result["legal"] = False
            result["reason"] = "candidate was pruned before loop-order search"
            results.append(result)
            continue

        best = evaluate_candidate(
            modules,
            architecture,
            schedule,
            layer,
            runtime,
            generated,
        )
        if best is None:
            result["legal"] = False
            result["reason"] = "no legal loop order"
            results.append(result)
            continue

        key, mapping, accesses = best
        result.update(
            {
                "legal": True,
                "runtime": key[0],
                "energy_score": key[1],
                "spatial_tile_count": (
                    args.oy // local_oy
                ) * (args.ox // local_ox),
                "loop_orders": mapping.loop_orders,
                "accesses": accesses,
            }
        )
        results.append(result)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
