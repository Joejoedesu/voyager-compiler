"""Conservative static slice hazards for analytical scheduling of selected ISA.

Only logical whole-root edges reconstructed from the version-1 Builder are
refined. Explicit extra edges and physical allocation reuse edges are retained.
Unknown/dynamic or large views fall back to whole-root overlap. No measured
schedule or hardware latency enters dependency construction.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from math import gcd, prod

import numpy as np


def intersects(a, b):
    """Intersection of two finite, positive-step integer progressions."""
    if not a or not b:
        return False
    lo, hi = max(a.start, b.start), min(a[-1], b[-1])
    if lo > hi:
        return False
    g = gcd(a.step, b.step)
    if (b.start - a.start) % g:
        return False
    modulus = b.step // g
    t = ((b.start - a.start) // g * pow(a.step // g, -1, modulus)) % modulus
    x, period = a.start + a.step * t, a.step * modulus
    x += ((lo - x + period - 1) // period) * period
    return x <= hi


@dataclass(frozen=True)
class Footprint:
    axes: tuple[range, ...] | None
    exact: bool = False

    def overlaps(self, other):
        if self.axes is None or other.axes is None:
            return True
        return all(intersects(a, b) for a, b in zip(self.axes, other.axes))

    def covers(self, other):
        if not self.exact or self.axes is None or other.axes is None:
            return False
        return all(
            not b
            or (
                bool(a)
                and b.start in a
                and b[-1] in a
                and (len(b) == 1 or b.step % a.step == 0)
            )
            for a, b in zip(self.axes, other.axes)
        )


UNKNOWN = Footprint(None)


def footprint(points, shape):
    """Conservative root-coordinate product; fast path for affine axis views."""
    axes, varying_axes = [], []
    for values in points:
        if not values.size:
            return Footprint(tuple(range(0) for _ in points), True)
        origin = (0,) * values.ndim
        start = int(values[origin])
        found = False
        for dim, size in enumerate(values.shape):
            if size <= 1:
                continue
            selection = list(origin)
            selection[dim] = slice(None)
            line = values[tuple(selection)]
            step = int(line[1]) - start
            if not step or not np.array_equal(
                line, start + step * np.arange(size)
            ):
                continue
            view_shape = [1] * values.ndim
            view_shape[dim] = size
            if not np.array_equal(
                values, np.broadcast_to(line.reshape(view_shape), values.shape)
            ):
                continue
            stop = start + step * (size - 1)
            axes.append(
                range(min(start, stop), max(start, stop) + 1, abs(step))
            )
            varying_axes.append(dim)
            found = True
            break
        if found:
            continue
        if np.all(values == start):
            axes.append(range(start, start + 1))
            continue
        break
    else:
        # Different root axes driven by the same view axis form a diagonal,
        # not a Cartesian product. Its product remains a safe overlap bound.
        return Footprint(
            tuple(axes), len(set(varying_axes)) == len(varying_axes)
        )

    # General reshape / advanced-index fallback, bounded by max_view_elements.
    axes, exact = [], True
    for values in points:
        unique = np.unique(values)
        step = int(np.gcd.reduce(np.diff(unique))) if len(unique) > 1 else 1
        axis = range(int(unique[0]), int(unique[-1]) + 1, step)
        axes.append(axis)
        exact &= len(axis) == len(unique)
    if exact:
        unique_count = np.unique(np.ravel_multi_index(points, shape)).size
        exact = prod(len(axis) for axis in axes) == unique_count
    return Footprint(tuple(axes), bool(exact))


def refine(program, *, max_view_elements=65536):
    """Return an analysis-only program and audit; preserve emitted source/storage."""
    from .instruction_plan import Builder, Tensor
    from .movement_search import coordinates

    checker = Builder.__new__(Builder)
    checker.program, checker.constants = program, {}
    for name, expr in program.indices.items():
        checker.constants[name] = checker.value(expr)
    stats = dict(
        refined_edges_removed=0,
        refined_edges_added=0,
        exact_accesses=0,
        conservative_accesses=0,
    )

    @lru_cache(maxsize=512)
    def access(expr):
        try:
            value = checker.value(expr)
            if not isinstance(value, Tensor):
                return None
            # Full roots require no coordinate materialization, even for HBM.
            if expr.op == "name" and not program.tensors[expr.value].alias:
                return expr.value, Footprint(
                    tuple(range(n) for n in value.shape), True
                )
            if prod(value.shape) > max_view_elements:
                return program.root(value.name), UNKNOWN
            root, points = coordinates(checker, expr)
            return root, footprint(points, program.tensors[root].shape)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            NotImplementedError,
        ):
            return None

    physical = defaultdict(set)
    if program.encoding_storage != "compiler":
        for before, after in program.reuse_edges():
            physical[after].add(before)
    old_writes, old_users = {}, defaultdict(set)
    writers, readers = defaultdict(list), defaultdict(list)
    instructions = []
    for index, ins in enumerate(program.instructions):
        read_roots = {program.root(n) for n in ins.reads}
        write_roots = {program.root(n) for n in ins.writes}
        old_required = {
            old_writes[r] for r in read_roots | write_roots if r in old_writes
        }
        for root in write_roots:
            old_required.update(old_users[root])
        required = (set(ins.dependencies) - old_required) | physical[index]
        reads, writes = defaultdict(list), defaultdict(list)
        for expr in (*ins.args, *(v for _, v in ins.kwargs)):
            hit = access(expr)
            if hit and hit[0] in read_roots:
                reads[hit[0]].append(hit[1])
        hit = access(ins.destination)
        if hit:
            writes[hit[0]].append(hit[1])
            if ins.accumulate:
                reads[hit[0]].append(hit[1])
        for roots, views in ((read_roots, reads), (write_roots, writes)):
            for root in roots:
                if not views[root]:
                    views[root].append(UNKNOWN)
                for f in views[root]:
                    stats[
                        "exact_accesses" if f.exact else "conservative_accesses"
                    ] += 1
        for root in read_roots:
            for f in reads[root]:
                required.update(
                    i for i, old in writers[root] if f.overlaps(old)
                )
        for root in write_roots:
            for f in writes[root]:
                required.update(
                    i
                    for i, old in writers[root] + readers[root]
                    if f.overlaps(old)
                )
            # Retire only accesses fully covered by an exact write; partial
            # overlap never discards hazards on the untouched remainder.
            writers[root] = [
                (i, f)
                for i, f in writers[root]
                if not any(w.covers(f) for w in writes[root])
            ]
            readers[root] = [
                (i, f)
                for i, f in readers[root]
                if not any(w.covers(f) for w in writes[root])
            ]
            writers[root].extend((index, f) for f in writes[root])
        for root in read_roots:
            readers[root].extend((index, f) for f in reads[root])
        stats["refined_edges_removed"] += len(set(ins.dependencies) - required)
        stats["refined_edges_added"] += len(required - set(ins.dependencies))
        instructions.append(replace(ins, dependencies=tuple(sorted(required))))
        for root in write_roots:
            old_writes[root], old_users[root] = index, set()
        for root in read_roots:
            old_users[root].add(index)
    return replace(program, instructions=instructions), stats
