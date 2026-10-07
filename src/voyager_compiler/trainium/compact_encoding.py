"""Select and verify affine encodings of already selected instruction repeats.

Only HBM address constants may vary. Local addresses, ISA operations, engines,
shapes and ordering must be identical in every encoded body. This pass runs at
selection time; emission applies its recorded encoding without choosing loops.
"""

import ast
import copy
from functools import lru_cache


def shift(rule, iteration):
    if isinstance(rule, int):
        return rule * iteration
    terms = rule.get("terms", (rule,))
    return rule["step"] * iteration + sum(
        t["jump"] * ((iteration + t["phase"]) // t["period"]) for t in terms
    )


def fit(values):
    """Fit affine addresses with up to two verified traversal wrap levels."""
    from collections import Counter

    base = values[0]
    differences = [b - a for a, b in zip(values, values[1:])]
    if len(set(differences)) == 1:
        return differences[0]
    # Traversal coordinates have a common stride and sparse wrap jumps.
    # Infer phases from observed jumps; scanning every possible phase was
    # quadratic in the outer traversal period.
    for step, _ in Counter(differences).most_common(2):
        residual = [d - step for d in differences]
        terms = []
        for _ in range(2):
            nonzero = [i for i, d in enumerate(residual) if d]
            if not nonzero:
                break
            found = False
            for period in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                if period >= len(values):
                    break
                phase = period - 1 - nonzero[0] % period
                boundary = [
                    i
                    for i in range(len(residual))
                    if (i + 1 + phase) // period != (i + phase) // period
                ]
                boundary_set = set(boundary)
                if any(i not in boundary_set for i in nonzero):
                    continue
                jump = Counter(residual[i] for i in boundary).most_common(1)[0][
                    0
                ]
                if not jump:
                    continue
                terms.append(dict(jump=jump, period=period, phase=phase))
                for i in boundary:
                    residual[i] -= jump
                found = True
                break
            if not found:
                break
        rule = dict(step=step, terms=terms)
        if (
            terms
            and not any(residual)
            and all(v == base + shift(rule, i) for i, v in enumerate(values))
        ):
            return rule
    return None


@lru_cache(maxsize=65536)
def _statement_signature(line, hbm):
    tree = ast.parse(line)
    values = []
    allowed = []

    def visit(node, address=False):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            values.append(node.value)
            allowed.append(address)
            node.value = 0
            return
        if isinstance(node, ast.Subscript):
            base = node.value
            while isinstance(base, (ast.Call, ast.Attribute)):
                base = base.func if isinstance(base, ast.Call) else base.value
            visit(node.value, False)
            visit(node.slice, isinstance(base, ast.Name) and base.id in hbm)
            return
        if isinstance(node, ast.Call):
            # arange extents and reshape dimensions must stay static.
            for child in ast.iter_child_nodes(node):
                visit(child, False)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, address)

    visit(tree)
    return ast.dump(tree), tuple(values), tuple(allowed)


def signature(lines, hbm):
    # Grouping neighbors must not repeatedly parse the same expanded body.
    # Each line is one selected instruction; concatenate its immutable syntax
    # key and address fields in source order, preserving exact verification.
    hbm = frozenset(hbm)
    parts = [_statement_signature(line, hbm) for line in lines]
    return (
        tuple(p[0] for p in parts),
        [v for p in parts for v in p[1]],
        [v for p in parts for v in p[2]],
    )


def select(program, lines):
    hbm = {n for n, t in program.tensors.items() if t.memory == "HBM"}
    candidates = []
    for region in program.repeated_regions:
        iterations = region["iterations"]
        if len(iterations) < 4:
            continue
        for prefix in range(min(4, len(iterations) - 3)):
            for group in (1, 2, 4, 8, 16, 32, 64):
                repeats = (len(iterations) - prefix) // group
                if repeats < 4:
                    continue
                first = iterations[prefix][0]
                block = iterations[prefix + group - 1][1] - first
                if not block:
                    continue
                while (
                    repeats >= 4
                    and iterations[prefix + repeats * group - 1][1]
                    != first + block * repeats
                ):
                    repeats -= 1
                if repeats < 4:
                    continue
                stop = first + block * repeats
                if any(
                    iterations[prefix + i * group][0] != first + i * block
                    or iterations[prefix + (i + 1) * group - 1][1]
                    != first + (i + 1) * block
                    for i in range(repeats)
                ):
                    continue
                key, base, allowed = signature(
                    lines[first : first + block], hbm
                )
                second_key, second, _ = signature(
                    lines[first + block : first + 2 * block], hbm
                )
                if second_key != key:
                    continue
                columns = [base, second]
                good = True
                for i in range(2, repeats):
                    other, values, _ = signature(
                        lines[first + i * block : first + (i + 1) * block], hbm
                    )
                    if other != key:
                        good = False
                        break
                    columns.append(values)
                if good:
                    delta = [fit(values) for values in zip(*columns)]
                    if any(
                        d is None or (d and not ok)
                        for d, ok in zip(delta, allowed)
                    ):
                        continue
                    candidates.append(
                        dict(
                            first=first,
                            block=block,
                            repeats=repeats,
                            deltas=delta,
                            loop_kind="sequential",
                        )
                    )
    selected = []
    for c in sorted(
        candidates, key=lambda c: c["block"] * (c["repeats"] - 1), reverse=True
    ):
        first = c["first"]
        stop = first + c["block"] * c["repeats"]
        if not any(
            first < x["first"] + x["block"] * x["repeats"] and x["first"] < stop
            for x in selected
        ):
            selected.append(c)
    return sorted(selected, key=lambda c: c["first"])


def encode(program, lines):
    """Mechanically encode recorded loops and reject any changed expansion."""
    hbm = {n for n, t in program.tensors.items() if t.memory == "HBM"}
    result = []
    cursor = 0
    for number, loop in enumerate(program.encoding_loops):
        first, block, repeats = (loop[k] for k in ("first", "block", "repeats"))
        stop = first + block * repeats
        if (
            first < cursor
            or block < 1
            or repeats < 1
            or stop > len(lines)
            or loop["loop_kind"] != "sequential"
        ):
            raise ValueError("Invalid selected loop encoding")
        body = lines[first : first + block]
        key, base, allowed = signature(body, hbm)
        delta = loop["deltas"]
        if len(delta) != len(base) or any(
            d and not ok for d, ok in zip(delta, allowed)
        ):
            raise ValueError(
                "Loop encoding changes a non-address instruction field"
            )
        for i in range(1, repeats):
            other, values, _ = signature(
                lines[first + i * block : first + (i + 1) * block], hbm
            )
            if other != key or values != [
                a + shift(d, i) for a, d in zip(base, delta)
            ]:
                raise ValueError(
                    "Loop encoding differs from selected instruction expansion"
                )
        result.extend(lines[cursor:first])
        variable = f"__repeat{number}"

        class Substitute(ast.NodeTransformer):
            def __init__(self):
                self.index = 0

            def visit_Constant(self, node):
                if type(node.value) is not int:
                    return node
                d = delta[self.index]
                self.index += 1
                if not d:
                    return node
                if isinstance(d, int):
                    expression = f"{node.value}+{variable}*{d}"
                else:
                    expression = f"{node.value}+{variable}*{d['step']}"
                    for term in d.get("terms", (d,)):
                        expression += f"+{term['jump']}*(({variable}+{term['phase']})//{term['period']})"
                return ast.parse(expression, mode="eval").body

        tree = Substitute().visit(ast.parse("\n".join(body)))
        result.append(f"for {variable} in nl.sequential_range({repeats}):")
        result.extend(
            "    " + line
            for line in ast.unparse(
                ast.fix_missing_locations(tree)
            ).splitlines()
        )
        cursor = stop
    return result + lines[cursor:]
