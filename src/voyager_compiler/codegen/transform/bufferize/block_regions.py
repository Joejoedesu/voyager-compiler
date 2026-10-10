"""Blocked matrix/reduction and gated-projection regions with explicit storage.

Structural contracts, not workload names. Schedules own row/feature traversal,
materialization and buffer generations; targets still realize individual tiles.
The materialized-softmax builder shares FA3's score commit and online state update.
"""

import operator
from dataclasses import dataclass
import torch
from torch._higher_order_ops.while_loop import while_loop
from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.subgraph import create_and_insert_subgraph
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import ShapeProp
from .attention_v3 import _FA3Pipeline
from .attention import _fuse_passes
from .ops import MemoryLevel, commit, oracle_disabled
from .pipeline import get_slot
from .utils import (
    voyager,
    _lenient_verifier,
    _finalize_exported_gm,
    effect_cond,
)
from .stream_regions import MATRIX, SOFTMAX, elide_contraction_padding

SRAM = int(MemoryLevel.SRAM)


def loop(n, action):
    def body(i):
        action(i)
        return (i + 1,)

    (end,) = while_loop(lambda i: i < n, body, (0,))
    torch._check(end == n)
    return end


def tile(x, offset, shape):
    return voyager.subview(x, offset, shape, [1] * len(shape))


def matmul(a, b, out, done, dependencies=()):
    def body(x, w, y):
        voyager.insert(x @ w, y)

    commit(body, [a, b, out], dependencies=dependencies, post=done)


@dataclass
class BlockRegion:
    kind: str
    nodes: tuple
    inputs: tuple
    shape: tuple


def discover(model):
    """Recognize closed matrix-softmax and gated two-projection DAGs."""
    found, used = [], set()
    mul = {torch.ops.aten.mul.Tensor}
    sigmoid = torch.ops.aten.sigmoid.default
    for out in reversed(list(model.graph.nodes)):
        if out in used or out.op != "call_function":
            continue
        if out.target in SOFTMAX:
            mm = out.args[0]
            if (
                mm.target not in MATRIX
                or get_arg_value(out, 1, "dim") % len(out.shape)
                != len(out.shape) - 1
            ):
                continue
            if len(mm.args[0].shape) not in (2, 3) or len(
                mm.args[1].shape
            ) != len(mm.args[0].shape):
                continue
            nodes = (mm, out)
            inputs = tuple(mm.args[:2])
            region = BlockRegion(
                "materialized_softmax", nodes, inputs, tuple(out.shape)
            )
        elif out.target in MATRIX and out.args[0].target in mul:
            product = out.args[0]
            leaves = []
            nodeset = {out}

            def flatten(n):
                if not isinstance(n, torch.fx.Node):
                    leaves.append(n)
                    return
                if n.target in mul:
                    nodeset.add(n)
                    for a in n.args:
                        flatten(a)
                else:
                    leaves.append(n)

            flatten(product)
            if any(not isinstance(n, torch.fx.Node) for n in leaves):
                continue
            sig = next((n for n in leaves if n.target == sigmoid), None)
            if len(leaves) != 3 or sig is None:
                continue
            gate = sig.args[0]
            up = next(
                (n for n in leaves if n is not sig and n is not gate), None
            )
            if (
                gate not in leaves
                or gate.target not in MATRIX
                or up is None
                or up.target not in MATRIX
            ):
                continue
            x, wg = gate.args[:2]
            if up.args[0] is not x or any(
                len(n.shape) != 2 for n in (x, wg, up.args[1], out.args[1])
            ):
                continue
            nodeset.update((gate, up, sig))
            nodes = tuple(n for n in model.graph.nodes if n in nodeset)
            inputs = (x, wg, up.args[1], out.args[1])
            region = BlockRegion(
                "gated_projection", nodes, inputs, tuple(out.shape)
            )
        else:
            continue
        if any(
            n is not out and any(u not in nodes for u in n.users) for n in nodes
        ):
            continue
        if any(n in nodes for n in inputs) or any(
            n.value.dtype != torch.float32 for n in inputs
        ):
            continue
        found.append(region)
        used.update(nodes)
    return list(reversed(found))


def validate_spec(region, spec):
    if spec["kind"] != region.kind:
        raise ValueError("Block schedule kind does not match its region")
    rows, features = spec["rows"], spec["features"]
    if (
        type(rows) is not int
        or type(features) is not int
        or min(rows, features) < 1
    ):
        raise ValueError("Block dimensions must be positive integers")
    if spec["slots"] not in (1, 2):
        raise ValueError("Block pipeline supports one or two slots")
    m = region.shape[-2]
    n = (
        region.shape[-1]
        if region.kind == "materialized_softmax"
        else region.inputs[1].shape[-1]
    )
    if m % rows or n % features:
        raise ValueError("Block schedule requires exact row/feature divisors")
    if region.kind == "materialized_softmax":
        if rows > 128:
            raise ValueError(
                "Softmax row layout requires at most 128 partitions"
            )
        if spec["storage"] not in ("sbuf", "hbm", "recompute"):
            raise ValueError("Unknown score materialization")
        a, b = region.inputs
        if a.shape[-1] != b.shape[-2] or b.shape[-1] != region.shape[-1]:
            raise ValueError("BMM dimensions disagree")
        if len(a.shape) == 3 and b.shape[0] not in (1, a.shape[0]):
            raise ValueError("Unsupported batch broadcast")
    else:
        if spec["slots"] != 1:
            raise ValueError(
                "Gated projection currently requires one buffer slot"
            )
        if type(spec["outputs"]) is not int or spec["outputs"] < 1:
            raise ValueError("Output block must be a positive integer")
        if spec["storage"] not in ("sbuf", "hbm") or spec["traversal"] not in (
            "row",
            "feature",
        ):
            raise ValueError(
                "Unknown gated projection materialization/traversal"
            )
        if spec["storage"] == "sbuf" and spec["traversal"] != "row":
            raise ValueError(
                "A retained output accumulator requires row-outer traversal"
            )
        x, wg, wu, wd = region.inputs
        if (
            tuple(wg.shape) != tuple(wu.shape)
            or x.shape[-1] != wg.shape[0]
            or wd.shape[0] != n
        ):
            raise ValueError("Gated projection dimensions disagree")
        if region.shape[-1] % spec["outputs"]:
            raise ValueError("Output block must divide output features")


def plan_block_regions(model, tiler):
    elide_contraction_padding(model)
    regions = discover(model)
    if not regions:
        raise ValueError(
            "No supported closed block region found after graph preparation"
        )
    choices = model.meta.get("block_region_choices")
    if choices is None or len(choices) != len(regions):
        raise ValueError(
            "Blocked regions require explicit selected schedule choices"
        )
    records = []
    for region, spec in zip(regions, choices):
        validate_spec(region, spec)
        grouped = create_and_insert_subgraph(list(region.nodes), model)
        # Subgraph input order follows dependency order, not semantic roles.
        inputs = list(grouped.all_input_nodes)
        spec = dict(
            spec, operand_order=[inputs.index(n) for n in region.inputs]
        )
        grouped.meta["block_region"] = spec
        records.append(
            dict(spec=spec, operations=[str(n.target) for n in region.nodes])
        )
    model.meta["block_regions"] = records
    model.graph.lint()
    model.recompile()


class MaterializedSoftmax(_FA3Pipeline):
    """FA3 online statistics, followed by full probability materialization.

    Full scores may live in SBUF slots, HBM, or be recomputed in the final pass.
    K remains whole; feature blocks are independent matmul outputs. Scale is one.
    """

    def __init__(self, a_shape, b_shape, spec):
        torch.nn.Module.__init__(self)
        self.a_shape, self.b_shape = tuple(a_shape), tuple(b_shape)
        self.spec = spec
        self.block_size = None
        self.scale = 1.0
        self.has_mask = False
        self.mask_scaled = False

    def forward(self, a, b):
        rank = len(self.a_shape)
        batch = 1 if rank == 2 else self.a_shape[0]
        m, k = self.a_shape[-2:]
        n = self.b_shape[-1]
        r, f, slots = (
            self.spec["rows"],
            self.spec["features"],
            self.spec["slots"],
        )
        blocks = n // f
        a = a.reshape(batch, m, k)
        b = b.reshape(1 if rank == 2 else self.b_shape[0], k, n)
        out = voyager.alloc([batch, m, n], a.dtype)
        scores_hbm = (
            voyager.alloc([batch, m, n], a.dtype)
            if self.spec["storage"] == "hbm"
            else None
        )
        q = voyager.alloc([r, k], a.dtype, SRAM)
        rhs = voyager.alloc([k, f], a.dtype, SRAM, slots)
        s = voyager.alloc([r, f], a.dtype, SRAM, slots)
        kept = (
            voyager.alloc([r, f], a.dtype, SRAM, blocks)
            if self.spec["storage"] == "sbuf"
            else None
        )
        mx = voyager.alloc([r, 1], a.dtype, SRAM)
        total = voyager.alloc([r, 1], a.dtype, SRAM)
        temp = voyager.alloc([r, 1], a.dtype, SRAM)
        alpha = voyager.alloc([r, 1], a.dtype, SRAM)
        inverse = voyager.alloc([r, 1], a.dtype, SRAM)
        qsem = voyager.zeros([], torch.int64)
        rsem = voyager.zeros([], torch.int64, num_slots=slots)
        done = voyager.zeros([], torch.int64)
        store = voyager.zeros([], torch.int64)

        def sweep(index):
            bi = index // (m // r)
            ri = index % (m // r)
            aa = tile(a, [bi, 0, 0], [1, m, k]).reshape(m, k)
            bb = tile(
                b,
                [0 if self.b_shape[0] == 1 or rank == 2 else bi, 0, 0],
                [1, k, n],
            ).reshape(k, n)
            oo = tile(out, [bi, 0, 0], [1, m, n]).reshape(m, n)
            hh = (
                tile(scores_hbm, [bi, 0, 0], [1, m, n]).reshape(m, n)
                if scores_hbm is not None
                else None
            )
            voyager.async_copy(aa, q, [ri], [r, k], qsem, [0])
            voyager.async_wait(qsem)
            if blocks != 1:
                self._reset(mx, total)

            def fetch(j):
                slot = j % slots
                torch._check(slot < slots)
                voyager.async_copy(
                    bb,
                    get_slot(rhs, slot),
                    [j],
                    [k, f],
                    get_slot(rsem, slot),
                    [1],
                )

            fetch(0)
            if blocks == 1:
                # The full normalization domain fits in one score tile. Use
                # the existing whole-row recipe without an online-state pass.
                ss = get_slot(s, 0)
                self._matmul_qk(
                    [q, get_slot(rhs, 0)], ss, done, [get_slot(rsem, 0)]
                )
                voyager.async_wait(done)
                voyager.insert(torch.softmax(ss, dim=-1), ss)
                voyager.async_copy(ss, oo, [ri, 0], [r, f], store, [0, 1])
                voyager.async_wait(store)
                return

            def first(j):
                slot = j % slots
                torch._check(slot < slots)
                if slots == 2:
                    effect_cond(j + 1 < blocks, lambda: fetch(j + 1))
                ss = get_slot(s, slot)
                self._matmul_qk(
                    [q, get_slot(rhs, slot)], ss, done, [get_slot(rsem, slot)]
                )
                # Preserve logits before FA3 exponentiates its score slot.
                if kept is not None or hh is not None:
                    voyager.async_wait(done)
                    if kept is not None:
                        voyager.insert(ss.clone(), get_slot(kept, j))
                    else:
                        voyager.async_copy(
                            ss, hh, [ri, j], [r, f], store, [0, 1]
                        )
                        voyager.async_wait(store)
                self._softmax(
                    ss,
                    ss,
                    mx,
                    total,
                    temp,
                    alpha,
                    done,
                    None,
                    None,
                    wait_scores=kept is None and hh is None,
                )
                if slots == 1:
                    effect_cond(j + 1 < blocks, lambda: fetch(j + 1))

            loop(blocks, first)
            voyager.insert(torch.reciprocal(total), inverse)
            if self.spec["storage"] == "recompute":
                fetch(0)

            def finish(j):
                slot = j % slots
                torch._check(slot < slots)
                ss = get_slot(s, slot)
                if kept is not None:
                    logits = get_slot(kept, j)
                elif hh is not None:
                    voyager.async_copy(hh, ss, [ri, j], [r, f], store, [0, 1])
                    voyager.async_wait(store)
                    logits = ss
                else:
                    if slots == 2:
                        effect_cond(j + 1 < blocks, lambda: fetch(j + 1))
                    self._matmul_qk(
                        [q, get_slot(rhs, slot)],
                        ss,
                        done,
                        [get_slot(rsem, slot)],
                    )
                    voyager.async_wait(done)
                    logits = ss
                voyager.insert(torch.exp(logits - mx) * inverse, ss)
                voyager.async_copy(ss, oo, [ri, j], [r, f], store, [0, 1])
                voyager.async_wait(store)
                if self.spec["storage"] == "recompute" and slots == 1:
                    effect_cond(j + 1 < blocks, lambda: fetch(j + 1))

            loop(blocks, finish)

        loop(batch * (m // r), sweep)
        return out.reshape((*self.a_shape[:-1], n))


class GatedProjection(torch.nn.Module):
    """Two projections + pointwise gate + contracting projection.

    HBM cut: feature-first or row-first producer, then a blocked final GEMM.
    SRAM: row-first feature chunks feed a retained final-output accumulator.
    """

    def __init__(self, shapes, spec):
        super().__init__()
        self.shapes = shapes
        self.spec = spec

    def forward(self, x, wg, wu, wd):
        m, k = self.shapes[0]
        n = self.shapes[1][1]
        d = self.shapes[3][1]
        r, f, t = self.spec["rows"], self.spec["features"], self.spec["outputs"]
        out = voyager.alloc([m, d], x.dtype)
        hidden = (
            voyager.alloc([m, n], x.dtype)
            if self.spec["storage"] == "hbm"
            else None
        )
        xx = voyager.alloc([r, k], x.dtype, SRAM)
        gg = voyager.alloc([k, f], x.dtype, SRAM)
        uu = voyager.alloc([k, f], x.dtype, SRAM)
        gate = voyager.alloc([r, f], x.dtype, SRAM)
        up = voyager.alloc([r, f], x.dtype, SRAM)
        h = voyager.alloc([r, f], x.dtype, SRAM)
        ds = voyager.alloc([f, d if hidden is None else t], x.dtype, SRAM)
        accum = voyager.alloc([r, d if hidden is None else t], x.dtype, SRAM)
        part = voyager.alloc([r, d if hidden is None else t], x.dtype, SRAM)
        load = voyager.zeros([], torch.int64)
        done = voyager.zeros([], torch.int64)
        store = voyager.zeros([], torch.int64)

        def weights(fi):
            voyager.async_copy(wg, gg, [fi], [k, f], load, [1])
            voyager.async_wait(load)
            voyager.async_copy(wu, uu, [fi], [k, f], load, [1])
            voyager.async_wait(load)

        def activation(ri):
            voyager.async_copy(x, xx, [ri], [r, k], load, [0])
            voyager.async_wait(load)

        def project():
            matmul(xx, gg, gate, done)
            voyager.async_wait(done)
            matmul(xx, uu, up, done)
            voyager.async_wait(done)
            voyager.insert(gate * torch.sigmoid(gate) * up, h)

        if hidden is not None:

            def producer(outer):
                if self.spec["traversal"] == "feature":
                    weights(outer)
                else:
                    activation(outer)

                def inner(inner):
                    fi, ri = (
                        (outer, inner)
                        if self.spec["traversal"] == "feature"
                        else (inner, outer)
                    )
                    if self.spec["traversal"] == "feature":
                        activation(ri)
                    else:
                        weights(fi)
                    project()
                    voyager.async_copy(
                        h, hidden, [ri, fi], [r, f], store, [0, 1]
                    )
                    voyager.async_wait(store)

                loop(
                    m // r if self.spec["traversal"] == "feature" else n // f,
                    inner,
                )

            loop(
                n // f if self.spec["traversal"] == "feature" else m // r,
                producer,
            )

            # Flatten the output grid: each tile owns a complete contraction.
            def final_tile(index):
                ri, oi = index // (d // t), index % (d // t)
                voyager.insert(torch.zeros_like(accum), accum)

                def contract(fi):
                    voyager.async_copy(
                        hidden, h, [ri, fi], [r, f], load, [0, 1]
                    )
                    voyager.async_wait(load)
                    voyager.async_copy(wd, ds, [fi, oi], [f, t], load, [0, 1])
                    voyager.async_wait(load)
                    matmul(h, ds, part, done)
                    voyager.async_wait(done)
                    voyager.insert(accum + part, accum)

                loop(n // f, contract)
                voyager.async_copy(accum, out, [ri, oi], [r, t], store, [0, 1])
                voyager.async_wait(store)

            loop((m // r) * (d // t), final_tile)
        else:

            def row(ri):
                activation(ri)
                voyager.insert(torch.zeros_like(accum), accum)

                def feature(fi):
                    weights(fi)
                    project()
                    voyager.async_copy(wd, ds, [fi], [f, d], load, [0])
                    voyager.async_wait(load)
                    matmul(h, ds, part, done)
                    voyager.async_wait(done)
                    voyager.insert(accum + part, accum)

                loop(n // f, feature)
                voyager.async_copy(accum, out, [ri], [r, d], store, [0])
                voyager.async_wait(store)

            loop(m // r, row)
        return out


def tag_loops(gm):
    # Export checks keep side-effecting nested loops live through Dynamo. After
    # finalization the assertions are gone; drop only their dead scalar tails.
    for node in reversed(list(gm.graph.nodes)):
        if (
            not node.users
            and node.op == "call_function"
            and node.target in (operator.eq, operator.getitem)
        ):
            gm.graph.erase_node(node)
    gm.recompile()
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            sub = getattr(gm, str(n.target), None)
            if isinstance(sub, torch.fx.GraphModule):
                tag_loops(sub)
        if n.target is torch.ops.higher_order.while_loop:
            cond = getattr(gm, str(n.args[0].target))
            test = next(
                x
                for x in cond.graph.nodes
                if x.op == "call_function" and x.target in (operator.lt,)
            )
            end = test.args[1]
            if not isinstance(end, int):
                raise ValueError("Block loop bound must be static")
            n.meta["loop_extents"] = [(0, end, 1)]


def build_block_region(node, *, tiler):
    spec = node.meta["block_region"]
    args = tuple(n.value.clone() for n in node.all_input_nodes)
    semantic = tuple(args[i] for i in spec["operand_order"])
    cls = (
        MaterializedSoftmax
        if spec["kind"] == "materialized_softmax"
        else GatedProjection
    )
    inner = (
        cls(*(tuple(x.shape for x in semantic) + (spec,)))
        if cls is MaterializedSoftmax
        else cls(tuple(tuple(x.shape) for x in semantic), spec)
    )

    class Ordered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            return self.inner(*(args[i] for i in spec["operand_order"]))

    with _lenient_verifier():
        gm = export_model(Ordered(), args)
    gm = _finalize_exported_gm(gm)
    tag_loops(gm)
    with oracle_disabled():
        ShapeProp(gm, recurse=True).propagate(*args)
    _fuse_passes(gm)
    return gm
