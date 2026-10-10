"""Static compiled-ISA adapter to Voyager's shared Trainium timing model.

Input contains ONLY static instructions and DMA descriptors. Runtime timestamps,
wait times, durations, engine utilization, and application latency are forbidden.
"""

import argparse
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    OperationEvent,
    RepeatedGraph,
    evaluate_graph,
)
from voyager_compiler.trainium import isa
from voyager_compiler.trainium.native_timing import evaluate as native_timing
from voyager_compiler.trainium.dependencies import engine_clock
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.movement import TransferPanel, transfer_graph
from voyager_compiler.trainium.timing import StreamTransposeTiming

WAIT = re.compile(r"S\[(\d+)\]\s*\([^)]*\)\s*(>=|==)(\d+)")
SET = re.compile(r"S\[(\d+)\]\s*\([^)]*\)\+\+@complete")
TENSOR = re.compile(
    r"\b(src\d*|dst|pred)=(?:(\w+)@)?(0x[0-9a-f]+)\[([^]]+)\]\[([^]]+)\]"
)
ENGINE = {
    "Tensor": "TensorE",
    "Vector": "VectorE",
    "Scalar": "ScalarE",
    "GpSimd": "GpSimdE",
    "Sync": "SyncE",
}
DTYPE = {
    "fp32": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
    "uint8": "uint8",
}


def fields(text):
    return {
        k: (
            dt,
            int(addr, 16),
            tuple(map(int, st.split(","))),
            tuple(map(int, sh.split(","))),
        )
        for k, dt, addr, st, sh in TENSOR.findall(text)
    }


def number(text, key):
    match = re.search(r"\b" + key + r"=(-?0x[0-9a-f]+|-?\d+)", text)
    if not match:
        raise ValueError(("missing field", key, text))
    return int(match[1], 0)


def effective_reduction_rank(operands, source):
    """Singleton native reduction axes do not create another reduction stage."""
    match = re.search(r"\bdim=([XYZW]+)", operands)
    if not match:
        return 1
    return (
        max(1, sum(source[3]["XYZW".index(axis)] > 1 for axis in match[1]))
        if source
        else len(match[1])
    )


def predict(
    data,
    *,
    hardware=None,
    in_order=True,
    context_model=False,
    descriptor_observer=None,
):
    # This is an allowlist, not a blacklist: new profile fields cannot leak in.
    if not set(data) <= {"instructions", "static_dma", "dma_audit"}:
        raise ValueError("Only static compiled metadata is accepted")
    allowed = {
        "opcode",
        "subgroup",
        "compiler_pc",
        "instruction_type",
        "operands",
        "compiler_opcode",
        "compiler_operands",
        "raw_bir_id",
        "bir_instruction_name",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "sbuf_read_bytes",
        "sbuf_write_bytes",
    }
    for ins in data["instructions"]:
        if not set(ins) <= allowed:
            raise ValueError("Only static instruction fields are accepted")
    dma_allowed = {
        "block_id",
        "dest",
        "dest_num_sb_partitions",
        "dest_offset",
        "dest_sb_partitions",
        "dest_steps",
        "dma_queue",
        "queue_type",
        "read_shape",
        "read_size",
        "semaphore_id",
        "source",
        "source_num_sb_partitions",
        "source_offset",
        "source_sb_partitions",
        "source_steps",
        "subgroup",
        "variable",
        "write_shape",
        "write_size",
    }
    for descriptor in data.get("static_dma", []):
        if not set(descriptor) <= dma_allowed:
            raise ValueError("Only static DMA fields are accepted")
    if not set(data.get("dma_audit", {})) <= {
        "aggregate_count",
        "read_bytes",
        "write_bytes",
    }:
        raise ValueError("Only static DMA audit fields are accepted")
    hw = hardware if hardware is not None else neuron_core(3)
    if hw.name != "trainium-v3":
        raise ValueError(
            "Compiled ISA adapter is pinned to Trainium2 / NeuronCore-v3"
        )
    # Native CAST and long dtype spellings are the same physical copy forms
    # already consumed by selected-ISA analysis. Keep this in the compiler,
    # rather than requiring an experiment-specific normalization wrapper.
    cast_count = 0
    normalized = []
    for ins in data["instructions"]:
        ins = dict(ins)
        if ins["opcode"] == "CAST":
            ins["opcode"] = "COPY"
            cast_count += 1
        for field in ("operands", "compiler_operands"):
            if field in ins:
                for long, short in (
                    ("bfloat16@", "bf16@"),
                    ("float16@", "fp16@"),
                    ("float32@", "fp32@"),
                ):
                    ins[field] = ins[field].replace(long, short)
        normalized.append(ins)
    data = dict(data, instructions=normalized)
    timing_descriptors = {}
    stream_context = StreamTransposeTiming()
    selected_law_counts = Counter()
    raw = []
    seen = {}
    deduplicated = 0
    for ins in data["instructions"]:
        key = (ins["subgroup"], ins["compiler_pc"])
        if key in seen:
            # A static DMA trigger has a late runtime bookkeeping write at the
            # same compiler PC. Retain the unique record with transfer bytes.
            old = seen[key]
            assert ins["opcode"] == old["opcode"] == "WRITE"
            assert ins["raw_bir_id"] == old["raw_bir_id"]
            if "hbm_read_bytes" in ins or "hbm_write_bytes" in ins:
                seen[key] = ins
            deduplicated += 1
        else:
            seen[key] = ins
    raw = sorted(seen.values(), key=lambda i: (i["subgroup"], i["compiler_pc"]))
    # Collapse exactly one NKI tensor instruction's backend FP32 expansion.
    groups = defaultdict(list)
    for i in raw:
        key = (
            ("tensor", i["raw_bir_id"])
            if i["opcode"] in ("LDWEIGHTS", "MATMUL")
            else (i["subgroup"], i["compiler_pc"])
        )
        groups[key].append(i)
    units = []
    for g in groups.values():
        g.sort(key=lambda i: i["compiler_pc"])
        units.append(
            {
                "items": g,
                "engine": g[0]["subgroup"],
                "pc": g[0]["compiler_pc"],
                "last_pc": g[-1]["compiler_pc"],
            }
        )
    units.sort(key=lambda u: (u["engine"], u["pc"]))
    nodes = []
    deps = []
    pending = []
    setters = defaultdict(list)
    last = {}
    missing = Counter()
    mapped = Counter()
    group_audit = Counter()
    bytes_read = bytes_written = 0
    fragmented_dma = 0

    def add(
        name,
        resource,
        issue=0,
        occupancy=0,
        latency=0,
        implementation="",
        edges=(),
        *,
        forward=None,
        read=None,
    ):
        idx = len(nodes)
        nodes.append(
            OperationEvent(
                name,
                resource,
                issue,
                occupancy,
                latency,
                implementation=implementation,
                forward_ns=forward,
                read_ns=read,
            )
        )
        deps.append(set(edges))
        return idx

    last_tensor_matmul = False
    last_tensor_loads = []
    retained_stationary = 0
    native_half_transposes = 0
    derived_operand_models = Counter()
    covered_setup = Counter()
    accumulator_writer = None
    accumulator_reader = None
    accumulator_edges = 0
    mask_writer = None
    mask_readers = []
    register_writers = {}
    register_readers = defaultdict(list)
    hidden_state_edges = 0
    for uid, u in enumerate(units):
        items = u["items"]
        i = items[-1]
        op = i["opcode"]
        eng = ENGINE[u["engine"]]
        text = " ".join(v["operands"] for v in items)
        waits = [(int(s), cmp, int(v)) for s, cmp, v in WAIT.findall(text)]
        semsets = [(int(s), 1) for s in SET.findall(text)]
        first = end = None
        issue_node = None
        native_cost = None
        if (
            op == "DMA_DIRECT2D"
            or i.get("compiler_opcode") == "PSEUDO_DMA_TRIGGER"
        ):
            if op == "DMA_DIRECT2D":
                meta = i.get("compiler_operands", i["operands"])
                sem = number(meta, "semaphore")
                increment = number(meta, "sem_increment")
                read = i.get("hbm_read_bytes", 0)
                write = i.get("hbm_write_bytes", 0)
                assert bool(read) != bool(write), (i, read, write)
                store = bool(write)
                nbytes = write or read
                side = "src" if store else "dst"
                pattern = re.search(
                    side + r"_pattern=\[([^]]+)\]\[([^]]+)\]", meta
                )
                strides = list(map(int, pattern[1].split(",")))
                counts = list(map(int, pattern[2].split(",")))
                assert 262144 in strides, (i, strides)
                partitions = math.prod(
                    n for s, n in zip(strides, counts) if s == 262144
                )
                assert 1 <= partitions <= 128
                # The byte count is compiler metadata, audited independently.
                assert nbytes % partitions == 0
            else:
                meta = i["compiler_operands"]
                block = number(meta, "block_id")
                queue = re.search(r"(q\w+)\s+block_id=", meta)[1]
                ds = [
                    d
                    for d in data["static_dma"]
                    if d["block_id"] == block and d["subgroup"] == queue
                ]
                assert len(ds) == 1, (queue, block, len(ds))
                d = ds[0]
                sem = int(re.search(r"S\[(\d+)\]", d["semaphore_id"])[1])
                increment = 16
                # Static DMA descriptors notify the 16-engine semaphore group.
                # Counter totals are checked independently against the trace;
                # measured notification times are never prediction inputs.
                read = i.get("hbm_read_bytes", 0)
                write = i.get("hbm_write_bytes", 0)
                assert bool(read) != bool(write)
                store = bool(write)
                nbytes = write or read
                partitions = (
                    d["source_num_sb_partitions"]
                    if store
                    else d["dest_num_sb_partitions"]
                )
                assert 1 <= partitions <= 128
            bytes_read += read
            bytes_written += write
            # Unit bytes avoid inventing a dtype for byte-oriented DMA geometry.
            ideal = (
                min(8, partitions)
                * (nbytes / partitions)
                / (hw.dram_bandwidth / 16)
            )
            panel_partitions, panel_free = partitions, nbytes // partitions
            if nbytes % partitions:
                from .native_dma import partition_bytes, payload_ns

                loads = partition_bytes(d, store, nbytes)
                ideal = payload_ns(loads, hw.dram_bandwidth)
                fragmented_dma += 1
                # This timing-only panel represents the packed byte stream;
                # physical per-partition loads were decoded above. It remains
                # one native request, not one request per descriptor fragment.
                panel_partitions, panel_free = 1, nbytes
            panel = TransferPanel(
                0,
                0,
                panel_partitions,
                panel_free,
                8,
                store,
                False,
                False,
                ideal,
            )
            graph = transfer_graph(hw, (panel,))
            offset = len(nodes)
            for n in graph.nodes:
                add(
                    f"u{uid}_{n.name}",
                    n.resource,
                    n.issue_ns,
                    n.occupancy_ns,
                    n.latency_ns,
                    n.implementation,
                    [(d.source + offset, d.milestone) for d in n.dependencies],
                )
            first = offset
            end = len(nodes) - 1
            issue_node = first
            setters[sem].append((u["engine"], u["pc"], end, increment))
            mapped["DMA"] += 1
        elif op in ("MATMUL", "LDWEIGHTS"):
            mm = [v for v in items if v["opcode"] == "MATMUL"]
            ld = [v for v in items if v["opcode"] == "LDWEIGHTS"]
            if not ld and len(mm) == 1:
                # The native compiler elides a redundant stationary load. The
                # most recent TensorE load still owns the stationary payload;
                # its waits/setters must not be duplicated for this matmul.
                if len(last_tensor_loads) != 1:
                    raise ValueError(
                        (
                            "matmul has no unambiguous retained stationary operand",
                            uid,
                        )
                    )
                ld = last_tensor_loads
                retained_stationary += 1
            assert len(mm) == len(ld) and len(mm) in (1, 2), (uid, items)
            last_tensor_loads = ld
            assert u["last_pc"] - u["pc"] + 1 == len(items), (
                uid,
                "noncontiguous tensor group",
            )
            shapes = fields(mm[0]["operands"])
            m = math.prod(shapes["src"][3])
            pair = re.search(r"(\d+)\*(\d+)\s*$", mm[0]["operands"])
            k, n = map(int, pair.groups())
            dt = DTYPE[shapes["src"][0]]
            trans = mm[0]["instruction_type"] == "TRANSPOSE"
            native_half_transpose = False
            if trans:
                assert len(mm) == 1
                if m > 128:
                    # A TRANSPOSE compiler tag can denote the wide identity
                    # matmul used by the native backend, not a legal NKI PF
                    # transpose tile. One stationary FP32 half and two moving
                    # halves execute two BF16-rate passes. Do not invent extra
                    # loads or authorize this form in the NKI lowering.
                    if not (
                        dt == "float32"
                        and k == n == 128
                        and m <= 512
                        and "fp32_mode=LOW " in ld[0]["operands"]
                        and "fp32_mode=LOW_HIGH " in mm[0]["operands"]
                    ):
                        raise ValueError(
                            "Uncharacterized wide native identity transpose"
                        )
                    expansion = isa.matmul(m, n, k, 16, hw, "bfloat16")
                    native_half_transpose = True
                    native_half_transposes += 1
                else:
                    expansion = isa.transpose(m, n, hw, "ScalarE", dt)
            else:
                assert len(mm) == (2 if dt == "float32" else 1)
                expansion = isa.matmul(
                    m,
                    n,
                    k,
                    32 if dt == "float32" else 16,
                    hw,
                    dt,
                    moving_stride=(
                        abs(shapes["src"][2][0])
                        if shapes["src"][3][1:] == (1, 1)
                        else None
                    ),
                    stationary_stride=(
                        abs(fields(ld[0]["operands"])["src"][2][0])
                        if fields(ld[0]["operands"])["src"][3][1:] == (1, 1)
                        else None
                    ),
                    streaming=last_tensor_matmul,
                )
            was_tensor_matmul = last_tensor_matmul
            last_tensor_matmul = not trans
            service = (
                expansion.tensor_cycles
                / hw.frequency
                * (2 if native_half_transpose else 1)
            )
            law_name = (
                expansion.timing_implementation or expansion.implementation
            )
            law = hw.timing_profile.operation(law_name)
            selected_law_counts[law_name] += 1
            occ, lat = law.evaluate(service) if law else (service, None)
            if expansion.timing_override is not None:
                occ, lat, _ = expansion.timing_override
            if context_model and not native_half_transpose:
                from .calibrated_isa import evaluate

                if op in ("MATMUL", "LDWEIGHTS"):
                    descriptor = dict(
                        opcode="nc_transpose" if trans else "nc_matmul",
                        engine=eng,
                        dtype=dt,
                        partitions=n,
                        free=m,
                        source_free=m,
                        source_memory="SBUF",
                        transpose=trans,
                        source_stride=(
                            abs(fields(ld[0]["operands"])["src"][2][0])
                            if trans
                            else abs(shapes["src"][2][0])
                        ),
                        destination_stride=1,
                        function="",
                    )
                    if not trans:
                        descriptor.update(
                            moving=m,
                            stationary=n,
                            contraction=k,
                            moving_stride=abs(shapes["src"][2][0]),
                            stationary_stride=abs(
                                fields(ld[0]["operands"])["src"][2][0]
                            ),
                            streaming=was_tensor_matmul,
                        )
                else:
                    src = fs.get("src") or fs.get("src0")
                    fun = re.search(
                        r"\b(EXP|SIGMOID|RSQRT|RECIPROCAL_SQRT|SQRT|SQUARE|LN|SILU|COPY|IDENTITY)\b",
                        i["operands"],
                    )
                    reduction = re.search(
                        r"\bop=(ADD|MAX|MIN)\b", i["operands"]
                    )
                    binary = re.search(
                        r"\bop=(ADD|SUBTRACT|MULTIPLY)\b", i["operands"]
                    )
                    pool = (
                        re.search(r"\b(AVERAGE|MAX|MIN)\b", i["operands"])
                        if op == "POOL"
                        else None
                    )
                    descriptor = dict(
                        opcode={
                            "COPY": "tensor_copy",
                            "ACTIVATE": "activation",
                            "RECIPROCAL": "reciprocal",
                            "TENSOR_REDUCE": "tensor_reduce",
                            "TENSOR_TENSOR": "tensor_tensor",
                            "POOL": "tensor_reduce",
                            "MEMSET": "memset",
                        }.get(op, op),
                        engine=eng,
                        dtype=dt,
                        partitions=(
                            number(i["operands"], "channels")
                            if "channels=" in i["operands"]
                            else 0
                        ),
                        free=free,
                        source_free=math.prod(src[3]) if src else 0,
                        source_memory=(
                            ("PSUM" if src[1] >= 0x2000000 else "SBUF")
                            if src
                            else None
                        ),
                        source_dtype=DTYPE.get(src[0], src[0]) if src else None,
                        source_stride=(
                            abs(src[2][0])
                            if src and all(x == 1 for x in src[3][1:])
                            else None
                        ),
                        destination_stride=(
                            abs(dst[2][0])
                            if all(x == 1 for x in dst[3][1:])
                            else None
                        ),
                        destination_memory=(
                            "PSUM" if dst[1] >= 0x2000000 else "SBUF"
                        ),
                        source_shape=src[3] if src else (),
                        source_strides=src[2] if src else (),
                        reduction_rank=effective_reduction_rank(
                            i["operands"], src
                        ),
                        destination_shape=dst[3],
                        destination_strides=dst[2],
                        broadcast_reads=sum(
                            bool(re.search(pattern, i["operands"]))
                            for pattern in (
                                r"\bscale=\[",
                                r"\b(?:bias_ptr=|imm=\[)",
                            )
                        ),
                        transpose=False,
                        function=(
                            fun[1].lower()
                            if fun
                            else (
                                reduction[1].lower()
                                if reduction
                                else (
                                    binary[1].lower()
                                    if binary
                                    else (
                                        "add"
                                        if pool and pool[1] == "AVERAGE"
                                        else pool[1].lower() if pool else ""
                                    )
                                )
                            )
                        ),
                    )
                if op == "TENSOR_TENSOR":
                    other = fs["src1"]
                    descriptor.update(
                        source1_dtype=DTYPE.get(other[0], other[0]),
                        source1_memory=(
                            "PSUM" if other[1] >= 0x2000000 else "SBUF"
                        ),
                        source1_shape=other[3],
                        source1_strides=other[2],
                    )
                timing_descriptors[f"u{uid}_{op}"] = descriptor
                override = evaluate(descriptor, occ, lat, None)
                if override is not None:
                    occ, lat, _ = override
                    if missing.get(law_name, 0):
                        missing[law_name] -= 1
                    selected_law_counts["context:" + descriptor["opcode"]] += 1
                    from .operand_timing import key, parameters

                    operand_model = parameters()["models"].get(
                        key(descriptor), {}
                    )
                    if operand_model.get("derived_from"):
                        derived_operand_models[key(descriptor)] += 1
            first = end = issue_node = add(
                f"u{uid}_{op}", eng, occ, occ, lat, law_name
            )
            mapped[expansion.implementation] += 1
            group_audit[
                f'{dt}:{"transpose" if trans else "matmul"}:{len(mm)}'
            ] += 1
        elif op == "MOVE":
            native_cost = native_timing(op, eng, {}, i["operands"], hw)
            first = end = issue_node = add(
                f"u{uid}_{op}",
                eng,
                native_cost.issue_ns,
                native_cost.issue_ns,
                native_cost.completion_ns,
                native_cost.implementation,
            )
            mapped[native_cost.implementation] += 1
        elif op == "LOAD_MASK_SELECT":
            native_cost = native_timing(op, eng, {}, i["operands"], hw)
            first = end = issue_node = add(
                f"u{uid}_{op}",
                eng,
                native_cost.issue_ns,
                native_cost.issue_ns,
                native_cost.completion_ns,
                native_cost.implementation,
            )
            mapped[native_cost.implementation] += 1
        elif op in ("EVENT_SEMAPHORE", "NOP", "ACT_TABLE_LOAD", "WRITE"):
            # No new per-kernel coefficient: existing fixed_kernel_ns covers
            # common setup. Semaphore waits remain graph edges.
            if op == "WRITE":
                raise ValueError(("unhandled write", i))
            first = end = issue_node = add(f"u{uid}_{op}", f"Control.{eng}")
            if op == "ACT_TABLE_LOAD":
                # One common table load is already included in the fixed
                # startup budget. Keep that accounting explicit, rather than
                # tagging it as a missing result-completion law.
                covered_setup["ACT_TABLE_LOAD"] += 1
        else:
            if op not in (
                "COPY",
                "TENSOR_TENSOR",
                "TENSOR_SCALAR",
                "TENSOR_REDUCE",
                "POOL",
                "RECIPROCAL",
                "ACTIVATE",
                "MEMSET",
                "STREAM_TRANSPOSE",
                "TENSOR_TENSOR_SCAN",
                "ACTIVATION_READ_ACCUMULATOR",
                "TENSOR_SCALAR_AFFINE_SELECT",
                "COPY_PREDICATED_SCALAR",
                "STREAM_SHUFFLE",
            ):
                raise ValueError(("unsupported compiled opcode", op, eng))
            fs = fields(i["operands"])
            dst = fs["dst"]
            dt = DTYPE.get(dst[0], dst[0])
            free = math.prod(dst[3])
            clock = engine_clock(hw, eng)
            service = max(64, free) / clock
            native_cost = native_timing(op, eng, fs, i["operands"], hw)
            if native_cost is not None:
                law_name = native_cost.implementation
                service = native_cost.issue_ns
            elif op == "COPY":
                mem = "PSUM" if fs["src"][1] >= 0x2000000 else "SBUF"
                law_name = f"nki.copy.{mem}.{dt}.{eng}"
            elif op == "TENSOR_TENSOR":
                law_name = f"nki.binary.{dt}.{eng}"
                service = max(64, 2 * free) / clock
            elif op == "TENSOR_SCALAR":
                law_name = f"nki.scalar.{dt}.{eng}"
            elif op in ("TENSOR_REDUCE", "POOL"):
                law_name = f"nki.reduce.{dt}.{eng}"
                service = max(64, math.prod(fs["src"][3])) / clock
            elif op == "RECIPROCAL":
                law_name = f"nki.reciprocal.{dt}.{eng}"
                service = max(64, 8 * free) / clock
            elif op == "ACTIVATE":
                law_name = f"nki.activation.{dt}.{eng}"
            elif op == "MEMSET":
                law_name = f"nki.memset.{dt}.{eng}"
            elif op == "STREAM_TRANSPOSE":
                law_name = stream_context.select(
                    dtype=dt,
                    source_dtype=DTYPE.get(fs["src"][0], fs["src"][0]),
                    engine=eng,
                    partitions=number(i["operands"], "channels"),
                    source_free=math.prod(fs["src"][3]),
                    destination_free=free,
                    source_address=fs["src"][1],
                    destination_address=dst[1],
                    strides=fs["src"][2] + dst[2],
                )
                if law_name is None:
                    law_name = f"UNSUPPORTED.STREAM_TRANSPOSE.{dt}.{eng}"
                    service = 0
                    missing[law_name] += 1
                else:
                    selected_law_counts[law_name] += 1
            else:
                raise ValueError(("unhandled opcode", i))
            if eng == "VectorE" and op != "STREAM_TRANSPOSE":
                stream_context.reset()
            law = hw.timing_profile.operation(law_name)
            occ, lat = law.evaluate(service) if law else (service, None)
            if native_cost is not None:
                occ, lat = native_cost.issue_ns, native_cost.completion_ns
                selected_law_counts[law_name] += 1
            if (
                (native_cost is None and law is None)
                or (native_cost is not None and lat is None)
            ) and not law_name.startswith("UNSUPPORTED"):
                missing[law_name] += 1
            if context_model and native_cost is None:
                from .calibrated_isa import evaluate

                if op in ("MATMUL", "LDWEIGHTS"):
                    descriptor = dict(
                        opcode="nc_transpose" if trans else "nc_matmul",
                        engine=eng,
                        dtype=dt,
                        partitions=n,
                        free=m,
                        source_free=m,
                        source_memory="SBUF",
                        transpose=trans,
                        source_stride=(
                            abs(fields(ld[0]["operands"])["src"][2][0])
                            if trans
                            else abs(shapes["src"][2][0])
                        ),
                        destination_stride=1,
                        function="",
                    )
                    if not trans:
                        descriptor.update(
                            moving=m,
                            stationary=n,
                            contraction=k,
                            moving_stride=abs(shapes["src"][2][0]),
                            stationary_stride=abs(
                                fields(ld[0]["operands"])["src"][2][0]
                            ),
                            streaming=was_tensor_matmul,
                        )
                else:
                    src = fs.get("src") or fs.get("src0")
                    fun = re.search(
                        r"\b(EXP|SIGMOID|RSQRT|RECIPROCAL_SQRT|SQRT|SQUARE|LN|SILU|COPY|IDENTITY)\b",
                        i["operands"],
                    )
                    reduction = re.search(
                        r"\bop=(ADD|MAX|MIN)\b", i["operands"]
                    )
                    binary = re.search(
                        r"\bop=(ADD|SUBTRACT|MULTIPLY)\b", i["operands"]
                    )
                    pool = (
                        re.search(r"\b(AVERAGE|MAX|MIN)\b", i["operands"])
                        if op == "POOL"
                        else None
                    )
                    descriptor = dict(
                        opcode={
                            "COPY": "tensor_copy",
                            "ACTIVATE": "activation",
                            "RECIPROCAL": "reciprocal",
                            "TENSOR_REDUCE": "tensor_reduce",
                            "TENSOR_TENSOR": "tensor_tensor",
                            "POOL": "tensor_reduce",
                            "MEMSET": "memset",
                        }.get(op, op),
                        engine=eng,
                        dtype=dt,
                        partitions=(
                            number(i["operands"], "channels")
                            if "channels=" in i["operands"]
                            else 0
                        ),
                        free=free,
                        source_free=math.prod(src[3]) if src else 0,
                        source_memory=(
                            ("PSUM" if src[1] >= 0x2000000 else "SBUF")
                            if src
                            else None
                        ),
                        source_dtype=DTYPE.get(src[0], src[0]) if src else None,
                        source_stride=(
                            abs(src[2][0])
                            if src and all(x == 1 for x in src[3][1:])
                            else None
                        ),
                        destination_stride=(
                            abs(dst[2][0])
                            if all(x == 1 for x in dst[3][1:])
                            else None
                        ),
                        destination_memory=(
                            "PSUM" if dst[1] >= 0x2000000 else "SBUF"
                        ),
                        source_shape=src[3] if src else (),
                        source_strides=src[2] if src else (),
                        reduction_rank=effective_reduction_rank(
                            i["operands"], src
                        ),
                        destination_shape=dst[3],
                        destination_strides=dst[2],
                        broadcast_reads=sum(
                            bool(re.search(pattern, i["operands"]))
                            for pattern in (
                                r"\bscale=\[",
                                r"\b(?:bias_ptr=|imm=\[)",
                            )
                        ),
                        transpose=False,
                        function=(
                            fun[1].lower()
                            if fun
                            else (
                                reduction[1].lower()
                                if reduction
                                else (
                                    binary[1].lower()
                                    if binary
                                    else (
                                        "add"
                                        if pool and pool[1] == "AVERAGE"
                                        else pool[1].lower() if pool else ""
                                    )
                                )
                            )
                        ),
                    )
                if op == "TENSOR_TENSOR":
                    other = fs["src1"]
                    descriptor.update(
                        source1_dtype=DTYPE.get(other[0], other[0]),
                        source1_memory=(
                            "PSUM" if other[1] >= 0x2000000 else "SBUF"
                        ),
                        source1_shape=other[3],
                        source1_strides=other[2],
                    )
                timing_descriptors[f"u{uid}_{op}"] = descriptor
                override = evaluate(descriptor, occ, lat, None)
                if override is not None:
                    occ, lat, _ = override
                    if missing.get(law_name, 0):
                        missing[law_name] -= 1
                    selected_law_counts["context:" + descriptor["opcode"]] += 1
                    from .operand_timing import key, parameters

                    operand_model = parameters()["models"].get(
                        key(descriptor), {}
                    )
                    if operand_model.get("derived_from"):
                        derived_operand_models[key(descriptor)] += 1
            first = end = issue_node = add(
                f"u{uid}_{op}",
                eng,
                occ,
                occ,
                lat,
                law_name,
                forward=native_cost.forward_ns if native_cost else None,
                read=native_cost.read_ns if native_cost else None,
            )
            mapped[law_name] += 1
        assert first is not None
        if op == "LOAD_MASK_SELECT":
            if mask_writer is not None:
                deps[first].add((mask_writer, "result"))
                hidden_state_edges += 1
            for reader in mask_readers:
                deps[first].add((reader, "read"))
                hidden_state_edges += 1
            mask_readers = []
            mask_writer = end
        elif op == "STREAM_SHUFFLE":
            if mask_writer is None:
                raise ValueError("Stream shuffle has no mask producer")
            deps[first].add((mask_writer, "result"))
            hidden_state_edges += 1
            mask_readers.append(end)
        write_reg = (
            re.search(r"\$R\[(\d+)\]=", i["operands"]) if op == "MOVE" else None
        )
        for reg in set(map(int, re.findall(r"\$R\[(\d+)\]", i["operands"]))):
            if write_reg and reg == int(write_reg[1]):
                if reg in register_writers:
                    deps[first].add((register_writers[reg], "result"))
                    hidden_state_edges += 1
                for reader in register_readers[reg]:
                    deps[first].add((reader, "read"))
                    hidden_state_edges += 1
                register_readers[reg] = []
                register_writers[reg] = end
            else:
                if reg not in register_writers:
                    raise ValueError("Register read has no modeled producer")
                deps[first].add((register_writers[reg], "result"))
                hidden_state_edges += 1
                register_readers[reg].append(end)
        # These registers are not SBUF/PSUM tensors. Their dependencies remain
        # necessary even when engine issue-order preservation is disabled.
        if eng == "ScalarE" and op == "ACTIVATION_READ_ACCUMULATOR":
            if accumulator_writer is None:
                raise ValueError("Accumulator read has no recognized producer")
            deps[first].add((accumulator_writer, "forward"))
            accumulator_reader = end
            accumulator_edges += 1
        elif eng == "ScalarE" and op == "ACTIVATE":
            command = re.search(r"\baccumulator_cmd=(\w+)", i["operands"])
            if accumulator_reader is not None:
                deps[first].add((accumulator_reader, "read"))
                accumulator_edges += 1
                accumulator_reader = None
            if command and command[1] in ("ZERO_ACCUMULATE", "ACCUMULATE"):
                if command[1] == "ACCUMULATE":
                    if accumulator_writer is None:
                        raise ValueError(
                            "Accumulator continuation has no producer"
                        )
                    deps[first].add((accumulator_writer, "forward"))
                    accumulator_edges += 1
                accumulator_writer = end
            else:
                accumulator_writer = None
        pending.append((first, waits))
        # Preserve each compiled engine's issue order without requiring previous
        # completion. Readiness is carried by the encoded hardware semaphores.
        if u["engine"] in last and in_order:
            deps[first].add((last[u["engine"]], "issue"))
        last[u["engine"]] = issue_node
        for sem, inc in semsets:
            setters[sem].append((u["engine"], u["last_pc"], end, inc))

    # Build prefix completion nodes, so >=N waits for all N setter completions,
    # including independent DMA requests; do not use observed completion order.
    prefixes = {}
    for sem, records in setters.items():
        assert len({r[0] for r in records}) == 1, (
            "multiple semaphore producers",
            sem,
            records[:5],
        )
        records.sort(key=lambda r: r[1])
        count = 0
        prev = None
        entries = []
        for engine, pc, node, inc in records:
            if inc is None:
                assert len(records) == 1
                # Static descriptor transfer has a unique producer. The exact
                # packet counter is irrelevant: all waits bind its completion.
                count = math.inf
            else:
                count += inc
            prev = add(
                f"sem{sem}_{pc}",
                f"Semaphore.{sem}",
                edges=[(node, "result")]
                + ([] if prev is None else [(prev, "result")]),
            )
            entries.append((count, prev))
        prefixes[sem] = entries
    wait_count = 0
    for node, waits in pending:
        for sem, cmp, target in waits:
            assert cmp == ">=", (
                "nonmonotonic semaphore wait",
                sem,
                cmp,
                target,
            )
            if target == 0:
                continue
            entries = prefixes.get(sem, [])
            producer = next(
                (n for count, n in entries if count >= target), None
            )
            if producer is None:
                raise ValueError(
                    (
                        "unresolved semaphore",
                        sem,
                        target,
                        entries[-1:] if entries else [],
                        nodes[node].name,
                    )
                )
            deps[node].add((producer, "result"))
            wait_count += 1
    # Stable topological order uses only encoded engine order/PC and semaphores.
    indegree = [len(d) for d in deps]
    succ = [[] for _ in nodes]
    for dst, edges in enumerate(deps):
        for src, m in edges:
            succ[src].append(dst)
    ready = [i for i, n in enumerate(indegree) if not n]
    heapq.heapify(ready)
    order = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for dst in succ[n]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, dst)
    if len(order) != len(nodes):
        raise ValueError(
            (
                "cycle",
                [
                    (nodes[i].name, [(nodes[s].name, m) for s, m in deps[i]])
                    for i, n in enumerate(indegree)
                    if n
                ][:8],
            )
        )
    remap = {old: new for new, old in enumerate(order)}
    graph = RepeatedGraph(
        tuple(
            replace(
                nodes[old],
                dependencies=tuple(
                    Dependency(remap[s], milestone=m)
                    for s, m in sorted(deps[old])
                ),
            )
            for old in order
        )
    )
    if descriptor_observer is not None:
        descriptor_observer(timing_descriptors)
    result = evaluate_graph(graph)
    unsupported = sum(
        v for k, v in missing.items() if k.startswith("UNSUPPORTED")
    )
    return {
        "prediction_us": (
            None
            if unsupported
            else (result.duration_ns + hw.timing_profile.fixed_kernel_ns) / 1000
        ),
        "partial_modeled_us": (
            (result.duration_ns + hw.timing_profile.fixed_kernel_ns) / 1000
            if unsupported
            else None
        ),
        "status": (
            "partial_unsupported_opcodes"
            if unsupported
            else (
                (
                    "modeled_with_analytical_domains"
                    if native_half_transposes or derived_operand_models
                    else "modeled"
                )
                if not result.unknown_latency
                else "mapped_with_incomplete_completion_laws"
            )
        ),
        "timing_profile": hw.timing_profile.name,
        "missing_laws": dict(missing),
        "unsupported_instructions": unsupported,
        "service_us": {k: v / 1000 for k, v in result.service_ns},
        "hbm_read_bytes": bytes_read,
        "hbm_write_bytes": bytes_written,
        "static_instructions": len(raw),
        "native_cast_alias_count": cast_count,
        "fragmented_dma_descriptors": fragmented_dma,
        "retained_stationary_matmuls": retained_stationary,
        "native_half_transpose_groups": native_half_transposes,
        "analytical_assumptions": (
            [
                "Wide FP32 LOW/LOW_HIGH identity matmul uses two BF16-rate passes; not independently characterized."
            ]
            if native_half_transposes
            else []
        ),
        "derived_operand_models": dict(derived_operand_models),
        "covered_setup_instructions": dict(covered_setup),
        "accumulator_state_dependencies": accumulator_edges,
        "hidden_mask_register_dependencies": hidden_state_edges,
        "deduplicated_runtime_writes": deduplicated,
        "tensor_groups": dict(group_audit),
        "mapped_operations": dict(mapped),
        "graph_events": len(nodes),
        "encoded_waits": wait_count,
        "unresolved_waits": 0,
        "unknown_completion_events": len(result.unknown_latency),
        "selected_timing_laws": dict(selected_law_counts),
        "preserve_engine_issue_order": in_order,
    }, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Timing-free compiled_static.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-model",
        action="store_true",
        help="Enable operand-aware characterization",
    )
    parser.add_argument(
        "--execution-model",
        choices=("baseline", "context-ready"),
        default="baseline",
        help="Apply static ordering/readiness policy after native ISA mapping",
    )
    args = parser.parse_args()
    import gzip

    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt") as stream:
        result, graph = predict(
            json.load(stream), context_model=args.context_model
        )
    if args.execution_model != "baseline":
        from .physical_context import transform

        graph, audit = transform(graph, args.execution_model)
        estimate = evaluate_graph(graph)
        result[
            (
                "partial_modeled_us"
                if result["unsupported_instructions"]
                else "prediction_us"
            )
        ] = (
            estimate.duration_ns + neuron_core(3).timing_profile.fixed_kernel_ns
        ) / 1000
        result["execution_model"] = args.execution_model
        result["ordering_audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
