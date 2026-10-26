"""Independent checks of Alpha's frozen projections and sealed payloads.

A buffered program intentionally has different placements and explicit copies
for split/concat views. Compare arithmetic fields and every payload exactly;
validate addresses against the new plan, rather than the old resident addresses.
This checker does not execute the Alpha C-model or physical-bank semantics.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def schedule_fields(path):
    return {
        key.strip(): value.strip()
        for line in path.read_text().splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def check_collateral(compiler, reference):
    compiler, reference = Path(compiler), Path(reference)
    directory, old = compiler / "alpha-cmod", reference / "alpha-cmod"
    report = json.loads((directory / "conversion_report.json").read_text())
    expected = json.loads((reference / "validation.json").read_text())
    bundle = json.loads((compiler / "bundle.json").read_text())
    plan = json.loads((compiler / "compilation_plan.json").read_text())
    bindings = json.loads((compiler / "buffer_bindings.json").read_text())
    invocations = json.loads((compiler / "target_invocations.json").read_text())
    program = json.loads((directory / "program.json").read_text())
    allocations = {b["buffer"]: b for b in plan["allocation"]["buffers"]}
    issues, differences = [], Counter()

    def check(condition, message):
        if not condition:
            issues.append(message)

    check(
        report["status"] == "complete" and not report["issues"],
        "conversion is incomplete",
    )
    check(
        report["emitted_layers"] == expected["emitted_layers"],
        "native layer count differs",
    )
    check(
        report["layer_kinds"] == expected["layer_kinds"],
        "native layer kinds differ",
    )
    check(
        (directory / "layers.txt").read_bytes()
        == (old / "layers.txt").read_bytes(),
        "native layer order differs",
    )
    check(
        program["allocation"] == plan["allocation"],
        "program allocation differs from canonical plan",
    )
    check(
        all(program[k] == bundle[k] for k in ("graph_digest", "plan_digest")),
        "program identity differs from bundle",
    )
    for name, sha in bundle["files"].items():
        path = compiler / name
        check(
            path.is_file() and digest(path) == sha, f"bundle checksum: {name}"
        )
    old_payloads = {p.relative_to(old) for p in old.glob("*/1_*.txt")}
    new_payloads = {
        p.relative_to(directory) for p in directory.glob("*/1_*.txt")
    }
    check(old_payloads == new_payloads, "payload file set differs")
    matched_payloads = 0
    for path in sorted(old_payloads & new_payloads):
        equal = digest(old / path) == digest(directory / path)
        check(equal, f"payload differs: {path}")
        matched_payloads += equal
    native = [s for s in program["steps"] if s["kind"] == "native"]
    host = [s for s in program["steps"] if s["kind"] == "host_output"]
    check(
        [s["layer"] for s in native] == [l["name"] for l in report["layers"]],
        "driver/native order differs",
    )
    check(
        len(host) == 6
        and {s["operation"] for s in host}
        == {"alpha_quant.dequantize.default", "aten.permute.default"},
        "unexpected host work",
    )
    check(
        all(
            s["kind"] in ("native", "host_output", "view", "buffer_operation")
            for s in program["steps"]
        ),
        "unaccounted executable operation",
    )
    check(
        sum(s["kind"] == "buffer_operation" for s in program["steps"])
        == report["buffer_operations"],
        "buffer operation count differs",
    )
    check(
        any(
            s["operation"] == "voyager.async_copy.default"
            for s in program["steps"]
        ),
        "missing explicit transfers",
    )
    check(
        all(
            report["claims"][k]
            for k in (
                "all_buffer_program_operations_accounted",
                "all_native_instructions_emitted",
                "connected_via_explicit_buffer_program",
            )
        ),
        "missing completeness claim",
    )

    # These fields describe old resident views. The new buffer graph materializes
    # those views with explicit copies; arithmetic dimensions/precision remain fixed.
    placement = {
        "is_from_split",
        "split_channel_idx",
        "split_cur_size",
        "split_idx",
        "split_total_size",
        "split_scale",
        "is_to_concat",
        "concat_channel_index_0",
        "concat_idx_0",
        "concat_size_0",
        "concat_total_size_0",
        "nmp_c2",
        "nmp_k2",
    }
    neutral = {
        "padding_c": "0",
        "padding_k": "0",
        "wt_binary_pt_loc": "0",
        "nonlinear_scale_shift": "0",
        "nonlinear_scale_factor": "1",
    }
    addresses_checked = 0
    for layer in report["layers"]:
        name, kind = layer["name"], layer["kind"]
        selected = invocations[layer["node"]]["target_schedule"]
        check(
            (selected["name"], selected["kind"]) == (name, kind),
            f"{name}: selection differs",
        )
        actual = schedule_fields(directory / name / "schedule0.txt")
        baseline = schedule_fields(old / name / "schedule0.txt")
        check(
            all(actual.get(k) == str(v) for k, v in selected["fields"].items()),
            f"{name}: emitter changed selected instruction fields",
        )
        check(
            actual == {k: str(v) for k, v in layer["schedule"].items()},
            f"{name}: schedule/report differs",
        )
        for index in (1, 10):
            check(
                schedule_fields(directory / name / f"schedule{index}.txt")
                == actual,
                f"{name}: schedule copies differ",
            )
        for key in actual.keys() | baseline.keys():
            a, b = actual.get(key), baseline.get(key)
            if a == b:
                continue
            differences[key] += 1
            allowed = key.endswith("_baseaddr") or key in placement
            allowed |= key in neutral and (a or neutral[key]) == (
                b or neutral[key]
            )
            # Resident group descriptors are layout metadata; DWC semantics are
            # already fixed by is_dwconv and the complete C/K dimensions.
            allowed |= (
                key in ("groups", "channels_per_group")
                and baseline.get("is_from_split") == "1"
            )
            check(
                allowed, f"{name}: arithmetic field {key} differs ({b} -> {a})"
            )
        ports = []
        for address in selected["addresses"]:
            buffer_name = bindings.get(address["buffer"], address["buffer"])
            buffer = allocations[buffer_name]
            memory, layout = buffer["memory_instance"], buffer["layout"]
            check(
                memory in ("global_activation", "global_weight"),
                f"{name}: unsupported memory",
            )
            if kind.startswith("nmp_"):
                check(
                    memory == "global_activation",
                    f"{name}: NMP weight-arena access",
                )
            cluster = layout["cluster"]
            # Alpha global banks have 1024 16-byte entries each.
            start = buffer["start"] // (cluster * 16) + address["entry_offset"]
            bank, entry = divmod(start, 1024)
            cells = {
                (b, e)
                for b, lo, hi in buffer["bank_ranges"]
                for e in range(lo, hi)
            }
            check(
                all(
                    (bank * cluster + lane, entry) in cells
                    for lane in range(cluster)
                ),
                f"{name}: address outside allocation",
            )
            encoded = 0x1000000 if memory == "global_activation" else 0x800000
            encoded |= (
                entry
                | (bank << 10)
                | ((address["bits"] == 8) << 19)
                | (cluster << 16)
            )
            check(
                int(actual[address["field"]], 16) == encoded,
                f"{name}: address differs from frozen placement",
            )
            ports.append((address["field"], buffer))
            addresses_checked += 1
        # Copy movements intentionally read/write one shuffle backing. All other
        # native ports require disjoint banks, even at distinct entry offsets.
        if kind != "nmp_copy":
            for i, (field, buffer) in enumerate(ports):
                for other_field, other in ports[i + 1 :]:
                    shared = {r[0] for r in buffer["bank_ranges"]} & {
                        r[0] for r in other["bank_ranges"]
                    }
                    check(
                        buffer["memory_instance"] != other["memory_instance"]
                        or not shared,
                        f"{name}: simultaneous ports {field}/{other_field} share banks",
                    )
    return dict(
        passed=not issues,
        issues=issues,
        emitted_layers=report["emitted_layers"],
        layer_kinds=report["layer_kinds"],
        payload_files=len(old_payloads),
        byte_identical_payloads=matched_payloads,
        addresses_checked=addresses_checked,
        schedule_difference_fields=dict(differences),
        host_output_nodes=len(host),
        buffer_operations=report["buffer_operations"],
        cmod_or_device_executed=False,
        scope="Exact payload/arithmetic parity and frozen buffer-program contracts; no physical execution",
    )
