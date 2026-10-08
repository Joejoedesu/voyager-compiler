"""Maintain a current findings panel and evidence-linked modeling history."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT.parent / "trainium-kernel-analysis-2026-10-07.html"
OUT = ROOT / "results/trainium/matmul-geometry-2026-10-08/modeling-history.json"


def main():
    entries = []

    def add(
        key, category, status, title, attempt, result, limit, evidence, anchor
    ):
        paths = [ROOT / p for p in evidence]
        assert all(p.exists() for p in paths), paths
        entries.append(
            dict(
                id=key,
                category=category,
                status=status,
                title=title,
                attempt=attempt,
                result=result,
                limit=limit,
                anchor=anchor,
                evidence=[
                    dict(
                        label=str(p.relative_to(ROOT)),
                        href="voyager-trainium-10-06-isa/"
                        + str(p.relative_to(ROOT)),
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                    for p in paths
                ],
            )
        )

    add(
        "analytical",
        "Timing model",
        "Retained with limits",
        "Analytical compute and bandwidth service",
        "Expand each logical operation into its physical ISA recipe, charge Tensor/Vector/Scalar engine service from shape and dtype, and price memory payload from bytes and bandwidth. Keep FP32 low/high backend expansion separate from logical calls so it is not multiplied twice.",
        "Provides consistent work, command and traffic accounting and useful resource lower bounds. Still supplies the fallback for uncharacterized matmul geometry.",
        "Equal FLOPs or equal bytes do not imply equal runtime. It misses issue, completion, stride, dependencies and finite backend admission. Engine-service totals are not profiler active times.",
        [
            "src/voyager_compiler/trainium/isa.py",
            "src/voyager_compiler/trainium/hardware.py",
        ],
        "hardware",
    )
    add(
        "dma",
        "Timing model",
        "Retained with limits",
        "DMA descriptors, dispatch, payload and notification",
        "Model legal physical DMA rectangles and their command count, shared request admission/issue, dispatch, payload service and completion notification. Charge boundary layout conversions and retained loads explicitly.",
        "Profile HBM-byte audits match completed full comparisons. Splitting a transfer into many commands can cost more despite identical payload.",
        "Descriptor geometry is distinct from SBUF strides read by a matmul. Finite submission queues and the measured delay from ready semaphore to a DMA trigger are not fully modeled.",
        [
            "src/voyager_compiler/trainium/movement.py",
            "src/voyager_compiler/trainium/timing_trainium2.json",
        ],
        "hardware",
    )
    add(
        "dag",
        "Timing model",
        "Retained with limits",
        "Dependency-aware engine scheduling",
        "Replace a sum/max of service totals with a graph of resource occupancy, issue, completion, data dependencies and shared DMA issue resources. Schedule dependency-ready operations with the shared ASAP evaluator.",
        "Represents engine overlap and distinguishes service from latency. Compact candidate graphs, selected-ISA graphs and compiled graphs can use the same primitive laws.",
        "The evaluator assumes ready commands can be admitted. It is not a finite-queue native scheduling simulator; a different graph can yield a different answer even with identical rates.",
        [
            "src/voyager_compiler/codegen/transform/tiling/execution.py",
            "src/voyager_compiler/trainium/dependencies.py",
        ],
        "flow",
    )
    add(
        "affine",
        "Timing model",
        "Retained with limits",
        "Isolated primitive completion laws and kernel overhead",
        "Characterize isolated matmul, copy, transpose, memset and DMA probes. Separate issue floor and service scale from completion base and completion scale; keep fixed device setup separate.",
        "The original FP32 matmul completion law was approximately 381 ns + 2 × analytical service. It describes isolated startup substantially better than using throughput as completion.",
        "An isolated completion law does not describe every sustained stream. Unknown completion laws stay explicit. No whole-application latency was used as a calibration target.",
        [
            "scripts/trainium_characterize.py",
            "scripts/trainium_calibrate.py",
            "src/voyager_compiler/trainium/timing.py",
        ],
        "hardware",
    )
    add(
        "vector",
        "Timing model",
        "Validated within scope",
        "Vector binary/scalar width calibration",
        "Measure contiguous FP32, 128-partition dependent chains at widths 128, 512 and 2048; hold out width 8192. Fit service slope and completion intercept independently.",
        "Service scale 0.5; completion intercepts about 158.667 ns for binary and 160.333 ns for scalar operations. Held-out completion errors were 0.5 ns and 0 ns.",
        "These tests do not validate arbitrary strides, reductions, other partition counts or whole-kernel overlap.",
        ["results/trainium/vector-characterization/calibration.json"],
        "hardware",
    )
    add(
        "scalar-copy",
        "Timing model",
        "Validated within scope",
        "ScalarE SBUF-copy calibration",
        "Use isolated contiguous SBUF copy chains, training widths 128/512/2048 and held-out width 8192.",
        "Measured 297/617/1897/7017 ns; the fitted 190.333 ns intercept plus the 1.2 GHz service term predicts the held-out case exactly.",
        "A successful contiguous calibration does not establish strided-copy timing. That extrapolation later misranked full movement candidates.",
        ["results/trainium/scalar-copy-characterization/calibration.json"],
        "hardware",
    )
    add(
        "stream-transpose",
        "Timing model",
        "Validated within scope",
        "STREAM_TRANSPOSE context and the 93 ns observation",
        "Characterize supported contiguous FP32 32×32 VectorE transposes and distinguish repeated quadrant context from switching context.",
        "The hardware profile records 93 ns steady issue and 234 ns switch issue, with 233 ns completion, for this narrow measured mode. Search chooses a legal chain; 93 ns is not a universal movement rate.",
        "Other sizes/dtypes/access patterns are not automatically admitted as calibrated modes. The pinned SDK also constrains the supported arena encoding.",
        [
            "results/trainium/model-integration-2026-10-07/calibration-provenance.json",
            "docs/trainium-movement-search.md",
        ],
        "search",
    )
    add(
        "short-k",
        "Timing model",
        "Validated within scope",
        "Separate short-K FP32 matmul law",
        "Test dependent and independent matmul streams and distinguish K64/N128/M512 from the full-K formula.",
        "A separate characterized short-K timing law is retained; the physical LDWEIGHTS/MATMUL expansion remains unchanged.",
        "This is a measured geometry-specific case, not a blanket penalty on every short reduction. Other short-K shapes keep the earlier analytical law.",
        [
            "src/voyager_compiler/trainium/timing_trainium2.json",
            "test/test_trainium_calibrated_isa.py",
        ],
        "hardware",
    )
    add(
        "compact",
        "Timing model",
        "Retained with limits",
        "Compact startup / steady / tail candidate scoring",
        "Evaluate repeated candidate graphs through startup, neighboring steady iterations and tail, with exact fallback. Track invariant parameter loads once and retain full selected-plan audits.",
        "Makes shared mapping search tractable without expanding every full executable. Saves both compact and selected-ISA estimates instead of hiding their disagreement.",
        "Compact overlap and recurrence do not always match physical realization. The latest GEMM512 audit demonstrates a ranking reversal at this boundary.",
        [
            "docs/trainium-selected-plan.md",
            "src/voyager_compiler/trainium/dependencies.py",
        ],
        "flow",
    )
    add(
        "physical",
        "Timing model",
        "Retained with limits",
        "Selected physical ISA, lifetimes and retirement",
        "Replay actual selected instructions and exact SBUF/PSUM placement/reuse dependencies. Preserve all outstanding readers before a region is overwritten; use PSUM bank rotation and capacity checks.",
        "Makes address reuse a scheduling cost rather than just a footprint reduction. Historical records missing reader-retirement edges are repaired only in diagnostic memory and labeled.",
        "Correct dependencies can serialize a tightly reused allocation. Source-order placement and backend issue scheduling are still approximations; legality does not establish optimal performance.",
        [
            "src/voyager_compiler/trainium/instruction_plan.py",
            "src/voyager_compiler/trainium/program_analysis.py",
            "scripts/trainium_replay_selected.py",
        ],
        "flow",
    )
    add(
        "compiled",
        "Diagnostic",
        "Diagnostic only",
        "Timing-free compiled-ISA replay",
        "Extract native NEFF instruction order, concrete operand access patterns, DMA descriptors and semaphore dependencies, then apply the same performance laws without reading profile durations or timestamps.",
        "Separates input-program/realization differences from rate errors. Historical RMSNorm-first plan/replay/hardware were 5.999/5.372/8.318 ms; replay did not close the gap by itself.",
        "This is a prediction for an already compiled binary, not a schedule found by Voyager and not a hardware measurement. Prior binaries are diagnostic comparators, not search candidates by default.",
        [
            "src/voyager_compiler/trainium/compiled_analysis.py",
            "results/trainium/movement-search-full-2026-10-07/schedule-gap/plan-replay-gap.json",
            "results/trainium/operand-reuse-2026-10-07/report.json",
        ],
        "flow",
    )
    add(
        "oracle-latency",
        "Diagnostic",
        "Diagnostic only",
        "Trace-conditioned completion substitutions",
        "Replace one category of modeled completion spans at a time with observed NEFF/NTFF spans, preserving the rest of the graph. Also substitute all spans together.",
        "In the historical RMSNorm-first audit, 5.372 ms became only 5.579 ms with all measured spans, still far below 8.318 ms hardware. Per-operation completion error alone was insufficient.",
        "These are oracle sensitivities, not independent predictions, calibrated production coefficients, or additive critical-path attributions.",
        [
            "results/trainium/neff-gap-2026-10-08/add_rmsnorm_matmul/audit.json",
            "scripts/trainium_neff_gap_audit.py",
        ],
        "neff-gap",
    )
    add(
        "oracle-admission",
        "Diagnostic",
        "Diagnostic only",
        "Observed issue spacing and DMA-admission substitutions",
        "Measure spacing between consecutive logical matmul groups and the interval from explicit DMA readiness to trigger; substitute those separately after the completion-span audit.",
        "Observed matmul spacing alone gave 6.317 ms; all spans plus spacing gave 6.516 ms; adding observed DMA admission gave 8.229 ms against 8.318 ms hardware.",
        "A near match with trace-conditioned inputs does not validate a predictive queue model. Neither a universal 900 ns DMA penalty nor a kernel-specific matmul issue rate was adopted.",
        ["results/trainium/neff-gap-2026-10-08/add_rmsnorm_matmul/audit.json"],
        "neff-gap",
    )
    add(
        "movement-search",
        "Search / realization",
        "Retained with limits",
        "Endpoint-constrained movement-chain search",
        "Search legal memory/layout/dtype routes, Tensor/Vector/Scalar alternatives, tiling, traversal and intermediate materialization; allocate and score each candidate with the physical ISA graph.",
        "Small GEMM improved 20→18 µs and small SwiGLU 75→69 µs. Some winning chains used two transposes to restore layout while changing resource use and reuse dependencies.",
        "Budgeted, fixed-software-schedule search is not exhaustive joint fusion/layout/tiling search. More legal choices are useful only if the cost model ranks them correctly.",
        [
            "docs/trainium-movement-search.md",
            "results/trainium/movement-search-2026-10-07/report.json",
        ],
        "search",
    )
    add(
        "movement-failure",
        "Diagnostic",
        "Open limitation",
        "Full-kernel strided-copy ranking failures",
        "Apply the movement choices to the six full benchmarks and compare selected-plan predictions, timing-free replay, device measurements and concrete copy strides.",
        "Full SwiGLU changed 53.008→55.458 ms despite a preferred plan score. Strided staging copies made a contiguous ScalarE timing assumption invalid; BMM unit-stride copies behaved differently.",
        "An instruction-count reduction or a lower contiguous copy estimate is insufficient. Broader stride-aware copy calibration remains distinct from the new matmul geometry law.",
        [
            "results/trainium/movement-search-full-2026-10-07/report.json",
            "docs/trainium-movement-search.md",
        ],
        "movement",
    )
    add(
        "reuse-fusion",
        "Search / realization",
        "Retained with limits",
        "Direct operands, payload reuse and pointwise fusion",
        "Expose staged/direct/reuse operand policies; reuse converted weight payload within a software GEMM invocation, with explicit lifetime and capacity charges. Enable shared pointwise fusion separately.",
        "Historical full RMSNorm-first improved 9.147→8.318 ms and GEMM-first 21.209→17.647 ms. SwiGLU reuse alone reached 47.719 ms; adding pointwise fusion reached 34.765 ms.",
        "These are program/search changes, not new timing coefficients. Generic reduction/GEMM fusion and compatible producer layouts were still missing at that stage; later row regions address a subset.",
        [
            "results/trainium/operand-reuse-2026-10-07/report.json",
            "docs/trainium-movement-search.md",
        ],
        "kernels",
    )
    add(
        "isa-fusion",
        "Search / realization",
        "Diagnostic only",
        "Matched fused-ISA microbenchmarks",
        "Compare PSUM-add, fused RMS scaling and scale+rsqrt against separate instructions in matched 32-step dependent hardware probes.",
        "All six pairs improved: PSUM-add 42→33 and 76→57 µs; RMS scaling 34→26 and 82→59 µs; scale+rsqrt 37→30 and 37→28 µs.",
        "These are local probes with shared setup, not end-to-end GEMM speedups. They did not introduce production timing constants or enable a default fusion policy.",
        ["results/trainium/isa-fusion-2026-10-08/measured/comparison.json"],
        "hardware",
    )
    add(
        "allocation-ablation",
        "Diagnostic",
        "Diagnostic only",
        "Strict addresses versus distinct/native allocation",
        "Hold software tiles and inputs fixed while comparing strict reuse, distinct SBUF storage and native allocation; inspect actual commands and waits.",
        "Address reuse introduces write-after-read waits and explains much of the historical GEMM regression. Distinct SBUF recovered overlap; native allocation also folded copies and removed clears.",
        "Distinct storage is not a scalable bounded policy. Native allocation changes realization, so it cannot isolate a timing coefficient or certify Voyager-assigned addresses. No matched convolution measurement was found.",
        [
            "results/trainium/relaxed-realization-2026-10-08/comparison.json",
            "results/trainium/gemm-regression-audit-2026-10-08/comparison.json",
        ],
        "flow",
    )
    add(
        "bounded-buffers",
        "Search / realization",
        "Retained with limits",
        "Bounded temporary-buffer rotation",
        "Add a positive temporary-buffer-depth multiplier that rotates legal SBUF slots by size class while preserving physical capacity, live-range exclusion and reuse completion edges.",
        "With identical software work, GEMM512 improved from 169 µs at depth1 to 76/62/49/46 µs at depths2/4/8/16. The selected ISA model sees address-induced dependencies; compact search does not fully represent them.",
        "This knob changes storage generations and overlap, not useful payload reuse. Depth is a pool multiplier, not a promise of that many simultaneously active panels.",
        ["results/trainium/bounded-buffering-2026-10-08/comparison.json"],
        "flow",
    )
    add(
        "maxpool",
        "Diagnostic",
        "Open limitation",
        "Maxpool coverage and recurrence audit",
        "Separate candidate exclusion from bad scoring: inspect exact-divisor tiling, two halo slots, conservative whole-root dependencies, and fixed-template hardware probes.",
        "Production 89-row tiling measured 1.016 ms; a legal masked-tail 128-row probe measured 0.729 ms but was excluded by exact-divisor enumeration. Distance-2 graph variants exposed another overlap mismatch.",
        "The 128-row probe is not a production-selected schedule. Distance-2/removal-of-reuse sensitivities are not measured speedups. Rolling halo reuse remains outside the tested search.",
        ["results/trainium/maxpool-gap-2026-10-08/analysis.json"],
        "row-region-update",
    )
    add(
        "row-regions",
        "Search / realization",
        "Retained with limits",
        "Generic row-region bufferization",
        "Compose row-independent pointwise/reduction/matrix operations with shared bufferization, invariant retention, explicit boundary traffic and row/matrix layout contracts, instead of one builder per operation order.",
        "The six residual/RMSNorm/GEMM orders gained common bufferized coverage. In eligible regions, intermediates stay in SBUF and one retained weight payload feeds multiple row tiles.",
        "A representable fused schedule still needs accurate timing and legal physical layouts. Whole-weight capacity, padding boundaries and multi-GEMM regions constrain coverage; BMM→softmax remains deferred.",
        [
            "results/trainium/row-regions-2026-10-08/report.json",
            "src/voyager_compiler/trainium/row_regions.py",
        ],
        "row-region-update",
    )
    add(
        "orientation",
        "Search / realization",
        "Retained with limits",
        "Weight- versus activation-stationary search",
        "Enumerate both TensorE orientations and charge their operand preparation, PSUM eviction and consumer-layout conversion using shared primitive laws.",
        "Made the alternative panel shapes expressible. With the original fixed weight representation, forced activation-stationary full kernels remained slower because repeated weight assembly/transpose dominated.",
        "Orientation alone cannot remove a physical layout mismatch. Those old forced-orientation results are historical and should not be confused with the later K-partitioned-weight runs.",
        [
            "results/trainium/orientation-2026-10-08/report.json",
            "src/voyager_compiler/trainium/orientation.py",
        ],
        "orientation-update",
    )
    add(
        "weight-layout",
        "Search / realization",
        "Retained with limits",
        "Joint invariant-weight layout and orientation",
        "Search generic versus K-partitioned invariant weights, load the final representation directly from HBM, and let both orientations consume checked views. Price partition rounding and boundary DMA.",
        "GEMM-first reached 6.016 ms with weights stationary and 3.567 ms with activations stationary; RMSNorm-first reached 2.456/2.744 ms. No extra converted weight buffer or hidden spill was needed.",
        "This exposed a pure ranking error: both schedules existed, but the old model preferred the slower GEMM-first orientation. The layout change itself was not a timing-model fix.",
        ["results/trainium/weight-layout-2026-10-08/report.json"],
        "weight-layout-update",
    )
    add(
        "accumulator-hypothesis",
        "Diagnostic",
        "Not established as cause",
        "Accumulator overlap and forwarding sensitivities",
        "Inspect one-row-tile graphs: weight/activation orientations exposed 16/4 independent output accumulators. Try forwarding at occupancy, serializing to one accumulator, and limiting GEMM accumulator slots to eight.",
        "Some extreme sensitivities changed the ranking, but the eight-slot diagnostic left the actual ranking unchanged. Excess overlap was not established as the cause of this wrong choice.",
        "The eight-slot test bounds only GEMM outputs, not every transpose temporary. One-slot serialization is a counterfactual, not the hardware configuration. These variants were not production calibration.",
        [
            "results/trainium/weight-layout-2026-10-08/ranking-dependency-diagnostic.json",
            "results/trainium/weight-layout-2026-10-08/ranking-trace-diagnostic.json",
        ],
        "matmul-geometry-update",
    )
    add(
        "spacing-hypothesis",
        "Diagnostic",
        "Diagnostic only",
        "Orientation-specific trace substitutions",
        "Compare complete FP32 groups in the two NEFFs, then substitute observed launch spacing or group span into the compact candidate graph independently.",
        "Old modeled spacing was 213/853 ns versus observed 436/858 ns. Spacing-only substitution changed whole-search scores from 4.698/5.283 ms to 6.513/5.290 ms and reversed the ranking.",
        "This established a missing effective throughput cost, but could not distinguish access geometry from issue or scheduling effects. No orientation-specific penalty was added.",
        [
            "results/trainium/weight-layout-2026-10-08/ranking-trace-diagnostic.json"
        ],
        "matmul-geometry-update",
    )
    add(
        "geometry-probes",
        "Timing model",
        "Validated within scope",
        "Controlled moving/stationary stride experiments",
        "Run direct-address hardware probes varying moving width and the free-axis strides of each operand independently. Preserve arithmetic and validate output correctness.",
        "For width128, moving stride1→16 changed spacing 223→437 ns. Stationary stride16 slowed narrow panels but was hidden by wider moving work. Twelve development probes and five fresh held-out probes support the feed-regime model.",
        "Rates describe the characterized FP32 geometry, not a universal SBUF-bank model. The five withheld issue errors are ≤6.50%; primitive accuracy does not prove whole-program accuracy.",
        [
            "results/trainium/matmul-geometry-2026-10-08/calibration.json",
            "scripts/trainium_geometry_calibration.py",
        ],
        "matmul-geometry-update",
    )
    add(
        "steady-overreach",
        "Timing model",
        "Superseded attempt",
        "Applying steady completion everywhere",
        "The first geometry-model version used sustained-stream completion for every supported matmul, including isolated calls and interrupted streams.",
        "It improved GEMM-first but worsened older standalone predictions. Existing traces showed about 799 ns isolated versus 496 ns sustained completion for contiguous width128.",
        "That broad assumption was replaced by explicit startup/steady context. Some absolute errors still remain after the correction; the failed initial results are retained.",
        [
            "results/trainium/matmul-geometry-2026-10-08/after-initial.json",
            "results/trainium/matmul-geometry-2026-10-08/existing-profiles/final-gemm512.json",
        ],
        "matmul-geometry-update",
    )
    add(
        "geometry-final",
        "Timing model",
        "Validated within scope",
        "Current geometry-aware feed and forwarding law",
        "Price moving/stationary feed paths by their maximum, add measured issue overhead and a moving-path completion tail, retain isolated startup bounds, and allow only true same-accumulator dependencies to forward early.",
        "GEMM-first selected-ISA predictions became 6.057/3.684 ms against 6.016/3.567 ms hardware. Fresh search now selects the faster orientation in both fused cases.",
        "Scope: FP32 K=N=128, moving width128–512, regular strides1/2/4/8/16. BF16, short-K and unknown geometries keep prior laws. Compiled replay has less producer-generation context than selected-ISA analysis.",
        [
            "results/trainium/matmul-geometry-2026-10-08/report.json",
            "src/voyager_compiler/trainium/timing.py",
        ],
        "matmul-geometry-update",
    )
    add(
        "ranking-validation",
        "Diagnostic",
        "Open limitation",
        "Broad validation and the remaining GEMM512 reversal",
        "Re-score 24 fixed historical programs; regenerate both fused searches and old/new standalone searches; measure standalone selections on the same core and re-score the old GEMM512 tile under the new model.",
        "Fresh GEMM128 stayed 17 µs, GEMM256 improved 42→37 µs, GEMM512 regressed 76→82 µs. New compact costs are 41.513/41.123 µs for old/new GEMM512 tiles; selected physical ISA costs are 63.176/71.860 µs, correctly ranking them.",
        "The remaining reversal is at the compact-candidate/realization boundary. RMSNorm-first activation prediction also worsened. More primitive calibration alone cannot close every overlap, dependency or search-coverage gap.",
        [
            "results/trainium/matmul-geometry-2026-10-08/report.json",
            "results/trainium/matmul-geometry-2026-10-08/after.json",
        ],
        "matmul-geometry-update",
    )
    current = json.loads(
        (
            ROOT / "results/trainium/matmul-geometry-2026-10-08/report.json"
        ).read_text()
    )
    profile = json.loads(
        (
            ROOT / "src/voyager_compiler/trainium/timing_trainium2.json"
        ).read_text()
    )
    data = dict(
        updated="2026-10-08",
        scope="Evidence-backed history of the Trainium performance-model investigation. Historical results retain their original schedules and timing scopes; diagnostic substitutions are not predictions. This documentation update runs no new benchmarks.",
        entries=entries,
        current=current,
        hardware_profile=profile,
    )
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    section = """<!-- MODELING_HISTORY_START -->
<style>
#modeling-history .mh-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:18px 0}
#modeling-history .mh-controls label{display:grid;gap:4px;font-weight:600}
#modeling-history input,#modeling-history select{padding:8px;border:1px solid var(--line);border-radius:6px;max-width:100%;background:white;color:var(--ink)}
#modeling-history .mh-entry{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:14px 18px}
#modeling-history .mh-entry summary{cursor:pointer;font-weight:700;overflow-wrap:anywhere}
#modeling-history .mh-entry dl{display:grid;grid-template-columns:130px 1fr;gap:10px;margin:15px 0}
#modeling-history .mh-entry dt{font-weight:700;color:var(--muted)}#modeling-history .mh-entry dd{margin:0;overflow-wrap:anywhere}
#modeling-history .mh-links a{display:block;overflow-wrap:anywhere;font-size:13px}
#modeling-history .mh-scroll{overflow-x:auto}#modeling-history .mh-snapshot{overflow-wrap:anywhere}
.mh-historical-note{border-left:3px solid var(--amber);padding:8px 12px;background:#fff6e8;font-size:13px;margin:12px 0}
@media(max-width:650px){#modeling-history .mh-entry dl{grid-template-columns:1fr;gap:4px}#modeling-history .mh-entry dd{margin-bottom:10px}#modeling-history .mh-controls>*{width:100%}}
</style>
<section id="modeling-history" class="card">
<span class="eyebrow">Current findings and modeling history · 8 October 2026</span>
<h2>What the model gets right, and where search still goes wrong</h2>
<p><strong>Operand geometry fixes the fused orientation choice. It does not yet make overall schedule ranking reliable.</strong> The latest GEMM512 experiment isolates a remaining disagreement between compact candidate scoring and the selected physical instruction graph.</p>
<div id="mh-current" class="mh-scroll"></div>
<p class="small">Search and selected-ISA columns use the new model but different input graphs. Fused hardware values are authenticated reuse after matching generated NKI and input arrays. Fresh standalone measurements below use core 0. All are device times, excluding host invocation.</p>
<div id="mh-ranking" class="mh-scroll"></div>
<div class="grid"><div><h3>Confirmed</h3><ul><li>Moving and stationary operand strides have different bottlenecks.</li><li>Startup completion differs from sustained-stream completion.</li><li>Address reuse and producer/consumer layouts change executable dependencies.</li><li>Fresh search chooses the faster orientation for both fused kernels.</li></ul></div><div><h3>Still unresolved</h3><ul><li>Compact search can disagree with its own physical-plan audit.</li><li>Finite admission queues and readiness-dependent pipeline restart remain approximate.</li><li>Strided-copy and some reduction/activation timing coverage remain incomplete.</li><li>Candidate coverage excludes useful cases such as the masked-tail maxpool probe; BMM→softmax work remains deferred.</li></ul></div></div>
<details><summary>Four quantities that must not be conflated</summary><ol><li><strong>Compact search prediction:</strong> a candidate schedule template, its modeled buffering, transfers and primitive costs.</li><li><strong>Selected-ISA prediction:</strong> Voyager's expanded instructions and physical reuse dependencies.</li><li><strong>Compiled-static replay:</strong> native NEFF order, operands and semaphores, with no measured durations supplied to prediction.</li><li><strong>Hardware measurement:</strong> actual execution of the correctness-checked binary. Profile-conditioned substitutions are separate diagnostic oracles.</li></ol></details>
<details><summary>Current hardware timing profile and its scope</summary><p>The baseline coefficient explorer later in this page is a historical snapshot. This is the latest hardware-owned profile, including the geometry law. The empirical constants are calibrated primitive parameters, not per-kernel adjustments. Search/realization changes are listed separately below.</p><pre id="mh-profile" style="max-height:460px;overflow:auto"></pre><p>Strict mode still assigns local addresses in Voyager. Native compilation produces the executable; native allocation was an explicit diagnostic alternative. No CPU simulator was used for these hardware experiments.</p></details>
<h3 style="margin-top:24px">Every recorded modeling approach and related experiment</h3>
<p>Filter by category or outcome, search terms such as <em>DMA</em>, <em>stride</em> or <em>forwarding</em>, and expand an entry for its hypothesis, evidence, result and limitation. “Validated within scope” describes the stated experiment, not universal model accuracy.</p>
<div class="mh-controls"><label>Category<select id="mh-category"><option>All categories</option></select></label><label>Outcome<select id="mh-status"><option>All outcomes</option></select></label><label>Search<input id="mh-query" type="search" placeholder="Search modeling history"></label><button type="button" id="mh-expand">Expand visible entries</button><button type="button" id="mh-collapse">Collapse visible entries</button></div>
<p id="mh-count" role="status" aria-live="polite"></p><div id="mh-entries"></div>
<p class="small">Seventeen geometry probes, twenty-four fixed-program reanalyses, fresh old/new standalone searches, 148 focused regression tests, and 19 subsequent affected-context checks underpin the latest update. Detailed tables: <a href="#matmul-geometry-update">geometry validation</a>. Earlier sections preserve their original data and are labeled historical.</p>
</section>
<script id="modeling-history-data" type="application/json">__DATA__</script>
<script>(()=>{const d=JSON.parse(document.getElementById('modeling-history-data').textContent);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const table=(h,r)=>'<table><thead><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+r.map(a=>'<tr>'+a.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';const ms=x=>(x/1000).toFixed(3),us=x=>x.toFixed(3);
document.getElementById('mh-current').innerHTML=table(['Current selected fused kernel','Orientation','Search ms','Selected ISA ms','Hardware ms'],d.current.fused_selected.map(r=>[r.case,r.orientation,ms(r.search_us),ms(r.selected_us),ms(r.hardware_us)]));
const g=d.current.gemm512_ranking_audit;document.getElementById('mh-ranking').innerHTML='<h3>Remaining GEMM512 ranking gap · both tiles scored with the new model</h3>'+table(['Tile M,N,K','Compact search µs','Selected physical ISA µs','Hardware µs'],[[g.old_tile.join(','),us(g.old_tile_new_model_search_us),us(g.old_tile_new_model_selected_us),us(g.old_tile_hardware_us)],[g.new_tile.join(','),us(g.new_tile_new_model_search_us),us(g.new_tile_new_model_selected_us),us(g.new_tile_hardware_us)]])+'<p>Compact search picks the new tile; the physical-plan audit correctly prefers the old tile. Fresh standalone outcomes: GEMM128 17→17 µs, GEMM256 42→37 µs, GEMM512 76→82 µs. No further matmul-rate adjustment was fitted to conceal this regression.</p>';
document.getElementById('mh-profile').textContent=JSON.stringify(d.hardware_profile,null,2);
const category=document.getElementById('mh-category'),status=document.getElementById('mh-status'),query=document.getElementById('mh-query'),host=document.getElementById('mh-entries');for(const [s,key] of [[category,'category'],[status,'status']])for(const v of [...new Set(d.entries.map(x=>x[key]))]){const o=document.createElement('option');o.textContent=v;s.appendChild(o);}
function render(){const term=query.value.toLowerCase();const rows=d.entries.filter(x=>(category.selectedIndex===0||x.category===category.value)&&(status.selectedIndex===0||x.status===status.value)&&[x.title,x.attempt,x.result,x.limit,x.category,x.status].join(' ').toLowerCase().includes(term));host.innerHTML=rows.map(x=>'<details class="mh-entry" id="mh-entry-'+esc(x.id)+'"><summary>'+esc(x.title)+' <span class="badge">'+esc(x.category)+'</span> <span class="badge '+(x.status==='Open limitation'||x.status==='Superseded attempt'?'limit':'policy')+'">'+esc(x.status)+'</span></summary><dl><dt>What we tried</dt><dd>'+esc(x.attempt)+'</dd><dt>What we learned</dt><dd>'+esc(x.result)+'</dd><dt>What it does not prove</dt><dd>'+esc(x.limit)+'</dd></dl><p><a href="#'+esc(x.anchor)+'">Related report section</a></p><div class="mh-links">'+x.evidence.map(e=>'<a href="'+esc(e.href)+'">'+esc(e.label)+'</a>').join('')+'</div></details>').join('');document.getElementById('mh-count').textContent=rows.length+' of '+d.entries.length+' approaches shown.';}
category.addEventListener('change',render);status.addEventListener('change',render);query.addEventListener('input',render);document.getElementById('mh-expand').addEventListener('click',()=>host.querySelectorAll('details').forEach(e=>e.open=true));document.getElementById('mh-collapse').addEventListener('click',()=>host.querySelectorAll('details').forEach(e=>e.open=false));render();})();</script>
<!-- MODELING_HISTORY_END -->
""".replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    page = HTML.read_text()
    embedded = lambda s: dict(
        re.findall(
            r'<script id="([^"]+)" type="application/json">(.*?)</script>',
            s,
            re.S,
        )
    )
    original = {
        k: v for k, v in embedded(page).items() if k != "modeling-history-data"
    }
    page = re.sub(
        r"<!-- MODELING_HISTORY_START -->.*?<!-- MODELING_HISTORY_END -->\n?",
        "",
        page,
        flags=re.S,
    )
    page = page.replace("<main>", "<main>\n" + section, 1)
    if 'href="#modeling-history"' not in page.split("</nav>")[0]:
        page = page.replace("<nav", "<nav", 1)
        page = re.sub(
            r"(<nav[^>]*>)",
            r'\1<a href="#modeling-history">Current findings & history</a>',
            page,
            count=1,
        )
    snapshots = [
        "weight-layout-update",
        "orientation-update",
        "row-region-update",
        "results",
        "neff-gap",
        "kernels",
        "hardware",
        "search",
    ]
    for ident in snapshots:
        pattern = r'(<section\b[^>]*\bid="' + re.escape(ident) + r'"[^>]*>)'
        note = (
            '<p class="mh-historical-note" data-history-note="'
            + ident
            + '"><strong>Historical snapshot.</strong> Results, policies and coefficients in this section describe that experiment. For the latest model and corrected findings, see <a href="#modeling-history">current findings and modeling history</a>.</p>'
        )
        if 'data-history-note="' + ident + '"' not in page:
            page = re.sub(pattern, lambda m: m[1] + note, page, count=1)
    assert all(
        embedded(page).get(k) == v for k, v in original.items()
    ), "Historical embedded evidence changed"
    HTML.write_text(page)
    print(
        json.dumps(
            dict(
                html=str(HTML),
                approaches=len(entries),
                historical_data_blocks_preserved=len(original),
                history=str(OUT),
            )
        )
    )


if __name__ == "__main__":
    main()
