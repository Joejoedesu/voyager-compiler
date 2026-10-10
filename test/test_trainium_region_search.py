"""Expanded selection: actual realization, prediction-only ranking and coverage."""
import json
import math
import pytest
import torch

import voyager_compiler as vc
from voyager_compiler.codegen.transform.bufferize import BufferizationOptions
from voyager_compiler.trainium.region_search import enumerate_candidates, choose_winner
from test_trainium_stream_regions import BMM, Chain, prepare


def test_enumeration_is_independent_of_compact_ranking():
    legal = [dict(legal=True, tile_rows=m, weight_residency=s, prediction_ns=1/m)
             for m in (16, 32, 64, 128) for s in ("resident", "stage")]
    a, coverage = enumerate_candidates([dict(candidates=legal)], row_count=2,
                                      orientations=("weights", "activations"))
    for c in legal:
        c["prediction_ns"] *= -1000
    b, _ = enumerate_candidates([dict(candidates=legal)], row_count=2,
                                orientations=("weights", "activations"))
    assert a == b
    assert len(a) == 8
    assert coverage[0]["omitted_rows"] == [32, 16]
    assert a[0]["regions"] == a[1]["regions"]
    assert a[0]["orientation"] != a[1]["orientation"]


def test_physical_score_controls_winner_and_failed_candidates_cannot_win():
    rows = [dict(id=0, status="valid", physical_prediction_ns=100,
                 compact_prediction_ns=1, hardware_us=1),
            dict(id=1, status="valid", physical_prediction_ns=50,
                 compact_prediction_ns=1000, hardware_us=1000),
            dict(id=2, status="rejected", physical_prediction_ns=0),
            dict(id=3, status="valid", physical_prediction_ns=math.inf)]
    assert choose_winner(rows)["id"] == 1
    with pytest.raises(ValueError, match="No region candidate"):
        choose_winner(rows[2:])


@pytest.mark.parametrize("name", ["bmm", "swiglu"])
def test_end_to_end_winner_is_the_scored_physical_program(tmp_path, name):
    torch.manual_seed(48)
    module = BMM() if name == "bmm" else Chain()
    shapes = ((2, 128, 64), (2, 64, 256)) if name == "bmm" else (
        (128, 64), (64, 128), (64, 128), (128, 64))
    args = tuple(torch.randn(*s)*0.1 for s in shapes)
    graph, context = prepare(module, args)
    before = context.record()
    vc.compile(graph, args, context=context, output_dir=tmp_path,
               dump_tensors=False, bufferization_options=BufferizationOptions(
                   stream_regions=True, stream_region_row_candidates=1))
    report = json.loads((tmp_path/'region-search.json').read_text())
    selected = json.loads((tmp_path/'selection.json').read_text())
    assert report['selected']['physical_prediction_ns'] == min(
        c['physical_prediction_ns'] for c in report['candidates'] if c['status']=='valid')
    assert report['realization_matches_scored_candidate']
    assert report['final_instructions_sha256'] == selected['instructions_sha256']
    assert report['selection_uses_compact_prediction'] is False
    assert report['selection_uses_measurement'] is False
    assert context.record() == before
    assert all(c['orientation'] == report['selected']['spec']['orientation']
               for c in selected['matrix_choices'])
    torch.testing.assert_close(graph(*args), module(*args), rtol=1e-3, atol=1e-3)
    context.realize(tmp_path)
    assert (tmp_path/'nki/program.py').exists()
    # Finalization must reject even a parseable change to the scored artifact.
    from pathlib import Path
    from voyager_compiler.trainium.region_search import reuse_scored_plan
    scored = Path(report['selected']['path'])
    with (scored/'instructions.json').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='artifact changed'):
        reuse_scored_plan(tmp_path, scored, context)


def test_invalid_search_bounds():
    for kwargs in [dict(stream_region_row_candidates=0),
                   dict(stream_region_search_budget=-1),
                   dict(stream_region_search='native-magic')]:
        with pytest.raises(ValueError):
            BufferizationOptions(stream_regions=True, **kwargs)


def test_budget_does_not_materialize_the_full_region_product():
    # 4**30 logical combinations; constructing a list before truncation is unsafe.
    regions = [dict(candidates=[dict(legal=True, tile_rows=m, weight_residency=s)
               for m in (64,128) for s in ("resident","stage")])] * 30
    specs, coverage = enumerate_candidates(regions, row_count=2,
        orientations=("weights","activations"), budget=3)
    assert len(specs) == 3
    assert math.prod(c['candidate_count'] for c in coverage) * 2 == 2 * 4**30


def test_expanded_discovery_does_not_require_a_finite_compact_score():
    from types import SimpleNamespace
    from voyager_compiler.codegen.transform.bufferize.stream_regions import plan_stream_regions
    graph, context = prepare(BMM(), (torch.randn(2,128,64),torch.randn(2,64,256)))
    evaluate = context.policy.stream_region_candidate
    def missing_compact(region, rows, residency):
        c = evaluate(region, rows, residency)
        if c.get('legal'): c['prediction_ns'] = math.inf
        return c
    context.policy.stream_region_candidate = missing_compact
    plan_stream_regions(graph, SimpleNamespace(mapping_policy=context.policy), discovery_only=True)
    assert graph.meta['stream_regions'][0]['status'] == 'selected'
