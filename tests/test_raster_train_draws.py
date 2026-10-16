import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_raster_train_draws")


def pair(module, left, right, lo=0, hi=0, length=25):
    row = importlib.import_module("run_strand_graph_pilot").pair_row
    return row(0, "test", left, right, lo, hi, length)


def test_zero_invariance_and_anchor_are_separate(module):
    counts = module.MetricCounts()
    a, b = ((1, 1, 1), 2), ((1, 1, 1, 2), 3)
    for row in (
        pair(module, a, b),
        pair(module, b, a),
        pair(module, a, a),
        pair(module, ((1,), 2), ((), 1)),
    ):
        counts.add(row)
    r = counts.report()
    assert r["counts"]["draws"] == 4
    assert r["counts"]["zero_nonidentical_endpoints"] == 3
    assert r["counts"]["zero_nonidentical_nonanchor"] == 2
    assert r["counts"]["zero_identical_endpoints"] == 1
    assert r["counts"]["anchor_pairs"] == 1
    assert r["distinct_unordered_endpoint_pairs"] == 3
    assert r["maximum_word_length"] == 4 and r["maximum_strands"] == 3


def test_weight_fractions_not_draw_fractions(module):
    a, b = ((1, 1, 1), 2), ((-1, -1, -1), 2)
    r = module.source_weight_fractions(
        {"original": [pair(module, a, a)], "sentinel": [pair(module, a, b, 2, 2)]}
    )
    assert r == dict(original=8 / 9, sentinel=1 / 9, exact_zero=8 / 9)


def test_source_slices_preserve_order(module):
    r = module.source_slices(list(range(32)))
    assert list(r) == ["original", "suffix", "graph", "sentinel"]
    assert [len(v) for v in r.values()] == [7, 8, 16, 1]
    assert [x for values in r.values() for x in values] == list(range(32))
    with pytest.raises(ValueError):
        module.source_slices(list(range(31)))
