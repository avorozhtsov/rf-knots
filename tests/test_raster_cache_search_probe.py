import copy
import importlib
from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_raster_cache_search")


def test_bounded_roots_and_two_cost_weights(monkeypatch):
    probe = module(monkeypatch)
    roots = probe.roots()
    assert [r["condition"] for r in roots] == list(range(4))
    assert {r["instance"]["cc_weight"] for r in roots} == {10, 1000}
    for root in roots:
        row = root["instance"]
        assert row["cap_moves"] in (4, 8)
        assert row["cap_crossing_changes"] == (0 if len(row["word"]) == 1 else 1)
        assert (row["growth_len"], row["growth_strands"]) == (2, 1)


def test_comparison_excludes_only_probabilities(monkeypatch):
    probe = module(monkeypatch)
    original = dict(
        rows=[
            dict(
                episodes=[
                    dict(
                        continuation=dict(probabilities=[0.25], actions=[42], stop_reason="horizon")
                    )
                ],
                L10=None,
            )
        ],
        counts={"episodes": 1},
        seconds=2,
    )
    other = copy.deepcopy(original)
    other["seconds"] = 1
    other["rows"][0]["episodes"][0]["continuation"]["probabilities"] = [0.2500001]
    assert probe.compare_runs(original, other)["passed"]
    assert original["rows"][0]["episodes"][0]["continuation"]["probabilities"] == [0.25]
    other["rows"][0]["episodes"][0]["continuation"]["actions"] = [43]
    result = probe.compare_runs(original, other)
    assert not result["passed"] and not result["equal_work_timing"]
    assert result["maximum_probability_error"] is None


def test_large_probability_or_count_difference_fails(monkeypatch):
    probe = module(monkeypatch)
    original = dict(
        rows=[dict(episodes=[dict(continuation=dict(probabilities=[0.25], actions=[42]))])],
        counts={"episodes": 1},
        seconds=2,
    )
    other = copy.deepcopy(original)
    other["rows"][0]["episodes"][0]["continuation"]["probabilities"] = [0.5]
    assert not probe.compare_runs(original, other)["passed"]
    other = copy.deepcopy(original)
    other["counts"]["episodes"] = 2
    assert not probe.compare_runs(original, other)["passed"]
