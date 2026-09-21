import importlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE
from rf_knots.evidence import UnknotWitness
from rf_knots.mcts_loop import Instance


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("summarize_graph_mixture_experiment")


def fixture_data(summary, monkeypatch, reference_extra=0, solution_extra=0):
    instance = Instance(
        "engineering", (1, 1, 1), 2, 1, cap_moves=128, growth_len=8, growth_strands=4
    )
    spec = instance.spec

    def witness(extra):
        actions = [
            a for _ in range(extra) for a in (spec.encode(INSERT, 0, 1, 1), spec.encode(REDUCE, 0))
        ]
        actions += [
            spec.encode(CROSSING_CHANGE, 0),
            spec.encode(REDUCE, 0),
            spec.encode(DESTABILIZE),
        ]
        return UnknotWitness.from_actions(instance.word, instance.strands, spec, actions)

    ref, sol = witness(reference_extra), witness(solution_extra)
    monkeypatch.setattr(summary, "verify_evaluation_pair", lambda r: ref)
    panel = {
        "validation": [
            {"left_fingerprint": [3, -2], "pair_prefix_length": ref.moves, "interval": {"upper": 1}}
        ]
    }
    roots = []
    rows = []
    for index, weight in enumerate((10, 1000)):
        roots.append(
            {
                "condition": index,
                "panel_index": 0,
                "fingerprint": [3, -2],
                "stratum": "within",
                "policy_stratum": "within",
                "instance": replace(instance, cc_weight=weight).to_dict(),
                "reference_L10": 10 + ref.moves,
                "reference_L1000": 1000 + ref.moves,
            }
        )
        rows.append(
            {
                "condition": index,
                "panel_index": 0,
                "cc_weight": weight,
                "solved": True,
                "witness": sol.to_dict(),
                "L10": 10 + sol.moves,
                "L1000": 1000 + sol.moves,
                "episodes": [],
            }
        )
    return panel, roots, {"graph": rows}


@pytest.mark.parametrize(
    "ref_extra,sol_extra,regret,optimal",
    [(0, 0, 0, True), (0, 1, 2, False), (1, 0, 0, True), (1, 1, None, False)],
)
def test_cost_gaps_and_proven_regrets_are_not_conflated(
    summary, monkeypatch, ref_extra, sol_extra, regret, optimal
):
    panel, roots, costs = fixture_data(summary, monkeypatch, ref_extra, sol_extra)
    result = summary.certify_costs(panel, roots, costs)
    for row in result["solutions"]["graph"]:
        assert row["gap_to_reference"] == 2 * (sol_extra - ref_extra)
        assert row["gap_to_lower_bound"] == 2 * sol_extra
        assert row["certified_regret"] == regret
        assert row["solution_certifiably_optimal"] is optimal


@pytest.mark.parametrize("change", ["cost", "identity", "cap", "solved", "alphabet"])
def test_replayed_costs_identity_and_resources_fail_closed(summary, monkeypatch, change):
    panel, roots, costs = fixture_data(summary, monkeypatch)
    if change == "cost":
        costs["graph"][0]["L10"] -= 1
    elif change == "identity":
        costs["graph"][0]["panel_index"] = 2
    elif change == "cap":
        roots[0]["instance"]["cap_crossing_changes"] = 0
    elif change == "solved":
        costs["graph"][0]["solved"] = False
    else:
        roots[0]["instance"]["growth_len"] += 2
    with pytest.raises(ValueError):
        summary.certify_costs(panel, roots, costs)


def test_censored_search_has_no_cost_not_zero(summary, monkeypatch):
    panel, roots, costs = fixture_data(summary, monkeypatch)
    for row in costs["graph"]:
        row.update(solved=False, witness=None, L10=None, L1000=None)
    assert summary.certify_costs(panel, roots, costs)["solutions"]["graph"] == []
    costs["graph"][0]["L10"] = 0
    with pytest.raises(ValueError, match="numeric cost"):
        summary.certify_costs(panel, roots, costs)


def test_partial_or_changed_matrix_cannot_be_reported_as_completed(summary, tmp_path):
    jobs = [{"name": "a"}]
    summary.require_final_matrix({"status": "completed", "jobs": jobs, "completed": [{}]}, jobs)
    for state in (
        {"status": "evaluating", "jobs": jobs, "completed": [{}]},
        {"status": "completed", "jobs": jobs, "completed": []},
        {"status": "completed", "jobs": [{"name": "other"}], "completed": [{}]},
    ):
        with pytest.raises(ValueError, match="complete declared"):
            summary.require_final_matrix(state, jobs)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "waiting", "jobs": [], "completed": []})
    )
    with pytest.raises(ValueError, match="complete declared"):
        summary.verified_inputs(tmp_path, tmp_path / "general")


def test_summary_preserves_both_families_views_and_censored_uniform(summary, monkeypatch):
    from evaluate_pure_braid_distance_panel import summaries
    from test_pure_braid_distance_evaluation import data

    panel, roots, costs = fixture_data(summary, monkeypatch)
    costs = {a: deepcopy(costs["graph"]) for a in (*summary.RUNS, "uniform")}
    for r in costs["uniform"]:
        r.update(solved=False, witness=None, L10=None, L1000=None)
    predictions, records = data()
    for p, r in zip(predictions, records, strict=True):
        p["left_fingerprint"] = r["left_fingerprint"]
    pure_stats = summaries(predictions, records)
    reports = {}
    for arm in summary.RUNS:
        reports[f"metric-{arm}-short-u"] = {
            "checkpoint_sha256": arm,
            **{v: {"predictions": deepcopy(predictions)} for v in ("validation", "size_shift")},
        }
    for arm in costs:
        reports[f"mcts-{arm}"] = {"counts": {"conditions": 2}, "seconds_committed_batches": 1.0}
    pure = {"initial": {"raw_sizes": {}}}
    comparison = {"arms": {a: pure_stats for a in summary.RUNS}}
    result = summary.summarize(
        reports, roots, costs, pure, comparison, panel, {"status": "unavailable"}
    )
    assert set(result["short_u"]["graph"]) == {"validation", "size_shift"}
    assert result["scope"]["promotion"] == "not_decided"
    assert result["original_general_source"]["status"] == "unavailable"
    assert result["search_comparisons"]["graph-vs-uniform"]["10"]["common_solved"] == 0
    text = summary.markdown(result)
    assert "not available" in text and "Constant1" in text and "not_performed" in text
    assert "Short-u exact2 MAE (count)" in text
    assert "General exact2 MAE (count)" in text
    assert text.count("1.000000 (1)") == 6
    for arm in summary.RUNS:
        del result["short_u"][arm]["validation"]["exact"]["2"]
    assert summary.markdown(result).count("not available (0)") == 3
