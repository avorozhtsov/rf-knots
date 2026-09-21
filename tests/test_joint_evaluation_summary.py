import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def summarizer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("summarize_joint_policy_metric_evaluation")


def prediction(index, group, value, lo=1, hi=1):
    return {
        "panel_index": index,
        "fingerprint_group": group,
        "prediction": value,
        "lower": lo,
        "upper": hi,
        "violation": max(lo - value, value - hi, 0),
    }


def test_constant_controls_use_the_same_equal_group_weights(summarizer):
    rows = [prediction(i, "many", 2, 2, 2) for i in range(9)] + [prediction(9, "one", 0, 0, 0)]
    stats = summarizer.metric_with_constants(rows)
    assert stats["equal_group_mean_interval_violation"] == 0
    assert stats["constants"]["0"]["mean_interval_violation"] == pytest.approx(1.8)
    assert stats["constants"]["0"]["equal_group_mean_interval_violation"] == 1
    assert stats["exact"]["2"]["count"] == 9


def test_paired_metric_is_candidate_minus_baseline_per_group(summarizer):
    a = [prediction(i, "a", 3) for i in range(9)] + [prediction(9, "b", 1)]
    b = [prediction(i, "a", 2) for i in range(9)] + [prediction(9, "b", 2)]
    result = summarizer.paired_metric(a, b)
    assert result["mean_group_error_delta"] == 0
    assert result["lower_error_groups"] == result["higher_error_groups"] == 1
    b[0]["panel_index"] = 200
    with pytest.raises(ValueError, match="labels/queries"):
        summarizer.paired_metric(a, b)


def test_search_censoring_common_only_costs_and_separate_lambdas(summarizer):
    baseline = [None, 1, None, 3, 4, 5, None]
    candidate = [None, None, 2, 2, 5, 5, 9000]
    roots, a, b = [], [], []
    for i, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        weight = 10 if i < 6 else 1000
        roots.append(
            {
                "condition": i,
                "panel_index": i,
                "instance": {"cc_weight": weight},
                "stratum": summarizer.STRATA[i % 4],
                "fingerprint": [i + 1, 0],
                "reference_L10": 15,
                "reference_L1000": 1005,
            }
        )
        common = {"condition": i, "panel_index": i, "cc_weight": weight}
        a.append({**common, "L10": left, "L1000": left})
        b.append({**common, "L10": right, "L1000": right})
    result = summarizer.paired_search(roots, a, b, 10)
    assert result["observations"] == 6
    assert result["baseline_solved"] == result["candidate_solved"] == 4
    assert result["common_solved"] == 3
    assert result["mean_cost_delta_common_only"] == 0
    assert result["outcomes"] == {
        "both_unsolved": 1,
        "lost_solution": 1,
        "newly_solved": 1,
        "lower_cost": 1,
        "higher_cost": 1,
        "equal_cost": 1,
    }
    assert summarizer.paired_search(roots, a, b, 1000)["common_solved"] == 0
    b[0]["cc_weight"] = 1000
    with pytest.raises(ValueError, match="identity/resources"):
        summarizer.paired_search(roots, a, b, 10)


def test_partial_controller_cannot_produce_final_summary(summarizer, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"status": "evaluating", "jobs": [], "completed": []}))
    with pytest.raises(ValueError, match="whole final"):
        summarizer.verified_inputs(path)


def test_complete_descriptive_matrix_keeps_views_costs_and_limits_separate(summarizer):
    record = {"interval": {"left": {"word": [1] * 220, "strands": 18}}}
    panels = {
        name: {"validation": [record], "size_shift": [record]}
        for name in summarizer.sequence.PANELS
    }
    reports = {}
    for arm in summarizer.sequence.RUNS:
        for name in panels:
            reports[f"metric-{arm}-{name}"] = {
                view: {"predictions": [prediction(0, "test-group", 1)]}
                for view in ("validation", "size_shift")
            }
    roots = [
        {
            "condition": index,
            "panel_index": 0,
            "instance": {"cc_weight": weight},
            "stratum": "both",
            "fingerprint": [3, 2],
            "reference_L10": 13,
            "reference_L1000": 1003,
        }
        for index, weight in enumerate((10, 1000))
    ]
    costs = {}
    for arm in (*summarizer.sequence.RUNS, "uniform"):
        costs[arm] = [
            {
                "condition": index,
                "panel_index": 0,
                "cc_weight": weight,
                "L10": None,
                "L1000": None,
                "episodes": [],
            }
            for index, weight in enumerate((10, 1000))
        ]
        reports[f"mcts-{arm}"] = {"counts": {"conditions": 2}, "seconds_committed_batches": 1}
    result = summarizer.summarize(reports, panels, roots, costs)
    assert len(result["metric"]) == len(result["paired_metric"]) == 12
    assert len(result["search_comparisons"]) == len(result["compute"]) == 4
    for item in result["search_comparisons"].values():
        assert set(item) == {"10", "1000"}
        assert item["10"]["strata"]["within"] is None
        assert item["10"]["mean_cost_delta_common_only"] is None
    assert "extra CC" in result["scope"]["resource_caveat"]
    assert result["scope"]["promotion"] == "not_decided"
    rendered = summarizer.markdown(result)
    assert "resource caps are not equal compute" in rendered
    assert "unavailable" in rendered
    reports["cost-resource-envelope"] = {
        "rows": [{"lambda": w, "extra_cc_improvement_not_ruled_out": False} for w in (10, 1000)]
    }
    certified = summarizer.summarize(reports, panels, roots, costs)
    assert certified["resource_envelope"]["10"]["extra_cc_improvement_not_ruled_out"] == 0
    assert "rule out improving the reference witness cost" in summarizer.markdown(certified)
    assert "does not prove optimal costs" in certified["scope"]["resource_caveat"]


def test_behavior_keeps_cost_conditions_separate_and_does_not_label_moves(summarizer):
    from rf_knots.actions import REDUCE
    from rf_knots.mcts_loop import Instance

    instance = Instance("behavior", (1, -1, 1), 2, 1, cap_moves=8, cc_weight=10)
    roots = [
        {"condition": 0, "instance": instance.to_dict()},
        {"condition": 1, "instance": {**instance.to_dict(), "cc_weight": 1000}},
    ]
    rows = [
        {
            "condition": 0,
            "cc_weight": 10,
            "episodes": [
                {
                    "continuation": {
                        "start": {"word": [1, -1, 1]},
                        "end": {"word": [1]},
                        "actions": [instance.spec.encode(REDUCE, 0)],
                        "stop_reason": "horizon",
                    }
                }
            ],
        },
        {"condition": 1, "cc_weight": 1000, "episodes": []},
    ]
    result = summarizer.search_behavior(roots, rows)
    assert result["10"]["continuation_action_kinds"] == {"REDUCE": 1}
    assert result["10"]["mean_length_change_end_minus_start"] == -2
    assert result["10"]["stop_reasons"] == {"horizon": 1}
    assert result["1000"]["mean_length_change_end_minus_start"] is None
    assert "labels" not in result["10"]
    rows[0]["condition"] = 1
    with pytest.raises(ValueError, match="identity/resources"):
        summarizer.search_behavior(roots, rows)


@pytest.mark.parametrize("lower", [0, 1])
def test_certified_regret_requires_exact_optimum_and_keeps_unsolved_censored(summarizer, lower):
    identity = {"q0": 2, "cc": 1, "moves": 3, "excess": 0}
    reference = {
        "panel_index": 0,
        "stratum": "within",
        "fingerprint": [3, 2],
        "identity": identity,
        "lower_cc": lower,
        "bounds": summarizer.reference_bounds(identity, lower, 64),
    }
    audit = {"status": "passed", "references": [reference]}
    roots = [
        {
            "condition": i,
            "panel_index": 0,
            "stratum": "within",
            "fingerprint": [3, 2],
            "instance": {"word": [1, 1, 1], "strands": 2, "cc_weight": w, "cap_moves": 64},
            "reference_L10": 13,
            "reference_L1000": 1003,
        }
        for i, w in enumerate((10, 1000))
    ]
    rows = [
        {
            "condition": i,
            "panel_index": 0,
            "cc_weight": w,
            "solved": i == 0,
            "L10": 15 if i == 0 else None,
            "L1000": 1005 if i == 0 else None,
        }
        for i, w in enumerate((10, 1000))
    ]
    result = summarizer.certified_costs(audit, roots, {"candidate": rows})
    solutions = result["arms"]["candidate"]
    assert len(solutions) == 1
    assert solutions[0]["certified_regret"] == (2 if lower == 1 else None)
    assert solutions[0]["gap_to_lower_bound"] == (2 if lower == 1 else 13)
    assert result["reference_summary"]["10"]["certified_reference_optima"] == lower
    rows[0]["L10"] = 0
    with pytest.raises(ValueError, match="violates certified"):
        summarizer.certified_costs(audit, roots, {"candidate": rows})
    rows[0]["L10"] = 15
    reference["bounds"]["10"]["certified_lower_cost"] += 1
    with pytest.raises(ValueError, match="root/bounds differ"):
        summarizer.certified_costs(audit, roots, {"candidate": rows})


def test_cost_floor_cannot_accept_an_empty_or_failed_audit(summarizer):
    with pytest.raises(ValueError, match="did not pass"):
        summarizer.certified_costs({"status": "failed"}, [], {})
    with pytest.raises(ValueError, match="coverage differs"):
        summarizer.certified_costs({"status": "passed", "references": []}, [{"panel_index": 0}], {})
