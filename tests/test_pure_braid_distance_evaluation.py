import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_pure_braid_distance_panel")


def data():
    records, predictions = [], []
    for i, (group, lo, hi, pred, n) in enumerate(
        ((101, 0, 0, 0, 3), (103, 1, 1, 1, 3), (103, 1, 2, 1, 3), (105, 2, 2, 1, 33))
    ):
        r = {
            "left_fingerprint": [group, 2],
            "interval": {"lower": lo, "upper": hi, "left": {"word": [1] * 100, "strands": n}},
        }
        records.append(r)
        predictions.append(
            {
                "panel_index": i,
                "fingerprint_group": f"{group}:2",
                "lower": lo,
                "upper": hi,
                "prediction": pred,
                "violation": max(lo - pred, pred - hi, 0),
            }
        )
    return predictions, records


def test_exact_zero_cannot_hide_exact_two_error_and_constants_are_group_weighted(evaluator):
    predictions, records = data()
    stats = evaluator.summaries(predictions, records)
    assert stats["all"]["mean_interval_violation"] == 0.25
    assert stats["all"]["equal_group_mean_interval_violation"] == pytest.approx(1 / 3)
    assert stats["positive_lower_bound"]["equal_group_mean_interval_violation"] == 0.5
    assert stats["exact_zero_coordinate"]["mean_interval_violation"] == 0
    assert stats["exact_two"]["mean_interval_violation"] == 1
    assert stats["exact_two"]["constants"]["2"]["mean_interval_violation"] == 0
    assert stats["raw_query_strata"]["strands_only"]["rows"] == 1
    assert stats["raw_query_strata"]["both"] == {"rows": 0, "statistics": None}


@pytest.mark.parametrize("change", ["group", "lower", "order", "nonfinite", "error"])
def test_prediction_coverage_identity_and_numbers_fail_closed(evaluator, change):
    predictions, records = data()
    bad = deepcopy(predictions)
    if change == "group":
        bad[0]["fingerprint_group"] = "other"
    elif change == "lower":
        bad[0]["lower"] = 2
    elif change == "order":
        bad.reverse()
    elif change == "nonfinite":
        bad[0]["prediction"] = float("nan")
    else:
        bad[0]["violation"] = 99
    with pytest.raises(ValueError):
        evaluator.summaries(bad, records)
