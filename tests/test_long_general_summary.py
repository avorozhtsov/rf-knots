import importlib
from copy import deepcopy
from pathlib import Path

import pytest
from test_long_general_evaluation import Toy

from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("summarize_long_general_experiment")


def metric_fixture():
    scoring = importlib.import_module("evaluate_long_general_distances")
    records = []
    for i, (lo, hi, length, strands) in enumerate(
        ((0, 0, 48, 12), (1, 1, 49, 13), (2, 2, 193, 33), (0, 1, 385, 34))
    ):
        endpoint = dict(word=[1] * length, strands=strands)
        records.append(
            dict(
                index=i,
                interval=dict(left=endpoint, right=endpoint, lower=lo, upper=hi),
                left_fingerprint=[3, 2],
                right_fingerprint=[3, 2],
            )
        )
    result = scoring.enrich(
        scoring.score_records(Toy().eval(), MetricEmbeddingObjective(), records), records
    )
    return result, records


def test_summary_recomputes_complete_metric_views(module):
    report, records = metric_fixture()
    verified = module.validate_metric_result(report, records)
    assert verified["rows"] == 4 and verified["certified_positive"]["rows"] == 2
    assert verified["certified_positive"]["constant1"]["mean_interval_violation"] == 0.5


@pytest.mark.parametrize(
    "field", ["size", "source", "positive", "strata", "constant", "group", "ranking"]
)
def test_summary_rejects_tampered_metric_statistics(module, field):
    report, records = metric_fixture()
    if field == "size":
        report["predictions"][0]["maximum_length"] += 1
    elif field == "source":
        report["predictions"][0]["source_index"] += 1
    elif field == "positive":
        report["certified_positive"]["rows"] += 1
    elif field == "strata":
        report["by_maximum_strands"].pop("34+")
    elif field == "constant":
        report["constant_baselines"]["1"]["mean_interval_violation"] += 1
    elif field == "group":
        report["by_fingerprint"]["group_count"] += 1
    else:
        report["ranking"]["comparable_pairs"] += 1
    with pytest.raises(ValueError):
        module.validate_metric_result(report, records)


def test_complete_matrix_not_partial_success(module):
    planned = module.jobs(Path("out"), Path("run"), Path("pgx"))
    state = dict(
        status="scored",
        jobs=planned,
        completed=[{k: j[k] for k in ("name", "output")} for j in planned],
    )
    module.require_complete_matrix(state, planned)
    for count in (0, 8, 17):
        with pytest.raises(ValueError):
            module.require_complete_matrix(
                {**state, "completed": state["completed"][:count]}, planned
            )
    bad = deepcopy(state)
    bad["completed"][0], bad["completed"][1] = bad["completed"][1], bad["completed"][0]
    with pytest.raises(ValueError):
        module.require_complete_matrix(bad, planned)


def test_record_and_group_error_deltas_are_distinct(module):
    a = [
        dict(
            panel_index=i, fingerprint_group="a" if i < 2 else "b", lower=1, upper=1, violation=1.0
        )
        for i in range(3)
    ]
    b = [{**r, "violation": v} for r, v in zip(a, (0.0, 0.0, 2.0), strict=True)]
    compared = module.compare_metrics(a, b)
    assert compared["mean_record_error_delta"] == pytest.approx(-1 / 3)
    assert compared["mean_group_error_delta"] == 0
    assert module.compare_metrics([], [])["mean_record_error_delta"] is None
    b[0]["lower"] = 0
    with pytest.raises(ValueError):
        module.compare_metrics(a, b)
