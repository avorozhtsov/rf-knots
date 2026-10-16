import importlib
from pathlib import Path

import pytest
import torch

from rf_knots.embedding_metric import MetricEmbeddingObjective


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("evaluate_raster_log_distances")


def test_rank_only_certified_disjoint_intervals(module):
    rows = [
        {"lower": 0, "upper": 0, "prediction": 0},
        {"lower": 1, "upper": 1, "prediction": 1},
        {"lower": 1, "upper": 2, "prediction": 1},
        {"lower": 2, "upper": 2, "prediction": 1},
    ]
    r = module.interval_ranking(rows)
    assert (r["comparable_pairs"], r["correct"], r["ties"]) == (4, 3, 1)
    assert r["accuracy_half_credit_ties"] == 0.875
    assert module.interval_ranking(rows[2:])["accuracy_half_credit_ties"] is None


def test_score_all_rows_and_restore_mode(module):
    class Net(torch.nn.Module):
        def encoder(self, words, lengths, strands):
            assert words.shape[0] == 1 and not torch.is_grad_enabled()
            return {"knot": torch.tensor([[1.0, 0.0]])}

    net = Net().train()
    records = []
    for i in range(3):
        records.append(
            {
                "interval": {
                    "left": {"word": [], "strands": 1},
                    "right": {"word": [1], "strands": 2},
                    "lower": i,
                    "upper": i,
                },
                "left_fingerprint": [1, 0],
                "right_fingerprint": [1, 0],
            }
        )
    result = module.score_records(net, MetricEmbeddingObjective(), records)
    assert net.training and result["encoded_endpoints"] == 6
    assert result["statistics"]["rows"] == 3
    assert result["statistics"]["mean_interval_violation"] == 1
    assert set(result["statistics"]["exact"]) == {"0", "1", "2"}
    assert [p["panel_index"] for p in result["predictions"]] == [0, 1, 2]
    with pytest.raises(ValueError, match="empty"):
        module.score_records(net, MetricEmbeddingObjective(), [])


def test_nonfinite_scoring_fails_and_restores_mode(module):
    class Net(torch.nn.Module):
        def encoder(self, *args):
            return {"knot": torch.tensor([[float("nan"), 0.0]])}

    net = Net().train()
    endpoint = {"word": [], "strands": 1}
    with pytest.raises(ValueError, match="nonfinite"):
        module.score_records(
            net, MetricEmbeddingObjective(), [{"interval": {"left": endpoint, "right": endpoint}}]
        )
    assert net.training
