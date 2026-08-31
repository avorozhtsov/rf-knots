import sqlite3

import pytest
import torch

from rf_knots.embedding_route_retrieval import (
    deterministic_rank,
    fit_ridge,
    predict_ridge,
    regression_metrics,
    summarize_ranks,
    validate_snapshot_contract,
)


def _metadata_db(path, values) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.executemany("INSERT INTO meta VALUES (?,?)", values.items())
    connection.commit()
    connection.close()


def test_snapshot_contract_rejects_mismatched_graph(tmp_path) -> None:
    proof = tmp_path / "proof.sqlite"
    _metadata_db(proof, {"schema": "proof"})
    pairs = tmp_path / "pairs.sqlite"
    _metadata_db(pairs, {"proof_sha256": "wrong"})
    identification = tmp_path / "identification.sqlite"
    _metadata_db(identification, {"graph_snapshot_sha256": "wrong"})
    lower = tmp_path / "lower.sqlite"
    _metadata_db(lower, {"schema": "lower"})

    with pytest.raises(ValueError, match="pairs and proof graph SHA mismatch"):
        validate_snapshot_contract(
            proof_path=proof,
            pairs_path=pairs,
            identification_path=identification,
            lower_bounds_path=lower,
        )


def test_deterministic_rank_uses_node_id_for_ties() -> None:
    scores = torch.tensor([0.5, 0.8, 0.8, 0.1])
    nodes = torch.tensor([9, 7, 3, 1])
    assert deterministic_rank(scores, nodes, 2) == 1
    assert deterministic_rank(scores, nodes, 1) == 2
    assert deterministic_rank(scores, nodes, 0) == 3


def test_rank_summary_includes_random_reference() -> None:
    summary = summarize_ranks([1, 20], [100, 100])
    assert summary["recall_at_1"] == pytest.approx(0.5)
    assert summary["recall_at_10"] == pytest.approx(0.5)
    assert summary["random_expected_recall_at_10"] == pytest.approx(0.1)


def test_ridge_probe_learns_a_linear_signal() -> None:
    features = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    targets = 2 * features[:, 0] + 3
    model = fit_ridge(features, targets, alpha=0.0)
    predictions = predict_ridge(model, features)
    metrics = regression_metrics(targets, predictions)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["pairwise_order_accuracy"] == pytest.approx(1.0)
