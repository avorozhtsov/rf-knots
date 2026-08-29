from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rf_knots.embedding_metric import (  # noqa: E402
    MetricBatch,
    MetricEmbeddingObjective,
    witness_priority_weight,
)


def test_long_cc01_witnesses_receive_monotone_priority() -> None:
    assert [witness_priority_weight("cc", 1, length) for length in (5, 6, 11, 21)] == [
        1,
        2,
        4,
        8,
    ]
    assert witness_priority_weight("cc", 2, 100) == 1
    assert witness_priority_weight("rm", 1, 100) == 1


def test_interval_objective_accepts_predictions_inside_interval() -> None:
    objective = MetricEmbeddingObjective()
    vector = torch.nn.functional.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    batch = MetricBatch(
        metric=("cc",),
        lower=torch.tensor([0.0]),
        upper=torch.tensor([1.0]),
        witness_semantic_length=torch.tensor([12]),
    )
    loss, diagnostics = objective(
        {"knot": vector, "isotopy": vector},
        {"knot": vector, "isotopy": vector},
        batch,
    )
    assert loss.item() == pytest.approx(0.0)
    assert diagnostics["cc_mean_predicted"].item() == pytest.approx(0.0, abs=1e-6)


def test_exact_positive_distance_penalizes_collapsed_embedding() -> None:
    objective = MetricEmbeddingObjective()
    vector = torch.nn.functional.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    batch = MetricBatch(
        metric=("cc",),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
        witness_semantic_length=torch.tensor([21]),
    )
    loss, _ = objective(
        {"knot": vector, "isotopy": vector},
        {"knot": vector, "isotopy": vector},
        batch,
    )
    assert loss.item() == pytest.approx(1.0)
