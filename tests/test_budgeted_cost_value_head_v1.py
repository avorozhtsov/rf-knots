import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from budgeted_cost_value_head_v1 import (
    BudgetedCostValueHead,
    estimate_local_simulations,
    optimistic_local_simulations,
    value_loss,
)


def test_internal_budget_uses_smoothed_path_share_not_root_remainder():
    assert estimate_local_simulations(101, []) == 101
    assert estimate_local_simulations(101, [([30, 10], [0.5, 0.5], 0)]) == 71
    assert (
        estimate_local_simulations(
            101,
            [([30, 10], [0.5, 0.5], 0), ([7, 3], [0.2, 0.8], 0)],
        )
        == 32.5
    )
    assert estimate_local_simulations(0, []) == 0
    with pytest.raises(ValueError, match="invalid tree"):
        estimate_local_simulations(10, [([2, 1], [0.4, 0.4], 0)])


def test_selection_optimism_is_separate_and_bounded_by_root_remainder():
    expected = estimate_local_simulations(101, [([7, 3], [0.2, 0.8], 0)])
    assert expected == 46
    assert optimistic_local_simulations(101, expected, additive=3) == 49
    assert optimistic_local_simulations(101, expected, relative=0.05) == pytest.approx(48.3)
    assert optimistic_local_simulations(100, 1.1) == 2
    assert optimistic_local_simulations(1, 1, additive=4) == 1
    assert optimistic_local_simulations(0, 0) == 0
    with pytest.raises(ValueError, match="invalid optimistic"):
        optimistic_local_simulations(10, 11)


def test_outputs_are_node_relative_and_within_caps():
    head = BudgetedCostValueHead(embedding_dim=4, width=16)
    encoded = {name: torch.randn(2, 4) for name in ("knot", "layout", "isotopy")}
    result = head(
        encoded,
        cc_left=[0, 3],
        moves_left=[10, 20],
        cc_weight=[10, 1000],
        local_simulations=[16, 128],
    )
    assert result["p_solve"].shape == (2,)
    assert torch.all((result["p_solve"] > 0) & (result["p_solve"] < 1))
    assert result["cc_remaining"][0].item() == 0
    assert 0 < result["cc_remaining"][1].item() < 3
    assert torch.all(result["moves_remaining"] < torch.tensor([10, 20]))


def test_censored_failures_do_not_train_cost_outputs():
    head = BudgetedCostValueHead(embedding_dim=4, width=16)
    encoded = {name: torch.randn(2, 4) for name in ("knot", "layout", "isotopy")}
    result = head(
        encoded,
        cc_left=[2, 3],
        moves_left=[10, 20],
        cc_weight=[10, 1000],
        local_simulations=[16, 128],
    )
    loss = value_loss(
        result,
        solved=[0, 1],
        cc=[float("nan"), 2],
        moves=[float("nan"), 12],
        cc_left=[2, 3],
        moves_left=[10, 20],
    )
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_success_cost_must_fit_remaining_resources():
    head = BudgetedCostValueHead(embedding_dim=4, width=16)
    encoded = {name: torch.randn(1, 4) for name in ("knot", "layout", "isotopy")}
    result = head(
        encoded,
        cc_left=[1],
        moves_left=[10],
        cc_weight=[10],
        local_simulations=[16],
    )
    with pytest.raises(ValueError, match="exceed"):
        value_loss(result, solved=[1], cc=[2], moves=[5], cc_left=[1], moves_left=[10])
