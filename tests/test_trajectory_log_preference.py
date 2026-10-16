import copy
import importlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def objective(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("trajectory_log_preference")


def test_reference_gradient_matches_original_local_direction(objective):
    logp = torch.tensor([-100.0, -1.0, -30.0], requires_grad=True)
    coefficients = torch.tensor([1.0, -0.5, -0.5])
    loss, score, mass = objective.trajectory_preference(logp, logp.detach(), coefficients)
    assert score == 0 and mass == 2
    torch.testing.assert_close(torch.autograd.grad(loss, logp)[0], -coefficients)


@pytest.mark.parametrize("delta", [-1000.0, -42.0, 0.0, 42.0, 1000.0])
def test_extreme_log_ratios_are_finite_and_wrong_preferences_still_get_gradient(objective, delta):
    # A loser whose log probability increased is an adverse preference change.
    logp = torch.tensor([-1200.0 + delta], requires_grad=True)
    coefficient = torch.tensor([-1.0])
    loss, _, _ = objective.trajectory_preference(logp, torch.tensor([-1200.0]), coefficient)
    gradient = torch.autograd.grad(loss, logp)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient.abs().max() <= 2
    if delta > 0:
        assert gradient.item() >= 1


def test_cancelled_common_prefix_has_no_influence(objective):
    reference = torch.tensor([-1.0, -2.0, -3.0])
    a = torch.tensor([-20.0, -3.0, -1.0], requires_grad=True)
    b = torch.tensor([-2000.0, -3.0, -1.0], requires_grad=True)
    coefficients = torch.tensor([0.0, 1.0, -1.0])
    left, _, _ = objective.trajectory_preference(a, reference, coefficients)
    right, _, _ = objective.trajectory_preference(b, reference, coefficients)
    assert left == right
    assert torch.autograd.grad(left, a)[0][0] == 0


def test_fully_cancelled_group_is_zero(objective):
    logp = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, score, mass = objective.trajectory_preference(logp, logp.detach(), torch.zeros_like(logp))
    assert loss == score == mass == 0
    assert not torch.autograd.grad(loss, logp)[0].any()


@pytest.mark.parametrize("microbatch", [1, 2, 3])
def test_two_pass_matches_full_group_parameter_gradient(objective, monkeypatch, microbatch):
    torch.manual_seed(95)
    net = torch.nn.Linear(3, 2)
    other = copy.deepcopy(net)
    requests = [(torch.tensor([1.0, float(i), -1.0]), None, (0, 1)) for i in range(3)]
    coefficients = torch.tensor([1.0, -0.5, -0.5])
    group = [(request, i % 2, float(coefficients[i])) for i, request in enumerate(requests)]

    def probabilities(model, rows):
        return model(torch.stack([row[0] for row in rows])).log_softmax(-1)

    monkeypatch.setattr(objective, "policy_log_probabilities", probabilities)
    selected = probabilities(net, requests)[torch.arange(3), [0, 1, 0]]
    reference = selected.detach() + torch.tensor([0.3, -0.2, 0.5])
    full, _, _ = objective.trajectory_preference(selected, reference, coefficients)
    (0.01 * full).backward()
    result = objective.relative_backward(other, (group, reference), microbatch=microbatch)
    assert result["relative_loss"] == pytest.approx(float(full.detach()), abs=1e-6)
    for a, b in zip(net.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)
