import importlib
import random
from pathlib import Path

import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.rollout_value import clipped_relative_loss


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("raster_pair_training")


class FixedDraw:
    def __init__(self, index):
        self.index = index

    def randrange(self, count):
        assert self.index < count
        return self.index


def test_sampled_relative_sum_has_correct_expectation_and_gradient(helper):
    values = torch.tensor([-1.2, -0.3, -0.8], requires_grad=True)
    old = torch.tensor([-1.1, -0.4, -0.9])
    coefficients = torch.tensor([1.0, -0.5, -0.5])
    full = clipped_relative_loss(values, old, coefficients)
    expected_grad = torch.autograd.grad(full, values)[0]
    group = [(i, 0, float(c)) for i, c in enumerate(coefficients)]
    sampled_losses = []
    for i in range(3):
        (sampled, reference), indices = helper.sample_relative(FixedDraw(i), (group, old), 1)
        assert indices == [i]
        sampled_losses.append(
            clipped_relative_loss(values[indices], reference, torch.tensor([sampled[0][2]]))
        )
    estimate = torch.stack(sampled_losses).mean()
    torch.testing.assert_close(estimate, full)
    torch.testing.assert_close(torch.autograd.grad(estimate, values)[0], expected_grad)


def test_replacement_sampling_preserves_repetitions_and_fixed_reference(helper):
    group = [(0, 0, 1.0), (1, 0, -1.0)]
    old = torch.tensor([-1.0, -2.0])
    (selected, reference), indices = helper.sample_relative(FixedDraw(1), (group, old), 8)
    assert indices == [1] * 8 and len(selected) == 8
    assert sum(row[2] for row in selected) == -2.0
    assert reference.tolist() == [-2.0] * 8
    with pytest.raises(ValueError):
        helper.sample_relative(FixedDraw(0), (group, old), 0)


def test_model_independent_draw_digest_and_rng_resume(helper):
    class Sampler:
        def sample(self, rng, count):
            return [
                helper.pair_row(
                    rng.randrange(100), "test", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6
                )
                for _ in range(count)
            ]

    instance = M.Instance("fixture", (1, 1, 1), 2, 1)
    state = M.SearchState(instance.word, 2, 0, 0)
    request = (instance, state, (instance.spec.encode(CROSSING_CHANGE, 0),))
    positives = [[(request, (0,))]]
    group = [(request, 0, 1.0), (request, 0, -1.0)]
    a, b = random.Random(94), random.Random(94)
    sampler = Sampler()
    first = helper.draw_batch(
        a, sampler, sampler, sampler, positives, [(group, torch.tensor([-1.0, -2.0]))]
    )
    second = helper.draw_batch(
        b, sampler, sampler, sampler, positives, [(group, torch.tensor([-3.0, -4.0]))]
    )
    assert len(first[0]) == 32 and len(first[1]) == 8 and len(first[2][0]) == 8
    assert first[3] == second[3] and a.getstate() == b.getstate()
    restored = random.Random()
    restored.setstate(a.getstate())
    args = (sampler, sampler, sampler, positives, [(group, torch.tensor([-1.0, -2.0]))])
    assert helper.draw_batch(a, *args)[3] == helper.draw_batch(restored, *args)[3]
