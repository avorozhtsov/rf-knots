"""The repaired F objective has a finite lower bound and useful gradient."""

import torch

from train_joint_f_r2_r4_bounded_v2 import pairwise_margin_loss


def test_pairwise_margin_is_bounded_and_rewards_positive_action():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = pairwise_margin_loss(torch.log_softmax(logits, dim=0),
                                [("good", 1.0)], [("bad", 1.0)],
                                {"good": 0, "bad": 1}, 1.0, 1.0)
    assert 0 < loss.item() < 2
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    separated = torch.tensor([1000.0, -1000.0])
    lower = pairwise_margin_loss(torch.log_softmax(separated, dim=0),
                                 [("good", 1.0)], [("bad", 1.0)],
                                 {"good": 0, "bad": 1}, 1.0, 1.0)
    assert 0 <= lower.item() < 1e-6


def test_pair_weights_are_normalized():
    logp = torch.log_softmax(torch.zeros(4), dim=0)
    loss = pairwise_margin_loss(logp, [(0, 1.0), (1, 3.0)],
                                [(2, 2.0), (3, 2.0)],
                                {0: 0, 1: 1, 2: 2, 3: 3}, 4.0, 4.0)
    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(0.5)))
