"""Calibrated pairwise F loss remains trainable at extreme confidence."""

import torch

from train_joint_f_r2_r4_smoothed_v3 import smoothed_pairwise_loss


def loss_for(logits):
    return smoothed_pairwise_loss(torch.log_softmax(logits, dim=0),
                                  [("good", 1.0)], [("bad", 1.0)],
                                  {"good": 0, "bad": 1}, 1.0, 1.0)


def test_finite_minimum_and_nonzero_extreme_gradients():
    for values in ((0.0, 0.0), (1000.0, -1000.0), (-1000.0, 1000.0)):
        logits = torch.tensor(values, requires_grad=True)
        loss = loss_for(logits)
        assert torch.isfinite(loss) and loss.item() >= 0
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().min().item() > 0.01
    calibrated = loss_for(torch.tensor([4.4, 0.0]))
    overconfident = loss_for(torch.tensor([1000.0, -1000.0]))
    assert calibrated < overconfident


def test_pair_weights_still_normalized():
    equal = torch.log_softmax(torch.zeros(4), dim=0)
    result = smoothed_pairwise_loss(equal, [(0, 1.0), (1, 3.0)],
                                    [(2, 2.0), (3, 2.0)],
                                    {0: 0, 1: 1, 2: 2, 3: 3}, 4.0, 4.0)
    expected = (0.98 * torch.nn.functional.softplus(torch.tensor(0.5))
                + 0.02 * torch.nn.functional.softplus(torch.tensor(-0.5)))
    assert torch.allclose(result, expected)
