"""Whole-trajectory log-space preference with shared-prefix cancellation intact.

This is a proposed replacement objective, NOT a change to the frozen94 run.
Coefficients must already be aggregated across identical state/action contexts.
"""

import torch
from pretrained_raster_policy import policy_log_probabilities


def trajectory_preference(logp, reference, coefficients):
    """2*C*softplus(-sum(c * log(pi/pi_ref))/C), C=sum(abs(c)).

    One common scale for the full cancelled comparison, not independent trajectory
    length normalization. At the reference policy its logp gradient is exactly -c;
    everywhere its magnitude is at most 2*abs(c). No importance-ratio exponential
    or hard clamp can create the old enormous constant-loss plateau.
    """
    reference = reference.to(logp)
    coefficients = coefficients.to(logp)
    if (
        logp.ndim != 1
        or reference.shape != logp.shape
        or coefficients.shape != logp.shape
        or not all(torch.isfinite(x).all() for x in (logp, reference, coefficients))
    ):
        raise ValueError("require finite matching one-dimensional preference vectors")
    mass = coefficients.abs().sum()
    score = (coefficients * (logp - reference)).sum()
    loss = 2 * mass * torch.nn.functional.softplus(-score / mass) if mass > 0 else score * 0
    return loss, score, mass


def relative_backward(net, relative, *, microbatch=1, weight=0.01):
    """Exact full-group gradient in two deterministic, bounded-memory passes."""
    if microbatch < 1 or weight < 0:
        raise ValueError("invalid relative-backward configuration")
    if any(
        isinstance(
            m,
            (
                torch.nn.modules.batchnorm._BatchNorm,
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            ),
        )
        for m in net.modules()
    ):
        raise ValueError("two-pass backward requires a deterministic stateless network")
    group, old = relative
    if not group or old.shape != (len(group),):
        raise ValueError("require one complete cancelled preference group")

    def selected(chunk):
        logp = policy_log_probabilities(net, [row[0] for row in chunk])
        if any(not 0 <= row[1] < len(row[0][2]) for row in chunk):
            raise ValueError("relative action outside candidate pool")
        return logp[torch.arange(len(chunk)), [row[1] for row in chunk]]

    with torch.no_grad():
        logs = torch.cat(
            [selected(group[i : i + microbatch]) for i in range(0, len(group), microbatch)]
        )
    leaf = logs.detach().requires_grad_()
    coefficients = leaf.new_tensor([row[2] for row in group])
    loss, score, mass = trajectory_preference(leaf, old, coefficients)
    derivative = torch.autograd.grad(loss, leaf)[0]
    if not torch.isfinite(loss) or (derivative.abs() > 2 * coefficients.abs() + 1e-6).any():
        raise ValueError("preference gradient bound failed")
    max_error = 0.0
    for start in range(0, len(group), microbatch):
        current = selected(group[start : start + microbatch])
        expected = logs[start : start + len(current)]
        torch.testing.assert_close(current.detach(), expected, rtol=1e-5, atol=5e-5)
        max_error = max(max_error, float((current.detach() - expected).abs().max()))
        (weight * (current * derivative[start : start + len(current)]).sum()).backward()
    return {
        "relative_loss": float(loss.detach()),
        "relative_log_score": float(score.detach()),
        "relative_coefficient_mass": float(mass),
        "relative_states": len(group),
        "two_pass_logp_max_abs_error": max_error,
    }
