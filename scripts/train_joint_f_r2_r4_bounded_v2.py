"""Bounded exact-context F repair of frozen R2/R4 trainer v1."""

import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import train_joint_f_r2_r4_v1 as base
from pretrained_raster_policy import policy_log_probabilities
from rf_knots.site_proposals import proposal_candidates


MARGIN = 0.5


def pairwise_margin_loss(logp, positive, negative, positions, pos_mass, neg_mass):
    result = None
    for positive_action, positive_weight in positive:
        for negative_action, negative_weight in negative:
            pair = F.softplus(MARGIN + logp[positions[negative_action]]
                              - logp[positions[positive_action]])
            weight = positive_weight * negative_weight / (pos_mass * neg_mass)
            result = pair * weight if result is None else result + pair * weight
    return result


def bounded_f_loss(net, instance, coefficients):
    contexts = {}
    for (state, action), value in coefficients.items():
        if value:
            contexts.setdefault(state, {})[action] = value
    if not contexts:
        raise ValueError("empty F signal")
    positive_mass = sum(value for actions in contexts.values()
                        for value in actions.values() if value > 0)
    if positive_mass <= 0:
        raise ValueError("F signal has no positive mass")
    total = None
    pair_count = 0
    for state, signed in contexts.items():
        positive = [(action, value) for action, value in signed.items() if value > 0]
        negative = [(action, -value) for action, value in signed.items() if value < 0]
        if not positive or not negative:
            raise ValueError("F context lost one side")
        pos_mass = sum(value for _, value in positive)
        neg_mass = sum(value for _, value in negative)
        if not math.isclose(pos_mass, neg_mass, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("F context coefficient masses differ")
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | set(signed)))
        logp = policy_log_probabilities(net, [(instance, state, actions)])[0]
        positions = {action: index for index, action in enumerate(actions)}
        context_loss = pairwise_margin_loss(logp, positive, negative, positions,
                                            pos_mass, neg_mass)
        pair_count += len(positive) * len(negative)
        context_loss = context_loss * (pos_mass / positive_mass)
        total = context_loss if total is None else total + context_loss
    if total is None or not torch.isfinite(total) or total.item() < 0:
        raise ValueError("nonfinite or negative bounded F loss")
    return total, pair_count


def train_step(net, objective, head, optimizer, group, step):
    instance, traces, coefficients = base.group_data(group)
    parameters = [*net.parameters(), *objective.parameters(), *head.parameters()]
    optimizer.zero_grad(set_to_none=True)
    f_loss, pair_count = bounded_f_loss(net, instance, coefficients)
    f_loss.backward()
    f_grads = base.gradient_snapshot(parameters)
    f_norm = base.encoder_norm(f_grads, parameters, net.encoder)
    if not math.isfinite(f_norm) or f_norm <= 1e-12:
        raise ValueError("zero/nonfinite F gradient: skip auxiliary update")
    best = min(traces, key=lambda t: (t["cost"], tuple(a for _, a in t["rows"])))
    optimizer.zero_grad(set_to_none=True)
    aux = base.auxiliary_loss(net, head, instance, best)
    aux.backward()
    aux_grads = base.gradient_snapshot(parameters)
    aux_norm = base.encoder_norm(aux_grads, parameters, net.encoder)
    aux_cos = base.encoder_cosine(f_grads, aux_grads, parameters, net.encoder)
    reference = traces[0]
    optimizer.zero_grad(set_to_none=True)
    rehearsal, imitation, distance = base.rehearsal_loss(
        net, objective, instance, reference, step)
    rehearsal.backward()
    rehearsal_grads = base.gradient_snapshot(parameters)
    rehearsal_norm = base.encoder_norm(rehearsal_grads, parameters, net.encoder)
    rehearsal_cos = base.encoder_cosine(f_grads, rehearsal_grads, parameters, net.encoder)
    aux_scale = min(1.0, base.CONFIG["max_aux_encoder_ratio"] * f_norm / aux_norm) if aux_norm else 0.0
    rehearsal_scale = (min(1.0, base.CONFIG["max_rehearsal_encoder_ratio"] * f_norm / rehearsal_norm)
                       if rehearsal_norm else 0.0)
    optimizer.zero_grad(set_to_none=True)
    for parameter, f_grad, aux_grad, rehearsal_grad in zip(
        parameters, f_grads, aux_grads, rehearsal_grads, strict=True
    ):
        pieces = [g * scale for g, scale in ((f_grad, 1.0),
                                             (aux_grad, aux_scale),
                                             (rehearsal_grad, rehearsal_scale))
                  if g is not None and scale]
        if pieces:
            parameter.grad = sum(pieces[1:], pieces[0])
    combined_norm = torch.nn.utils.clip_grad_norm_(
        parameters, base.CONFIG["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"F_loss": float(f_loss.detach()), "aux_loss": float(aux.detach()),
            "imitation_loss": imitation, "distance_loss": distance,
            "F_encoder_grad_norm": f_norm, "aux_encoder_grad_norm": aux_norm,
            "rehearsal_encoder_grad_norm": rehearsal_norm,
            "aux_F_cosine": aux_cos, "rehearsal_F_cosine": rehearsal_cos,
            "aux_scale": aux_scale, "rehearsal_scale": rehearsal_scale,
            "combined_grad_norm": float(combined_norm),
            "sampled_F_terms": len(coefficients), "F_pairs": pair_count}


if __name__ == "__main__":
    base.PROTOCOL = Path("research/305-f-bounded-pairwise-repair-v1.md")
    base.CONFIG["F_loss"] = "exact-context-pairwise-softplus"
    base.CONFIG["F_margin"] = MARGIN
    base.CONFIG["v2_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base.train_step = train_step
    base.main()
