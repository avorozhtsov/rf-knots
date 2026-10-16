"""Training and terminal-cost-search interfaces for the unified raster network.

No scalar value is invented; legacy value-consuming MCTS fails explicitly.
No persistent neural cache survives a word or parameter update.
"""

from dataclasses import dataclass

import numpy as np
import torch
from mcts_training_supervision import set_valued_policy_loss

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior
from rf_knots.rollout_value import clipped_relative_loss
from rf_knots.site_head import SiteHead


@dataclass(frozen=True)
class PriorOnlyResult:
    priors: np.ndarray

    @property
    def value(self):
        raise RuntimeError("This network has no value prediction; use terminal-only cost MCTS")


def validate_requests(requests):
    for instance, state, actions in requests:
        if (
            not 0 <= state.used_cc <= instance.cap_crossing_changes
            or not 0 <= state.used_moves < instance.cap_moves
            or not actions
            or len(set(actions)) != len(actions)
        ):
            raise ValueError("invalid policy resource context or action pool")
        if any(
            not reference.is_legal(
                instance.spec,
                state.word,
                state.strands,
                a,
                state.used_cc < instance.cap_crossing_changes,
            )
            for a in actions
        ):
            raise ValueError("illegal policy action")


def policy_log_probabilities(net, requests, *, heads=None):
    if not requests:
        raise ValueError("empty training policy batch")
    validate_requests(requests)
    device = next(net.parameters()).device
    batch = M.build_batch(requests, device, budget_dim=net.budget_dim)
    if heads is not None:
        heads = torch.as_tensor(heads, device=device)
        if heads.shape != (len(requests),) or heads.dtype not in (torch.int32, torch.int64):
            raise ValueError("require one integer head position per request")
    output = net(**batch, heads=heads)
    if not torch.equal(output["effective"], batch["site_mask"]):
        raise ValueError("network masked an audited action")
    if not torch.isfinite(output["logits"][output["effective"]]).all():
        raise ValueError("nonfinite policy logits")
    return SiteHead.masked_log_policy(output["logits"], output["effective"])


class RasterPolicyEvaluator:
    batched = True

    def __init__(self, net, *, microbatch=2):
        if microbatch < 1:
            raise ValueError("positive microbatch required")
        self.net, self.microbatch = net, microbatch
        self.calls = self.forwards = self.action_scores = 0

    @torch.no_grad()
    def evaluate(self, requests, *, heads=None):
        if not requests:
            return []
        if heads is not None and len(heads) != len(requests):
            raise ValueError("head/request coverage differs")
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][1].word))
        results = [None] * len(requests)
        was_training = self.net.training
        self.net.eval()
        try:
            for start in range(0, len(order), self.microbatch):
                indices = order[start : start + self.microbatch]
                chunk = [requests[i] for i in indices]
                chunk_heads = None if heads is None else [heads[i] for i in indices]
                logp = policy_log_probabilities(self.net, chunk, heads=chunk_heads)
                probabilities = logp.exp().cpu().numpy()
                self.forwards += 1
                self.calls += len(chunk)
                self.action_scores += sum(len(r[2]) for r in chunk)
                for row, index in enumerate(indices):
                    p = probabilities[row, : len(requests[index][2])].astype(np.float64)
                    if not np.isfinite(p).all() or p.sum() <= 0:
                        raise ValueError("invalid policy probability; no uniform fallback")
                    results[index] = PriorOnlyResult(p / p.sum())
        finally:
            self.net.train(was_training)
        return results


class RawKnotDistance:
    """Raw diagram embeddings, refreshed per call; no invariant/normalized cache."""

    def __init__(self, net, objective, *, microbatch=2):
        if microbatch < 1:
            raise ValueError("positive microbatch required")
        self.net, self.objective, self.microbatch = net, objective, microbatch
        self.evaluations = self.forwards = 0

    @torch.no_grad()
    def __call__(self, states):
        if not states:
            return []
        device = next(self.net.parameters()).device
        keys = [(tuple(s.word), s.strands) for s in states]
        unique = list(dict.fromkeys([((), 1), *keys]))
        embeddings = {}
        was_training = self.net.training
        self.net.eval()
        try:
            for begin in range(0, len(unique), self.microbatch):
                chunk = unique[begin : begin + self.microbatch]
                padded = max(1, max(len(word) for word, _ in chunk))
                words = torch.tensor(
                    [list(w) + [0] * (padded - len(w)) for w, _ in chunk],
                    dtype=torch.long,
                    device=device,
                )
                lengths = torch.tensor([len(w) for w, _ in chunk], device=device)
                strands = torch.tensor([n for _, n in chunk], device=device)
                vectors = self.net.encoder(words, lengths, strands)["knot"]
                if not torch.isfinite(vectors).all():
                    raise ValueError("nonfinite knot embedding")
                embeddings.update(zip(chunk, vectors, strict=True))
                self.evaluations += len(chunk)
                self.forwards += 1
            anchor = embeddings[((), 1)]
            cosine = torch.stack([(embeddings[key] * anchor).sum() for key in keys])
            distances = self.objective.scale("cc") * self.objective.cc_distance(cosine.clamp(-1, 1))
            if not torch.isfinite(distances).all() or (distances < 0).any():
                raise ValueError("invalid learned distance")
            return distances.cpu().tolist()
        finally:
            self.net.train(was_training)


class RasterDistancePrior(EmbeddingDistancePrior):
    def evaluate(self, requests):
        return [PriorOnlyResult(result.priors) for result in super().evaluate(requests)]


@torch.no_grad()
def reference_preferences(net, groups, *, microbatch=2):
    if microbatch < 1:
        raise ValueError("positive microbatch required")
    was_training = net.training
    net.eval()
    result = []
    try:
        for group in groups:
            logs = []
            for begin in range(0, len(group), microbatch):
                chunk = group[begin : begin + microbatch]
                logp = policy_log_probabilities(net, [row[0] for row in chunk])
                logs.extend(
                    logp[torch.arange(len(chunk)), [row[1] for row in chunk]].cpu().tolist()
                )
            result.append((group, torch.tensor(logs)))
    finally:
        net.train(was_training)
    return result


def policy_backward(
    net, positives, relative, *, microbatch=2, positive_weight=0.1, relative_weight=0.01, clip=0.2
):
    """Same set-positive mean and unnormalized relative sum across microbatches."""
    if not positives or microbatch < 1 or positive_weight < 0 or relative_weight < 0:
        raise ValueError("invalid policy loss inputs")
    positive_loss = relative_loss = 0.0
    for begin in range(0, len(positives), microbatch):
        chunk = positives[begin : begin + microbatch]
        logp = policy_log_probabilities(net, [row[0] for row in chunk])
        mask = torch.zeros_like(logp, dtype=torch.bool)
        for i, (_, good) in enumerate(chunk):
            if not good or any(index < 0 or index >= len(chunk[i][0][2]) for index in good):
                raise ValueError("invalid positive action indices")
            mask[i, list(good)] = True
        loss = set_valued_policy_loss(logp, mask) * (len(chunk) / len(positives))
        (positive_weight * loss).backward()
        positive_loss += float(loss.detach())
    if relative is not None:
        group, old = relative
        if old.shape != (len(group),) or not torch.isfinite(old).all():
            raise ValueError("reference probabilities do not cover relative group")
        for begin in range(0, len(group), microbatch):
            chunk = group[begin : begin + microbatch]
            logp = policy_log_probabilities(net, [row[0] for row in chunk])
            if any(not 0 <= row[1] < len(row[0][2]) for row in chunk):
                raise ValueError("relative action index outside pool")
            selected = logp[torch.arange(len(chunk)), [row[1] for row in chunk]]
            coefficients = torch.tensor([row[2] for row in chunk], device=logp.device)
            loss = clipped_relative_loss(
                selected, old[begin : begin + len(chunk)].to(logp.device), coefficients, clip
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite relative loss")
            (relative_weight * loss).backward()
            relative_loss += float(loss.detach())
    return {"positive_loss": positive_loss, "relative_loss": relative_loss}
