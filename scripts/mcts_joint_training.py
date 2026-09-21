"""Joint policy/metric training helpers; scalar value is deliberately not a target."""

import hashlib
import json
from pathlib import Path

import torch
from balanced_metric_sampling import GroupBalancedMetricSampler, prepare_rows
from build_mcts_training_supervision import digest, short_u_views
from mcts_training_supervision import set_valued_policy_loss
from run_strand_graph_pilot import load_frozen_corpus, pair_row

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.evidence import UnknotWitness
from rf_knots.rollout_value import clipped_relative_loss
from rf_knots.site_head import SiteHead


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_from_dict(row):
    return M.SearchState(tuple(row["word"]), row["strands"], row["used_cc"], row["used_moves"])


def policy_forward(net, requests):
    batch = M.build_batch(requests, "cpu", budget_dim=net.budget_dim)
    logits, _, effective = net(**batch)
    if not torch.equal(effective, batch["site_mask"]):
        raise ValueError("network masked an audited training action")
    if not torch.isfinite(logits[effective]).all():
        raise ValueError("nonfinite training policy logits")
    return SiteHead.masked_log_policy(logits, effective)


def load_policy_targets(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("supervision preparation incomplete")
    positives, preferences = [], []
    for item in manifest["conditions"]:
        path = directory / item["file"]
        if sha(path) != item["sha256"]:
            raise ValueError("policy target shard changed")
        data = json.loads(path.read_text())
        if data["condition"] != len(positives):
            raise ValueError("policy condition coverage differs")
        instance = M.Instance(**{**data["instance"], "word": tuple(data["instance"]["word"])})
        states, group = {}, []
        for row in data["positives"]:
            state = state_from_dict(row["state"])
            actions = tuple(row["actions"])
            good = tuple(row["positive_actions"])
            if (
                not 0 <= state.used_moves < instance.cap_moves
                or not 0 <= state.used_cc <= instance.cap_crossing_changes
                or not actions
                or not good
                or len(set(actions)) != len(actions)
                or len(set(good)) != len(good)
                or state in states
                or not set(good) <= set(actions)
            ):
                raise ValueError("invalid positive action coverage/resources")
            if any(
                not reference.is_legal(
                    instance.spec,
                    state.word,
                    state.strands,
                    action,
                    state.used_cc < instance.cap_crossing_changes,
                )
                for action in actions
            ):
                raise ValueError("illegal action in training pool")
            request = (instance, state, actions)
            states[state] = request
            group.append((request, tuple(actions.index(a) for a in good)))
        if not group:
            raise ValueError("empty positive condition")
        positives.append(group)
        relative = []
        for row in data["preferences"]:
            state = state_from_dict(row["state"])
            request = states[state]
            if tuple(row["actions"]) != request[2] or row["coefficient_denominator"] <= 0:
                raise ValueError("preference pool or coefficient differs")
            relative.append(
                (
                    request,
                    request[2].index(row["action"]),
                    row["coefficient_numerator"] / row["coefficient_denominator"],
                )
            )
        if relative:
            preferences.append(relative)
    return positives, preferences


@torch.no_grad()
def reference_preferences(net, groups, micro_batch=64):
    result = []
    for group in groups:
        probabilities = []
        for begin in range(0, len(group), micro_batch):
            chunk = group[begin : begin + micro_batch]
            logp = policy_forward(net, [r[0] for r in chunk])
            probabilities.extend(logp[torch.arange(len(chunk)), [r[1] for r in chunk]].tolist())
        result.append((group, torch.tensor(probabilities)))
    return result


def policy_backward(
    net, positives, relative, *, positive_weight=0.1, relative_weight=0.01, micro_batch=64, clip=0.2
):
    logp = policy_forward(net, [r[0] for r in positives])
    mask = torch.zeros_like(logp, dtype=torch.bool)
    for row, (_, indices) in enumerate(positives):
        mask[row, list(indices)] = True
    loss = set_valued_policy_loss(logp, mask)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite set-valued policy loss")
    (positive_weight * loss).backward()
    relative_loss = 0.0
    if relative is not None:
        group, old = relative
        for begin in range(0, len(group), micro_batch):
            chunk = group[begin : begin + micro_batch]
            logp = policy_forward(net, [r[0] for r in chunk])
            selected = logp[torch.arange(len(chunk)), [r[1] for r in chunk]]
            coefficients = torch.tensor([r[2] for r in chunk])
            loss_r = clipped_relative_loss(
                selected, old[begin : begin + len(chunk)], coefficients, clip
            )
            if not torch.isfinite(loss_r):
                raise ValueError("nonfinite relative preference loss")
            (relative_weight * loss_r).backward()
            relative_loss += float(loss_r.detach())
    return {"positive_loss": float(loss.detach()), "relative_loss": relative_loss}


def load_metric_samplers(corpus_path, supervision_path):
    raw, corpus = load_frozen_corpus(corpus_path)
    old_rows = prepare_rows(corpus["train"][0], raw["train"], bounds="determinant")
    original = GroupBalancedMetricSampler(old_rows, raw["train"])
    index = json.loads((supervision_path / "short-u-views.json").read_text())
    if index["right_endpoint"] != {"word": [], "strands": 1}:
        raise ValueError("metric suffix endpoint is not the unknot")
    sources = {}
    for witness_id, payload in index["witnesses"].items():
        if digest(payload) != witness_id:
            raise ValueError("metric source witness hash differs")
        witness = UnknotWitness.from_dict(payload)
        sources[witness_id] = {r["suffix_index"]: r for r in short_u_views(witness, witness_id)}
    new_rows, records = [], []
    for n, row in enumerate(index["views"]):
        if sources[row["witness_id"]][row["suffix_index"]] != row:
            raise ValueError("metric suffix bound/view differs from source replay")
        new_rows.append(
            pair_row(
                n,
                "audited-mcts-short-u",
                (tuple(row["word"]), row["strands"]),
                ((), 1),
                row["lower"],
                row["upper"],
                row["remaining_moves"],
            )
        )
        records.append(
            {"split": "train", "left_fingerprint": row["fingerprint"], "right_fingerprint": [1, 0]}
        )
    return original, GroupBalancedMetricSampler(new_rows, records)


def sample_step(rng, old_sampler, new_sampler, positive_groups, relative_groups):
    """Same draws in both arms; control also consumes unused policy draws."""
    metrics = old_sampler.sample(rng, 31) + new_sampler.sample(rng, 32)
    positives = [rng.choice(rng.choice(positive_groups)) for _ in range(32)]
    relative = rng.choice(relative_groups) if relative_groups else None
    return metrics, positives, relative
