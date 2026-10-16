"""Real-network TRAIN gradient check for bounded tournament term sampling."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from pretrained_raster_policy import reference_preferences
from tournament_s10_dataset_v1 import load_groups
from tournament_term_sampler_v1 import sample_coefficients
from trajectory_log_preference import relative_backward

from rf_knots import mcts_loop as M
from rf_knots.site_proposals import proposal_candidates


def requests(instance, coefficients):
    pools = {}
    for state, action in coefficients:
        pools.setdefault(state, set()).add(action)
    rows = []
    for (state, action), coefficient in sorted(
        coefficients.items(),
        key=lambda item: (item[0][0].used_moves, item[0][0].used_cc, item[0][1]),
    ):
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | pools[state]))
        rows.append(((instance, state, actions), actions.index(action), coefficient))
    return rows


def gradient(net, rows):
    net.zero_grad(set_to_none=True)
    relative = reference_preferences(net, [rows], microbatch=1)[0]
    relative_backward(net, relative, microbatch=1, weight=1.0)
    result = [p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p.cpu())
              for p in net.encoder.parameters()]
    net.zero_grad(set_to_none=True)
    return result


def compare(exact, estimate):
    dot = sum(float((a * b).sum()) for a, b in zip(exact, estimate, strict=True))
    a2 = sum(float(a.square().sum()) for a in exact)
    b2 = sum(float(b.square().sum()) for b in estimate)
    return {"cosine": dot / math.sqrt(a2 * b2), "norm_ratio": math.sqrt(b2 / a2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    args = parser.parse_args()
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench source is missing")
    sys.path.insert(0, str(pgx_src))
    if not torch.backends.mps.is_available():
        raise RuntimeError("native MPS required for saved-checkpoint gradient gate")
    torch.set_num_threads(1)
    roots, groups, _ = load_groups()
    candidates = []
    for (base, weight), witnesses in sorted(groups.items()):
        root = roots[base * 2]
        instance = M.Instance(
            **{**root["instance"], "word": tuple(root["instance"]["word"]), "cc_weight": weight}
        )
        traces = verified_success_traces(instance, witnesses)
        signals = _groups(traces, "D", seed=0, k=4, near_margin=2, gap_scale=weight, gap_cap=4)
        if signals:
            candidates.append((len(signals[0]["coefficients"]), base, weight, instance, signals[0]))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    length, base, weight, instance, signal = candidates[len(candidates) // 2]
    net, _ = load_actor()
    net.to("mps")
    net.eval()
    exact = gradient(net, requests(instance, signal["coefficients"]))
    results = []
    for budget in (64, 128):
        estimates = []
        counts = []
        for seed in range(4):
            sampled = sample_coefficients(signal["coefficients"], terms=budget, seed=seed)
            counts.append(len(sampled))
            estimates.append(gradient(net, requests(instance, sampled)))
        mean = [sum(parts) / len(parts) for parts in zip(*estimates, strict=True)]
        results.append({"budget": budget, "sampled_terms": counts, **compare(exact, mean)})
    print(json.dumps({"base_root_index": base, "lambda": weight, "exact_terms": length,
                      "four_draw_mean": results}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
