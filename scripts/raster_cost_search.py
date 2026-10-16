"""Protocol97 raw-raster cost-search batch; no value, oracle incumbent or persistent cache."""

import json
from collections import Counter
from dataclasses import asdict

import torch
from audit_cost_mcts_continuations import audit_episode
from pretrained_raster_policy import RasterDistancePrior, RasterPolicyEvaluator, RawKnotDistance

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches


def run_batch(roots, net, objective, begin, config):
    """Caller owns immutable checkpoint verification and per-root protocol70 caps."""
    if not roots:
        raise ValueError("empty search batch")
    instances = [
        M.Instance(**{**r["instance"], "word": tuple(r["instance"]["word"])}) for r in roots
    ]
    if any(
        not torch.isfinite(t).all()
        for t in [*net.state_dict().values(), *objective.state_dict().values()]
    ):
        raise ValueError("nonfinite checkpoint tensors")
    torch.manual_seed(config["seed"] + begin)
    heuristic = RawKnotDistance(net, objective, microbatch=2)
    tree_evaluator = RasterDistancePrior(heuristic, temperature=config["temperature"])
    policy_evaluator = RasterPolicyEvaluator(net, microbatch=2)
    policy = BatchedPolicyContinuation(
        policy_evaluator,
        horizon=config["horizon"],
        insertion_candidates=config["insertion_candidates"],
    )
    trees = run_continuation_cost_searches(
        instances,
        tree_evaluator,
        policy,
        simulations=config["simulations"],
        seed=config["seed"] + begin,
        widening_batch=config["widening_batch"],
        c_puct=config["c_puct"],
    )
    rows, counts = [], Counter()
    for root, tree in zip(roots, trees, strict=True):
        episodes = []
        for simulation, prefix, continuation in tree.continuations:
            episode = json.loads(
                json.dumps(
                    {
                        "simulation": simulation,
                        "tree_prefix": list(prefix),
                        "continuation": asdict(continuation),
                    }
                )
            )
            steps, scores, _ = audit_episode(tree.instance, episode, config)
            counts["episodes"] += 1
            counts["continuation_moves"] += steps
            counts["full_path_moves"] += steps + len(prefix)
            counts["policy_action_scores"] += scores
            episodes.append(episode)
        witness = tree.best_witness
        if witness is not None:
            witness.verify()
            if (witness.start.word, witness.start.strands) != (
                tree.instance.word,
                tree.instance.strands,
            ):
                raise ValueError("solution starts at a different query")
            if (
                witness.crossing_changes > tree.instance.cap_crossing_changes
                or witness.moves > tree.instance.cap_moves
            ):
                raise ValueError("solution exceeds common resources")
        rows.append(
            {
                "condition": root["condition"],
                "panel_index": root["panel_index"],
                "cc_weight": tree.instance.cc_weight,
                "search_seed": config["seed"] + begin,
                "tree_seed": tree.seed,
                "episodes": episodes,
                "simulations": tree.simulations,
                "expansions": tree.expansions,
                "maximum_tree_depth": tree.maximum_tree_depth,
                "maximum_observed_depth": tree.maximum_observed_depth,
                "incumbents": tree.incumbents,
                "solved": witness is not None,
                "witness": witness.to_dict() if witness else None,
                "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
            }
        )
        counts["conditions"] += 1
        counts["solved_conditions"] += int(witness is not None)
    if counts["continuation_moves"] != policy.state_evaluations:
        raise ValueError("policy-state accounting differs from replay")
    if counts["policy_action_scores"] != policy.action_scores:
        raise ValueError("policy-action accounting differs from replay")
    if policy_evaluator.calls != policy.state_evaluations:
        raise ValueError("actual neural-state accounting differs")
    if policy_evaluator.action_scores != policy.action_scores:
        raise ValueError("actual neural-action accounting differs")
    counts.update(
        {
            "policy_neural_evaluations": policy_evaluator.calls,
            "policy_neural_batches": policy_evaluator.forwards,
            "tree_embedding_evaluations": heuristic.evaluations,
            "tree_action_scores": tree_evaluator.action_scores,
            "tree_embedding_batches": heuristic.forwards,
        }
    )
    return rows, counts
