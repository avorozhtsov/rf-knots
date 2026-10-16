"""Protocol139 search: fixed rollout policy, learned-distance or uniform tree prior.

The frozen97 batch is untouched. Local tile reuse is optional, inference-only,
and must pass the declared real-checkpoint parity gate before TEST dispatch.
"""

import json
from collections import Counter
from dataclasses import asdict

import torch
from audit_cost_mcts_continuations import audit_episode
from pretrained_raster_policy import RasterDistancePrior, RasterPolicyEvaluator, RawKnotDistance
from probe_raster_cache_full_network import state_digest
from probe_raster_tile_cache import TileCacheProbe

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches


class ZeroDistance:
    """Uniform tree ordering without any encoder forward or fabricated value."""

    evaluations = forwards = 0

    def __call__(self, states):
        return [0.0] * len(states)


def search(roots, net, objective, begin, config, prior):
    instances = [
        M.Instance(**{**r["instance"], "word": tuple(r["instance"]["word"])}) for r in roots
    ]
    torch.manual_seed(config["seed"] + begin)
    heuristic = (
        RawKnotDistance(net, objective, microbatch=2) if prior == "distance" else ZeroDistance()
    )
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
                    dict(
                        simulation=simulation,
                        tree_prefix=list(prefix),
                        continuation=asdict(continuation),
                    )
                )
            )
            steps, scores, _ = audit_episode(tree.instance, episode, config)
            counts.update(
                episodes=1,
                continuation_moves=steps,
                full_path_moves=steps + len(prefix),
                policy_action_scores=scores,
            )
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
            dict(
                condition=root["condition"],
                panel_index=root["panel_index"],
                cc_weight=tree.instance.cc_weight,
                search_seed=config["seed"] + begin,
                tree_seed=tree.seed,
                episodes=episodes,
                simulations=tree.simulations,
                expansions=tree.expansions,
                maximum_tree_depth=tree.maximum_tree_depth,
                maximum_observed_depth=tree.maximum_observed_depth,
                incumbents=tree.incumbents,
                solved=witness is not None,
                witness=witness.to_dict() if witness else None,
                L10=10 * witness.crossing_changes + witness.moves if witness else None,
                L1000=1000 * witness.crossing_changes + witness.moves if witness else None,
            )
        )
        counts.update(conditions=1, solved_conditions=int(witness is not None))
    if (
        counts["continuation_moves"] != policy.state_evaluations
        or counts["policy_action_scores"] != policy.action_scores
        or policy_evaluator.calls != policy.state_evaluations
        or policy_evaluator.action_scores != policy.action_scores
    ):
        raise ValueError("actual policy neural work differs from replay")
    counts.update(
        policy_neural_evaluations=policy_evaluator.calls,
        policy_neural_batches=policy_evaluator.forwards,
        tree_embedding_evaluations=heuristic.evaluations,
        tree_action_scores=tree_evaluator.action_scores,
        tree_embedding_batches=heuristic.forwards,
    )
    return rows, counts


def run_batch(roots, net, objective, begin, config, *, prior="distance", cached=False):
    if not roots or prior not in ("distance", "uniform"):
        raise ValueError("nonempty roots and explicit distance/uniform prior required")
    if any(
        not torch.isfinite(t).all()
        for t in [*net.state_dict().values(), *objective.state_dict().values()]
    ):
        raise ValueError("nonfinite model/objective")
    if any(
        p.device.type != "cpu" or p.dtype != torch.float32
        for p in [*net.parameters(), *objective.parameters()]
    ):
        raise ValueError("protocol139 search requires CPU float32")
    before = state_digest(net, objective)
    was_training = net.training
    net.eval()
    original_forward = net.encoder.tiles.forward
    cache = TileCacheProbe(net.encoder.tiles, capacity=512) if cached else None
    if cache is not None:
        net.encoder.tiles.forward = cache
    try:
        with torch.no_grad():
            rows, counts = search(roots, net, objective, begin, config, prior)
        if state_digest(net, objective) != before:
            raise ValueError("search changed model/objective state")
        if cache is not None:
            if cache.invalidations:
                raise ValueError("tile cache namespace changed during fixed-checkpoint search")
            counts.update(
                cache_logical_tiles=cache.logical_tiles,
                cache_computed_tiles=cache.computed_tiles,
                cache_hits=cache.hits,
            )
        return rows, counts
    finally:
        if cache is not None:
            net.encoder.tiles.forward = original_forward
        net.train(was_training)
