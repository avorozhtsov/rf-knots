"""Protocol92 real-weight training/search bridge gate, without optimizer updates."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from audit_cost_mcts_continuations import audit_episode
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import load_model
from mcts_joint_training import load_policy_targets
from pretrained_raster_graph import RasterGraphPolicy
from pretrained_raster_policy import (
    RasterDistancePrior,
    RasterPolicyEvaluator,
    RawKnotDistance,
    policy_backward,
    reference_preferences,
)
from pretrained_raster_tiles import PretrainedRasterTiles

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import BatchedPolicyContinuation, run_continuation_cost_searches
from rf_knots.embedding_metric import MetricEmbeddingObjective

INITIAL = Path("artifacts/pretrained-raster-graph-initialization-v1")
SUPERVISION = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
CONFIG = {
    "seed": 20261046,
    "simulations": 4,
    "horizon": 8,
    "insertion_candidates": 4,
    "widening_batch": 4,
    "microbatch": 2,
    "temperature": 1.0,
}


def inputs():
    initial = json.loads((INITIAL / "report.json").read_text())
    certificate = json.loads((SUPERVISION / "certificate-audit.json").read_text())
    if initial["status"] != "passed" or certificate["status"] != "passed":
        raise ValueError("require passed initialization and policy certificate")
    if sha(INITIAL / "initialization.pt") != initial["checkpoint_sha256"]:
        raise ValueError("initial checkpoint hash differs")
    pins = dict(initial["inputs"])
    for path, digest in certificate["inputs"].items():
        if path in pins and pins[path] != digest:
            raise ValueError("conflicting lineage")
        pins[path] = digest
    for path in (
        INITIAL / "report.json",
        INITIAL / "initialization.pt",
        SUPERVISION / "certificate-audit.json",
        Path(__file__),
        Path("scripts/pretrained_raster_policy.py"),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/mcts_training_supervision.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("tests/test_pretrained_raster_policy.py"),
        Path("research/92-pretrained-raster-policy-search-gate.md"),
    ):
        pins[str(path.resolve())] = sha(path)
    check_pins(pins)
    return pins


def gradient_gate(net):
    positives, preferences = load_policy_targets(SUPERVISION)
    ranked = sorted(
        (len(row[0][1].word), condition, index)
        for condition, group in enumerate(positives)
        for index, row in enumerate(group)
        if len(row[1]) > 1
    )
    chosen = [positives[c][i] for _, c, i in ranked[:4]] + [positives[0][0]]
    group_index = min(
        range(len(preferences)), key=lambda i: (sum(len(r[0][1].word) for r in preferences[i]), i)
    )
    relative = reference_preferences(net, [preferences[group_index]], microbatch=2)[0]
    started = time.perf_counter()
    losses = policy_backward(net, chosen, relative, microbatch=2)
    elapsed = time.perf_counter() - started
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in net.named_parameters()
    }
    groups = {
        "pretrained_tiles": "encoder.tiles.",
        "projection": "encoder.project.",
        "graph": "encoder.message.",
        "multiscale": "encoder.context.",
        "pool": "encoder.pool.",
        "global_heads": "encoder.heads.",
        "position": "position.",
        "policy": "head.",
    }
    magnitudes = {}
    for group, prefix in groups.items():
        grads = [g for name, g in gradients.items() if name.startswith(prefix)]
        if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise ValueError(f"missing or nonfinite {group} gradient")
        magnitudes[group] = sum(float(g.abs().sum()) for g in grads)
        if magnitudes[group] <= 0:
            raise ValueError(f"zero {group} gradient")
    net.zero_grad(set_to_none=True)
    singles = policy_backward(net, chosen, relative, microbatch=1)
    max_error = 0.0
    for name, p in net.named_parameters():
        expected = gradients[name]
        if expected is None:
            if p.grad is not None:
                raise ValueError("gradient coverage changed with microbatch")
        else:
            torch.testing.assert_close(p.grad, expected, rtol=1e-3, atol=5e-5)
            max_error = max(max_error, float((p.grad - expected).abs().max()))
    for name in losses:
        if abs(losses[name] - singles[name]) > 5e-5:
            raise ValueError("microbatch loss comparison differs")
    net.zero_grad(set_to_none=True)
    return {
        "positive_selection": [{"condition": c, "index": i} for _, c, i in ranked[:4]]
        + [{"condition": 0, "index": 0}],
        "positive_lengths": [len(r[0][1].word) for r in chosen],
        "positive_set_sizes": [len(r[1]) for r in chosen],
        "relative_group_index": group_index,
        "relative_states": len(relative[0]),
        "relative_total_word_length": sum(len(r[0][1].word) for r in relative[0]),
        "losses": losses,
        "single_losses": singles,
        "gradient_l1_by_group": magnitudes,
        "microbatch_gradient_max_abs_error": max_error,
        "positive_and_relative_backward_seconds": elapsed,
    }


def search_gate(net):
    instances = [
        M.Instance(
            f"engineering-{len(word)}-{weight}",
            word,
            2,
            cc,
            cap_moves=moves,
            growth_len=2,
            growth_strands=1,
            cc_weight=weight,
        )
        for word, cc, moves in (((1,), 0, 4), ((1, 1, 1), 1, 8))
        for weight in (10, 1000)
    ]
    heuristic = RawKnotDistance(net, MetricEmbeddingObjective(), microbatch=2)
    prior = RasterDistancePrior(heuristic, temperature=CONFIG["temperature"])
    evaluator = RasterPolicyEvaluator(net, microbatch=2)
    continuation = BatchedPolicyContinuation(
        evaluator, horizon=CONFIG["horizon"], insertion_candidates=CONFIG["insertion_candidates"]
    )
    started = time.perf_counter()
    trees = run_continuation_cost_searches(
        instances,
        prior,
        continuation,
        simulations=CONFIG["simulations"],
        seed=CONFIG["seed"],
        widening_batch=CONFIG["widening_batch"],
    )
    elapsed = time.perf_counter() - started
    rows, steps, scores = [], 0, 0
    for instance, tree in zip(instances, trees, strict=True):
        episodes = []
        for simulation, prefix, result in tree.continuations:
            episode = json.loads(
                json.dumps(
                    {
                        "simulation": simulation,
                        "tree_prefix": list(prefix),
                        "continuation": asdict(result),
                    }
                )
            )
            n, s, _ = audit_episode(instance, episode, CONFIG)
            steps, scores = steps + n, scores + s
            episodes.append(episode)
        witness = tree.best_witness
        if witness is not None:
            witness.verify()
            if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
                raise ValueError("wrong witness root")
            if instance.cc_weight * witness.crossing_changes + witness.moves != tree.best_cost:
                raise ValueError("best witness cost mismatch")
            if (
                witness.moves > instance.cap_moves
                or witness.crossing_changes > instance.cap_crossing_changes
            ):
                raise ValueError("witness outside root caps")
        elif len(instance.word) == 1:
            raise ValueError("tree failed to discover the direct terminal transition")
        rows.append(
            {
                "instance": instance.to_dict(),
                "episodes": episodes,
                "simulations": tree.simulations,
                "incumbents": tree.incumbents,
                "witness": witness.to_dict() if witness else None,
                "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
            }
        )
    if steps != continuation.state_evaluations or steps != evaluator.calls:
        raise ValueError("policy evaluation count differs from replay")
    if scores != continuation.action_scores or scores != evaluator.action_scores:
        raise ValueError("policy score count differs from replay")
    return {
        "rows": rows,
        "wall_seconds": elapsed,
        "policy_neural_samples": evaluator.calls,
        "policy_forward_batches": evaluator.forwards,
        "policy_action_scores": scores,
        "raw_tree_encodings_including_anchors": heuristic.evaluations,
        "raw_tree_forward_batches": heuristic.forwards,
        "tree_action_scores": prior.action_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pins = inputs()
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    q = load_model("q254", baseline, pins)
    net = RasterGraphPolicy(PretrainedRasterTiles(q.network.representation)).train()
    payload = torch.load(INITIAL / "initialization.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"], strict=True)
    training = gradient_gate(net)
    print(json.dumps({"gradient_gate": training}), flush=True)
    search = search_gate(net)
    if any(
        not torch.equal(value, net.state_dict()[name]) for name, value in payload["model"].items()
    ):
        raise ValueError("gate changed initialization tensors")
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "passed",
        "config": CONFIG,
        "inputs": pins,
        "training": training,
        "search": search,
        "optimizer_steps": 0,
        "trained_checkpoint_saved": False,
        "scope": "Real-weight interface gate, not learned distance/solver quality",
    }
    write_new(args.out / "report.json", report)
    print(
        json.dumps(
            {
                "status": "passed",
                "search_wall_seconds": search["wall_seconds"],
                "policy_neural_samples": search["policy_neural_samples"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
