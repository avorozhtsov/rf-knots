"""Score resource-conditioned pi0 value on evaluation-only completed episodes."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.rollout_value import verify_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--run", type=Path, help="completed trained checkpoint; otherwise initial pi0 value"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an evaluation")
    data = json.loads((args.data / "manifest.json").read_text())
    if data["status"] != "completed" or data.get("purpose") != "evaluation-only":
        raise ValueError("require complete evaluation-only data")
    for name in ("roots", "actor"):
        path = args.data / ("roots.json" if name == "roots" else "actor.pt")
        if sha(path) != data[f"{name}_sha256"]:
            raise ValueError("evaluation data hash mismatch")
    behaviour = torch.load(args.data / "actor.pt", map_location="cpu", weights_only=False)
    constant = None
    if args.run:
        run = json.loads((args.run / "manifest.json").read_text())
        checkpoint_sha = sha(args.run / "checkpoint.pt")
        if run["status"] != "completed" or checkpoint_sha != run["checkpoint_sha256"]:
            raise ValueError("require completed hash-verified training")
        training_data = Path(run["config"]["data"])
        actor = torch.load(training_data / "actor.pt", map_location="cpu", weights_only=False)
        if actor["architecture"] != behaviour["architecture"] or any(
            not torch.equal(value, behaviour["model"][key]) for key, value in actor["model"].items()
        ):
            raise ValueError("label behaviour differs from value training actor")
        checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
        constant = json.loads((args.run / "before.json").read_text())["rollout_value"][
            "constant_training_root_rate"
        ]
    else:
        checkpoint, checkpoint_sha = behaviour, data["actor_sha256"]
    torch.set_num_threads(1)
    net = M.SitePolicyValueNet(**checkpoint["architecture"]).eval()
    net.load_state_dict(checkpoint["model"], strict=True)
    roots = json.loads((args.data / "roots.json").read_text())
    episodes = defaultdict(list)
    for shard in data["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("episode shard hash mismatch")
        for episode in json.loads(path.read_text()):
            if episode["eligible"]:
                raise ValueError("evaluation data cannot be training-eligible")
            episodes[episode["group"]].append(episode)
    rows = []
    if not roots or set(episodes) != set(range(len(roots))):
        raise ValueError("missing/extra evaluation groups")
    fingerprints = {}
    with torch.no_grad():
        for group, root in enumerate(roots):
            problem = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            key = problem.word, problem.strands
            if key not in fingerprints:
                fingerprints[key] = invariants(*key)
            fp = fingerprints[key]
            if list(fp) != root["fingerprint"] or owner(fp) not in ("validation", "test"):
                raise ValueError("root is not in the claimed heldout knot group")
            trials = episodes[group]
            if sorted(e["seed"] for e in trials) != list(range(data["config"]["rollouts"])):
                raise ValueError("missing/duplicate evaluation trials")
            for episode in trials:
                verify_episode(episode, problem)
            state = M.SearchState(problem.word, problem.strands, 0, 0)
            batch = M.build_batch([(problem, state, ())], budget_dim=net.budget_dim)
            _, logit, _ = net(**batch)
            probability = float(logit.sigmoid()[0])
            if not math.isfinite(probability):
                raise ValueError("nonfinite value")
            rows.append(
                {
                    "group": group,
                    "panel_index": root["panel_index"],
                    "fingerprint_group": [root["fingerprint"][0], abs(root["fingerprint"][1])],
                    "cc_weight": problem.cc_weight,
                    "probability": probability,
                    "trials": len(trials),
                    "successes": sum(e["success"] for e in trials),
                }
            )

    def metrics(rows, prediction=None):
        count = sum(r["trials"] for r in rows)
        brier, log_loss = 0.0, 0.0
        for r in rows:
            q = r["probability"] if prediction is None else prediction
            a, b = r["successes"], r["trials"] - r["successes"]
            brier += a * (1 - q) ** 2 + b * q**2
            log_loss -= a * math.log(max(q, 1e-7)) + b * math.log(max(1 - q, 1e-7))
        return {
            "trials": count,
            "successes": sum(r["successes"] for r in rows),
            "brier": brier / count,
            "log_loss": log_loss / count,
        }

    report = {
        "status": "completed",
        "data_sha256": sha(args.data / "manifest.json"),
        "checkpoint_sha256": checkpoint_sha,
        "run": str(args.run) if args.run else "initial-pi0",
        "summary": metrics(rows),
        "constant_half": metrics(rows, 0.5),
        "constant_training_rate": {"probability": constant, **metrics(rows, constant)}
        if constant is not None
        else None,
        "by_cost_weight": {
            str(w): metrics([r for r in rows if r["cc_weight"] == w])
            for w in sorted({r["cc_weight"] for r in rows})
        },
        "by_fingerprint_group": {
            str(fp): metrics([r for r in rows if tuple(r["fingerprint_group"]) == fp])
            for fp in sorted({tuple(r["fingerprint_group"]) for r in rows})
        },
        "rows": rows,
        "code_sha256": sha(Path(__file__)),
        "caveat": "Frozen pi0 single-rollout success, NOT MCTS success or pi1 calibration. "
        "Repeated representations/seeds/lambdas are correlated, not independent knot types.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(args.output)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
