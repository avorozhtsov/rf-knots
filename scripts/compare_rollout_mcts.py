"""Replay-audited fixed-budget rollout portfolio vs one fixed MCTS seed."""

import argparse
import hashlib
import json
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.rollout_value import verify_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mcts-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite comparison")
    data = json.loads((args.data / "manifest.json").read_text())
    report = json.loads(args.mcts_report.read_text())
    if (
        data["status"] != "completed"
        or data.get("purpose") != "evaluation-only"
        or report["status"] != "completed"
    ):
        raise ValueError("require completed evaluation artifacts")

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    if digest(args.data / "source-panel.json") != report["panel_sha256"]:
        raise ValueError("different root panels")
    if digest(args.data / "roots.json") != data["roots_sha256"]:
        raise ValueError("root hash changed")
    episodes = []
    for shard in data["shards"]:
        path = args.data / shard["file"]
        if digest(path) != shard["sha256"]:
            raise ValueError("episode hash changed")
        episodes.extend(json.loads(path.read_text()))
    roots = json.loads((args.data / "roots.json").read_text())
    weight = report["config"].get("cc_weight", 1000)
    budget = report["config"]["simulations"]
    rows = []
    for selected in report["selected"]:
        index = selected["panel_index"]
        matches = [
            (g, r)
            for g, r in enumerate(roots)
            if r["panel_index"] == index and r["instance"]["cc_weight"] == weight
        ]
        if len(matches) != 1:
            raise ValueError("missing/duplicate rollout root")
        group, root = matches[0]

        def make(r):
            return M.Instance(**{**r, "word": tuple(r["word"])})

        problem, baseline = make(root["instance"]), make(selected["resource_caps"])

        def signature(x):
            return (
                x.word,
                x.strands,
                x.cap_moves,
                x.cap_crossing_changes,
                x.spec.max_len,
                x.spec.max_strands,
            )

        if signature(problem) != signature(baseline):
            raise ValueError("resource contract mismatch")
        trials = sorted((e for e in episodes if e["group"] == group), key=lambda e: e["seed"])
        if [e["seed"] for e in trials] != list(range(data["config"]["rollouts"])):
            raise ValueError("episode seed mismatch")
        spent, completed, best, partial = 0, 0, None, False
        for episode in trials:
            if spent + episode["moves"] > budget:
                # Its unseen suffix must not contribute a hindsight success.
                spent, partial = budget, True
                break
            verify_episode(episode, problem)
            spent += episode["moves"]
            completed += 1
            if episode["success"]:
                w = UnknotWitness.from_actions(
                    problem.word, problem.strands, problem.spec, episode["actions"]
                )
                w.verify()
                if (
                    best is None
                    or weight * w.crossing_changes + w.moves
                    < weight * best.crossing_changes + best.moves
                ):
                    best = w
        mcts_rows = [
            r for r in report["search"]["rows"] if r["instance"] == baseline.uid and r["seed"] == 0
        ]
        if len(mcts_rows) != 1:
            raise ValueError("missing/duplicate MCTS seed zero")
        old = mcts_rows[0]
        if old["solved"]:
            w = UnknotWitness.from_dict(old["witness"])
            w.verify()
            if w.start.word != problem.word or w.start.strands != problem.strands:
                raise ValueError("MCTS witness root mismatch")
            states = [w.start] + [s.after for s in w.steps]
            if (
                w.moves > problem.cap_moves
                or w.crossing_changes > problem.cap_crossing_changes
                or any(
                    len(s.word) > problem.spec.max_len or s.strands > problem.spec.max_strands
                    for s in states
                )
            ):
                raise ValueError("MCTS witness resource violation")
            if any(
                old[f"L{weight}"] != weight * w.crossing_changes + w.moves for weight in (10, 1000)
            ):
                raise ValueError("MCTS stored costs differ from replay")
        elif old["witness"] is not None or old["L10"] is not None or old["L1000"] is not None:
            raise ValueError("failed MCTS row must retain censored costs")
        rows.append(
            {
                "panel_index": index,
                "fingerprint_group": selected["fingerprint"],
                "mcts_solved": old["solved"],
                "mcts_L10": old["L10"],
                "mcts_L1000": old["L1000"],
                "rollout_solved": best is not None,
                "neural_state_evaluations": spent,
                "completed_episodes": completed,
                "partial_episode_censored": partial,
                "rollout_L10": 10 * best.crossing_changes + best.moves if best else None,
                "rollout_L1000": 1000 * best.crossing_changes + best.moves if best else None,
                "rollout_witness": best.to_dict() if best else None,
            }
        )
    result = {
        "status": "passed",
        "mcts_report_sha256": digest(args.mcts_report),
        "data_manifest_sha256": digest(args.data / "manifest.json"),
        "budget_per_root": budget,
        "mcts_seed": 0,
        "cc_weight": weight,
        "mcts_checkpoint_sha256": report["checkpoint_sha256"],
        "rollout_actor_parent_sha256": data["inputs"][
            str(Path(data["config"]["run"]) / "checkpoint.pt")
        ],
        "rows": rows,
        "mcts_solved": sum(r["mcts_solved"] for r in rows),
        "rollout_solved": sum(r["rollout_solved"] for r in rows),
        "caveat": "One portfolio per root vs MCTS seed 0, not multiple MCTS seeds pooled. "
        "Same maximum neural-state evaluations: each MCTS simulation uses at most one. "
        "Action scores, wall time and proposal algorithms differ. "
        "Correlated roots from the declared evaluation panel; "
        "best of completed rollouts vs first-found MCTS, not optimal-cost proof.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
