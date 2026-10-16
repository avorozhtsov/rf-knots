"""Frozen final-head calibration against independent protocol265 DEV attempts."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from budgeted_cost_value_head_v1 import BudgetedCostValueHead
from build_mcts_training_supervision import atomic, sha
from rf_knots.evidence import UnknotWitness


TRAIN = Path("artifacts/root-budget16-multioutput-labels-v1/labels.json")
HEAD = Path("artifacts/root-budget16-multioutput-value-head-v1")
TEACHER = Path("artifacts/root-budget16-value-dev-teacher-v1")
PANELS = {
    "short": Path("artifacts/mcts-tournament-dev-panel-v1/panel.json"),
    "long": Path("artifacts/mcts-tournament-long-dev-panel-v1/panel.json"),
}
PROTOCOL = Path("research/265-root-budget16-value-dev-calibration-v1.md")


def read(path):
    return json.loads(path.read_text())


def brier(predictions, targets):
    return sum((p - y) ** 2 for p, y in zip(predictions, targets, strict=True)) / len(targets)


def logloss(predictions, targets):
    return -sum(y * math.log(max(p, 1e-8)) + (1 - y) * math.log(max(1 - p, 1e-8))
                for p, y in zip(predictions, targets, strict=True)) / len(targets)


def summary(rows, constants):
    probability = [r["p_solve"] for r in rows]
    truth = [r["solved"] for r in rows]
    fixed = [constants["p_solve"]] * len(rows)
    successful = [r for r in rows if r["solved"]]
    bins = []
    for bin_index in range(5):
        members = [r for r in rows if min(4, int(r["p_solve"] * 5)) == bin_index]
        bins.append({"count": len(members),
                     "mean_prediction": sum(r["p_solve"] for r in members) / len(members) if members else None,
                     "observed_fraction": sum(r["solved"] for r in members) / len(members) if members else None})
    result = {
        "attempts": len(rows), "solved": sum(truth),
        "model_brier": brier(probability, truth),
        "constant_brier": brier(fixed, truth),
        "model_logloss": logloss(probability, truth),
        "constant_logloss": logloss(fixed, truth),
        "calibration_bins": bins,
        "teacher_cpu_seconds": sum(r["cpu_seconds"] for r in rows),
        "teacher_neural_evaluations": sum(r["neural_evaluations"] for r in rows),
    }
    if successful:
        for field, prediction_field, baseline in (
            ("cc", "cc_remaining", "cc_remaining"),
            ("moves", "moves_remaining", "moves_remaining"),
        ):
            result[f"{field}_mae_model_on_success"] = sum(
                abs(r[prediction_field] - r[field]) for r in successful
            ) / len(successful)
            result[f"{field}_mae_constant_on_success"] = sum(
                abs(constants[baseline] - r[field]) for r in successful
            ) / len(successful)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("calibration report exists; never overwrite")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical dependency missing")
    sys.path.insert(0, str(pgx_src))
    teacher, trained, train = read(TEACHER / "manifest.json"), read(HEAD / "manifest.json"), read(TRAIN)
    if (teacher["status"] != "completed" or teacher["next_task"] != 256
            or trained["status"] != "completed" or trained["optimizer_steps"] != 512
            or len(train["labels"]) != 2560):
        raise ValueError("teacher/head/label completion missing")
    for manifest in (teacher, trained):
        if any(sha(Path(p)) != h for p, h in manifest["contract"]["inputs"].items()):
            raise ValueError("frozen source pin changed")
    entry = trained["checkpoints"][-1]
    checkpoint_path = HEAD / entry["file"]
    if entry["step"] != 512 or sha(checkpoint_path) != entry["sha256"]:
        raise ValueError("final head checkpoint differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["contract_sha256"] != trained["contract_sha256"] or checkpoint["step"] != 512:
        raise ValueError("head state identity differs")
    head = BudgetedCostValueHead().eval()
    head.load_state_dict(checkpoint["head"])
    net, _ = load_actor()
    net.eval()
    train_by_lambda = defaultdict(list)
    for row in train["labels"]:
        train_by_lambda[row["lambda"]].append(row)
    constants = {}
    for lam, rows in train_by_lambda.items():
        successes = [r for r in rows if r["solved"]]
        constants[lam] = {
            "p_solve": len(successes) / len(rows),
            "cc_remaining": sum(r["cc_remaining"] for r in successes) / len(successes),
            "moves_remaining": sum(r["moves_remaining"] for r in successes) / len(successes),
        }
    panels = {band: read(path) for band, path in PANELS.items()}
    predictions = {}
    with torch.no_grad():
        for band, panel in panels.items():
            for row_index, row in enumerate(panel["rows"]):
                for instance in row["instances"]:
                    lam = instance["cc_weight"]
                    word = instance["word"]
                    encoded = net.encoder(
                        torch.tensor([word], dtype=torch.long),
                        torch.tensor([len(word)]),
                        torch.tensor([instance["strands"]]),
                    )
                    result = head(
                        encoded, cc_left=[instance["cap_crossing_changes"]],
                        moves_left=[instance["cap_moves"]], cc_weight=[lam],
                        local_simulations=[16],
                    )
                    predictions[(band, row_index, lam)] = {
                        "p_solve": float(result["p_solve"][0]),
                        "cc_remaining": float(result["cc_remaining"][0]),
                        "moves_remaining": float(result["moves_remaining"][0]),
                    }
    rows = []
    for position, entry in enumerate(teacher["shards"]):
        path = TEACHER / entry["file"]
        if entry["file"] != f"attempt-{position:03d}.json" or sha(path) != entry["sha256"]:
            raise ValueError("DEV teacher shard changed")
        source = read(path)
        if source["task"] != position:
            raise ValueError("DEV task order changed")
        band, row_index, lam = source["band"], source["row_index"], source["lambda"]
        instance = next(x for x in panels[band]["rows"][row_index]["instances"]
                        if x["cc_weight"] == lam)
        outcome = source["result"]
        if outcome["simulations"] != 16 or bool(outcome["witness"]) != bool(outcome["solved"]):
            raise ValueError("DEV teacher budget/outcome differs")
        cc = moves = None
        if outcome["solved"]:
            witness = UnknotWitness.from_dict(outcome["witness"])
            witness.verify()
            if (witness.start.word, witness.start.strands) != (tuple(instance["word"]), instance["strands"]):
                raise ValueError("DEV witness start differs")
            cc, moves = witness.crossing_changes, witness.moves
        rows.append({
            "band": band, "lambda": lam, "row_index": row_index,
            "solved": int(outcome["solved"]), "cc": cc, "moves": moves,
            "cpu_seconds": source["cpu_seconds"],
            "neural_evaluations": source["counts"].get("policy_neural_evaluations", 0)
                                  + source["counts"].get("tree_embedding_evaluations", 0),
            **predictions[(band, row_index, lam)],
        })
    report = {
        "schema": "rf-knots-root-budget16-value-dev-calibration-v1",
        "inputs": {str(p): sha(p) for p in (
            TEACHER / "manifest.json", HEAD / "manifest.json", checkpoint_path,
            TRAIN, PROTOCOL, Path(__file__), *PANELS.values(),
        )},
        "train_only_constants": constants,
        "slices": {f"{band}-L{lam}": summary(
            [r for r in rows if r["band"] == band and r["lambda"] == lam], constants[lam])
            for band in PANELS for lam in (10, 1000)},
        "all_attempts": len(rows),
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "report.json", report)
    print(json.dumps({"attempts": len(rows), "slices": report["slices"],
                      "report_sha256": sha(args.out / "report.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
