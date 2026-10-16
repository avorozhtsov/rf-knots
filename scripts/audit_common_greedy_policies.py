"""Independent transcript/native replay for the completed protocol82 diagnostic."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import CONFIG, GATE, load_model

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, PASS, ActionSpec
from rf_knots.evidence import UnknotWitness


def state(payload):
    return tuple(payload["word"]), payload["strands"]


def audit_row(row, root, scientist=None):
    spec = ActionSpec(48, 12)
    witness = UnknotWitness.from_dict(root["witness"])
    current = witness.start.word, witness.start.strands
    actions, cc, repeated, seen = [], 0, 0, {current}
    transcript = row["transcript"]
    if len(transcript) > CONFIG["decisions"]:
        raise ValueError("decision cap exceeded")
    game = transition = None
    if scientist is not None:
        from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem

        game = FixedWordGame(
            scientist.game,
            KnotItem(str(root["index"]), len(current[0]), *current),
            row["cc_weight"],
        )
        transition = game.reset(CONFIG["seed"])
    for step in transcript:
        if current == ((), 1) or state(step["before"]) != current:
            raise ValueError("broken state chain or action after solved")
        action = step["semantic_action"]
        if game is not None:
            if transition.terminated:
                raise ValueError("native action after terminal")
            native = step["native_action"]
            head = int(transition.state.base_state.head)
            if head != step["head_before"] or not transition.legal_actions[native]:
                raise ValueError("native head/action mismatch")
            mapped = scientist.game.underlying_action(native, head, len(current[0]))
            if mapped != action:
                raise ValueError("native action maps to another semantic action")
            transition = game.step(transition.state, native)
            raw = game.unwrap(transition.state)
            actual = tuple(int(g) for g in np.asarray(raw._word) if g), int(np.asarray(raw._n))
            if actual != state(step["after"]):
                raise ValueError("native replay differs from stored after-state")
        elif step["native_action"] is not None or step["head_before"] is not None:
            raise ValueError("RF trajectory contains native controller data")
        if action is not None:
            kind = spec.decode(action)[0]
            if kind == PASS or not reference.is_legal(spec, *current, action, cc < 2):
                raise ValueError("illegal, PASS, or excess-CC action")
            current = reference.apply(spec, *current, action)
            actions.append(action)
            cc += int(kind == CROSSING_CHANGE)
            repeated += int(current in seen)
            seen.add(current)
        elif scientist is None:
            raise ValueError("RF decision without a semantic action")
        if state(step["after"]) != current:
            raise ValueError("reference replay differs from stored after-state")
    solved = current == ((), 1)
    if (
        row["solved"] != solved
        or state(row["final"]) != current
        or row["used_cc"] != cc
        or row["semantic_moves"] != len(actions)
        or row["neural_samples"] != len(transcript)
        or row["forward_batches"] != len(transcript)
        or row["controller_only_decisions"] != len(transcript) - len(actions)
        or row["repeated_semantic_states"] != repeated
        or row["scored_actions"] != sum(s["legal_scores"] for s in transcript)
    ):
        raise ValueError("episode counters or terminal claims differ from replay")
    if solved:
        replay = UnknotWitness.from_actions(
            witness.start.word, witness.start.strands, spec, actions
        )
        replay.verify()
        if row["witness"] != replay.to_dict() or row["cost"] != row["cc_weight"] * cc + len(
            actions
        ):
            raise ValueError("successful witness or objective mismatch")
        if row["stop_reason"] != "solved":
            raise ValueError("solved episode mislabeled")
    elif row["witness"] is not None or row["cost"] is not None:
        raise ValueError("failure assigned a witness or artificial objective")
    elif game is not None and transition.terminated:
        if row["stop_reason"] != str(transition.termination_reason):
            raise ValueError("native termination reason mismatch")
    elif len(transcript) == CONFIG["decisions"]:
        if row["stop_reason"] != "decision_cap":
            raise ValueError("incorrect external decision-cap label")
    else:
        raise ValueError("early failure without native termination or exhausted clock")
    return {
        "solved": solved,
        "decisions": len(transcript),
        "semantic_moves": len(actions),
        "cc": cc,
        "repeated_semantic_states": repeated,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed" or manifest["config"] != CONFIG:
        raise ValueError("require complete frozen run")
    pins = dict(manifest["inputs"])
    for name, expected in manifest["episodes"].items():
        pins[str((args.run / name).resolve())] = expected
    for path in (manifest_path, Path(__file__)):
        pins[str(path.resolve())] = sha(path)
    check_pins(pins)
    roots = {r["index"]: r for r in json.loads((GATE / "panel.json").read_text())["rows"]}
    expected_grid = {(m, i, w) for m in CONFIG["models"] for i in roots for w in CONFIG["weights"]}
    rows = [json.loads((args.run / p).read_text()) for p in manifest["episodes"]]
    grid = [(r["model"], r["index"], r["cc_weight"]) for r in rows]
    if len(grid) != len(set(grid)) or set(grid) != expected_grid:
        raise ValueError("missing, duplicate, or extra conditions")
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    summary = {}
    for label in CONFIG["models"]:
        model = None if label == "rf-joint57" else load_model(label, baseline, pins)
        subset = [r for r in rows if r["model"] == label]
        for row in subset:
            audit_row(row, roots[row["index"]], model)
        for weight in CONFIG["weights"]:
            cells = [r for r in subset if r["cc_weight"] == weight]
            summary[f"{label}-L{weight}"] = {
                "solved_indices": [r["index"] for r in cells if r["solved"]],
                "stop_reasons": dict(Counter(r["stop_reason"] for r in cells)),
                "neural_samples": sum(r["neural_samples"] for r in cells),
                "controller_only_decisions": sum(r["controller_only_decisions"] for r in cells),
                "semantic_moves": sum(r["semantic_moves"] for r in cells),
                "repeated_semantic_states": sum(r["repeated_semantic_states"] for r in cells),
                "costs_of_successes_only": {
                    str(r["index"]): r["cost"] for r in cells if r["solved"]
                },
            }
    check_pins(pins)
    report = {
        "status": "passed",
        "conditions": len(rows),
        "inputs": pins,
        "summary": summary,
        "scope": "Replay audit; TRAIN greedy diagnostic, not independent MCTS/embedding quality",
    }
    write_new(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
