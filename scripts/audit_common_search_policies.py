"""Replay native discovered proofs and all RF continuations from protocol84."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from audit_cost_mcts_continuations import audit_episode
from check_qrun_common_witnesses import check_pins, sha, write_new
from evaluate_common_greedy_policies import GATE, load_model, raw_state
from evaluate_common_search_policies import CONFIG, GREEDY

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, PASS, ActionSpec
from rf_knots.evidence import UnknotWitness


def native_replay(model, original, weight, native_actions, transcript=None):
    from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem

    start = original.start.word, original.start.strands
    fixed = FixedWordGame(model.game, KnotItem("audit", len(start[0]), *start), weight)
    t = fixed.reset(CONFIG["seed"])
    spec, word_state, actions, cc = model.game.spec, start, [], 0
    if len(native_actions) > 128:
        raise ValueError("native proof/trajectory exceeds declared horizon")
    for index, native in enumerate(native_actions):
        if t.terminated or not t.legal_actions[native]:
            raise ValueError("illegal native action or action after termination")
        head = int(t.state.base_state.head)
        action = model.game.underlying_action(native, head, len(word_state[0]))
        if transcript is not None:
            step = transcript[index]
            if (
                step["head_before"] != head
                or step["semantic_action"] != action
                or (tuple(step["before"]["word"]), step["before"]["strands"]) != word_state
            ):
                raise ValueError("native executed transcript prefix mismatch")
        if action is not None:
            kind = spec.decode(action)[0]
            if kind == PASS or not reference.is_legal(spec, *word_state, action, cc < 2):
                raise ValueError("native path violates shared semantic caps")
            word_state = reference.apply(spec, *word_state, action)
            cc += int(kind == CROSSING_CHANGE)
            actions.append(action)
        t = fixed.step(t.state, native)
        if raw_state(fixed, t.state) != word_state:
            raise ValueError("native/reference states disagree")
        if int(np.asarray(fixed.unwrap(t.state)._crossing_changes)) != cc:
            raise ValueError("native/reference CC counters disagree")
        if transcript is not None:
            after = transcript[index]["after"]
            if (tuple(after["word"]), after["strands"]) != word_state:
                raise ValueError("native executed transcript endpoint mismatch")
    witness = None
    if word_state == ((), 1):
        witness = UnknotWitness.from_actions(*start, spec, actions)
        witness.verify()
    return witness, actions, t


def audit_native(model, row, original):
    transcript = row["executed_transcript"]
    executed, _, terminal = native_replay(
        model,
        original,
        row["cc_weight"],
        [s["native_action"] for s in transcript],
        transcript,
    )
    if row["executed_solved"] != (executed is not None):
        raise ValueError("executed solve claim mismatch")
    reason = str(terminal.termination_reason) if terminal.terminated else "decision_cap"
    if row["stop_reason"] != reason or (not terminal.terminated and len(transcript) != 128):
        raise ValueError("native stopping condition mismatch")
    best_cost, last_transition = float("inf"), 0
    for best in row["incumbents"]:
        witness, actions, _ = native_replay(
            model, original, row["cc_weight"], best["native_actions"]
        )
        if witness is None or witness.to_dict() != best["witness"]:
            raise ValueError("native discovered proof does not replay")
        cost = row["cc_weight"] * witness.crossing_changes + witness.moves
        if cost != best["cost"] or actions != best["semantic_actions"] or cost >= best_cost:
            raise ValueError("incumbent objective/prefix mismatch")
        if (
            not last_transition
            < best["discovery_transition"]
            <= row["virtual_and_executed_transitions"]
        ):
            raise ValueError("discovery counter not monotone")
        best_cost, last_transition = cost, best["discovery_transition"]
    expected = row["incumbents"][-1] if row["incumbents"] else None
    if row["best"] != expected or row["solved"] != (expected is not None):
        raise ValueError("best proof selection mismatch")
    if executed is not None and (
        expected is None
        or best_cost > row["cc_weight"] * executed.crossing_changes + executed.moves
    ):
        raise ValueError("executed solution missing from discovered proofs")
    if (
        row["actual_neural_samples"] != row["inference_cache"]["misses"]
        or row["actual_forward_batches"] != row["actual_neural_samples"]
        or row["actual_neural_samples"] > 128 * 9
        or row["search_config"]["simulations"] != 8
    ):
        raise ValueError("native inference accounting/config mismatch")
    return None if expected is None else best_cost


def audit_rf(row, original):
    root = row["roots"][0]
    data = root["instance"]
    instance = M.Instance(**{**data, "word": tuple(data["word"])})
    result, config = row["search"], row["rf_config"]
    if (
        instance.word != original.start.word
        or instance.strands != original.start.strands
        or instance.spec != ActionSpec(48, 12)
        or instance.cap_crossing_changes != 2
        or instance.cap_moves != 128
        or instance.cc_weight != row["cc_weight"]
        or config["simulations"] != 8
        or config["insertion_candidates"] != 1056
        or config["horizon"] != 128
        or not 0 <= result["simulations"] <= 8
    ):
        raise ValueError("RF resource/root contract differs")
    steps = scores = last_sim = 0
    for episode in result["episodes"]:
        if not last_sim < episode["simulation"] <= result["simulations"]:
            raise ValueError("RF simulation sequence differs")
        last_sim = episode["simulation"]
        moves, evaluated, _ = audit_episode(instance, episode, config)
        steps += moves
        scores += evaluated
    counts = row["counts"]
    if (
        steps != counts["continuation_moves"]
        or steps != counts["policy_neural_evaluations"]
        or scores != counts["policy_action_scores"]
        or len(result["episodes"]) != counts["episodes"]
    ):
        raise ValueError("RF replay/inference counters differ")
    if row["solved"] != result["solved"] or result["solved"] != (result["witness"] is not None):
        raise ValueError("RF solve claim differs")
    if result["witness"] is None:
        if result["L10"] is not None or result["L1000"] is not None or result["incumbents"]:
            raise ValueError("RF failure assigned cost/incumbent")
        return None
    witness = UnknotWitness.from_dict(result["witness"])
    witness.verify()
    if (
        witness.start != original.start
        or witness.crossing_changes > 2
        or witness.moves > 128
        or (witness.max_len, witness.max_strands) != (48, 12)
    ):
        raise ValueError("RF solution violates root/caps")
    for weight in (10, 1000):
        if result[f"L{weight}"] != weight * witness.crossing_changes + witness.moves:
            raise ValueError("RF witness cost differs")
    best = float("inf")
    for incumbent in result["incumbents"]:
        if (
            incumbent["cost"] >= best
            or incumbent["cost"] != row["cc_weight"] * incumbent["cc"] + incumbent["moves"]
        ):
            raise ValueError("RF incumbent sequence differs")
        best = incumbent["cost"]
    if best != result[f"L{row['cc_weight']}"]:
        raise ValueError("RF final incumbent cost differs")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mp = args.run / "manifest.json"
    manifest = json.loads(mp.read_text())
    if manifest["status"] != "completed" or manifest["config"] != CONFIG:
        raise ValueError("require full frozen search matrix")
    pins = dict(manifest["inputs"])
    for name, expected in manifest["conditions"].items():
        pins[str((args.run / name).resolve())] = expected
    for p in (mp, Path(__file__)):
        pins[str(p.resolve())] = sha(p)
    check_pins(pins)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    roots = {r["index"]: r for r in json.loads((GATE / "panel.json").read_text())["rows"]}
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    rows = [json.loads((args.run / n).read_text()) for n in manifest["conditions"]]
    grid = [(r["model"], r["index"], r["cc_weight"]) for r in rows]
    expected = {(m, i, w) for m in CONFIG["models"] for i in roots for w in CONFIG["weights"]}
    if len(grid) != len(set(grid)) or set(grid) != expected:
        raise ValueError("missing/extra/duplicate search conditions")
    summaries = {}
    greedy = json.loads((GREEDY / "replay-audit.json").read_text())
    for label in CONFIG["models"]:
        model = None if label == "rf-joint57" else load_model(label, baseline, pins)
        subset = [r for r in rows if r["model"] == label]
        costs = {}
        for row in subset:
            original = UnknotWitness.from_dict(roots[row["index"]]["witness"])
            cost = audit_rf(row, original) if model is None else audit_native(model, row, original)
            costs[row["index"], row["cc_weight"]] = cost
        for weight in CONFIG["weights"]:
            key = f"{label}-L{weight}"
            cells = [r for r in subset if r["cc_weight"] == weight]
            solved = sorted(r["index"] for r in cells if r["solved"])
            old = greedy["summary"][key]
            summary = {
                "solved_indices": solved,
                "costs_of_successes_only": {str(i): costs[i, weight] for i in solved},
                "new_vs_greedy": sorted(set(solved) - set(old["solved_indices"])),
                "lost_vs_greedy": sorted(set(old["solved_indices"]) - set(solved)),
                "common_success_cost_changes": {
                    str(i): costs[i, weight] - old["costs_of_successes_only"][str(i)]
                    for i in set(solved) & set(old["solved_indices"])
                },
            }
            if model is None:
                summary["policy_neural_samples"] = sum(
                    r["counts"]["policy_neural_evaluations"] for r in cells
                )
                summary["tree_embedding_samples"] = sum(
                    r["counts"]["tree_embedding_evaluations"] for r in cells
                )
                summary["continuation_stops"] = dict(
                    Counter(
                        e["continuation"]["stop_reason"]
                        for r in cells
                        for e in r["search"]["episodes"]
                    )
                )
            else:
                summary["actual_neural_samples"] = sum(r["actual_neural_samples"] for r in cells)
                summary["executed_solved"] = sum(r["executed_solved"] for r in cells)
                summary["discovered_only_indices"] = [
                    r["index"] for r in cells if r["solved"] and not r["executed_solved"]
                ]
            summaries[key] = summary
    check_pins(pins)
    report = {
        "status": "passed",
        "conditions": len(rows),
        "inputs": pins,
        "summary": summaries,
        "scope": "TRAIN search diagnostic, unequal algorithms/compute, no embedding promotion",
    }
    write_new(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
