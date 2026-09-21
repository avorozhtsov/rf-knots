"""Replay every successful AND censored trajectory before using rollout data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots.cost_mcts_rollout import advance
from rf_knots.evidence import UnknotWitness
from rf_knots.site_proposals import proposal_candidates


def state_from_dict(row):
    return M.SearchState(tuple(row["word"]), row["strands"], row["used_cc"], row["used_moves"])


def audit_episode(instance, episode, config):
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    for action in episode["tree_prefix"]:
        state = advance(instance, state, action)
    result = episode["continuation"]
    assert state == state_from_dict(result["start"]), "prefix/start mismatch"
    visited, scored = {state.key}, 0
    assert len(result["actions"]) == len(result["probabilities"]), "probability coverage"
    assert len(result["actions"]) <= config["horizon"], "rollout horizon exceeded"
    for action, probability in zip(result["actions"], result["probabilities"], strict=True):
        assert state.key != M.SOLVED, "continued past the terminal state"
        candidates = proposal_candidates(instance, state, config["insertion_candidates"])
        candidates = {a: advance(instance, state, a) for a in candidates}
        candidates = {a: s for a, s in candidates.items() if s.key not in visited}
        assert action in candidates, "action outside visited-filtered proposal pool"
        assert 0 < probability <= 1, "invalid chosen probability"
        scored += len(candidates)
        state = candidates[action]
        visited.add(state.key)
    assert state == state_from_dict(result["end"]), "endpoint replay mismatch"
    assert scored == result["scored_actions"], "action-score accounting mismatch"
    reason = result["stop_reason"]
    assert (state.key == M.SOLVED) == (reason == "solved"), "terminal label mismatch"
    if reason == "move_cap":
        assert state.used_moves == instance.cap_moves, "false move cap"
    elif reason == "horizon":
        assert len(result["actions"]) == config["horizon"], "false horizon"
        assert state.used_moves < instance.cap_moves, "move cap precedes horizon"
    elif reason == "candidate_pool_exhausted":
        assert state.used_moves < instance.cap_moves, "move cap precedes empty pool"
        candidates = proposal_candidates(instance, state, config["insertion_candidates"])
        assert not any(advance(instance, state, a).key not in visited for a in candidates)
    elif reason != "solved":
        raise AssertionError("unknown continuation stop reason")
    if reason == "solved":
        witness = UnknotWitness.from_actions(
            instance.word,
            instance.strands,
            instance.spec,
            tuple(episode["tree_prefix"] + result["actions"]),
        )
        witness.verify()
        assert witness.moves == state.used_moves and witness.crossing_changes == state.used_cc
    return len(result["actions"]), scored, state.used_moves


def audit_report(report):
    assert report["status"] == "completed" and report["current"] is None
    config = report["config"]
    instances = [
        M.Instance(**{k: v for k, v in row.items() if k != "word"}, word=tuple(row["word"]))
        for row in report["instances"]
    ]
    assert [a["mode"] for a in report["arms"]] == ["uniform", "parent", "candidate"]
    summary = []
    for arm in report["arms"]:
        assert len(arm["rows"]) == len(instances) == len(report["selected"])
        total_steps, total_scores, episodes, solved = 0, 0, 0, 0
        for instance, row, selected in zip(instances, arm["rows"], report["selected"], strict=True):
            assert row["corpus_index"] == selected["corpus_index"]
            assert list(instance.word) == selected["start"]["word"]
            assert instance.strands == selected["start"]["strands"]
            assert instance.cc_weight == config["cc_weight"]
            assert 0 <= row["simulations"] <= config["simulations"]
            assert row["maximum_tree_depth"] <= row["maximum_observed_depth"] <= instance.cap_moves
            stops, previous_sim = Counter(), 0
            for episode in row["episodes"]:
                assert previous_sim < episode["simulation"] <= row["simulations"]
                previous_sim = episode["simulation"]
                steps, scores, depth = audit_episode(instance, episode, config)
                assert depth <= row["maximum_observed_depth"]
                assert len(episode["tree_prefix"]) <= row["maximum_tree_depth"]
                stops[episode["continuation"]["stop_reason"]] += 1
                total_steps += steps
                total_scores += scores
                episodes += 1
            assert dict(stops) == row["continuation_stops"]
            assert row["solved"] == (row["witness"] is not None)
            if row["solved"]:
                witness = UnknotWitness.from_dict(row["witness"])
                witness.verify()
                assert (
                    witness.start.word == instance.word
                    and witness.start.strands == instance.strands
                )
                assert witness.max_len == instance.spec.max_len
                assert witness.max_strands == instance.spec.max_strands
                assert witness.moves <= instance.cap_moves
                assert witness.crossing_changes <= instance.cap_crossing_changes
                assert row["L10"] == 10 * witness.crossing_changes + witness.moves
                assert row["L1000"] == 1000 * witness.crossing_changes + witness.moves
                best, last_sim = float("inf"), 0
                for incumbent in row["incumbents"]:
                    assert 0 <= incumbent["cc"] <= instance.cap_crossing_changes
                    assert 0 <= incumbent["moves"] <= instance.cap_moves
                    assert last_sim <= incumbent["simulation"] <= row["simulations"]
                    assert (
                        incumbent["cost"]
                        == instance.cc_weight * incumbent["cc"] + incumbent["moves"]
                    )
                    assert incumbent["cost"] < best
                    best, last_sim = incumbent["cost"], incumbent["simulation"]
                assert best == instance.cc_weight * witness.crossing_changes + witness.moves
                solved += 1
            else:
                assert row["L10"] is None and row["L1000"] is None and not row["incumbents"]
        assert total_steps == arm["continuation_state_evaluations"]
        assert total_scores == arm["continuation_action_scores"]
        assert arm["continuation_neural_evaluations"] == (
            0 if arm["mode"] == "uniform" else total_steps
        )
        summary.append(
            {
                "mode": arm["mode"],
                "episodes": episodes,
                "steps_replayed": total_steps,
                "solved_roots": solved,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    for name, expected in report["inputs"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    for name, expected in report["sources"].items():
        path = Path(name)
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        archived = args.report.parent / "source_snapshot" / relative
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    summary = audit_report(report)
    result = {
        "status": "passed",
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arms": summary,
        "scope": "replay and resource accounting, not independent scientific promotion",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
