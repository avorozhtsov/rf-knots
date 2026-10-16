"""Independently replay and audit a pathwise MCTS preference overlay."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from mcts_training_supervision import verified_success_traces

from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import owner, path_fingerprints
from rf_knots.evidence import UnknotWitness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_key(state):
    return (tuple(state.word), state.strands, state.used_cc, state.used_moves)


def json_key(state):
    return (tuple(state["word"]), state["strands"], state["used_cc"], state["used_moves"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite audit")
    overlay = json.loads((args.overlay / "manifest.json").read_text())
    collection = json.loads((args.data / "manifest.json").read_text())
    audit = json.loads(args.audit.read_text())
    original = json.loads((args.original / "manifest.json").read_text())
    if (
        overlay["status"] != "prepared_unverified"
        or collection["status"] != "completed"
        or audit["status"] != "passed"
        or original["status"] != "completed"
    ):
        raise ValueError("upstream status gate failed")
    if any(sha(Path(path)) != checksum for path, checksum in overlay["inputs"].items()):
        raise ValueError("overlay input/source pin differs")
    if overlay["inputs"].get(str(args.audit)) != sha(args.audit):
        raise ValueError("overlay audit identity differs")
    if overlay["inputs"].get(str(args.original / "manifest.json")) != sha(
        args.original / "manifest.json"
    ):
        raise ValueError("overlay original target identity differs")
    roots = json.loads((args.data / "roots.json").read_text())
    gates = {row["condition"]: row for row in audit["conditions"]}
    old_entries = {row["condition"]: row for row in original["conditions"]}
    new_entries = {row["condition"]: row for row in overlay["conditions"]}
    heldout = set(map(tuple, collection["heldout_query_groups"]))
    if len(new_entries) != len(overlay["conditions"]):
        raise ValueError("duplicate preference condition")
    expected_indices = {
        index for index, row in old_entries.items() if row["distinct_success_costs"] > 1
    }
    if set(new_entries) != expected_indices:
        raise ValueError("preference condition coverage differs")
    rows = {}
    for shard in collection["shards"]:
        path = args.data / shard["file"]
        if sha(path) != shard["sha256"]:
            raise ValueError("collection shard differs")
        for row in json.loads(path.read_text()):
            index = row["condition"]
            if index in rows:
                raise ValueError("duplicate collection condition")
            rows[index] = row
    if set(rows) != set(range(len(roots))):
        raise ValueError("collection coverage differs")
    counts = Counter()
    for index in sorted(new_entries):
        entry = new_entries[index]
        path = args.overlay / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("preference shard differs")
        payload = json.loads(path.read_text())
        old_entry = old_entries[index]
        old_path = args.original / old_entry["file"]
        if sha(old_path) != old_entry["sha256"]:
            raise ValueError("original target shard differs")
        old = json.loads(old_path.read_text())
        root, row, gate = roots[index], rows[index], gates[index]
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        if payload["condition"] != index or payload["root"] != root["instance"]:
            raise ValueError("preference root differs")
        if payload["old_target_sha256"] != old_entry["sha256"]:
            raise ValueError("old target identity differs")
        witnesses = [UnknotWitness.from_dict(root["reference_witness"])]
        for episode in row["episodes"]:
            if episode["eligible"] and episode["continuation"]["stop_reason"] == "solved":
                actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
                witnesses.append(UnknotWitness.from_actions(
                    instance.word, instance.strands, instance.spec, actions
                ))
        if gate["best_witness_train_owned"]:
            witnesses.append(UnknotWitness.from_dict(row["witness"]))
        for witness in witnesses:
            witness.verify()
            if any(
                owner(fp) != "train" or (fp[0], abs(fp[1])) in heldout
                for fp in path_fingerprints(witness)
            ):
                raise ValueError("non-TRAIN witness entered preference group")
        witness_hashes = sorted({
            hashlib.sha256(json.dumps(w.to_dict(), sort_keys=True).encode()).hexdigest()
            for w in witnesses
        })
        if payload["witness_sha256"] != witness_hashes:
            raise ValueError("source witness identity differs")
        traces = verified_success_traces(instance, witnesses)
        if payload["unique_success_paths"] != len(traces):
            raise ValueError("unique path count differs")
        costs = {trace["cost"] for trace in traces}
        if payload["distinct_success_costs"] != len(costs) or len(costs) < 2:
            raise ValueError("distinct completed costs differ")
        best = min(costs)
        winners = sum(trace["cost"] == best for trace in traces)
        losers = len(traces) - winners
        expected = defaultdict(Fraction)
        for trace in traces:
            coefficient = Fraction(1, winners) if trace["cost"] == best else Fraction(-1, losers)
            for state, action in trace["rows"]:
                expected[state_key(state), action] += coefficient
        expected = {key: value for key, value in expected.items() if value}
        pools = {json_key(item["state"]): item["actions"] for item in old["positives"]}
        actual = {}
        for item in payload["preferences"]:
            key = json_key(item["state"]), item["action"]
            if key in actual or item["actions"] != pools.get(key[0]):
                raise ValueError("duplicate term or changed action candidate pool")
            if item["action"] not in item["actions"]:
                raise ValueError("preferred action missing from candidate pool")
            actual[key] = Fraction(
                item["coefficient_numerator"], item["coefficient_denominator"]
            )
        if actual != expected:
            raise ValueError("recomputed path preference coefficients differ")
        if old["preferences"] and payload["preferences"] != old["preferences"]:
            raise ValueError("old admitted preference changed")
        counts["preference_conditions"] += 1
        counts["preference_terms"] += len(actual)
        counts["recovered_conditions"] += int(not old["preferences"])
        counts["recovered_terms"] += len(actual) if not old["preferences"] else 0
        counts["replayed_source_witnesses"] += len(witnesses)
    for key in (
        "preference_conditions", "preference_terms", "recovered_conditions", "recovered_terms"
    ):
        if overlay["counts"].get(key) != counts[key]:
            raise ValueError(f"manifest count differs: {key}")
    report = {
        "schema": "rf-knots-mcts-pathwise-preference-audit-v1",
        "status": "passed",
        "overlay_manifest_sha256": sha(args.overlay / "manifest.json"),
        "auditor_sha256": sha(Path(__file__)),
        "counts": dict(counts),
        "scope": "complete TRAIN-owned successes only; no model training or search quality claim",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(counts), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
