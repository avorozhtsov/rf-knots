"""Independent final source/replay/partition/dedup gate for ordinary graph pairs."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_pair_supply import heldout_queries
from build_mcts_training_supervision import atomic, digest, sha
from build_ordinary_graph_metric_corpus import EDGE_SQL, enumerate_paths
from burau_integer_check import determinant
from probe_ordinary_graph_pairs import SOURCE, ordinary

from rf_knots.embedding_replay_corpus import invariants, owner


def inspect_database(db, source, contract, progress=None):
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("database integrity failed")
    metadata = dict(db.execute("SELECT key,value FROM meta"))
    if json.loads(metadata["contract"]) != contract:
        raise ValueError("database contract differs")
    edges = list(source.execute(EDGE_SQL))
    selection = enumerate_paths(edges)
    config = contract["config"]
    if (len(edges), len(selection) - len(edges)) != (
        config["single_edges"],
        config["double_paths"],
    ):
        raise ValueError("source coverage differs")
    if int(metadata["next_index"]) != len(selection):
        raise ValueError("incomplete preparation")
    excluded = set(map(tuple, contract["excluded_groups"]))
    states, needed = {}, {r for p in selection for r in p["representations"]}
    for rep, payload, n, det, sigma, membership, key in db.execute(
        "SELECT * FROM states ORDER BY rep"
    ):
        word = tuple(json.loads(payload))
        encoded, split = source.execute(
            "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
        ).fetchone()
        if split != "train" or ordinary(bytes(encoded)) != (word, n):
            raise ValueError("source state differs")
        fp = invariants(word, n)
        if fp != (int(det), sigma) or fp[0] != determinant(word, n) or owner(fp) != membership:
            raise ValueError("state invariant/owner differs")
        # Canonicalize independently of the builder's canonical helper.
        rotation = min(word[p:] + word[:p] for p in range(len(word))) if word else ()
        expected = json.dumps((n, rotation), separators=(",", ":"))
        if key != expected:
            raise ValueError("signed cyclic canonical form differs")
        states[rep] = (word, n, fp, membership, key)
        if progress and len(states) % 4096 == 0:
            progress({"states_checked": len(states)})
    if set(states) != needed:
        raise ValueError("source state coverage differs")
    best, counts, eligible_reps = {}, Counter(), set()
    rows = db.execute("SELECT * FROM paths ORDER BY path_index")
    checked = 0
    for index, (row, selected) in enumerate(zip(rows, selection, strict=True)):
        idx, payload, positions_json, lo, hi, witness_cc, zero, reason, key = row
        positions = json.loads(positions_json)
        if idx != index or json.loads(payload) != selected:
            raise ValueError("source path enumeration differs")
        path = [states[r] for r in selected["representations"]]
        if len(positions) != len(path) - 1 or witness_cc != len(positions):
            raise ValueError("CC path coverage differs")
        for a, b, p in zip(path[:-1], path[1:], positions, strict=True):
            if (
                type(p) is not int
                or not 0 <= p < len(a[0])
                or a[1] != b[1]
                or a[0][:p] + (-a[0][p],) + a[0][p + 1 :] != b[0]
            ):
                raise ValueError("literal crossing change failed")
        a, b = path[0], path[-1]
        lower = max(abs(a[2][1] - b[2][1]) // 2, int(a[2][0] != b[2][0]))
        equal = a[4] == b[4]
        upper = 0 if equal else len(positions)
        if (lo, hi, zero) != (lower, upper, int(equal)) or lo > hi:
            raise ValueError("path bound/canonical-zero certificate differs")
        why = (
            "other_owner"
            if any(s[3] != "train" for s in path)
            else "historical_query"
            if any((s[2][0], abs(s[2][1])) in excluded for s in path)
            else "eligible"
        )
        if reason != why or key != digest(sorted((a[4], b[4]))):
            raise ValueError("path exclusion/pair key differs")
        counts[reason] += 1
        if reason == "eligible":
            eligible_reps.update(selected["representations"])
            value = (upper, index, lower)
            if key not in best or value < best[key]:
                best[key] = value
        checked += 1
        if progress and checked % 8192 == 0:
            progress({"paths_checked": checked})
    actual = {
        key: (upper, index, lower)
        for key, index, lower, upper in db.execute(
            "SELECT pair_key,path_index,lower_bound,upper_bound FROM training_pairs"
        )
    }
    if actual != best:
        raise ValueError("dedup representatives differ from best certified paths")
    sizes = [states[r] for r in eligible_reps]
    endpoint_groups = set()
    for _, index, _ in best.values():
        reps = selection[index]["representations"]
        endpoint_groups.add(
            tuple(sorted((states[r][2][0], abs(states[r][2][1])) for r in (reps[0], reps[-1])))
        )
    return {
        "states": len(states),
        "paths": dict(counts),
        "deduplicated_pairs": len(best),
        "intervals": dict(Counter(f"{lo}-{hi}" for hi, _, lo in best.values())),
        "support": {
            "eligible_state_representations": len(eligible_reps),
            "eligible_invariant_groups": sorted({(s[2][0], abs(s[2][1])) for s in sizes}),
            "endpoint_group_pairs": len(endpoint_groups),
            "eligible_min_length": min((len(s[0]) for s in sizes), default=None),
            "eligible_max_length": max((len(s[0]) for s in sizes), default=None),
            "eligible_min_strands": min((s[1] for s in sizes), default=None),
            "eligible_max_strands": max((s[1] for s in sizes), default=None),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite full corpus audit")
    manifest_path, corpus_path = args.run / "manifest.json", args.run / "corpus.sqlite"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed" or manifest["train_ready"] is not False:
        raise ValueError("require completed preparation, not partial progress")
    contract = manifest["contract"]
    if sha(corpus_path) != manifest["corpus_sha256"]:
        raise ValueError("completed corpus SHA differs")
    pins = dict(contract["inputs"])
    merge_pins(
        pins, {str(p): sha(p) for p in (Path(__file__).resolve(), manifest_path, corpus_path)}
    )
    check_pins(pins)
    panels = [json.loads(Path(p).read_text()) for p in pins if p.endswith("/panel.json")]
    if sorted(map(list, heldout_queries(panels))) != contract["excluded_groups"]:
        raise ValueError("historical query exclusion set differs")
    source = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    db = sqlite3.connect(corpus_path.resolve().as_uri() + "?mode=ro", uri=True)
    actual = inspect_database(db, source, contract, lambda p: print(json.dumps(p), flush=True))
    db.close()
    source.close()
    # Normalize only JSON tuple/list representation, not scientific values.
    if json.loads(json.dumps(actual)) != manifest["summary"]:
        raise ValueError("published corpus support differs")
    check_pins(pins)
    atomic(
        args.out,
        {
            "status": "passed",
            "train_ready": True,
            "inputs": pins,
            "summary": actual,
            "historical_panels_checked": len(panels),
            "scope": "Ordinary general-Gordian metric pairs; not u labels, "
            "optimal policy actions, independent knot counts or large-knot evidence.",
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "pairs": actual["deduplicated_pairs"],
                "intervals": actual["intervals"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
