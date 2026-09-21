"""Transactional full-source ordinary CC-pair corpus preparation, protocol67."""

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_pair_supply import heldout_queries
from build_mcts_training_supervision import atomic, digest, sha
from burau_integer_check import determinant
from probe_ordinary_graph_pairs import SOURCE, SOURCE_SHA, ordinary, replay_cc

from rf_knots.embedding_replay_corpus import canonical, invariants, owner

PROBE_AUDIT = Path("artifacts/ordinary-graph-pair-supply-v1/partition-replay-audit.json")
CONFIG = {
    "commit_paths": 256,
    "single_edges": 62086,
    "double_paths": 28857,
    "canonical_zero": True,
    "dedup": "unordered-signed-cyclic-endpoints-v1",
}
EDGE_SQL = """SELECT p.pair_id,p.left_representation,p.right_representation
FROM pairs p JOIN representations l ON l.representation_id=p.left_representation
JOIN representations r ON r.representation_id=p.right_representation
WHERE p.metric='cc' AND p.split='train' AND l.split='train' AND r.split='train'
AND p.witness_semantic_length=1 AND p.provenance IN
('replayed_cc_upper_bound','replayed_cc_distinct_attested_knots') ORDER BY p.pair_id"""


def enumerate_paths(edges):
    adjacency = defaultdict(list)
    result = []
    for pair, a, b in edges:
        result.append({"representations": [a, b], "pairs": [pair]})
        adjacency[a].append((b, pair))
        adjacency[b].append((a, pair))
    for middle in sorted(adjacency):
        neighbors = sorted(adjacency[middle])
        if len(neighbors) != len({a for a, _ in neighbors}):
            raise ValueError("duplicate source CC neighbor")
        for (a, first), (b, second) in combinations(neighbors, 2):
            result.append({"representations": [a, middle, b], "pairs": [first, second]})
    return result


def initialize(db, contract):
    db.executescript("""
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE states(rep INTEGER PRIMARY KEY, word TEXT NOT NULL, strands INTEGER NOT NULL,
      det TEXT NOT NULL, sigma INTEGER NOT NULL, owner TEXT NOT NULL, canonical TEXT NOT NULL);
    CREATE TABLE paths(path_index INTEGER PRIMARY KEY, source TEXT NOT NULL,
      cc_positions TEXT NOT NULL, lower_bound INTEGER NOT NULL, upper_bound INTEGER NOT NULL,
      witness_cc INTEGER NOT NULL, cyclic_zero INTEGER NOT NULL, reason TEXT NOT NULL,
      pair_key TEXT NOT NULL);
    CREATE TABLE training_pairs(pair_key TEXT PRIMARY KEY, path_index INTEGER NOT NULL,
      lower_bound INTEGER NOT NULL, upper_bound INTEGER NOT NULL);
    CREATE INDEX paths_reason ON paths(reason);
    """)
    db.executemany(
        "INSERT INTO meta VALUES(?,?)",
        [("contract", json.dumps(contract, sort_keys=True)), ("next_index", "0")],
    )
    db.commit()


def state(db, source, rep):
    stored = db.execute(
        "SELECT word,strands,det,sigma,owner,canonical FROM states WHERE rep=?", (rep,)
    ).fetchone()
    if stored:
        return (
            tuple(json.loads(stored[0])),
            stored[1],
            (int(stored[2]), stored[3]),
            stored[4],
            stored[5],
        )
    row = source.execute(
        "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
    ).fetchone()
    if row is None or row[1] != "train":
        raise ValueError("missing/non-TRAIN source state")
    word, n = ordinary(bytes(row[0]))
    fp = invariants(word, n)
    if fp[0] != determinant(word, n):
        raise ValueError("independent determinant mismatch")
    membership, key = owner(fp), json.dumps(canonical(word, n), separators=(",", ":"))
    db.execute(
        "INSERT INTO states VALUES(?,?,?,?,?,?,?)",
        (rep, json.dumps(word), n, str(fp[0]), fp[1], membership, key),
    )
    return word, n, fp, membership, key


def process_path(db, source, index, selected, excluded):
    states = [state(db, source, r) for r in selected["representations"]]
    positions = [
        replay_cc((a[0], a[1]), (b[0], b[1])) for a, b in zip(states[:-1], states[1:], strict=True)
    ]
    a, b = states[0], states[-1]
    lower = max(abs(a[2][1] - b[2][1]) // 2, int(a[2][0] != b[2][0]))
    cyclic_zero = a[4] == b[4]
    upper = 0 if cyclic_zero else len(positions)
    if lower > upper:
        raise ValueError("ordinary bound contradicts replay/cyclic equality")
    reason = (
        "other_owner"
        if any(s[3] != "train" for s in states)
        else "historical_query"
        if any((s[2][0], abs(s[2][1])) in excluded for s in states)
        else "eligible"
    )
    key = digest(sorted((a[4], b[4])))
    db.execute(
        "INSERT INTO paths VALUES(?,?,?,?,?,?,?,?,?)",
        (
            index,
            json.dumps(selected, sort_keys=True),
            json.dumps(positions),
            lower,
            upper,
            len(positions),
            int(cyclic_zero),
            reason,
            key,
        ),
    )
    if reason == "eligible":
        previous = db.execute(
            "SELECT lower_bound,upper_bound FROM training_pairs WHERE pair_key=?", (key,)
        ).fetchone()
        if previous and previous[0] != lower:
            raise ValueError("equivalent pair lower bounds differ")
        if previous is None or upper < previous[1]:
            db.execute(
                "INSERT OR REPLACE INTO training_pairs VALUES(?,?,?,?)", (key, index, lower, upper)
            )


def summary(db):
    return {
        "states": db.execute("SELECT count(*) FROM states").fetchone()[0],
        "paths": dict(db.execute("SELECT reason,count(*) FROM paths GROUP BY reason")),
        "deduplicated_pairs": db.execute("SELECT count(*) FROM training_pairs").fetchone()[0],
        "intervals": {
            f"{lo}-{hi}": count
            for lo, hi, count in db.execute(
                "SELECT lower_bound,upper_bound,count(*) FROM training_pairs GROUP BY 1,2"
            )
        },
    }


def final_support(db):
    states = {
        rep: (len(json.loads(word)), n, (int(det), abs(sigma)))
        for rep, word, n, det, sigma in db.execute("SELECT rep,word,strands,det,sigma FROM states")
    }
    eligible_reps, endpoint_groups = set(), set()
    for (payload,) in db.execute("SELECT source FROM paths WHERE reason='eligible'"):
        eligible_reps.update(json.loads(payload)["representations"])
    for (payload,) in db.execute(
        "SELECT p.source FROM training_pairs t JOIN paths p ON p.path_index=t.path_index"
    ):
        reps = json.loads(payload)["representations"]
        endpoint_groups.add(tuple(sorted((states[reps[0]][2], states[reps[-1]][2]))))
    sizes = [states[r] for r in eligible_reps]
    return {
        "eligible_state_representations": len(eligible_reps),
        "eligible_invariant_groups": sorted({s[2] for s in sizes}),
        "endpoint_group_pairs": len(endpoint_groups),
        "eligible_min_length": min((s[0] for s in sizes), default=None),
        "eligible_max_length": max((s[0] for s in sizes), default=None),
        "eligible_min_strands": min((s[1] for s in sizes), default=None),
        "eligible_max_strands": max((s[1] for s in sizes), default=None),
    }


def prepare(directory, source_path, contract, excluded, *, resume=False):
    """DB transaction owns progress; manifest is a replaceable progress view."""
    if directory.exists() != resume:
        raise ValueError("new output required, or explicit resume")
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_pins(contract["inputs"])
    manifest_path, db_path = directory / "manifest.json", directory / "corpus.sqlite"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["contract"] != contract:
            raise ValueError("resume contract differs")
        if previous["status"] == "completed":
            if sha(db_path) != previous["corpus_sha256"]:
                raise ValueError("completed corpus changed")
            print("already completed, corpus hash verified")
            return
    source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
    if (
        dict(source.execute("SELECT key,value FROM meta")).get("schema")
        != "unknotdb-embedding-pairs-v3"
    ):
        raise ValueError("unsupported source schema")
    edges = list(source.execute(EDGE_SQL))
    paths = enumerate_paths(edges)
    config = contract["config"]
    if len(edges) != config["single_edges"] or len(paths) != len(edges) + config["double_paths"]:
        raise ValueError("full source enumeration coverage differs")
    if not resume:
        for i, name in enumerate(contract["inputs"]):
            path = Path(name)
            if path.suffix in {".py", ".md"}:
                snapshot = directory / "source_snapshot" / f"{i:04d}-{path.name}"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, snapshot)
    db = sqlite3.connect(db_path)
    if not resume:
        initialize(db, contract)
    metadata = dict(db.execute("SELECT key,value FROM meta"))
    if json.loads(metadata["contract"]) != contract:
        raise ValueError("database contract differs")
    next_index = int(metadata["next_index"])
    if (
        not 0 <= next_index <= len(paths)
        or db.execute("SELECT count(*) FROM paths").fetchone()[0] != next_index
    ):
        raise ValueError("invalid committed path coverage")

    def publish(status, info, **extra):
        atomic(
            manifest_path,
            {
                "status": status,
                "pid": os.getpid(),
                "contract": contract,
                "next_index": next_index,
                "total_paths": len(paths),
                "summary": info,
                "train_ready": False,
                **extra,
            },
        )
        print(json.dumps({"status": status, "next_index": next_index, **info}), flush=True)

    publish("preparing", summary(db))
    while next_index < len(paths):
        end = min(len(paths), next_index + config["commit_paths"])
        with db:
            for index in range(next_index, end):
                process_path(db, source, index, paths[index], excluded)
            db.execute("UPDATE meta SET value=? WHERE key='next_index'", (str(end),))
        next_index = end
        publish("preparing", summary(db))
    final = {**summary(db), "support": final_support(db)}
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("prepared database integrity failed")
    db.close()
    source.close()
    check_pins(contract["inputs"])
    publish("completed", final, corpus_sha256=sha(db_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    audit = json.loads(PROBE_AUDIT.read_text())
    if audit["status"] != "passed":
        raise ValueError("require passed ordinary graph probe")
    panels = [Path(p) for p in audit["inputs"] if p.endswith("/panel.json")]
    if len(panels) != audit["historical_panels_checked"]:
        raise ValueError("historical panel coverage differs")
    excluded = heldout_queries([json.loads(p.read_text()) for p in panels])
    paths = [
        Path(__file__).resolve(),
        Path("research/67-full-ordinary-graph-corpus-protocol.md"),
        Path("scripts/audit_ordinary_graph_pair_supply.py"),
        PROBE_AUDIT,
    ]
    pins = dict(audit["inputs"])
    merge_pins(pins, {str(p): sha(p) for p in paths})
    merge_pins(pins, {str(SOURCE): SOURCE_SHA})
    check_pins(pins)
    contract = {
        "schema": "rf-knots-ordinary-graph-metric-corpus-v1",
        "config": CONFIG,
        "inputs": pins,
        "excluded_groups": sorted(map(list, excluded)),
    }
    prepare(args.out, SOURCE, contract, excluded, resume=args.resume)


if __name__ == "__main__":
    main()
