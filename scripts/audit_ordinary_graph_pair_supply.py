"""Verify graph-pair probe source states, replay, bounds, and held-out exclusion.

Only identifies eligible ordinary-distance traces. Does not supply a full
unknotting witness, an optimal policy action, or a knot-identity certificate.
"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant
from probe_ordinary_graph_pairs import CONFIG, SOURCE, ordinary, select_paths

from rf_knots.embedding_replay_corpus import invariants, owner

PROBE = Path("artifacts/ordinary-graph-pair-supply-v1/report.json")
PANELS = tuple(sorted(Path("artifacts").glob("*/panel.json")))
PROVENANCE = {"replayed_cc_upper_bound", "replayed_cc_distinct_attested_knots"}


def trace_bounds(row):
    states, positions = row["states"], row["cc_positions"]
    if len(states) != len(positions) + 1 or len(positions) not in (1, 2):
        raise ValueError("invalid trace coverage")
    fingerprints = []
    for i, state in enumerate(states):
        word, n = tuple(state["word"]), state["strands"]
        fp = invariants(word, n)
        if fp[0] != determinant(word, n):
            raise ValueError("independent determinant differs")
        fingerprints.append(list(fp))
        if i < len(positions):
            p = positions[i]
            if type(p) is not int or not 0 <= p < len(word):
                raise ValueError("invalid CC position")
            after = states[i + 1]
            # Independent literal one-letter flip, not the producer's replay helper.
            if (
                after["strands"] != n
                or tuple(after["word"]) != word[:p] + (-word[p],) + word[p + 1 :]
            ):
                raise ValueError("trace is not a literal CC")
    a, b = fingerprints[0], fingerprints[-1]
    lower = max(abs(a[1] - b[1]) // 2, int(a[0] != b[0]))
    owners = [owner(tuple(f)) for f in fingerprints]
    if (row["lower"], row["upper"], row["fingerprints"], row["owners"], row["all_owner_train"]) != (
        lower,
        len(positions),
        fingerprints,
        owners,
        all(o == "train" for o in owners),
    ) or lower > len(positions):
        raise ValueError("trace labels/partition differ")
    return {(f[0], abs(f[1])) for f in fingerprints}


def heldout_queries(panels):
    groups = set()
    for panel in panels:
        if not isinstance(panel.get("validation"), list):
            raise ValueError("unknown historical panel schema")
        for row in panel["validation"]:
            fp = row["left_fingerprint"]
            groups.add((fp[0], abs(fp[1])))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite supply audit")
    data = json.loads(PROBE.read_text())
    if data["status"] != "completed" or data["config"] != CONFIG or data["train_ready"]:
        raise ValueError("require completed diagnostic-only probe")
    sources = [
        Path(__file__).resolve(),
        Path("scripts/probe_ordinary_graph_pairs.py"),
        PROBE,
        *PANELS,
    ]
    pins = {**data["inputs"], **{str(p): sha(p) for p in sources}}
    check_pins(pins)
    excluded = heldout_queries([json.loads(p.read_text()) for p in PANELS])
    db = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    edges = list(
        db.execute("""SELECT pair_id,left_representation,right_representation FROM pairs
        WHERE metric='cc' AND split='train' AND witness_semantic_length=1 AND provenance IN
        ('replayed_cc_upper_bound','replayed_cc_distinct_attested_knots') ORDER BY pair_id""")
    )
    selection = select_paths(edges, CONFIG)
    if len(data["rows"]) != len(selection) or data["next_index"] != len(selection):
        raise ValueError("incomplete fixed probe")
    eligible, groups, counts = [], set(), Counter()
    for index, (row, selected) in enumerate(zip(data["rows"], selection, strict=True)):
        if row["source"] != selected:
            raise ValueError("fixed source selection differs")
        reps = selected["representations"]
        if len(reps) != len(row["states"]):
            raise ValueError("representation coverage differs")
        for rep, state in zip(reps, row["states"], strict=True):
            encoded, split = db.execute(
                "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
            ).fetchone()
            if split != "train" or ordinary(bytes(encoded)) != (
                tuple(state["word"]),
                state["strands"],
            ):
                raise ValueError("source state/split differs")
        for i, pair in enumerate(selected["pairs"]):
            a, b, metric, split, provenance, length = db.execute(
                "SELECT left_representation,right_representation,metric,split,provenance,"
                "witness_semantic_length FROM pairs WHERE pair_id=?",
                (pair,),
            ).fetchone()
            if (
                set((a, b)) != set(reps[i : i + 2])
                or metric != "cc"
                or split != "train"
                or provenance not in PROVENANCE
                or length != 1
            ):
                raise ValueError("source edge provenance differs")
        path_groups = trace_bounds(row)
        if row["all_owner_train"] and not path_groups & excluded:
            eligible.append(index)
            groups.update(path_groups)
            counts[f"{row['lower']}-{row['upper']}"] += 1
    db.close()
    check_pins(pins)
    atomic(
        args.out,
        {
            "status": "passed",
            "inputs": pins,
            "traces_checked": len(selection),
            "eligible_indices": eligible,
            "eligible_groups": sorted(groups),
            "eligible_intervals": dict(counts),
            "excluded_query_groups": sorted(excluded),
            "historical_panels_checked": len(PANELS),
            "scope": "Ordinary CC traces only; all states owner-TRAIN and absent "
            "from the pinned historical query groups. Known reference endpoints may "
            "be shared. Fingerprint grouping is conservative, not exact knot identity. "
            "No full unknotting witnesses or policy-optimality labels supplied.",
        },
    )
    print(json.dumps({"status": "passed", "eligible": len(eligible), "intervals": dict(counts)}))


if __name__ == "__main__":
    main()
