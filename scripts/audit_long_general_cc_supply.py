"""Post-exit consumer for147; selected mathematical evidence is recomputed."""

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_long_general_cc_supply import (
    CONFIG,
    SCHEMA,
    construct,
    group,
    load_shards,
    pair_key,
    provenance,
    select,
    supply_summary,
)
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant

from rf_knots import reference
from rf_knots.embedding_replay_corpus import invariants, owner


@lru_cache(maxsize=1024)
def checked_invariants(word, strands):
    if reference.num_components(word, strands) != 1:
        raise ValueError("selected state is not a knot")
    fp = invariants(word, strands)
    if fp[0] != determinant(word, strands):
        raise ValueError("Seifert/Burau determinant disagreement")
    return fp


def verify_pair(row, split, training, excluded):
    states, positions = row["states"], row["cc_positions"]
    if not 2 <= len(states) <= 3 or not isinstance(positions, list):
        raise ValueError("invalid selected path coverage")
    fps = []
    for state in states:
        n, word = state["strands"], state["word"]
        if type(n) is not int or n < 1 or not isinstance(word, list) or not word:
            raise ValueError("invalid selected state")
        if any(type(g) is not int or g == 0 or abs(g) >= n for g in word):
            raise ValueError("invalid selected generator")
        fps.append(list(checked_invariants(tuple(word), n)))
    if fps != row["fingerprints"]:
        raise ValueError("selected fingerprints differ")
    a, b = states[0], states[-1]
    rotations = [
        i
        for i in range(len(a["word"]))
        if a["strands"] == b["strands"] and a["word"][i:] + a["word"][:i] == b["word"]
    ]
    if positions:
        if len(states) != len(positions) + 1:
            raise ValueError("selected CC coverage differs")
        for before, after, position in zip(states[:-1], states[1:], positions, strict=True):
            word = before["word"]
            if type(position) is not int or not 0 <= position < len(word):
                raise ValueError("invalid selected CC position")
            expected = word[:position] + [-word[position]] + word[position + 1 :]
            if after != dict(word=expected, strands=before["strands"]):
                raise ValueError("selected CC replay differs")
    elif len(states) != 2 or not rotations:
        raise ValueError("selected zero-CC certificate absent")
    lower = max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0]))
    upper = 0 if rotations else len(positions)
    if not 0 <= lower <= upper <= 2:
        raise ValueError("selected bounds inconsistent")
    expected_interval = dict(
        metric="cc",
        left=a,
        right=b,
        lower=lower,
        upper=upper,
        witness_semantic_length=len(positions),
    )
    if (
        any(type(row[k]) is not int for k in ("lower", "upper"))
        or (row["lower"], row["upper"]) != (lower, upper)
        or row["interval"] != expected_interval
        or row["cyclic_rotation"] != (min(rotations) if rotations else None)
        or row["left_fingerprint"] != fps[0]
        or row["right_fingerprint"] != fps[-1]
        or row["split"] != split
        or row["pair_key"] != pair_key(row)
    ):
        raise ValueError("selected redundant fields differ")
    if any(owner(fp) == "quarantine" for fp in fps) or owner(fps[0]) != split:
        raise ValueError("selected ownership differs")
    if split == "train":
        if any(owner(fp) != "train" or group(fp) in excluded for fp in fps):
            raise ValueError("non-TRAIN or excluded path in TRAIN")
    elif group(fps[0]) in training | excluded:
        raise ValueError("selected held-out query was exposed")
    band = row["source"]["length_band"]
    lo, hi = CONFIG["length_bands"][band]
    if any(
        not lo <= len(s["word"]) <= hi
        or not CONFIG["min_strands"] <= s["strands"] <= CONFIG["max_strands"]
        for s in states
    ):
        raise ValueError("selected size band differs")
    return group(fps[0])


def consume(root):
    corpus_path, progress_path = root / "corpus.json", root / "progress.json"
    corpus, progress = (json.loads(p.read_text()) for p in (corpus_path, progress_path))
    pins, groups, training, excluded = provenance()
    contract = dict(config=CONFIG, inputs=pins)
    if (
        corpus["schema"] != SCHEMA
        or progress["schema"] != SCHEMA
        or corpus["contract"] != contract
        or progress["contract"] != contract
        or progress["status"] != "completed"
        or progress["next_index"] != CONFIG["attempts"]
        or progress["corpus_sha256"] != sha(corpus_path)
        or corpus["construction_complete"] is not True
        or corpus["optimizer_ready"] is not False
        or corpus["training_groups"] != {k: [list(g) for g in sorted(v)] for k, v in groups.items()}
        or corpus["excluded_query_groups"] != [list(g) for g in sorted(excluded)]
        or corpus["model_forwards"] != 0
        or corpus["optimizer_steps"] != 0
    ):
        raise ValueError("completed corpus provenance/coverage differs")
    rows = load_shards(root, progress, replay=False)
    for row in rows:
        if (row["source"], row["states"], row["cc_positions"]) != construct(row["index"]):
            raise ValueError("attempt construction differs")
    records, decisions = select(rows, training, excluded)
    summary, gaps = supply_summary(records)
    if (
        corpus["records"] != records
        or corpus["selection_decisions"] != decisions
        or corpus["summary"] != summary
        or progress["summary"] != summary
        or corpus["support_gaps"] != gaps
        or progress["support_gaps"] != gaps
        or corpus["minimum_support_ready"] != (not gaps)
        or progress["minimum_support_ready"] != (not gaps)
        or corpus["selection_counts"] != dict(Counter(decisions))
        or progress["selection_counts"] != dict(Counter(decisions))
    ):
        raise ValueError("selection or support arithmetic differs")
    query_groups, optimized_groups, keys = {}, set(training), set()
    checked = 0
    for split, selected in records.items():
        query_groups[split] = set()
        for row in selected:
            g = verify_pair(row, split, training, excluded)
            query_groups[split].add(g)
            if row["pair_key"] in keys:
                raise ValueError("duplicate canonical pair across splits")
            keys.add(row["pair_key"])
            if split == "train":
                optimized_groups.update(map(group, row["fingerprints"]))
            checked += 1
            if checked % 128 == 0:
                print(json.dumps(dict(selected_certificates_checked=checked)), flush=True)
    if (
        query_groups["validation"] & optimized_groups
        or query_groups["test"] & optimized_groups
        or query_groups["validation"] & query_groups["test"]
    ):
        raise ValueError("query split overlap")
    artifacts = [corpus_path, progress_path, *(root / s["file"] for s in progress["shards"])]
    sources = [Path(__file__).resolve(), Path("tests/test_long_general_cc_audit.py")]
    merge_pins(pins, {str(p): sha(p) for p in artifacts + sources})
    check_pins(pins)
    return dict(
        status="passed",
        training_ready=not gaps,
        minimum_support_ready=not gaps,
        corpus=str(corpus_path),
        corpus_sha256=sha(corpus_path),
        inputs=pins,
        attempts=len(rows),
        selected_certificates_checked=checked,
        summary=summary,
        support_gaps=gaps,
        optimizer_steps=0,
        model_forwards=0,
        scope="All attempt constructions/hash coverage checked; all selected pairs "
        "replayed with recomputed integer invariants. Rejected invariants were checked "
        "by the pinned builder, not repeated by this consumer. Query-group isolation "
        "is not full Q254 pretraining independence or a knot-type count.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("audit output must be new")
    report = consume(args.data)
    atomic(args.out, report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"inputs", "summary"}}), flush=True
    )


if __name__ == "__main__":
    main()
