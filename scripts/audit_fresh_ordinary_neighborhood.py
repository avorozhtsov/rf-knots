"""Verify protocol72 diagnostic draws against source bytes and independent CC replay."""

import argparse
import json
import random
import sqlite3
from collections import Counter
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_test_panel import trace_bounds
from build_mcts_training_supervision import atomic, sha
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from probe_fresh_ordinary_neighborhood import CONFIG
from probe_fresh_ordinary_neighborhood import SOURCE as PATH_SOURCE
from probe_ordinary_graph_pairs import SOURCE, ordinary

from rf_knots.embedding_replay_corpus import invariants, owner


def endpoint_key(state):
    w = tuple(state["word"])
    return state["strands"], min([w[k:] + w[:k] for k in range(len(w))] or [()])


def check_draw(row, index, references):
    rng = random.Random(CONFIG["seed"] + index)
    rep = rng.choice(sorted(references))
    word, n = references[rep]
    positions = rng.sample(range(len(word)), 1 + index % 2)
    if (
        row["index"] != index
        or row["source"] != {"reference_rep": rep, "positions_from_reference": positions}
        or row["cc_positions"] != list(reversed(positions))
        or row["states"][-1] != {"word": list(word), "strands": n}
    ):
        raise ValueError("deterministic source/draw/reference differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite diagnostic audit")
    report = json.loads(args.report.read_text())
    if (
        report["status"] != "completed"
        or report["contract"]["config"] != CONFIG
        or report["evaluation_ready"] is not False
        or len(report["rows"]) != CONFIG["attempts"]
    ):
        raise ValueError("require completed diagnostic-only fixed probe")
    pins, _, training, historical = partition_inputs()
    merge_pins(pins, report["contract"]["inputs"])
    merge_pins(pins, {str(p): sha(p) for p in (args.report, Path(__file__).resolve())})
    check_pins(pins)
    progress = json.loads((PATH_SOURCE / "progress.json").read_text())
    ids = sorted({r for row in progress["rows"] for r in row["source"]["representations"]})
    db = sqlite3.connect(SOURCE.as_uri() + "?mode=ro", uri=True)
    references = {}
    for rep in ids:
        encoded, split = db.execute(
            "SELECT encoding,split FROM representations WHERE representation_id=?", (rep,)
        ).fetchone()
        if split not in {"validation", "test"}:
            raise ValueError("source reference is TRAIN")
        word, n = ordinary(bytes(encoded))
        if len(word) >= 2:
            references[rep] = word, n
    db.close()
    if len(references) != report["contract"]["source_representations"]:
        raise ValueError("source representation coverage differs")

    @cache
    def fingerprint(word, n):
        return invariants(word, n)

    @cache
    def independent_det(word, n):
        return determinant(word, n)

    reasons, groups, unique = Counter(), set(), {}
    for i, row in enumerate(report["rows"]):
        check_draw(row, i, references)
        trace_bounds(row, fingerprint, independent_det)
        a, b = row["fingerprints"][0], row["fingerprints"][-1]
        group = a[0], abs(a[1])
        if owner(a) != "test":
            expected = "other_owner"
        elif group in training:
            expected = "optimizer_collision"
        elif group in historical:
            expected = "historical_query"
        elif owner(b) == "quarantine":
            expected = "quarantine_reference"
        else:
            expected = "eligible"
        if row["reason"] != expected:
            raise ValueError("query ownership/exclusion reason differs")
        reasons[expected] += 1
        if expected == "eligible":
            groups.add(group)
            key = tuple(sorted(endpoint_key(s) for s in (row["states"][0], row["states"][-1])))
            interval = row["lower"], row["upper"]
            if key in unique and unique[key][0] != interval[0]:
                raise ValueError("equivalent endpoint lower bounds inconsistent")
            if key not in unique or interval[1] < unique[key][1]:
                unique[key] = interval
        if (i + 1) % 128 == 0:
            print(json.dumps({"checked": i + 1, "total": len(report["rows"])}), flush=True)
    summary = {
        "attempts": len(report["rows"]),
        "reasons": dict(reasons),
        "eligible_query_groups": [list(g) for g in sorted(groups)],
        "eligible_paths": reasons["eligible"],
        "unique_endpoint_pairs": len(unique),
        "unique_pair_intervals": dict(Counter(f"{lo}-{hi}" for lo, hi in unique.values())),
    }
    if report["summary"] != summary:
        raise ValueError("probe summary/deduplication differs")
    check_pins(pins)
    atomic(
        args.out,
        {
            "status": "passed",
            "inputs": pins,
            "summary": summary,
            "evaluation_ready": False,
            "scope": "Source bytes/splits and deterministic draws checked independently; "
            "every signed CC path/bound replayed. Supply diagnostic only, "
            "not a model result or permission to reuse these query groups as fresh TEST.",
        },
    )
    print(
        json.dumps(
            {"status": "passed", "eligible_groups": len(groups), "unique_pairs": len(unique)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
