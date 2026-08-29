#!/usr/bin/env python3
"""Measure exact graph-based rejection of a virtual embedding pair stream."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from rf_knots.embedding_curriculum_io import load_million_pair_curriculum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--gordian", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.examples <= 0 or args.examples % 20:
        raise ValueError("examples must be a positive multiple of twenty")

    protocol = sqlite3.connect(f"file:{args.protocol}?mode=ro", uri=True)
    anchors = {
        str(row[0]) for row in protocol.execute("SELECT knot_id FROM anchors")
    }
    cc0_representations = {
        int(row[0])
        for row in protocol.execute(
            "SELECT representation_id FROM quarantine_representations "
            "WHERE cc_radius=0"
        )
    }
    one_representations = {
        int(row[0])
        for row in protocol.execute(
            "SELECT representation_id FROM quarantine_representations "
            "WHERE cc_radius=1"
        )
    }
    protocol.close()
    raw = sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True)
    anchor_representations = {
        int(row[0])
        for row in raw.execute(
            "SELECT representation_id FROM representations WHERE knot_id IN ("
            + ",".join("?" for _ in anchors)
            + ")",
            sorted(anchors),
        )
    }
    raw.close()
    cc0_representations -= anchor_representations
    one_representations -= anchor_representations | cc0_representations

    curriculum, pools = load_million_pair_curriculum(
        args.pairs,
        args.geodesic,
        args.hard_cases,
        args.gordian,
        size=args.examples,
        seed=args.seed,
    )
    rejected = Counter()
    rejected_strata: dict[str, Counter] = {}
    for index in range(args.examples):
        spec = curriculum.spec(index)
        endpoints = (spec.left_representation, spec.right_representation)
        if any(value in anchor_representations for value in endpoints):
            reason = "anchor_identity"
        elif any(value in cc0_representations for value in endpoints):
            reason = "cc0_closure"
        elif any(value in one_representations for value in endpoints):
            reason = "verified_one_cc_buffer"
        else:
            reason = "accepted_before_embedding_gate"
        rejected[reason] += 1
        rejected_strata.setdefault(spec.stratum, Counter())[reason] += 1
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(json.dumps({"examined": index + 1, "counts": rejected}), flush=True)

    excluded = args.examples - rejected["accepted_before_embedding_gate"]
    report = {
        "schema": "rf-knots-embedding-validation-leakage-audit-v0",
        "examples": args.examples,
        "seed": args.seed,
        "source_pools": pools,
        "quarantine": {
            "anchor_knot_ids": len(anchors),
            "anchor_representations": len(anchor_representations),
            "cc0_representations_beyond_anchor": len(cc0_representations),
            "one_cc_buffer_representations": len(one_representations),
        },
        "counts": dict(rejected),
        "excluded_before_embedding_gate": excluded,
        "excluded_fraction": excluded / args.examples,
        "by_stratum": {
            name: dict(counts) for name, counts in sorted(rejected_strata.items())
        },
        "embedding_near_duplicate_gate": "not_yet_applied",
    }
    output = args.protocol.with_name("leakage-audit-10m-v0.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
