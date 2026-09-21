"""Audit an existing retrieval report without scoring or selecting a new model.

Distinct provenance rows for the same directed edge are not independent trials.
This audit also checks snapshot membership and overlap with checkpoint-selection
probes. It deliberately does not certify the complete training ancestry.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics

from rf_knots.embedding_route_retrieval import load_hidden_exact_cc1_edges, sha256


def unique_cases(rows):
    result = {}
    for row in rows:
        key = (row["edge_id"], row["source_node"], row["target_node"])
        if key in result:
            previous = result[key]
            for field in ("embedding_rank", "structural_rank", "candidate_pool", "cosine"):
                if previous[field] != row[field]:
                    raise ValueError(f"conflicting duplicate {key}: {field}")
        else:
            result[key] = row
    return list(result.values())


def rank_summary(rows, field):
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "median_rank": statistics.median(r[field] for r in rows),
        "mrr": statistics.mean(1 / r[field] for r in rows),
        **{f"recall_at_{k}": statistics.mean(r[field] <= k for r in rows)
           for k in (1, 10, 100, 1000)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "proof", "pairs", "hard-cases", "training-report", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite an audit")
    report = json.loads(args.report.read_text())
    training = json.loads(args.training_report.read_text())
    hashes = {key: sha256(path) for key, path in {
        "proof": args.proof, "pairs": args.pairs, "hard_cases": args.hard_cases,
        "report": args.report, "training_report": args.training_report,
    }.items()}
    for key in ("proof", "pairs"):
        if report["inputs"][f"{key}_sha256"] != hashes[key]:
            raise ValueError(f"snapshot mismatch: {key}")
    for key in ("pairs", "hard_cases"):
        if training[f"{key}_sha256"] != hashes[key]:
            raise ValueError(f"training snapshot mismatch: {key}")
    if training["checkpoint_sha256"] != report["inputs"]["checkpoint_sha256"]:
        raise ValueError("training and retrieval checkpoints differ")
    raw = report["retrospective_test_reserved"]["cases"]
    if any(r["source_split"] != "test" for r in raw):
        raise ValueError("test report contains a non-test source")
    rows = unique_cases(raw)
    edges = load_hidden_exact_cc1_edges(args.proof, args.pairs)
    lookup = defaultdict(list)
    for edge in edges:
        if edge.source_split == "test":
            lookup[(edge.edge_id, edge.source_node, edge.target_node)].append(edge)
    observed = {(r["edge_id"], r["source_node"], r["target_node"]) for r in rows}
    if observed != set(lookup):
        raise ValueError("test report does not cover every unique source edge")
    with sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM meta"))
        reps = {int(i): (bytes(mirror), split, knot) for i, mirror, split, knot in
                db.execute("SELECT representation_id,mirror_key,split,knot_id FROM representations")}
        train_keys = {reps[int(i)][0] for pair in db.execute(
            "SELECT left_representation,right_representation FROM pairs WHERE split='train'"
        ) for i in pair}
    with sqlite3.connect(f"file:{args.hard_cases}?mode=ro", uri=True) as db:
        probes = list(db.execute(
            "SELECT case_id,left_representation,right_representation FROM hard_case_pairs"
        ))
    probe_keys = defaultdict(set)
    for case, left, right in probes:
        for i in (left, right):
            probe_keys[reps[int(i)][0]].add(case)
    overlaps = []
    for key, cases in sorted(lookup.items()):
        source_keys = {reps[e.source_representation][0] for e in cases}
        overlaps.append({
            "edge_id": key[0], "source_node": key[1], "target_node": key[2],
            "source_in_natural_train_pairs": bool(source_keys & train_keys),
            "source_in_selection_probes": sorted({case for k in source_keys for case in probe_keys[k]}),
        })
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_node"]].append(row)
    source_macro = {}
    for field in ("embedding_rank", "structural_rank"):
        summaries = [rank_summary(group, field) for group in by_source.values()]
        source_macro[field] = {
            key: statistics.mean(s[key] for s in summaries)
            for key in ("mrr", "recall_at_1", "recall_at_10", "recall_at_100", "recall_at_1000")
        }
    exposed = {r['source_node'] for r in overlaps if r['source_in_selection_probes']}
    probe_disjoint = [r for r in rows if r['source_node'] not in exposed]
    result = {
        "status": "completed", "input_sha256": hashes,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "metric_contract": metadata["cc_metric"],
        "raw_rows": len(raw), "unique_edges": len(rows), "unique_sources": len(by_source),
        "edge_weighted": {f: rank_summary(rows, f) for f in ("embedding_rank", "structural_rank")},
        "source_macro": source_macro, "source_membership_audit": overlaps,
        "selection_exposed_sources": sorted(exposed),
        "probe_disjoint_diagnostic": {
            f: rank_summary(probe_disjoint, f) for f in ("embedding_rank", "structural_rank")
        },
        "unique_cases": rows,
        "limitations": [
            "Snapshot metric identifies mirror orbits; this does not validate ordinary oriented Gordian distance.",
            "Source membership audit covers natural pair training and hard-case selection probes only; ancestral, geodesic and synthetic exposure remains unaudited.",
            "Probe-disjoint statistics are a post-audit sensitivity check, not a newly reserved test.",
            "Candidate pool is conditioned on the known target U upper bound; retrieval is not a measured proof improvement.",
            "Multiple edges and sources can share knot type; counts are not independent topology trials.",
        ],
        "code_sha256": {str(p): sha256(p) for p in [Path(__file__), Path('src/rf_knots/embedding_route_retrieval.py')]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('unique_cases', 'source_membership_audit')}))


if __name__ == '__main__':
    main()
