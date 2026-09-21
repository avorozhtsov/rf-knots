"""Extend an immutable full rollout-path audit to a new independently audited panel.

This does not claim to replay episodes again. It verifies every byte-pinned
dependency of the prior full replay certificate and compares the new query
groups with its complete optimizer-eligible path group set.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_replay_corpus import owner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_set(rows):
    return {tuple(row) for row in rows}


def check_sets(queries, rollout_groups, metric_groups):
    if not queries or queries & rollout_groups or queries & metric_groups:
        raise ValueError("empty queries or training-path intersection")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-audit", type=Path, required=True)
    p.add_argument("--base-audit-sha256", required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--metric-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite certificate extension")
    if sha(args.base_audit) != args.base_audit_sha256:
        raise ValueError("prior full-replay certificate differs from trusted pin")
    base, metric, panel = (json.loads(path.read_text()) for path in
                           (args.base_audit, args.metric_audit, args.panel))
    if base["status"] != "passed" or metric["status"] != "passed" or not panel["complete"]:
        raise ValueError("require passed full audits and adequate panel")
    if metric["inputs"][str(args.panel)] != sha(args.panel):
        raise ValueError("new panel differs from metric audit")
    pins = {}
    for source in (base["inputs"], metric["inputs"]):
        for name, digest in source.items():
            if name in pins and pins[name] != digest:
                raise ValueError("conflicting shared dependency versions")
            pins[name] = digest
    pins.update({str(path): sha(path) for path in
                 (args.base_audit, args.metric_audit, args.panel, Path(__file__))})
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError("audited input/source bytes changed")
    queries = {(r["left_fingerprint"][0], abs(r["left_fingerprint"][1]))
               for view in ("validation", "size_shift") for r in panel[view]}
    if queries != group_set(metric["test_query_groups"]) or any(owner(fp) != "test" for fp in queries):
        raise ValueError("new query coverage/ownership differs from metric certificate")
    rollouts, metrics = group_set(base["optimizer_path_groups"]), group_set(metric["train_path_groups"])
    check_sets(queries, rollouts, metrics)
    result = {"status": "passed", "inputs": pins,
              "base_full_replay_counts": base["counts"], "new_query_groups": sorted(queries),
              "rollout_path_groups": sorted(rollouts), "metric_path_groups": sorted(metrics),
              "query_rollout_intersection": [], "query_metric_intersection": [],
              "verification": "Reused byte-verified full replay certificate; no episode replay claimed for this extension. "
                              "Both metric and rollout training paths are covered; groups are not knot IDs."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".partial")
    temp.write_text(json.dumps(result, indent=2) + "\n"); temp.replace(args.output)
    print(json.dumps({"status": "passed", "queries": len(queries),
                      "rollout_path_groups": len(rollouts), "metric_path_groups": len(metrics)}))


if __name__ == "__main__":
    main()
