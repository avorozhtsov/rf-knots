"""Extend the full-witness short-u audit to the complete protocol69 graph lineage."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from graph_evaluation_partition import partition_inputs


def stratum(length, strands, *, max_length=254, max_strands=21):
    return {
        (False, False): "within",
        (True, False): "length_only",
        (False, True): "strands_only",
        (True, True): "both",
    }[length > max_length, strands > max_strands]


def inspect(panel, training, excluded):
    expected = {
        "seed": 20261032,
        "attempts": 10000,
        "max_strands": 33,
        "min_depth": 24,
        "max_depth": 96,
    }
    if (
        not panel["complete"]
        or not panel["construction_complete"]
        or panel["schema"] != "rf-knots-gordian-pair-panel-v1"
        or panel["pair_kind"] != "short-u"
        or {k: panel["contract"][k] for k in expected} != expected
    ):
        raise ValueError("short-u prospective construction differs")
    queries, classes, metric_strata, policy_strata = Counter(), Counter(), Counter(), Counter()
    for row in panel["validation"]:
        fp = row["left_fingerprint"]
        g = fp[0], abs(fp[1])
        if g in training | excluded:
            raise ValueError("query collides with optimizer or prior TEST")
        p = row["interval"]
        if p["right"].get("word") != [] or p["right"].get("strands") != 1:
            raise ValueError("not an explicit unknot endpoint")
        queries[g] += 1
        classes[str(p["upper"])] += 1
        metric_strata[stratum(len(p["left"]["word"]), p["left"]["strands"])] += 1
        policy_strata[
            stratum(len(p["left"]["word"]), p["left"]["strands"], max_length=208, max_strands=17)
        ] += 1
    if len(panel["validation"]) < 32 or len(queries) < 8 or any(classes[c] < 8 for c in ("1", "2")):
        raise ValueError("insufficient fixed-budget short-u supply")
    if any(v > 4 for v in queries.values()) or any(v > 32 for v in classes.values()):
        raise ValueError("short-u selection caps differ")
    return {
        "test_query_groups": sorted(queries),
        "witness_cc_classes": dict(classes),
        "raw_metric_size_strata": dict(metric_strata),
        "raw_policy_size_strata": dict(policy_strata),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite panel audit")
    panel = json.loads(args.panel.read_text())
    base = json.loads(args.base_audit.read_text())
    if base["status"] != "passed" or base["inputs"].get(str(args.panel)) != sha(args.panel):
        raise ValueError("require full-witness base audit for this exact panel")
    pins, groups, training, excluded = partition_inputs()
    merge_pins(pins, base["inputs"])
    merge_pins(pins, panel["contract"]["inputs"])
    merge_pins(
        pins, {str(p): sha(p) for p in (args.base_audit, args.panel, Path(__file__).resolve())}
    )
    check_pins(pins)
    result = inspect(panel, training, excluded)
    if base["test_query_groups"] != [list(g) for g in result["test_query_groups"]]:
        raise ValueError("query identity coverage differs from base replay audit")
    result.update(
        status="passed",
        evaluation_ready=True,
        inputs=pins,
        training_groups={k: sorted(v) for k, v in groups.items()},
        query_train_intersection=[],
        query_historical_intersection=[],
        scope="Base audit replays every panel witness and certifies historical optimizer "
        "path separation. This extension includes every ordinary graph TRAIN path "
        "group, audited metric suffix and sentinel. Fingerprints are conservative "
        "groups, not knot identities; zero-CC lifted views are not new knot types.",
    )
    check_pins(pins)
    atomic(args.out, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"inputs", "training_groups"}}),
        flush=True,
    )


if __name__ == "__main__":
    main()
