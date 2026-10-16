"""Exact F-signal supply audit for the completed Q-run on-policy teacher."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
TEACHER = Path("artifacts/qrun-rung1-onpolicy-teacher-v1")
AUDIT = Path("artifacts/qrun-rung1-onpolicy-teacher-audit-v2/report.json")
ROSTER_SHA = "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506"
TEACHER_MANIFEST_SHA = "658c128c11fed26a639994c5d54af8ad31550f7f189be52c144cbb76c192f6a2"
AUDIT_SHA = "ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new audit report output must not exist")
    if (sha(ROSTER) != ROSTER_SHA or sha(TEACHER / "manifest.json") != TEACHER_MANIFEST_SHA
            or sha(AUDIT) != AUDIT_SHA):
        raise ValueError("pinned roster, teacher or independent audit differs")
    manifest = json.loads((TEACHER / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["next_task"] != 2000:
        raise ValueError("teacher incomplete")
    roots = [row for stage in json.loads(ROSTER.read_text())["stages"] for row in stage]
    if len(roots) != 1000:
        raise ValueError("roster coverage differs")
    counters = defaultdict(Counter)
    cost_roots, f_roots = set(), set()
    for position, entry in enumerate(manifest["shards"]):
        path = TEACHER / entry["file"]
        if entry["file"] != f"condition-{position:04d}.json" or sha(path) != entry["sha256"]:
            raise ValueError(f"teacher shard differs: {position}")
        row = json.loads(path.read_text())
        root_index, weight_index = divmod(position, 2)
        root = roots[root_index]
        weight = (10, 1000)[weight_index]
        stage_key = f"stage{root['stage']}_L{weight}"
        counter = counters[stage_key]
        counter["attempts"] += 1
        instance_dict = root["instances"][weight_index]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        reference = UnknotWitness.from_dict(root["witness"])
        reference.verify()
        reference_cost = weight * reference.crossing_changes + reference.moves
        witnesses = [reference]
        seen = {tuple(step.action for step in reference.steps)}
        for episode, admitted in zip(row["result"]["episodes"],
                                     row["result"]["episodes_train_owned"], strict=True):
            if not admitted or episode["continuation"]["stop_reason"] != "solved":
                continue
            actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
            witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                  instance.spec, actions)
            witness.verify()
            key = tuple(step.action for step in witness.steps)
            if key not in seen:
                seen.add(key)
                witnesses.append(witness)
        traces = verified_success_traces(instance, witnesses)
        if any(trace["cost"] > reference_cost for trace in traces):
            cost_roots.add(root_index)
            counter["cost_separated_groups"] += 1
        if len({trace["cost"] for trace in traces}) < 2:
            continue
        signal = _groups(traces, "F", seed=20261006, k=4,
                         near_margin=2, gap_scale=weight, gap_cap=4)
        if signal and any(group["coefficients"] for group in signal):
            f_roots.add(root_index)
            counter["F_usable_groups"] += 1
            counter["F_terms"] += sum(len(group["coefficients"]) for group in signal)
    report = {"schema": "rf-knots-qrun-F-supply-audit-v1",
              "inputs": {str(path): sha(path) for path in
                         (ROSTER, TEACHER / "manifest.json", AUDIT, Path(__file__),
                          Path("scripts/mcts_tournament_signals_v1.py"),
                          Path("research/273-f-dominant-curriculum-two-architecture-plan-v1.md"))},
              "distinct_cost_separated_roots": len(cost_roots),
              "distinct_F_usable_roots": len(f_roots),
              "by_stage_lambda": {key: dict(value) for key, value in sorted(counters.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"report_sha256": sha(args.out),
                      "distinct_F_usable_roots": len(f_roots),
                      "by_stage_lambda": report["by_stage_lambda"]}, sort_keys=True))


if __name__ == "__main__":
    main()
