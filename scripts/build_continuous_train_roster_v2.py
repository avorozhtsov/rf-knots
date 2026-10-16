"""Exclude all prior candidate groups before selecting residual TRAIN supply."""

import argparse
import sys
from pathlib import Path

import build_continuous_train_roster_v1 as base
from build_mcts_training_supervision import atomic, sha

PARENT = Path("artifacts/continuous-train-roster-v1/roster.json")
PARENT_SHA = "cea36964728381466158315252e6db2ea4f64eab17f3ba39e87f11d0a70586a0"
BANK = Path("artifacts/continuous-additive-f-bank-v6")
BANK_SHA = "bc3787e021b48d83caae6c8836debcdb20143202b4e6ee786f56e5d571af46d4"
BANK_AUDIT = Path("artifacts/continuous-additive-f-bank-audit-v6/report.json")
BANK_AUDIT_SHA = "08b45f1f860cd3245d9e2c25e20a8892677bc864cee4cd52a3c0903549649bb7"
PROTOCOL = Path("research/381-residual-corpus-train-supply-v2.md")
ORIGINAL_EXCLUSIONS = base.exclusions


def exclusions():
    groups, states = ORIGINAL_EXCLUSIONS()
    groups.update(tuple(row["query_group"]) for row in base.read(PARENT)["rows"])
    for entry in base.read(BANK / "manifest.json")["roots"]:
        path = BANK / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("current bank root pin differs")
        groups.add(tuple(base.read(path)["query_group"]))
    return groups, states


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    for path, expected in ((PARENT, PARENT_SHA), (BANK / "manifest.json", BANK_SHA),
                           (BANK_AUDIT, BANK_AUDIT_SHA)):
        if sha(path) != expected:
            raise ValueError("frozen residual source input differs")
    if sha(Path(base.__file__)) != "cc8da36424d4698af6beac20e176b868869539beac22914dc67b7a5c99e98eab":
        raise ValueError("prior builder source changed")
    base.exclusions = exclusions
    base.PROTOCOL = PROTOCOL
    base.main()
    path = args.out / "roster.json"
    roster = base.read(path)
    roster["schema"] = "rf-knots-continuous-residual-TRAIN-roster-v2"
    for pin in (PARENT, BANK / "manifest.json", BANK_AUDIT, Path(__file__)):
        roster["inputs"][str(pin)] = sha(pin)
    roster["all_parent_candidates_excluded"] = 256
    atomic(path, roster)
    print({"residual_roster_sha256": sha(path), "roots": len(roster["rows"]),
           "stages": roster["stage_counts"]}, flush=True)


if __name__ == "__main__":
    main()
