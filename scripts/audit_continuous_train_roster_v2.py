"""Independent full replay of residual candidates and prior-group exclusions."""

import argparse
from pathlib import Path

import audit_continuous_train_roster_v1 as prior
import build_continuous_train_roster_v2 as residual
from build_mcts_training_supervision import atomic, sha


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    prior.source.exclusions = residual.exclusions
    prior.main()
    path = args.out / "report.json"
    report = prior.source.read(path)
    report["schema"] = "rf-knots-continuous-residual-TRAIN-roster-audit-v2"
    report["prior_auditor_sha256"] = report["auditor_sha256"]
    report["auditor_sha256"] = sha(Path(__file__))
    report["parent256_candidate_groups_excluded"] = True
    atomic(path, report)
    print({"residual_report_sha256": sha(path), "roots": report["roots_replayed"]}, flush=True)


if __name__ == "__main__":
    main()
