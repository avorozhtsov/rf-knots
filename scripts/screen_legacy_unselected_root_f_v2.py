"""Corrected legacy B16 exact-root screen: do not self-exclude its TRAIN source."""

import hashlib
import json
import sys
from pathlib import Path

import screen_legacy_unselected_root_f_v1 as prior
from build_f_curriculum_roster_v1 import (
    F_DEV, LONG, LONG_DEV, SHORT_DEV, SOURCE, TEST150, group,
)
from build_mcts_training_supervision import atomic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_exclusions():
    """Historical exclusions except the source ROOTS term itself."""
    long, source = prior.read(LONG), prior.read(SOURCE)
    excluded = {tuple(row) for row in long["excluded_query_groups"]}
    excluded.update(tuple(row) for row in source["heldout_query_groups"])
    for panel in (SHORT_DEV, LONG_DEV, F_DEV):
        excluded.update(tuple(row["query_group"])
                        for row in prior.read(panel)["rows"])
    excluded.update(group(row["left_fingerprint"])
                    for row in prior.read(TEST150)["validation"])
    return excluded


def main():
    # The v1 screen imported a roster helper whose union includes ROOTS, the
    # very TRAIN source under inspection. Keep v1 and its failed report intact.
    if sha(Path(prior.__file__)) != "24de536c1783903bb861dbd8a4c6035e7af96af09048755ae0f1eaffa3d0373c":
        raise ValueError("pinned prior screen source changed")
    prior.excluded_groups = external_exclusions
    prior.main()
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    report["schema"] = "rf-knots-legacy-unselected-b16-exact-root-F-screen-v2"
    report["prior_screen_source_sha256"] = report["screen_source_sha256"]
    report["screen_source_sha256"] = sha(Path(__file__))
    report["external_exclusion_sha256"] = {
        str(path): sha(path) for path in (LONG, SOURCE, SHORT_DEV,
                                          LONG_DEV, F_DEV, TEST150)
    }
    report["correction"] = "historical roster helper included this screen's own TRAIN roots"
    atomic(report_path, report)
    print(json.dumps({"corrected_report_sha256": sha(report_path),
                      "new_groups": report["distinct_new_train_root_groups"],
                      "decision": report["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
