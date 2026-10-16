"""Separate post-exit consumer of completed136/138 inputs; no model scoring."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from resource_evaluation_inputs import load_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("short-u", "general"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    panel, pins, panel_path = load_panel(args.panel)
    for path in (
        Path(__file__).resolve(),
        Path("scripts/resource_evaluation_inputs.py"),
        Path("research/139-resource-units-evaluation-protocol.md"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    records = panel["validation"]
    bounds = Counter(f"{r['interval']['lower']}..{r['interval']['upper']}" for r in records)
    groups = {(r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in records}
    endpoints = [r["interval"][side] for r in records for side in ("left", "right")]
    check_pins(pins)
    result = dict(
        status="passed",
        evaluation_ready=True,
        panel=str(panel_path),
        panel_sha256=sha(panel_path),
        inputs=pins,
        selected_pairs=len(records),
        lifted_views=len(panel["size_shift"]),
        query_groups=len(groups),
        intervals=dict(bounds),
        lengths=[min(len(s["word"]) for s in endpoints), max(len(s["word"]) for s in endpoints)],
        strands=[min(s["strands"] for s in endpoints), max(s["strands"] for s in endpoints)],
        model_forwards=0,
        optimizer_steps=0,
        scope="Post-exit selected-record replay and partition/attempt-coverage "
        "consumption. Unselected general attempts were replayed by the builder, "
        "not re-replayed here. Query groups are not knot identities; full Q254 "
        "pretraining independence is not established.",
    )
    write_new(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}), flush=True)


if __name__ == "__main__":
    main()
