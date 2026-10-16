"""Protocol139 CPU distances for audited135 models and fresh136/138 tests."""

import argparse
import json
import sys
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from evaluate_raster_log_distances import score_records
from probe_raster_cache_full_network import state_digest
from resource_evaluation_inputs import load_model, load_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--arm", choices=("raw", "units"), required=True)
    parser.add_argument("--stage", choices=("initial", "final"), required=True)
    parser.add_argument("--panel", choices=("short-u", "general"), required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    panel, pins, panel_path = load_panel(args.panel)
    net, objective, model_pins, checkpoint = load_model(
        args.run, args.pair_audit, args.arm, args.stage
    )
    merge_pins(pins, model_pins)
    for path in (
        Path(__file__).resolve(),
        Path("scripts/resource_evaluation_inputs.py"),
        Path("scripts/evaluate_raster_log_distances.py"),
        Path("scripts/evaluate_strand_graph_panel.py"),
        Path("scripts/summarize_clean_paired_evaluation.py"),
        Path("scripts/probe_raster_cache_full_network.py"),
        Path("research/139-resource-units-evaluation-protocol.md"),
        Path("tests/test_resource_evaluation_inputs.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    before = state_digest(net, objective)
    splits = ("validation", "size_shift") if args.panel == "short-u" else ("validation",)
    results = {split: score_records(net, objective, panel[split]) for split in splits}
    if state_digest(net, objective) != before:
        raise ValueError("metric evaluation changed model/objective state")
    check_pins(pins)
    write_new(
        args.out,
        dict(
            status="completed",
            arm=args.arm,
            stage=args.stage,
            panel=str(panel_path),
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            inputs=pins,
            results=results,
            state_sha256=before,
            weights_unchanged=True,
            device="cpu",
            threads=1,
            microbatch=1,
            scope="Fresh synthetic query-separated TEST; Q pretraining exposure "
            "unknown; no optimizer or search step",
        ),
    )
    print(
        json.dumps(
            dict(
                status="completed",
                arm=args.arm,
                stage=args.stage,
                panel=args.panel,
                counts={k: len(panel[k]) for k in splits},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
