"""149 checkpoint-specific cache parity on non-TEST engineering fixtures."""

import argparse
import json
import sys
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from check_resource_search_cache import check_model
from long_general_evaluation_inputs import load_inputs, load_model

MODELS = (("control", "initial"), ("control", "final"), ("long", "final"))
SCHEMA = "rf-knots-long-general-cache-parity-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".partial").exists():
        raise ValueError("new cache-gate output required")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    parent, audit, pins = load_inputs(args.run, args.pair_audit)
    for name in (
        "scripts/check_long_general_search_cache.py",
        "scripts/check_resource_search_cache.py",
        "scripts/resource_cost_search.py",
        "scripts/probe_raster_tile_cache.py",
        "scripts/probe_raster_cache_search.py",
        "scripts/probe_raster_cache_full_network.py",
        "tests/test_long_general_search.py",
    ):
        path = Path(name).resolve()
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    results = {}
    for arm, stage in MODELS:
        net, objective, checkpoint, identity = load_model(args.run, parent, audit, pins, arm, stage)
        result = check_model(net, objective)
        result.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha(checkpoint),
            shared_initial_identity=identity,
        )
        results[f"{arm}-{stage}"] = result
        check_pins(pins)
        print(json.dumps(dict(arm=arm, stage=stage, status="passed")), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(
        args.out,
        dict(
            schema=SCHEMA,
            status="passed",
            inputs=pins,
            models=results,
            training_contract_sha256=parent["contract_sha256"],
            cache_capacity=512,
            device="cpu",
            threads=1,
            torch_version=torch.__version__,
            scope="Unchanged139 engineering fixtures and tolerances, three149 models. "
            "No TEST scoring or scientific quality/speed claim.",
        ),
    )


if __name__ == "__main__":
    main()
