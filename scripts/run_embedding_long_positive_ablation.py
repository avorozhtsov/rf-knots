#!/usr/bin/env python3
"""Evaluate long exact-positive CC augmentation from a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from run_embedding_geometry_ablation import atomic_json, sha256, verified_report

SCHEMA = "rf-knots-embedding-long-positive-ablation-v0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--gordian", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--examples", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--validation-pairs", type=int, default=2048)
    parser.add_argument("--gordian-validation-pairs", type=int, default=974)
    parser.add_argument("--curve-points", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_paths = {
        "pairs": args.pairs,
        "geodesic": args.geodesic,
        "hard_cases": args.hard_cases,
        "gordian": args.gordian,
        "validation_protocol": args.validation_protocol,
    }
    missing = [
        str(path) for path in (*source_paths.values(), args.parent_checkpoint) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing long-positive inputs: {missing}")
    parent_report = json.loads(args.parent_checkpoint.with_name("report.json").read_text())
    parent_sha = sha256(args.parent_checkpoint)
    if parent_report.get("checkpoint_sha256") != parent_sha:
        raise ValueError("parent report/checkpoint SHA mismatch")
    report_keys = {
        "pairs": "pairs_sha256",
        "geodesic": "geodesic_sha256",
        "hard_cases": "hard_cases_sha256",
        "gordian": "gordian_sha256",
        "validation_protocol": "validation_protocol_sha256",
    }
    for name, report_key in report_keys.items():
        if parent_report.get(report_key) != sha256(source_paths[name]):
            raise ValueError(f"parent/source SHA mismatch for {name}")
    expected_contract = (
        int(parent_report["config"]["validation_pairs"]),
        int(parent_report["config"]["protected_and_gordian_pairs"]),
    )
    if args.validation_pairs != expected_contract[0]:
        raise ValueError("validation-pairs differs from frozen parent contract")
    parent_payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    parent_lineage = int(
        parent_payload.get("lineage_examples_seen", parent_payload["examples_seen"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.pt"
    progress = args.output_dir / "progress.pt"
    report = args.output_dir / "report.json"
    log = args.output_dir / "train.log"
    manifest_path = args.output_dir / "ablation.json"
    manifest = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "parent_selection_score": float(parent_report["selection_score"]),
        "held_out_test_used": False,
        "requested_training_examples": args.examples,
        "curriculum_change": {
            "long_exact_positive_fraction": 0.5,
            "minimum_insertions_per_endpoint": 60,
            "maximum_insertions_per_endpoint": 120,
        },
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "result": None,
        "selected": None,
    }
    atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    existing = verified_report(checkpoint, report)
    if existing is None:
        config = parent_payload["config"]
        steps = (args.examples + args.batch_size - 1) // args.batch_size
        trainer = Path(__file__).with_name("calibrate_embedding_training.py")
        command = [
            sys.executable,
            str(trainer),
            "--pairs",
            str(args.pairs),
            "--geodesic",
            str(args.geodesic),
            "--hard-cases",
            str(args.hard_cases),
            "--gordian",
            str(args.gordian),
            "--validation-protocol",
            str(args.validation_protocol),
            "--output",
            str(report),
            "--checkpoint",
            str(checkpoint),
            "--progress-checkpoint",
            str(progress),
            "--checkpoint-selection",
            "best",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--learning-rate-schedule",
            "cosine",
            "--minimum-learning-rate-ratio",
            str(args.minimum_learning_rate_ratio),
            "--depth",
            str(config["depth"]),
            "--width",
            str(config["width"]),
            "--embedding-dim",
            str(config["embedding_dim"]),
            "--workers",
            str(args.workers),
            "--steps",
            str(steps),
            "--warmup-steps",
            str(min(args.warmup_steps, max(1, steps - 1))),
            "--virtual-size",
            str(max(20, (steps * args.batch_size) // 20 * 20)),
            "--seed",
            str(args.seed),
            "--validation-pairs",
            str(args.validation_pairs),
            "--gordian-validation-pairs",
            str(args.gordian_validation_pairs),
            "--curve-every-steps",
            str(max(1, steps // args.curve_points)),
            "--checkpoint-every-steps",
            str(max(1, steps // args.curve_points)),
        ]
        if progress.exists():
            progress_payload = torch.load(progress, map_location="cpu", weights_only=False)
            if int(progress_payload.get("completed_steps", 0)) < steps:
                command.extend(("--resume", str(progress)))
            else:
                raise RuntimeError("complete progress checkpoint lacks verified report")
        else:
            command.extend(
                (
                    "--parent-checkpoint",
                    str(args.parent_checkpoint),
                    "--parent-lineage-examples-seen",
                    str(parent_lineage),
                    "--reset-optimizer",
                )
            )
        if args.amp:
            command.append("--amp")
        with log.open("a") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"long-positive arm failed; see {log}")
        existing = verified_report(checkpoint, report)
    if existing is None:
        raise RuntimeError("long-positive arm produced no verified report")
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": existing["checkpoint_sha256"],
        "selection_score": float(existing["selection_score"]),
        "selected_examples_seen": int(existing["selected_examples_seen"]),
        "trained_examples_seen": int(existing["config"]["steps"])
        * int(existing["config"]["batch_size"]),
        "lineage_examples_seen": int(existing["lineage_examples_seen"]),
        "validation_loss": float(existing["selected_validation_loss"]),
        "protected": existing["selected_protected"],
    }
    manifest["result"] = result
    if result["selection_score"] < manifest["parent_selection_score"]:
        manifest["status"] = "completed_improved"
        manifest["selected"] = result
    else:
        manifest["status"] = "completed_no_improvement"
        manifest["selected"] = {
            "checkpoint": str(args.parent_checkpoint.resolve()),
            "checkpoint_sha256": parent_sha,
            "selection_score": manifest["parent_selection_score"],
        }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
