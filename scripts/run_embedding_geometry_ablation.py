#!/usr/bin/env python3
"""Compare CC distance geometries from one frozen embedding checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from rf_knots.embedding_metric import CC_GEOMETRIES

SCHEMA = "rf-knots-embedding-geometry-ablation-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def deterministic_seed(parent_sha256: str, geometry: str, master_seed: int) -> int:
    material = f"{parent_sha256}:{geometry}:{master_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)


def verified_report(checkpoint: Path, report: Path) -> dict | None:
    if not checkpoint.exists() or not report.exists():
        return None
    payload = json.loads(report.read_text())
    if payload.get("status") != "completed":
        return None
    if payload.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"completed report does not match checkpoint: {report}")
    return payload


def run_geometry(
    args: argparse.Namespace,
    geometry: str,
    parent_payload: dict,
    parent_sha: str,
    parent_lineage: int,
) -> dict:
    directory = args.output_dir / geometry
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    progress = directory / "progress.pt"
    report = directory / "report.json"
    log = directory / "train.log"
    existing = verified_report(checkpoint, report)
    if existing is not None:
        return record_from_report(geometry, checkpoint, report, existing, reused=True)

    config = parent_payload["config"]
    steps = (args.examples_each + args.batch_size - 1) // args.batch_size
    actual_examples = steps * args.batch_size
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
        args.learning_rate_schedule,
        "--minimum-learning-rate-ratio",
        str(args.minimum_learning_rate_ratio),
        "--cc-geometry",
        geometry,
        "--cc-reference-cosine",
        str(args.cc_reference_cosine),
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
        str(max(20, actual_examples // 20 * 20)),
        "--seed",
        str(deterministic_seed(parent_sha, geometry, args.master_seed)),
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
            raise RuntimeError(f"complete progress checkpoint lacks report: {progress}")
    else:
        command.extend(
            (
                "--parent-checkpoint",
                str(args.parent_checkpoint),
                "--parent-lineage-examples-seen",
                str(parent_lineage),
                "--reset-optimizer",
                "--reset-objective",
            )
        )
    if args.amp:
        command.append("--amp")
    with log.open("a") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"geometry arm failed: {geometry}; see {log}")
    result = verified_report(checkpoint, report)
    if result is None:
        raise RuntimeError(f"geometry arm produced no verified report: {geometry}")
    return record_from_report(geometry, checkpoint, report, result, reused=False)


def record_from_report(
    geometry: str, checkpoint: Path, report: Path, payload: dict, *, reused: bool
) -> dict:
    return {
        "geometry": geometry,
        "status": "completed",
        "reused": reused,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "report": str(report.resolve()),
        "selection_score": float(payload["selection_score"]),
        "selected_examples_seen": int(payload["selected_examples_seen"]),
        "trained_examples_seen": int(payload["config"]["steps"])
        * int(payload["config"]["batch_size"]),
        "lineage_examples_seen": int(payload["lineage_examples_seen"]),
        "validation_loss": float(payload["selected_validation_loss"]),
        "protected": payload["selected_protected"],
        "validation_pairs": int(payload["config"]["validation_pairs"]),
        "protected_and_gordian_pairs": int(payload["config"]["protected_and_gordian_pairs"]),
    }


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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--examples-each", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate-schedule", choices=("constant", "cosine"), default="cosine"
    )
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--cc-reference-cosine", type=float, default=0.4)
    parser.add_argument("--master-seed", type=int, default=20260830)
    parser.add_argument("--validation-pairs", type=int, default=2048)
    parser.add_argument("--gordian-validation-pairs", type=int, default=974)
    parser.add_argument("--curve-points", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.examples_each <= 0 or args.batch_size <= 0 or args.curve_points <= 0:
        raise ValueError("examples, batch size, and curve points must be positive")
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
        raise FileNotFoundError(f"missing geometry-ablation inputs: {missing}")
    parent_report_path = args.parent_checkpoint.with_name("report.json")
    parent_report = json.loads(parent_report_path.read_text())
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
    manifest = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": parent_sha,
        "parent_selection_score": float(parent_report["selection_score"]),
        "held_out_test_used": False,
        "geometries": list(CC_GEOMETRIES),
        "requested_training_examples": len(CC_GEOMETRIES) * args.examples_each,
        "config": {
            "batch_size": args.batch_size,
            "examples_each": args.examples_each,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": args.learning_rate_schedule,
            "cc_reference_cosine": args.cc_reference_cosine,
        },
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "records": [],
        "selected": None,
    }
    manifest_path = args.output_dir / "ablation.json"
    atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    for geometry in CC_GEOMETRIES:
        record = run_geometry(args, geometry, parent_payload, parent_sha, parent_lineage)
        if (record["validation_pairs"], record["protected_and_gordian_pairs"]) != expected_contract:
            raise RuntimeError("geometry arm validation contract differs from parent")
        manifest["records"].append(record)
        atomic_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)
    candidates = manifest["records"] + [
        {
            "geometry": "parent",
            "selection_score": float(parent_report["selection_score"]),
            "checkpoint": str(args.parent_checkpoint.resolve()),
            "checkpoint_sha256": parent_sha,
        }
    ]
    selected = min(candidates, key=lambda row: (row["selection_score"], row["geometry"]))
    manifest["selected"] = selected
    manifest["actual_training_examples"] = sum(
        row["trained_examples_seen"] for row in manifest["records"]
    )
    manifest["relative_improvement_percent"] = (
        (float(parent_report["selection_score"]) - float(selected["selection_score"]))
        / float(parent_report["selection_score"])
        * 100
    )
    manifest["status"] = (
        "completed_improved" if selected["geometry"] != "parent" else "completed_no_improvement"
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
