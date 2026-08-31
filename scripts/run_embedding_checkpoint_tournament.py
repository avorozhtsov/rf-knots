#!/usr/bin/env python3
"""Fork one embedding checkpoint, train branches, and retain the best trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

SCHEMA = "rf-knots-embedding-checkpoint-tournament-v0"


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


def branch_seed(parent_sha256: str, rung: int, master_seed: int, branch: int) -> int:
    material = f"{parent_sha256}:{rung}:{master_seed}:{branch}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)


def select_best(records: list[dict]) -> dict:
    completed = [row for row in records if row.get("status") == "completed"]
    if not completed:
        raise RuntimeError("tournament has no completed branch")
    return min(completed, key=lambda row: (float(row["selection_score"]), row["branch"]))


def parent_lineage_examples(checkpoint_path: Path, payload: dict) -> int:
    """Resolve lineage from a checkpoint, with a verified legacy-report fallback."""
    lineage = payload.get("lineage_examples_seen")
    if lineage is not None:
        return int(lineage)
    report_path = checkpoint_path.with_name("report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report.get("status") == "completed"
            and report.get("checkpoint_sha256") == sha256(checkpoint_path)
            and report.get("lineage_examples_seen") is not None
        ):
            return int(report["lineage_examples_seen"])
    return int(payload["examples_seen"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--gordian", type=Path, required=True)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--branches", type=int, default=5)
    parser.add_argument("--rung", type=int, default=1)
    parser.add_argument("--rung-examples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--learning-rate-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument(
        "--checkpoint-selection", choices=("final", "best"), default="final"
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--master-seed", type=int, default=20260829)
    parser.add_argument("--validation-pairs", type=int, default=2048)
    parser.add_argument("--gordian-validation-pairs", type=int, default=974)
    parser.add_argument("--curve-points", type=int, default=10)
    args = parser.parse_args()
    if args.branches < 1 or args.rung < 1 or args.rung_examples <= 0:
        raise ValueError("branches, rung, and rung-examples must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.curve_points <= 0:
        raise ValueError("curve points must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= args.minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must lie in [0, 1]")
    steps = (args.rung_examples + args.batch_size - 1) // args.batch_size
    actual_examples = steps * args.batch_size
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    parent_sha256 = sha256(args.parent_checkpoint)
    parent_lineage = parent_lineage_examples(args.parent_checkpoint, parent)
    config = parent["config"]
    optimizer_lr = (
        args.learning_rate
        if args.learning_rate is not None
        else float(parent["optimizer"]["param_groups"][0]["lr"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "tournament.json"
    records = []
    source_sha256 = {
        "pairs": sha256(args.pairs),
        "geodesic": sha256(args.geodesic),
        "hard_cases": sha256(args.hard_cases),
        "gordian": sha256(args.gordian),
        "validation_protocol": sha256(args.validation_protocol),
    }

    def publish(status: str, best: dict | None = None) -> None:
        atomic_json(
            manifest_path,
            {
                "schema": SCHEMA,
                "status": status,
                "parent_checkpoint": str(args.parent_checkpoint.resolve()),
                "parent_checkpoint_sha256": parent_sha256,
                "parent_lineage_examples_seen": parent_lineage,
                "rung": args.rung,
                "branches": args.branches,
                "master_seed": args.master_seed,
                "requested_rung_examples": args.rung_examples,
                "actual_rung_examples": actual_examples,
                "batch_size": args.batch_size,
                "steps": steps,
                "device": args.device,
                "optimization": {
                    "learning_rate": optimizer_lr,
                    "learning_rate_schedule": args.learning_rate_schedule,
                    "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
                    "reset_optimizer": args.reset_optimizer,
                    "checkpoint_selection": args.checkpoint_selection,
                    "amp_bfloat16": args.amp,
                },
                "source_sha256": source_sha256,
                "selection_contract": (
                    "lowest report.selection_score; deterministic branch-index tie break"
                ),
                "branch_records": records,
                "selected": best,
            },
        )

    publish("running")
    trainer = Path(__file__).with_name("calibrate_embedding_training.py")
    for branch in range(args.branches):
        directory = args.output_dir / f"branch-{branch:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "report.json"
        checkpoint_path = directory / "checkpoint.pt"
        progress_checkpoint_path = directory / "progress.pt"
        seed = branch_seed(parent_sha256, args.rung, args.master_seed, branch)
        record = {
            "branch": branch,
            "seed": seed,
            "status": "running",
            "report": str(report_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
        records.append(record)
        publish("running")
        if report_path.exists() and checkpoint_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("status") == "completed" and report.get("checkpoint_sha256") == sha256(
                checkpoint_path
            ):
                record.update(
                    status="completed",
                    selection_score=float(report["selection_score"]),
                    checkpoint_sha256=report["checkpoint_sha256"],
                    selected_completed_steps=int(
                        report.get("selected_completed_steps", steps)
                    ),
                    selected_lineage_examples_seen=int(report["lineage_examples_seen"]),
                    trained_lineage_examples_seen=int(
                        report.get(
                            "trained_lineage_examples_seen",
                            report["lineage_examples_seen"],
                        )
                    ),
                    reused=True,
                )
                publish("running")
                continue
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
            str(report_path),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(optimizer_lr),
            "--learning-rate-schedule",
            args.learning_rate_schedule,
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
            str(min(10, max(1, steps - 1))),
            "--virtual-size",
            str(max(20, actual_examples // 20 * 20)),
            "--seed",
            str(seed),
            "--validation-pairs",
            str(args.validation_pairs),
            "--gordian-validation-pairs",
            str(args.gordian_validation_pairs),
            "--curve-every-steps",
            str(max(1, steps // args.curve_points)),
            "--checkpoint-every-steps",
            str(max(1, steps // args.curve_points)),
            "--checkpoint-selection",
            args.checkpoint_selection,
        ]
        if args.amp:
            command.append("--amp")
        if args.checkpoint_selection == "best":
            command.extend(("--progress-checkpoint", str(progress_checkpoint_path)))
        partial = None
        resume_path = (
            progress_checkpoint_path
            if progress_checkpoint_path.exists()
            else checkpoint_path
        )
        if resume_path.exists():
            partial = torch.load(resume_path, map_location="cpu", weights_only=False)
        if partial and int(partial.get("completed_steps", 0)) < steps:
            command.extend(("--resume", str(resume_path)))
        else:
            if args.reset_optimizer:
                command.append("--reset-optimizer")
            command.extend(
                (
                    "--parent-checkpoint",
                    str(args.parent_checkpoint),
                    "--parent-lineage-examples-seen",
                    str(parent_lineage),
                )
            )
        log_path = directory / "train.log"
        with log_path.open("a") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            record.update(status="failed", returncode=completed.returncode)
            publish("failed")
            raise RuntimeError(f"branch {branch} failed; see {log_path}")
        report = json.loads(report_path.read_text())
        record.update(
            status="completed",
            selection_score=float(report["selection_score"]),
            checkpoint_sha256=sha256(checkpoint_path),
            selected_completed_steps=int(report.get("selected_completed_steps", steps)),
            selected_lineage_examples_seen=int(report["lineage_examples_seen"]),
            trained_lineage_examples_seen=int(
                report.get("trained_lineage_examples_seen", report["lineage_examples_seen"])
            ),
            reused=False,
        )
        publish("running")
        print(json.dumps(record, sort_keys=True), flush=True)
    best = select_best(records)
    publish("completed", best)
    print(json.dumps(json.loads(manifest_path.read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
