#!/usr/bin/env python3
"""Spend a fixed single-GPU budget on staged embedding optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCHEMA = "rf-knots-embedding-gpu-budget-v0"


@dataclass(frozen=True)
class Arm:
    name: str
    learning_rate: float
    schedule: str
    reset_optimizer: bool
    requested_examples: int


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


def deterministic_seed(parent_sha256: str, stage: int, name: str, seed: int) -> int:
    material = f"{parent_sha256}:{stage}:{name}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)


def checkpoint_lineage(path: Path, payload: dict) -> int:
    if payload.get("lineage_examples_seen") is not None:
        return int(payload["lineage_examples_seen"])
    report_path = path.with_name("report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report.get("status") == "completed"
            and report.get("checkpoint_sha256") == sha256(path)
            and report.get("lineage_examples_seen") is not None
        ):
            return int(report["lineage_examples_seen"])
    return int(payload["examples_seen"])


def verified_report(checkpoint: Path, report: Path) -> dict | None:
    if not checkpoint.exists() or not report.exists():
        return None
    payload = json.loads(report.read_text())
    if payload.get("status") != "completed":
        return None
    if payload.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"completed report does not match checkpoint: {report}")
    return payload


def parent_score(checkpoint: Path) -> float:
    report_path = checkpoint.with_name("report.json")
    if not report_path.exists():
        raise ValueError(f"parent report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("parent report/checkpoint SHA mismatch")
    return float(report["selection_score"])


def stage_one_arms(examples_each: int) -> tuple[Arm, ...]:
    return (
        Arm("preserve-cosine-lr1e4", 1e-4, "cosine", False, examples_each),
        Arm("reset-cosine-lr15e5", 1.5e-4, "cosine", True, examples_each),
        Arm("reset-cosine-lr1e4", 1e-4, "cosine", True, examples_each),
        Arm("reset-cosine-lr5e5", 5e-5, "cosine", True, examples_each),
        Arm("reset-constant-lr1e4", 1e-4, "constant", True, examples_each),
        Arm("reset-constant-lr5e5", 5e-5, "constant", True, examples_each),
    )


def run_arm(
    *,
    args: argparse.Namespace,
    stage: int,
    arm: Arm,
    parent_checkpoint: Path,
    directory: Path,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    progress = directory / "progress.pt"
    report = directory / "report.json"
    log = directory / "train.log"
    reused = verified_report(checkpoint, report)
    if reused is not None:
        return {
            "stage": stage,
            "arm": asdict(arm),
            "status": "completed",
            "reused": True,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": reused["checkpoint_sha256"],
            "report": str(report.resolve()),
            "selection_score": float(reused["selection_score"]),
            "selected_examples_seen": int(reused["selected_examples_seen"]),
            "trained_examples_seen": int(reused["config"]["steps"])
            * int(reused["config"]["batch_size"]),
            "lineage_examples_seen": int(reused["lineage_examples_seen"]),
            "validation_pairs": int(reused["config"]["validation_pairs"]),
            "protected_and_gordian_pairs": int(
                reused["config"]["protected_and_gordian_pairs"]
            ),
        }

    parent_payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    parent_sha = sha256(parent_checkpoint)
    lineage = checkpoint_lineage(parent_checkpoint, parent_payload)
    config = parent_payload["config"]
    steps = (arm.requested_examples + args.batch_size - 1) // args.batch_size
    actual_examples = steps * args.batch_size
    seed = deterministic_seed(parent_sha, stage, arm.name, args.master_seed)
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
        str(arm.learning_rate),
        "--learning-rate-schedule",
        arm.schedule,
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
    ]
    resume = None
    if progress.exists():
        progress_payload = torch.load(progress, map_location="cpu", weights_only=False)
        if int(progress_payload.get("completed_steps", 0)) < steps:
            resume = progress
    if resume:
        command.extend(("--resume", str(resume)))
    else:
        command.extend(
            (
                "--parent-checkpoint",
                str(parent_checkpoint),
                "--parent-lineage-examples-seen",
                str(lineage),
            )
        )
        if arm.reset_optimizer:
            command.append("--reset-optimizer")
    if args.amp:
        command.append("--amp")
    with log.open("a") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"GPU budget arm failed: {directory}; see {log}")
    payload = verified_report(checkpoint, report)
    if payload is None:
        raise RuntimeError(f"GPU budget arm produced no verified report: {directory}")
    return {
        "stage": stage,
        "arm": asdict(arm),
        "status": "completed",
        "reused": False,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "report": str(report.resolve()),
        "selection_score": float(payload["selection_score"]),
        "selected_examples_seen": int(payload["selected_examples_seen"]),
        "trained_examples_seen": actual_examples,
        "lineage_examples_seen": int(payload["lineage_examples_seen"]),
        "validation_pairs": int(payload["config"]["validation_pairs"]),
        "protected_and_gordian_pairs": int(
            payload["config"]["protected_and_gordian_pairs"]
        ),
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
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--master-seed", type=int, default=20260830)
    parser.add_argument("--validation-pairs", type=int, default=2048)
    parser.add_argument("--gordian-validation-pairs", type=int, default=974)
    parser.add_argument("--curve-points", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--stage1-examples-each", type=int, default=500_000)
    parser.add_argument("--stage2-examples-each", type=int, default=2_000_000)
    parser.add_argument("--stage3-examples", type=int, default=3_000_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-noncomparable-validation", action="store_true")
    args = parser.parse_args()
    positive = (
        args.batch_size,
        args.curve_points,
        args.stage1_examples_each,
        args.stage2_examples_each,
        args.stage3_examples,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("batch, curve, and stage sizes must be positive")
    if not 0 <= args.minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must lie in [0, 1]")
    source_paths = {
        "pairs": args.pairs,
        "geodesic": args.geodesic,
        "hard_cases": args.hard_cases,
        "gordian": args.gordian,
        "validation_protocol": args.validation_protocol,
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing GPU budget inputs: {missing}")
    if not args.parent_checkpoint.exists():
        raise FileNotFoundError(args.parent_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "budget.json"
    root_score = parent_score(args.parent_checkpoint)
    root_sha = sha256(args.parent_checkpoint)
    parent_report = json.loads(args.parent_checkpoint.with_name("report.json").read_text())
    parent_validation_pairs = int(parent_report["config"]["validation_pairs"])
    parent_protected_pairs = int(
        parent_report["config"]["protected_and_gordian_pairs"]
    )
    if (
        args.validation_pairs != parent_validation_pairs
        and not args.allow_noncomparable_validation
    ):
        raise ValueError(
            "validation-pairs differs from the parent report; use the frozen count or "
            "explicitly allow a non-comparable smoke run"
        )
    source_report_keys = {
        "pairs": "pairs_sha256",
        "geodesic": "geodesic_sha256",
        "hard_cases": "hard_cases_sha256",
        "gordian": "gordian_sha256",
        "validation_protocol": "validation_protocol_sha256",
    }
    for name, report_key in source_report_keys.items():
        if parent_report.get(report_key) != sha256(source_paths[name]):
            raise ValueError(f"parent/source SHA mismatch for {name}")
    stage1 = stage_one_arms(args.stage1_examples_each)
    requested_budget = (
        len(stage1) * args.stage1_examples_each
        + 2 * args.stage2_examples_each
        + args.stage3_examples
    )
    manifest = {
        "schema": SCHEMA,
        "status": "dry_run" if args.dry_run else "running",
        "device": args.device,
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": root_sha,
        "parent_selection_score": root_score,
        "requested_training_examples": requested_budget,
        "held_out_test_used": False,
        "source_sha256": {key: sha256(path) for key, path in source_paths.items()},
        "config": {
            "batch_size": args.batch_size,
            "workers": args.workers,
            "amp_bfloat16": args.amp,
            "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
            "curve_points": args.curve_points,
        },
        "stage1_plan": [asdict(arm) for arm in stage1],
        "records": [],
        "selected": None,
    }
    atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    records = manifest["records"]

    def validate_contract(record: dict) -> None:
        if not args.allow_noncomparable_validation and (
            record["validation_pairs"] != parent_validation_pairs
            or record["protected_and_gordian_pairs"] != parent_protected_pairs
        ):
            raise RuntimeError("GPU arm validation contract differs from the parent")

    for arm in stage1:
        record = run_arm(
            args=args,
            stage=1,
            arm=arm,
            parent_checkpoint=args.parent_checkpoint,
            directory=args.output_dir / "stage-01" / arm.name,
        )
        validate_contract(record)
        records.append(record)
        atomic_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)
    finalists = sorted(records, key=lambda row: (row["selection_score"], row["arm"]["name"]))[
        :2
    ]
    stage2_records = []
    for rank, finalist in enumerate(finalists):
        source_arm = Arm(**finalist["arm"])
        arm = Arm(
            f"rank{rank}-{source_arm.name}",
            max(2e-5, source_arm.learning_rate * 0.5),
            "cosine",
            False,
            args.stage2_examples_each,
        )
        record = run_arm(
            args=args,
            stage=2,
            arm=arm,
            parent_checkpoint=Path(finalist["checkpoint"]),
            directory=args.output_dir / "stage-02" / arm.name,
        )
        validate_contract(record)
        records.append(record)
        stage2_records.append(record)
        atomic_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)
    champion = min(
        stage2_records, key=lambda row: (row["selection_score"], row["arm"]["name"])
    )
    source_arm = Arm(**champion["arm"])
    final_arm = Arm(
        f"champion-{source_arm.name}",
        max(1e-5, source_arm.learning_rate * 0.5),
        "cosine",
        False,
        args.stage3_examples,
    )
    final_record = run_arm(
        args=args,
        stage=3,
        arm=final_arm,
        parent_checkpoint=Path(champion["checkpoint"]),
        directory=args.output_dir / "stage-03" / final_arm.name,
    )
    validate_contract(final_record)
    records.append(final_record)
    candidates = records + [
        {
            "stage": 0,
            "selection_score": root_score,
            "checkpoint": str(args.parent_checkpoint.resolve()),
            "checkpoint_sha256": root_sha,
            "status": "parent",
        }
    ]
    selected = min(candidates, key=lambda row: (row["selection_score"], row["stage"]))
    manifest["selected"] = selected
    manifest["actual_training_examples"] = sum(
        int(record["trained_examples_seen"]) for record in records
    )
    manifest["relative_improvement_percent"] = (
        (root_score - float(selected["selection_score"])) / root_score * 100
    )
    manifest["status"] = (
        "completed_improved" if selected["stage"] else "completed_no_improvement"
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
