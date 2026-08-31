#!/usr/bin/env python3
"""Run one reproducible end-to-end embedding calibration workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rf_knots.embedding_augmentation import (
    MaterializedSyntheticPair,
    SyntheticEmbeddingIterableDataset,
    SyntheticPairSpec,
    collate_embedding_pairs,
    scramble_zero_cc,
)
from rf_knots.embedding_curriculum_io import load_million_pair_curriculum
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from rf_knots.embedding_gordian import load_gordian_pairs
from rf_knots.embedding_metric import CC_GEOMETRIES, MetricBatch, MetricEmbeddingObjective
from rf_knots.embedding_supervision import UnknotDBEmbeddingSidecar

SCHEMA = "rf-knots-embedding-calibration-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state(device: torch.device) -> dict:
    state = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if device.type == "mps":
        state["torch_mps_rng_state"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(payload: dict, device: torch.device) -> bool:
    """Restore a stochastic trajectory; return False for legacy checkpoints."""
    if "python_random_state" not in payload or "torch_cpu_rng_state" not in payload:
        return False
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if device.type == "mps" and "torch_mps_rng_state" in payload:
        torch.mps.set_rng_state(payload["torch_mps_rng_state"])
    elif device.type == "cuda" and "torch_cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return True


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def scheduled_learning_rate(
    base_learning_rate: float,
    schedule: str,
    minimum_ratio: float,
    completed_steps: int,
    total_steps: int,
) -> float:
    """Return the deterministic LR for the next optimizer step."""
    if base_learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= minimum_ratio <= 1:
        raise ValueError("minimum LR ratio must lie in [0, 1]")
    if not 0 <= completed_steps < total_steps:
        raise ValueError("completed steps must identify a pending optimizer step")
    if schedule == "constant":
        return base_learning_rate
    if schedule != "cosine":
        raise ValueError(f"unsupported learning-rate schedule {schedule!r}")
    progress = completed_steps / max(total_steps - 1, 1)
    multiplier = minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return base_learning_rate * multiplier


def compute_selection_score(validation_loss: float, protected: dict) -> float:
    """Frozen validation contract used for checkpoint and branch selection."""
    return (
        0.25 * validation_loss
        + 0.50 * protected["group_macro_interval_violation"]
        + 0.25 * protected["exact_mae"]
    )


def restore_scaler_state(scaler: torch.amp.GradScaler, state: dict | None) -> bool:
    """Restore AMP scaling only when both source and destination use a scaler."""
    if not scaler.is_enabled() or not state:
        return False
    scaler.load_state_dict(state)
    return True


def training_checkpoint(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    completed_steps: int,
    lineage_base_examples: int,
    lineage_parent_sha256: str | None,
    lineage_parent_checkpoint: Path | None,
    best_selection_score: float | None,
    best_completed_steps: int | None,
    best_validation_loss: float | None,
    best_protected: dict | None,
) -> dict:
    """Build one self-contained, resumable training checkpoint."""
    return {
        "schema": SCHEMA,
        "completed_steps": completed_steps,
        "examples_seen": completed_steps * args.batch_size,
        "lineage_examples_seen": lineage_base_examples + completed_steps * args.batch_size,
        "parent_checkpoint_sha256": lineage_parent_sha256,
        "parent_checkpoint": (
            str(lineage_parent_checkpoint.resolve()) if lineage_parent_checkpoint else None
        ),
        "model": model.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": vars(args),
        "best_selection_score": best_selection_score,
        "best_completed_steps": best_completed_steps,
        "best_validation_loss": best_validation_loss,
        "best_protected": best_protected,
        **capture_rng_state(device),
    }


def validation_probe(path: Path, limit: int) -> list[MaterializedSyntheticPair]:
    buckets = defaultdict(list)
    with UnknotDBEmbeddingSidecar(path) as sidecar:
        for pair in sidecar.pairs("validation"):
            buckets[(pair.metric, pair.lower, pair.upper)].append(pair)
        selected = []
        keys = sorted(buckets)
        while len(selected) < limit and any(buckets.values()):
            for key in keys:
                if buckets[key] and len(selected) < limit:
                    selected.append(buckets[key].pop(0))
        rows = []
        for index, pair in enumerate(selected):
            left = sidecar.representation(pair.left_representation)
            right = sidecar.representation(pair.right_representation)
            spec = SyntheticPairSpec(
                index,
                "validation_replay",
                pair.metric,
                pair.left_representation,
                pair.right_representation,
                pair.lower,
                pair.upper,
                pair.witness_semantic_length,
                0,
                0,
                1.0,
            )
            rows.append(
                MaterializedSyntheticPair(
                    spec,
                    pair.metric,
                    left.word,
                    right.word,
                    left.strands,
                    right.strands,
                    pair.witness_semantic_length,
                )
            )
    return rows


def hard_probe(pairs_path: Path, hard_cases_path: Path) -> list[MaterializedSyntheticPair]:
    hard = sqlite3.connect(f"file:{hard_cases_path}?mode=ro", uri=True)
    source_rows = list(
        hard.execute(
            "SELECT case_id,left_representation,right_representation,metric,"
            "distance_lower,distance_upper,witness_semantic_length "
            "FROM hard_case_pairs ORDER BY case_id,left_representation,"
            "right_representation"
        )
    )
    hard.close()
    rows = []
    with UnknotDBEmbeddingSidecar(pairs_path) as sidecar:
        for index, row in enumerate(source_rows):
            case_id, left_id, right_id, metric, lower, upper, length = row
            left = sidecar.representation(int(left_id))
            right = sidecar.representation(int(right_id))
            spec = SyntheticPairSpec(
                index,
                f"protected:{case_id}",
                str(metric),
                int(left_id),
                int(right_id),
                float(lower),
                float(upper),
                int(length),
                0,
                0,
                1.0,
            )
            rows.append(
                MaterializedSyntheticPair(
                    spec,
                    str(metric),
                    left.word,
                    right.word,
                    left.strands,
                    right.strands,
                    int(length),
                )
            )
    return rows


def protocol_validation_probe(
    pairs_path: Path, protocol_path: Path
) -> list[MaterializedSyntheticPair]:
    """Build exact CC=0 probes from validation anchors without touching test."""
    protocol = sqlite3.connect(f"file:{protocol_path}?mode=ro", uri=True)
    knot_ids = {
        str(row[0])
        for row in protocol.execute("SELECT knot_id FROM anchors WHERE split='validation'")
    }
    protocol.close()
    pairs = sqlite3.connect(f"file:{pairs_path}?mode=ro", uri=True)
    placeholders = ",".join("?" for _ in knot_ids)
    representation_ids = [
        int(row[0])
        for row in pairs.execute(
            "SELECT representation_id FROM representations WHERE knot_id IN ("
            + placeholders
            + ") ORDER BY representation_id",
            sorted(knot_ids),
        )
    ]
    pairs.close()
    rows = []
    with UnknotDBEmbeddingSidecar(pairs_path) as sidecar:
        for index, representation_id in enumerate(representation_ids):
            root = sidecar.representation(representation_id)
            variant = scramble_zero_cc(
                root,
                0xA11D17 + index,
                1 + index % 6,
            )
            spec = SyntheticPairSpec(
                index,
                "protocol_validation_cc0",
                "cc",
                representation_id,
                representation_id,
                0.0,
                0.0,
                variant.semantic_length,
                0xA11D17 + index,
                0,
                1.0,
            )
            rows.append(
                MaterializedSyntheticPair(
                    spec,
                    "cc",
                    root.word,
                    variant.word,
                    root.strands,
                    variant.strands,
                    variant.semantic_length,
                )
            )
    return rows


def gordian_probe(
    pairs_path: Path,
    gordian_path: Path,
    *,
    split: str,
    limit: int,
) -> list[MaterializedSyntheticPair]:
    """Build a fixed knot-identity-held-out Gordian challenge probe."""
    buckets = defaultdict(list)
    for pair in load_gordian_pairs(gordian_path, split):
        buckets[(pair.lower, pair.upper)].append(pair)
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    rows = []
    with UnknotDBEmbeddingSidecar(pairs_path) as sidecar:
        for index, pair in enumerate(selected):
            left_id = pair.left.representation_ids[index % len(pair.left.representation_ids)]
            right_id = pair.right.representation_ids[
                (index * 3 + 1) % len(pair.right.representation_ids)
            ]
            left = sidecar.representation(left_id)
            right = sidecar.representation(right_id)
            left_depth, right_depth = index % 7, (index * 5 + 2) % 7
            left_variant = scramble_zero_cc(
                left, 0xF1ED + index, left_depth, mirror=pair.left.mirror
            )
            right_variant = scramble_zero_cc(
                right, 0xF1ED + index * 7, right_depth, mirror=pair.right.mirror
            )
            stratum = (
                f"gordian_{split}_exact3" if pair.exact else f"gordian_{split}_lower{pair.lower:g}"
            )
            spec = SyntheticPairSpec(
                index,
                stratum,
                "cc",
                left_id,
                right_id,
                pair.lower,
                pair.upper,
                0,
                0xF1ED + index,
                0xF1ED + index * 7,
                1.0,
                pair.left.mirror,
                pair.right.mirror,
                left_depth,
                right_depth,
            )
            rows.append(
                MaterializedSyntheticPair(
                    spec,
                    "cc",
                    left_variant.word,
                    right_variant.word,
                    left_variant.strands,
                    right_variant.strands,
                    left_variant.semantic_length + right_variant.semantic_length,
                )
            )
    return rows


@torch.no_grad()
def evaluate(
    model,
    objective,
    rows: list[MaterializedSyntheticPair],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for start in range(0, len(rows), batch_size):
        batch = move(collate_embedding_pairs(rows[start : start + batch_size]), device)
        left = model(batch["left_word"], batch["left_length"], batch["left_strands"])
        right = model(batch["right_word"], batch["right_length"], batch["right_strands"])
        loss, _ = objective(
            left,
            right,
            MetricBatch(
                batch["metric"],
                batch["lower"],
                batch["upper"],
                batch["witness_semantic_length"],
            ),
        )
        losses.append(float(loss.cpu()))
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate_hard_metrics(
    model,
    objective,
    rows: list[MaterializedSyntheticPair],
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    groups = defaultdict(list)
    violations = []
    exact_errors = []
    for start in range(0, len(rows), batch_size):
        batch = move(collate_embedding_pairs(rows[start : start + batch_size]), device)
        left = model(batch["left_word"], batch["left_length"], batch["left_strands"])
        right = model(batch["right_word"], batch["right_length"], batch["right_strands"])
        for index, metric in enumerate(batch["metric"]):
            head = "knot" if metric == "cc" else "isotopy"
            cosine = (left[head][index] * right[head][index]).sum()
            predicted = float((objective.scale(metric) * (1 - cosine)).cpu())
            lower = float(batch["lower"][index].cpu())
            upper = float(batch["upper"][index].cpu())
            violation = max(lower - predicted, predicted - upper, 0.0)
            violations.append(violation)
            if lower == upper:
                exact_errors.append(abs(predicted - lower))
            key = f"{batch['stratum'][index]}:{metric}:{lower:g}-{upper:g}"
            groups[key].append((predicted, violation))
    model.train()
    return {
        "pairs": len(rows),
        "mean_interval_violation": sum(violations) / len(violations),
        "exact_mae": sum(exact_errors) / len(exact_errors),
        "group_macro_interval_violation": sum(
            sum(value[1] for value in values) / len(values) for values in groups.values()
        )
        / len(groups),
        "groups": {
            key: {
                "pairs": len(values),
                "mean_predicted": sum(value[0] for value in values) / len(values),
                "mean_interval_violation": sum(value[1] for value in values) / len(values),
            }
            for key, values in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--gordian", type=Path)
    parser.add_argument("--validation-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning-rate-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-objective", action="store_true")
    parser.add_argument("--cc-geometry", choices=CC_GEOMETRIES, default="chord")
    parser.add_argument("--cc-reference-cosine", type=float, default=0.4)
    parser.add_argument("--cc-rank-weight", type=float, default=0.0)
    parser.add_argument("--cc-rank-margin", type=float, default=0.25)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--virtual-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--validation-pairs", type=int, default=2048)
    parser.add_argument("--gordian-validation-pairs", type=int, default=2048)
    parser.add_argument("--curve-every-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--progress-checkpoint", type=Path)
    parser.add_argument("--checkpoint-selection", choices=("final", "best"), default="final")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-lineage-examples-seen", type=int)
    args = parser.parse_args()
    if args.steps <= args.warmup_steps:
        raise ValueError("steps must exceed warmup steps")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= args.minimum_learning_rate_ratio <= 1:
        raise ValueError("minimum learning-rate ratio must lie in [0, 1]")
    if args.checkpoint_selection == "best" and not args.checkpoint:
        raise ValueError("best checkpoint selection requires --checkpoint")
    if args.progress_checkpoint and not args.checkpoint:
        raise ValueError("progress checkpoint requires --checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")

    if args.resume and args.parent_checkpoint:
        raise ValueError("resume and parent-checkpoint are mutually exclusive")
    if args.resume and args.reset_optimizer:
        raise ValueError("resume cannot reset optimizer state")
    if args.resume and args.reset_objective:
        raise ValueError("resume cannot reset objective state")
    if args.reset_objective and not args.reset_optimizer:
        raise ValueError("reset-objective requires reset-optimizer")
    if not -1.0 < args.cc_reference_cosine < 1.0:
        raise ValueError("cc-reference-cosine must lie strictly inside (-1, 1)")
    if args.cc_rank_weight < 0:
        raise ValueError("cc-rank-weight must be non-negative")
    if args.cc_rank_margin <= 0:
        raise ValueError("cc-rank-margin must be positive")
    if args.parent_lineage_examples_seen is not None and not args.parent_checkpoint:
        raise ValueError("parent lineage override requires parent-checkpoint")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_source = args.resume or args.parent_checkpoint
    source_payload = (
        torch.load(checkpoint_source, map_location="cpu", weights_only=False)
        if checkpoint_source
        else None
    )
    resume_payload = source_payload if args.resume else None
    parent_payload = source_payload if args.parent_checkpoint else None
    start_step = int(resume_payload["completed_steps"]) if resume_payload else 0
    if parent_payload:
        lineage_base_examples = int(
            args.parent_lineage_examples_seen
            if args.parent_lineage_examples_seen is not None
            else parent_payload.get("lineage_examples_seen") or parent_payload["examples_seen"]
        )
    elif resume_payload:
        lineage_base_examples = (
            int(resume_payload.get("lineage_examples_seen", resume_payload["examples_seen"]))
            - start_step * args.batch_size
        )
    else:
        lineage_base_examples = 0
    lineage_parent_checkpoint = args.parent_checkpoint
    if lineage_parent_checkpoint is None and resume_payload:
        original_parent = resume_payload.get("parent_checkpoint") or resume_payload.get(
            "config", {}
        ).get("parent_checkpoint")
        if original_parent:
            lineage_parent_checkpoint = Path(original_parent)
    lineage_parent_sha256 = (
        sha256(args.parent_checkpoint)
        if args.parent_checkpoint
        else resume_payload.get("parent_checkpoint_sha256")
        if resume_payload
        else None
    )
    if start_step >= args.steps:
        raise ValueError("resume checkpoint already reached requested steps")
    curriculum_started = time.perf_counter()
    curriculum, pools = load_million_pair_curriculum(
        args.pairs,
        args.geodesic,
        args.hard_cases,
        args.gordian,
        args.validation_protocol,
        size=args.virtual_size,
        seed=args.seed,
    )
    curriculum_seconds = time.perf_counter() - curriculum_started
    dataset = SyntheticEmbeddingIterableDataset(
        args.pairs,
        curriculum,
        infinite=True,
        start_position=start_step * args.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_embedding_pairs,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=True,
    )
    model = ScalableBraidEmbeddingEncoder(
        width=args.width, depth=args.depth, embedding_dim=args.embedding_dim
    ).to(device)
    objective = MetricEmbeddingObjective(
        cc_geometry=args.cc_geometry,
        cc_rank_weight=args.cc_rank_weight,
        cc_rank_margin=args.cc_rank_margin,
    ).to(device)
    if args.reset_objective:
        objective.calibrate_cc_scale(args.cc_reference_cosine)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if source_payload:
        source_config = source_payload["config"]
        expected_architecture = (args.width, args.depth, args.embedding_dim)
        source_architecture = (
            int(source_config["width"]),
            int(source_config["depth"]),
            int(source_config["embedding_dim"]),
        )
        if expected_architecture != source_architecture:
            raise ValueError("checkpoint and requested encoder architectures differ")
        source_cc_geometry = source_config.get("cc_geometry", "chord")
        if source_cc_geometry != args.cc_geometry and not args.reset_objective:
            raise ValueError(
                "checkpoint CC geometry differs; use --reset-objective and --reset-optimizer"
            )
        model.load_state_dict(source_payload["model"])
        if not args.reset_objective:
            objective.load_state_dict(source_payload["objective"])
        if resume_payload or not args.reset_optimizer:
            optimizer.load_state_dict(source_payload["optimizer"])
            restore_scaler_state(scaler, source_payload.get("scaler"))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    probe = validation_probe(args.pairs, args.validation_pairs)
    protected_probe = hard_probe(args.pairs, args.hard_cases)
    if args.validation_protocol:
        protected_probe.extend(protocol_validation_probe(args.pairs, args.validation_protocol))
    if args.gordian:
        protected_probe.extend(
            gordian_probe(
                args.pairs,
                args.gordian,
                split="validation",
                limit=args.gordian_validation_pairs,
            )
        )
    validation_before = evaluate(model, objective, probe, args.batch_size, device)
    protected_before = evaluate_hard_metrics(
        model, objective, protected_probe, args.batch_size, device
    )
    iterator = iter(loader)
    rng_state_restored = restore_rng_state(resume_payload, device) if resume_payload else False
    step_records = []
    curve = [
        {
            "completed_steps": start_step,
            "examples_seen": start_step * args.batch_size,
            "validation_loss": validation_before,
            "protected": protected_before,
        }
    ]
    initial_loss = None
    best_selection_score = (
        float(resume_payload["best_selection_score"])
        if resume_payload and resume_payload.get("best_selection_score") is not None
        else None
    )
    best_completed_steps = (
        int(resume_payload["best_completed_steps"])
        if resume_payload and resume_payload.get("best_completed_steps") is not None
        else None
    )
    best_validation_loss = (
        float(resume_payload["best_validation_loss"])
        if resume_payload and resume_payload.get("best_validation_loss") is not None
        else None
    )
    best_protected = resume_payload.get("best_protected") if resume_payload else None
    started = time.perf_counter()
    for step in range(start_step, args.steps):
        fetch_started = time.perf_counter()
        batch = move(next(iterator), device)
        synchronize(device)
        fetch_seconds = time.perf_counter() - fetch_started
        train_started = time.perf_counter()
        current_learning_rate = scheduled_learning_rate(
            args.learning_rate,
            args.learning_rate_schedule,
            args.minimum_learning_rate_ratio,
            step,
            args.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp,
        ):
            left = model(batch["left_word"], batch["left_length"], batch["left_strands"])
            right = model(batch["right_word"], batch["right_length"], batch["right_strands"])
            loss, diagnostics = objective(
                left,
                right,
                MetricBatch(
                    batch["metric"],
                    batch["lower"],
                    batch["upper"],
                    batch["witness_semantic_length"],
                ),
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        train_seconds = time.perf_counter() - train_started
        value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = value
        step_records.append(
            {
                "step": step,
                "loss": value,
                "learning_rate": current_learning_rate,
                "fetch_seconds": fetch_seconds,
                "train_seconds": train_seconds,
                "diagnostics": {key: float(value.cpu()) for key, value in diagnostics.items()},
            }
        )
        completed_steps = step + 1
        if args.curve_every_steps and completed_steps % args.curve_every_steps == 0:
            point = {
                "completed_steps": completed_steps,
                "examples_seen": completed_steps * args.batch_size,
                "validation_loss": evaluate(model, objective, probe, args.batch_size, device),
                "protected": evaluate_hard_metrics(
                    model,
                    objective,
                    protected_probe,
                    args.batch_size,
                    device,
                ),
            }
            curve.append(point)
            point_score = compute_selection_score(point["validation_loss"], point["protected"])
            point["selection_score"] = point_score
            if args.checkpoint_selection == "best" and (
                best_selection_score is None or point_score < best_selection_score
            ):
                best_selection_score = point_score
                best_completed_steps = completed_steps
                best_validation_loss = point["validation_loss"]
                best_protected = point["protected"]
                atomic_torch_save(
                    args.checkpoint,
                    training_checkpoint(
                        args=args,
                        model=model,
                        objective=objective,
                        optimizer=optimizer,
                        scaler=scaler,
                        device=device,
                        completed_steps=completed_steps,
                        lineage_base_examples=lineage_base_examples,
                        lineage_parent_sha256=lineage_parent_sha256,
                        lineage_parent_checkpoint=lineage_parent_checkpoint,
                        best_selection_score=best_selection_score,
                        best_completed_steps=best_completed_steps,
                        best_validation_loss=best_validation_loss,
                        best_protected=best_protected,
                    ),
                )
            print(
                json.dumps(
                    {
                        "progress": point["completed_steps"],
                        "examples_seen": point["examples_seen"],
                        "validation_loss": point["validation_loss"],
                        "protected_exact_mae": point["protected"]["exact_mae"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        progress_checkpoint = args.progress_checkpoint or (
            args.checkpoint if args.checkpoint_selection == "final" else None
        )
        if (
            progress_checkpoint
            and args.checkpoint_every_steps
            and completed_steps % args.checkpoint_every_steps == 0
        ):
            atomic_torch_save(
                progress_checkpoint,
                training_checkpoint(
                    args=args,
                    model=model,
                    objective=objective,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    completed_steps=completed_steps,
                    lineage_base_examples=lineage_base_examples,
                    lineage_parent_sha256=lineage_parent_sha256,
                    lineage_parent_checkpoint=lineage_parent_checkpoint,
                    best_selection_score=best_selection_score,
                    best_completed_steps=best_completed_steps,
                    best_validation_loss=best_validation_loss,
                    best_protected=best_protected,
                ),
            )
    synchronize(device)
    total_seconds = time.perf_counter() - started
    validation_after = evaluate(model, objective, probe, args.batch_size, device)
    protected_after = evaluate_hard_metrics(
        model, objective, protected_probe, args.batch_size, device
    )
    final_selection_score = compute_selection_score(validation_after, protected_after)
    if curve[-1]["completed_steps"] != args.steps:
        curve.append(
            {
                "completed_steps": args.steps,
                "examples_seen": args.steps * args.batch_size,
                "lineage_examples_seen": (lineage_base_examples + args.steps * args.batch_size),
                "parent_checkpoint_sha256": (lineage_parent_sha256),
                "validation_loss": validation_after,
                "protected": protected_after,
                "selection_score": final_selection_score,
            }
        )
    else:
        curve[-1]["selection_score"] = final_selection_score
    if args.checkpoint_selection == "best" and (
        best_selection_score is None or final_selection_score < best_selection_score
    ):
        best_selection_score = final_selection_score
        best_completed_steps = args.steps
        best_validation_loss = validation_after
        best_protected = protected_after
        atomic_torch_save(
            args.checkpoint,
            training_checkpoint(
                args=args,
                model=model,
                objective=objective,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                completed_steps=args.steps,
                lineage_base_examples=lineage_base_examples,
                lineage_parent_sha256=lineage_parent_sha256,
                lineage_parent_checkpoint=lineage_parent_checkpoint,
                best_selection_score=best_selection_score,
                best_completed_steps=best_completed_steps,
                best_validation_loss=best_validation_loss,
                best_protected=best_protected,
            ),
        )
    if args.checkpoint and args.checkpoint_selection == "final":
        best_selection_score = final_selection_score
        best_completed_steps = args.steps
        best_validation_loss = validation_after
        best_protected = protected_after
        atomic_torch_save(
            args.checkpoint,
            training_checkpoint(
                args=args,
                model=model,
                objective=objective,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                completed_steps=args.steps,
                lineage_base_examples=lineage_base_examples,
                lineage_parent_sha256=lineage_parent_sha256,
                lineage_parent_checkpoint=lineage_parent_checkpoint,
                best_selection_score=best_selection_score,
                best_completed_steps=best_completed_steps,
                best_validation_loss=best_validation_loss,
                best_protected=best_protected,
            ),
        )
    measured = step_records[min(args.warmup_steps, len(step_records) - 1) :]
    measured_seconds = sum(row["fetch_seconds"] + row["train_seconds"] for row in measured)
    measured_examples = len(measured) * args.batch_size
    selected_score = (
        best_selection_score if args.checkpoint_selection == "best" else final_selection_score
    )
    selected_completed_steps = (
        best_completed_steps if args.checkpoint_selection == "best" else args.steps
    )
    selected_validation_loss = (
        best_validation_loss if args.checkpoint_selection == "best" else validation_after
    )
    selected_protected = best_protected if args.checkpoint_selection == "best" else protected_after
    if selected_completed_steps is None or selected_validation_loss is None:
        raise RuntimeError("checkpoint selection produced no completed training point")
    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "Apple Metal"
            if device.type == "mps"
            else None
        ),
        "host": platform.node(),
        "machine": platform.machine(),
        "config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": args.learning_rate_schedule,
            "minimum_learning_rate_ratio": args.minimum_learning_rate_ratio,
            "reset_optimizer": args.reset_optimizer,
            "reset_objective": args.reset_objective,
            "cc_geometry": args.cc_geometry,
            "cc_reference_cosine": args.cc_reference_cosine,
            "cc_rank_weight": args.cc_rank_weight,
            "cc_rank_margin": args.cc_rank_margin,
            "depth": args.depth,
            "width": args.width,
            "embedding_dim_each": args.embedding_dim,
            "workers": args.workers,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "amp_bfloat16": args.amp,
            "validation_pairs": len(probe),
            "protected_and_gordian_pairs": len(protected_probe),
            "seed": args.seed,
            "infinite_stream": True,
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "virtual_pairs": len(curriculum),
        "start_step": start_step,
        "resume_rng_state_restored": rng_state_restored,
        "source_pools": pools,
        "curriculum_load_seconds": curriculum_seconds,
        "total_loop_seconds": total_seconds,
        "measured_examples": measured_examples,
        "measured_seconds": measured_seconds,
        "examples_per_second": measured_examples / measured_seconds,
        "mean_fetch_seconds": sum(row["fetch_seconds"] for row in measured) / len(measured),
        "mean_train_seconds": sum(row["train_seconds"] for row in measured) / len(measured),
        "initial_loss": initial_loss,
        "final_loss": step_records[-1]["loss"],
        "minimum_loss": min(row["loss"] for row in step_records),
        "validation_loss_before": validation_before,
        "validation_loss_after": validation_after,
        "validation_improvement": validation_before - validation_after,
        "selection_score": selected_score,
        "selection_score_contract": (
            "0.25*validation_loss + 0.50*protected_group_macro_interval_violation "
            "+ 0.25*protected_exact_mae; lower is better"
        ),
        "protected_before": protected_before,
        "protected_after": protected_after,
        "checkpoint_selection": args.checkpoint_selection,
        "selected_completed_steps": selected_completed_steps,
        "selected_examples_seen": selected_completed_steps * args.batch_size,
        "selected_validation_loss": selected_validation_loss,
        "selected_protected": selected_protected,
        "final_selection_score": final_selection_score,
        "learning_curve": curve,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "progress_checkpoint": (
            str(args.progress_checkpoint.resolve()) if args.progress_checkpoint else None
        ),
        "progress_checkpoint_sha256": (
            sha256(args.progress_checkpoint)
            if args.progress_checkpoint and args.progress_checkpoint.exists()
            else None
        ),
        "parent_checkpoint": (
            str(lineage_parent_checkpoint.resolve()) if lineage_parent_checkpoint else None
        ),
        "parent_checkpoint_sha256": lineage_parent_sha256,
        "lineage_examples_seen": (
            lineage_base_examples + selected_completed_steps * args.batch_size
        ),
        "trained_lineage_examples_seen": (lineage_base_examples + args.steps * args.batch_size),
        "pairs_sha256": sha256(args.pairs),
        "geodesic_sha256": sha256(args.geodesic),
        "hard_cases_sha256": sha256(args.hard_cases),
        "gordian_sha256": sha256(args.gordian) if args.gordian else None,
        "validation_protocol_sha256": (
            sha256(args.validation_protocol) if args.validation_protocol else None
        ),
        "steps": step_records,
    }
    atomic_json(args.output, payload)
    summary = {
        key: payload[key]
        for key in (
            "device",
            "parameters",
            "examples_per_second",
            "mean_fetch_seconds",
            "mean_train_seconds",
            "initial_loss",
            "final_loss",
            "minimum_loss",
            "validation_loss_before",
            "validation_loss_after",
            "validation_improvement",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
