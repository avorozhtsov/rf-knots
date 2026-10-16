"""Protocol135: resumable matched MPS training, raw versus fixed resource units."""

import argparse
import fcntl
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic
from check_qrun_common_witnesses import check_pins, sha
from graph_evaluation_partition import partition_inputs
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import policy_backward, reference_preferences
from raster_log_training import (
    CONFIG as BASE_CONFIG,
)
from raster_log_training import (
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    draw_batch,
    inputs_and_sources,
    matched_initializations,
    relative_due,
    validate_relative_counts,
)
from raster_resource_units import RESOURCE_NAMES, RESOURCE_UNITS, with_resource_units
from run_strand_graph_pilot import load_frozen_corpus, metric_loss
from train_mcts_policy_metric import finite_optimizer, finite_state
from trajectory_log_preference import relative_backward

from rf_knots.embedding_metric import MetricEmbeddingObjective, witness_priority_weight

CONFIG = {
    **BASE_CONFIG,
    "protocol": 135,
    "seed": 20261071,
    "device": "mps",
    "initialization": "q254",
    "arms": ["units", "raw"],
    "resource_names": list(RESOURCE_NAMES),
    "resource_units": list(RESOURCE_UNITS),
    "dev_steps": [0, 2048],
    "automatic_mps_fallback": False,
}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_checkpoint(directory, payload):
    """Publish a content-addressed immutable checkpoint before its manifest."""
    temporary = directory / "checkpoint.partial"
    torch.save(cpu_tree(payload), temporary)
    checksum = sha(temporary)
    name = f"step-{payload['steps']:06d}-{checksum[:16]}.pt"
    target = directory / name
    if target.exists() and sha(target) != checksum:
        raise ValueError("checkpoint name collision")
    temporary.replace(target)
    return {"file": name, "sha256": checksum, "steps": payload["steps"]}


def verify_checkpoint(directory, entry, contract, arm):
    path = directory / entry["file"]
    if path.parent != directory or sha(path) != entry["sha256"]:
        raise ValueError("committed checkpoint identity differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload["contract_sha256"] != contract
        or payload["arm"] != arm
        or payload["steps"] != entry["steps"]
        or payload["config"] != CONFIG
    ):
        raise ValueError("checkpoint feature/config/step contract differs")
    return payload


def optimize(net, objective, optimizer, rows, positives, relative):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    weights = [
        witness_priority_weight("cc", r.spec.distance_upper, r.witness_semantic_length)
        for r in rows
    ]
    metric = 0.0
    for row, weight in zip(rows, weights, strict=True):
        loss, _ = metric_loss(net, objective, [row], "mps")
        scaled = loss * (weight / sum(weights))
        if not torch.isfinite(scaled):
            raise ValueError("nonfinite metric loss")
        scaled.backward()
        metric += float(scaled.detach())
    losses = policy_backward(
        net, positives, None, microbatch=1, positive_weight=CONFIG["positive_weight"]
    )
    if relative is not None:
        losses.update(
            relative_backward(
                net,
                relative,
                microbatch=1,
                weight=CONFIG["relative_weight"] * CONFIG["relative_every"],
            )
        )
    norm = torch.nn.utils.clip_grad_norm_(
        [*net.parameters(), *objective.parameters()], CONFIG["grad_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {"metric_loss": metric, **losses, "gradient_norm_before_clip": float(norm)}


@torch.no_grad()
def development(net, objective, rows, records):
    net.eval()
    predictions = []
    for row, record in zip(rows, records, strict=True):
        _, predicted = metric_loss(net, objective, [row], "mps")
        p = float(predicted[0])
        if not torch.isfinite(predicted).all():
            raise ValueError("nonfinite DEV prediction")
        lo, hi = row.spec.distance_lower, row.spec.distance_upper
        predictions.append(
            dict(
                index=row.spec.index,
                lower=lo,
                upper=hi,
                prediction=p,
                violation=max(lo - p, p - hi, 0),
                constant1_violation=max(lo - 1, 1 - hi, 0),
                query_group=[record["left_fingerprint"][0], abs(record["left_fingerprint"][1])],
            )
        )
    groups = {}
    for p in predictions:
        groups.setdefault(tuple(p["query_group"]), []).append(p)
    return dict(
        role="historical DEV, not fresh TEST",
        records=predictions,
        count=len(predictions),
        groups=len(groups),
        mean_interval_violation=sum(r["violation"] for r in predictions) / len(predictions),
        equal_group_interval_violation=sum(
            sum(r["violation"] for r in g) / len(g) for g in groups.values()
        )
        / len(groups),
        constant1_mean_violation=sum(r["constant1_violation"] for r in predictions)
        / len(predictions),
        exact_groups={
            str(lo): dict(
                count=len(g), mean_interval_violation=sum(r["violation"] for r in g) / len(g)
            )
            for lo in sorted({r["lower"] for r in predictions if r["lower"] == r["upper"]})
            if (g := [r for r in predictions if r["lower"] == r["upper"] == lo])
        },
    )


def run_arm(arm, root, net0, architecture, data, dev, contract, pins, sources):
    directory = root / arm
    directory.mkdir(exist_ok=True)
    path = directory / "manifest.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "arm": arm,
            "contract_sha256": contract,
            "status": "preparing",
            "completed_steps": 0,
        }
    )
    if state["contract_sha256"] != contract or state["arm"] != arm:
        raise ValueError("arm contract differs")
    if state["status"] == "completed":
        verify_checkpoint(directory, state["checkpoint"], contract, arm)
        for name, checksum in state["development_sha256"].items():
            if sha(directory / name) != checksum:
                raise ValueError("completed development report changed")
        return state
    state.update(pid=os.getpid(), status="preparing", phase="initial_reference")
    atomic(path, state)
    net = with_resource_units(net0, arm).to("mps")
    objective = MetricEmbeddingObjective().to("mps")
    original, suffix, graph, positives, raw_groups = data
    preferences = reference_preferences(net, raw_groups, microbatch=1)
    reference_digest = digest([old.tolist() for _, old in preferences])
    if state.get("reference_sha256", reference_digest) != reference_digest:
        raise ValueError("initial reference probabilities changed")
    state["reference_sha256"] = reference_digest
    optimizer = torch.optim.AdamW(
        [*net.parameters(), *objective.parameters()],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    rng = random.Random(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.mps.manual_seed(CONFIG["seed"])
    completed, trace = 0, digest([])
    counts = [0] * len(preferences)
    if "checkpoint" in state:
        payload = verify_checkpoint(directory, state["checkpoint"], contract, arm)
        net.load_state_dict(payload["model"], strict=True)
        objective.load_state_dict(payload["objective"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["rng"])
        torch.set_rng_state(payload["torch_rng"])
        torch.mps.set_rng_state(payload["mps_rng"])
        completed, trace, counts = (payload[k] for k in ("steps", "draw_digest", "relative_counts"))
        validate_relative_counts(counts, len(preferences), completed)

    def checkpoint(step):
        finite_state(net, objective)
        finite_optimizer(optimizer)
        torch.mps.synchronize()
        return save_checkpoint(
            directory,
            dict(
                model=net.state_dict(),
                objective=objective.state_dict(),
                optimizer=optimizer.state_dict(),
                architecture=architecture,
                config=CONFIG,
                arm=arm,
                steps=step,
                rng=rng.getstate(),
                torch_rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state(),
                draw_digest=trace,
                relative_counts=counts.copy(),
                contract_sha256=contract,
            ),
        )

    if "initial_checkpoint" not in state:
        if completed:
            raise ValueError("trained state missing initial checkpoint identity")
        state["initial_checkpoint"] = checkpoint(0)
        state["checkpoint"] = state["initial_checkpoint"]
        atomic(path, state)
    if "development_sha256" not in state:
        if completed:
            raise ValueError("training advanced without initial DEV")
        report = development(net, objective, *dev)
        report.update(checkpoint=state["initial_checkpoint"], contract_sha256=contract, arm=arm)
        atomic(directory / "dev-000000.json", report)
        state["development_sha256"] = {"dev-000000.json": sha(directory / "dev-000000.json")}
        atomic(path, state)
    started = time.monotonic()
    state.update(status="training", phase="optimizer")
    atomic(path, state)
    for step in range(completed, CONFIG["steps"]):
        rows, chosen, relative, step_digest, group_index = draw_batch(
            rng, original, suffix, graph, positives, preferences
        )
        if relative_due(step):
            counts[group_index] += 1
        else:
            relative = None
        losses = optimize(net, objective, optimizer, rows, chosen, relative)
        trace = digest([trace, step_digest])
        if (step + 1) % CONFIG["checkpoint_interval"] == 0:
            entry = checkpoint(step + 1)
            if (step + 1) % 256 == 0:
                check_pins(pins)
                check_pins(sources)
            state.update(
                checkpoint=entry,
                completed_steps=step + 1,
                draw_digest=trace,
                relative_counts=counts.copy(),
                **losses,
                seconds_this_run=time.monotonic() - started,
            )
            atomic(directory / f"metrics-{step + 1:06d}.json", state)
            atomic(path, state)
            print(
                json.dumps(
                    {
                        k: state[k]
                        for k in (
                            "arm",
                            "completed_steps",
                            "metric_loss",
                            "positive_loss",
                            "seconds_this_run",
                        )
                    }
                ),
                flush=True,
            )
    state.update(phase="final_development")
    atomic(path, state)
    report = development(net, objective, *dev)
    report.update(checkpoint=state["checkpoint"], contract_sha256=contract, arm=arm)
    atomic(directory / "dev-002048.json", report)
    state["development_sha256"]["dev-002048.json"] = sha(directory / "dev-002048.json")
    check_pins(pins)
    check_pins(sources)
    state.update(status="completed", phase="completed")
    atomic(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS fallback must be disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    pins, sources = inputs_and_sources()
    partition_pins, _, training, _ = partition_inputs()
    for name, checksum in partition_pins.items():
        if name in pins and pins[name] != checksum:
            raise ValueError("conflicting partition provenance")
        pins[name] = checksum
    for name in (
        "scripts/raster_resource_units.py",
        "scripts/train_raster_resource_pair.py",
        "tests/test_raster_resource_units.py",
        "research/135-raster-resource-units-training-protocol.md",
    ):
        sources[str(Path(name).resolve())] = sha(Path(name))
    contract = digest(dict(config=CONFIG, inputs=pins, sources=sources))
    manifest = (
        json.loads(path.read_text())
        if args.resume
        else dict(
            config=CONFIG, inputs=pins, sources=sources, contract_sha256=contract, completed_arms=[]
        )
    )
    if manifest["contract_sha256"] != contract:
        raise ValueError("resume contract changed")
    if not args.resume:
        snapshot = args.out / "source_snapshot"
        snapshot.mkdir()
        for i, name in enumerate(sources):
            shutil.copy2(name, snapshot / f"{i:03d}-{Path(name).name}")
    manifest.update(
        status="preparing", pid=os.getpid(), phase="verify_data", torch_version=torch.__version__
    )
    atomic(path, manifest)
    raw, corpus = load_frozen_corpus(CORPUS)
    if len(raw["validation"]) != 256:
        raise ValueError("unexpected historical DEV coverage")
    dev_groups = {
        (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in raw["validation"]
    }
    if dev_groups & training:
        raise ValueError("historical DEV query overlaps audited optimizer lineage")
    manifest["development"] = dict(
        count=256, query_groups=len(dev_groups), optimizer_query_overlap=[], role="historical DEV"
    )
    atomic(path, manifest)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    data = (original, suffix, load_graph_sampler(GRAPH), *load_policy_targets(SUPERVISION))
    net, states, architecture, _ = matched_initializations(pins)
    net.load_state_dict(states["q254"], strict=True)
    results = []
    for arm in CONFIG["arms"]:
        manifest.update(status="training", current_arm=arm, phase="arm")
        atomic(path, manifest)
        results.append(
            run_arm(
                arm,
                args.out,
                net,
                architecture,
                data,
                (corpus["validation"][0], raw["validation"]),
                contract,
                pins,
                sources,
            )
        )
        if arm not in manifest["completed_arms"]:
            manifest["completed_arms"].append(arm)
        atomic(path, manifest)
    if results[0]["draw_digest"] != results[1]["draw_digest"]:
        raise ValueError("paired data traces differ")
    check_pins(pins)
    check_pins(sources)
    manifest.update(
        status="completed",
        phase="training_and_development_completed",
        current_arm=None,
        independent_search_evaluation="pending",
        independent_distance_evaluation="pending",
    )
    atomic(path, manifest)


if __name__ == "__main__":
    main()
