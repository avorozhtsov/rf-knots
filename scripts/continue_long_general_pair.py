"""Protocol248: immutable-lineage, native-MPS continuation in matched rounds."""

import argparse
import copy
import fcntl
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import train_long_general_pair as base
from train_raster_resource_pair import cpu_tree

SCHEMA = "rf-knots-long-general-continuation-v1"
CONFIG = dict(
    parent_steps=2048,
    round_steps=1024,
    checkpoint_interval=64,
    minimum_training_seconds=72 * 3600,
    maximum_training_seconds=96 * 3600,
    dev_interval=4096,
    arms=["control", "long"],
    device="mps",
    test_selection=False,
    optimizer_reset=False,
)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def load_checkpoint(directory, entry, contract, arm):
    if (
        Path(entry["file"]).name != entry["file"]
        or base.sha(directory / entry["file"]) != entry["sha256"]
    ):
        raise ValueError("checkpoint path/hash differs")
    payload = torch.load(directory / entry["file"], map_location="cpu", weights_only=False)
    if (
        payload["contract_sha256"] != contract
        or payload["arm"] != arm
        or payload["steps"] != entry["steps"]
        or payload["continuation_config"] != CONFIG
        or payload["config"] != base.CONFIG
    ):
        raise ValueError("continuation checkpoint contract differs")
    base.validate_relative_counts(
        payload["relative_counts"], payload["relative_groups"], payload["steps"]
    )
    return payload


def restore(payload, net, objective, optimizer, rng, long_rng):
    net.load_state_dict(payload["model"], strict=True)
    objective.load_state_dict(payload["objective"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    rng.setstate(payload["rng"])
    long_rng.setstate(payload["long_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.mps.set_rng_state(payload["mps_rng"])
    for key, actual in (
        ("model", net.state_dict()),
        ("objective", objective.state_dict()),
        ("optimizer", optimizer.state_dict()),
        ("rng", rng.getstate()),
        ("long_rng", long_rng.getstate()),
        ("torch_rng", torch.get_rng_state()),
        ("mps_rng", torch.mps.get_rng_state()),
    ):
        if not equal_tree(cpu_tree(actual), payload[key]):
            raise ValueError(f"immediate restore not bitwise identical: {key}")
    base.finite_state(net, objective)
    base.finite_optimizer(optimizer)


def train_block(
    arm, root, parent_root, parent, net0, architecture, data, dev, target, contract, pins
):
    directory = root / arm
    directory.mkdir(exist_ok=True)
    path = directory / "manifest.json"
    state = (
        base.read(path)
        if path.exists()
        else dict(
            arm=arm,
            contract_sha256=contract,
            completed_steps=CONFIG["parent_steps"],
            seconds_committed=0.0,
            metrics_history=[],
            development_sha256={},
        )
    )
    if state["contract_sha256"] != contract or state["arm"] != arm:
        raise ValueError("arm contract differs")
    for item in state["metrics_history"]:
        if (
            Path(item["file"]).name != item["file"]
            or base.sha(directory / item["file"]) != item["sha256"]
        ):
            raise ValueError("metrics hash differs")
    expected = list(
        range(
            CONFIG["parent_steps"] + CONFIG["checkpoint_interval"],
            state["completed_steps"] + 1,
            CONFIG["checkpoint_interval"],
        )
    )
    if [x["steps"] for x in state["metrics_history"]] != expected:
        raise ValueError("checkpoint history gap")
    for name, checksum in state["development_sha256"].items():
        if Path(name).name != name or base.sha(directory / name) != checksum:
            raise ValueError("DEV hash differs")
    if state["completed_steps"] == target:
        load_checkpoint(directory, state["checkpoint"], contract, arm)
        return state
    if not CONFIG["parent_steps"] <= state["completed_steps"] < target:
        raise ValueError("round target regression")
    state.update(status="preparing", phase="original_initial_policy_reference", pid=os.getpid())
    base.atomic(path, state)
    net = base.with_resource_units(copy.deepcopy(net0), "raw").to("mps")
    objective = base.MetricEmbeddingObjective().to("mps")
    original, suffix, graph, long_source, positives, raw_groups = data
    preferences = base.reference_preferences(net, raw_groups, microbatch=1)
    reference = base.digest([old.tolist() for _, old in preferences])
    if reference != parent[arm]["reference_sha256"]:
        raise ValueError("reference must remain original Q254, not the trained endpoint")
    optimizer = torch.optim.AdamW(
        [*net.parameters(), *objective.parameters()],
        lr=base.CONFIG["lr"],
        weight_decay=base.CONFIG["weight_decay"],
    )
    rng, long_rng = random.Random(), random.Random()
    if "checkpoint" in state:
        payload = load_checkpoint(directory, state["checkpoint"], contract, arm)
    else:
        payload = base.verify_checkpoint(
            parent_root / arm, parent[arm]["checkpoint"], parent["contract_sha256"], arm
        )
    if payload["architecture"] != architecture or payload["relative_groups"] != len(preferences):
        raise ValueError("architecture or relative groups differ")
    if payload["steps"] != state["completed_steps"]:
        raise ValueError("manifest/checkpoint step disagreement")
    restore(payload, net, objective, optimizer, rng, long_rng)
    state.update(
        restore_bitwise_passed=True,
        reference_sha256=reference,
        status="training",
        phase="optimizer",
        target_steps=target,
    )
    base.atomic(path, state)
    started, previous_seconds = time.monotonic(), state["seconds_committed"]

    def checkpoint(step):
        base.finite_state(net, objective)
        base.finite_optimizer(optimizer)
        torch.mps.synchronize()
        payload.update(
            model=net.state_dict(),
            objective=objective.state_dict(),
            optimizer=optimizer.state_dict(),
            steps=step,
            rng=rng.getstate(),
            long_rng=long_rng.getstate(),
            torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state(),
            contract_sha256=contract,
            continuation_config=CONFIG,
            parent_checkpoint=parent[arm]["checkpoint"],
        )
        return base.save_checkpoint(directory, payload)

    if "checkpoint" not in state:
        state["initial_checkpoint"] = checkpoint(payload["steps"])
        state["checkpoint"] = state["initial_checkpoint"]
        base.atomic(path, state)
    for step in range(state["completed_steps"], target):
        draw = base.paired_draw(
            arm, rng, long_rng, original, suffix, graph, long_source, positives, preferences
        )
        relative = draw.relative if base.relative_due(step) else None
        if relative is not None:
            payload["relative_counts"][draw.group_index] += 1
        losses = base.optimize(net, objective, optimizer, draw.metrics, draw.positives, relative)
        payload["shared_draw_digest"] = base.digest(
            [payload["shared_draw_digest"], draw.shared_digest]
        )
        payload["selected_draw_digest"] = base.digest(
            [payload["selected_draw_digest"], draw.selected_digest]
        )
        base.count_batch(payload["size_counters"], draw, relative)
        if (step + 1) % CONFIG["checkpoint_interval"] == 0:
            if (step + 1) % 1024 == 0:
                base.check_pins(pins)
            state.update(
                checkpoint=checkpoint(step + 1),
                completed_steps=step + 1,
                shared_draw_digest=payload["shared_draw_digest"],
                selected_draw_digest=payload["selected_draw_digest"],
                relative_counts=payload["relative_counts"].copy(),
                size_counters=payload["size_counters"].copy(),
                **losses,
                seconds_committed=previous_seconds + time.monotonic() - started,
            )
            metric = directory / f"metrics-{step + 1:06d}-{base.digest(state)[:16]}.json"
            if metric.exists() and base.read(metric) != state:
                raise ValueError("immutable metrics collision")
            if not metric.exists():
                base.atomic(metric, state)
            state["metrics_history"].append(
                dict(file=metric.name, sha256=base.sha(metric), steps=step + 1)
            )
            base.atomic(path, state)
            print(
                json.dumps(
                    {
                        k: state[k]
                        for k in (
                            "arm",
                            "completed_steps",
                            "metric_loss",
                            "positive_loss",
                            "seconds_committed",
                        )
                    }
                ),
                flush=True,
            )
    if target % CONFIG["dev_interval"] == 0:
        for label, values in dev.items():
            name = f"dev-{label}-{target:06d}.json"
            if name not in state["development_sha256"]:
                report = base.development(net, objective, *values)
                report.update(
                    role="DEV only; no TEST selection or promotion",
                    arm=arm,
                    checkpoint=state["checkpoint"],
                    contract_sha256=contract,
                )
                if (directory / name).exists() and base.read(directory / name) != report:
                    raise ValueError("uncommitted DEV differs; preserve for diagnosis")
                if not (directory / name).exists():
                    base.atomic(directory / name, report)
                state["development_sha256"][name] = base.sha(directory / name)
    state.update(status="round_completed", phase="waiting_for_matched_arm")
    base.atomic(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("out", "parent", "parent-audit", "long-data", "execution-gate", "pgx-root"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    if (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("native MPS required; no CPU fallback")
    base.checked_loss_contract()
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    parent = base.read(args.parent / "manifest.json")
    audit = base.read(args.parent_audit)
    if parent["status"] != "completed" or audit["status"] != "passed":
        raise ValueError("completed, independently audited parent required")
    pins = dict(parent["contract"]["inputs"])
    base.merge_pins(pins, audit["inputs"])
    for path in (
        args.parent / "manifest.json",
        args.parent_audit,
        Path(__file__).resolve(),
        Path("research/248-long-general-multiday-continuation-protocol.md").resolve(),
        Path("tests/test_long_general_continuation.py").resolve(),
    ):
        base.merge_pins(pins, {str(path): base.sha(path)})
    for arm in CONFIG["arms"]:
        parent[arm] = base.verify_completed_arm(args.parent / arm, parent["contract_sha256"], arm)
        for path in (
            args.parent / arm / "manifest.json",
            args.parent / arm / parent[arm]["checkpoint"]["file"],
        ):
            base.merge_pins(pins, {str(path): base.sha(path)})
    base.check_pins(pins)
    _, gate_pins = base.execution_gate(args.execution_gate, args.long_data)
    base.merge_pins(pins, gate_pins)
    contract_payload = dict(
        config=CONFIG,
        parent_contract=parent["contract_sha256"],
        inputs=pins,
        torch_version=torch.__version__,
        final_test="new independent panel required",
    )
    contract = base.digest(contract_payload)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    state = (
        base.read(path)
        if args.resume
        else dict(
            schema=SCHEMA,
            contract=contract_payload,
            contract_sha256=contract,
            completed_round_steps=CONFIG["parent_steps"],
            started_at=time.time(),
        )
    )
    if (
        state["schema"] != SCHEMA
        or state["contract"] != contract_payload
        or state["contract_sha256"] != contract
    ):
        raise ValueError("resume contract differs")
    if state.get("status") == "completed":
        for arm in CONFIG["arms"]:
            item = base.read(args.out / arm / "manifest.json")
            load_checkpoint(args.out / arm, item["checkpoint"], contract, arm)
        print("completed continuation verified; not restarted", flush=True)
        return
    snapshot = args.out / "source_snapshot"
    if not args.resume:
        snapshot.mkdir()
        for name, checksum in pins.items():
            if Path(name).suffix == ".py" or name.endswith(
                "248-long-general-multiday-continuation-protocol.md"
            ):
                target = snapshot / f"{checksum[:16]}-{Path(name).name}"
                shutil.copy2(name, target)
    for name, checksum in pins.items():
        if Path(name).suffix == ".py" or name.endswith(
            "248-long-general-multiday-continuation-protocol.md"
        ):
            if base.sha(snapshot / f"{checksum[:16]}-{Path(name).name}") != checksum:
                raise ValueError("source snapshot differs")
    state.update(status="preparing", phase="train_only_data_load", pid=os.getpid())
    base.atomic(path, state)
    long_source, fresh_dev, long_pins = base.load_long_source(args.long_data)
    if any(pins.get(k) != v for k, v in long_pins.items()):
        raise ValueError("TRAIN source differs from bound parent")
    raw, corpus = base.load_frozen_corpus(base.CORPUS)
    original, suffix = base.load_metric_samplers(base.CORPUS, base.SUPERVISION)
    data = (
        original,
        suffix,
        base.load_graph_sampler(base.GRAPH),
        long_source,
        *base.load_policy_targets(base.SUPERVISION),
    )
    net, states, architecture, _ = base.matched_initializations(pins)
    net.load_state_dict(states["q254"], strict=True)
    if base.digest(contract_payload) != contract:
        raise ValueError("initialization added unbound inputs")
    dev = dict(historical=(corpus["validation"][0], raw["validation"]), fresh_general=fresh_dev)
    while True:
        target = state["completed_round_steps"] + CONFIG["round_steps"]
        results = []
        for arm in CONFIG["arms"]:
            state.update(
                status="training", phase="matched_round", current_arm=arm, target_steps=target
            )
            base.atomic(path, state)
            results.append(
                train_block(
                    arm,
                    args.out,
                    args.parent,
                    parent,
                    net,
                    architecture,
                    data,
                    dev,
                    target,
                    contract,
                    pins,
                )
            )
        if (
            results[0]["shared_draw_digest"] != results[1]["shared_draw_digest"]
            or results[0]["reference_sha256"] != results[1]["reference_sha256"]
        ):
            raise ValueError("paired draw traces/reference diverged")
        base.check_pins(pins)
        seconds = sum(x["seconds_committed"] for x in results)
        state.update(
            completed_round_steps=target,
            paired_training_seconds=seconds,
            over_96h=seconds > CONFIG["maximum_training_seconds"],
        )
        base.atomic(path, state)
        if seconds >= CONFIG["minimum_training_seconds"]:
            break
    state.update(
        status="completed",
        phase="training_completed_not_promoted",
        current_arm=None,
        independent_continuation_audit="pending",
        fresh_independent_test="pending",
    )
    base.atomic(path, state)


if __name__ == "__main__":
    main()
