"""Final-only tensor, provenance and matched-RNG audit for protocol-57 arms."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, sha
from train_mcts_policy_metric import ACTOR_SHA

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective

RUNS = {name: Path(f"artifacts/mcts-policy-metric-{name}-v1") for name in ("joint", "control")}
EXPECTED_CONFIG = {
    "steps": 1024,
    "seed": 20261023,
    "device": "cpu",
    "threads": 1,
    "lr": 2.5e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "clip": 0.2,
    "positive_weight": 0.1,
    "relative_weight": 0.01,
    "policy_micro_batch": 64,
    "metric_original": 31,
    "metric_suffix": 32,
    "policy_batch": 32,
}


def inspect_checkpoint(checkpoint, initial, *, mode, steps=1024):
    if mode not in ("joint", "metric-control"):
        raise ValueError("unknown training arm")
    if checkpoint["steps"] != steps or checkpoint["architecture"] != initial["architecture"]:
        raise ValueError("final step or architecture differs")
    net = M.SitePolicyValueNet(**checkpoint["architecture"])
    net.load_state_dict(checkpoint["model"], strict=True)
    net.value.requires_grad_(False)
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.load_state_dict(checkpoint["objective"], strict=True)
    optimizer = torch.optim.AdamW(
        [p for p in [*net.parameters(), *objective.parameters()] if p.requires_grad],
        lr=2.5e-5,
        weight_decay=0.01,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    tensors = [*net.state_dict().values(), *objective.state_dict().values()]
    steps_seen = []
    for state in optimizer.state.values():
        tensors.extend(v for v in state.values() if isinstance(v, torch.Tensor))
        step = float(state["step"])
        if not 1 <= step <= steps or step != int(step) or (state["exp_avg_sq"] < 0).any():
            raise ValueError("invalid Adam moments or update counter")
        steps_seen.append(int(step))
    if not steps_seen or any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("empty optimizer or nonfinite tensors")
    if any(g["lr"] != 2.5e-5 or g["weight_decay"] != 0.01 for g in optimizer.param_groups):
        raise ValueError("optimizer configuration differs")
    changed = {
        prefix: sum(
            not torch.equal(v, initial["model"][k])
            for k, v in checkpoint["model"].items()
            if k.startswith(prefix + ".")
        )
        for prefix in ("encoder", "head", "value")
    }
    if changed["encoder"] == 0 or changed["value"] != 0:
        raise ValueError("encoder did not train or frozen value head changed")
    if (mode == "joint" and changed["head"] == 0) or (
        mode == "metric-control" and changed["head"] != 0
    ):
        raise ValueError("policy parameters contradict arm contract")
    return {
        "finite_tensors": len(tensors),
        "changed_model_tensors": changed,
        "optimizer_parameters_with_state": len(steps_seen),
        "optimizer_step_range": [min(steps_seen), max(steps_seen)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite training audit")
    torch.set_num_threads(1)
    manifests, checkpoints, reports, pins = {}, {}, {}, {}
    for name, run in RUNS.items():
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mode = "joint" if name == "joint" else "metric-control"
        if manifest["config"] != {**EXPECTED_CONFIG, "mode": mode}:
            raise ValueError("configuration differs from protocol 57")
        if (
            manifest["status"] != "completed"
            or manifest["completed_steps"] != 1024
            or manifest["config"]["steps"] != 1024
            or manifest["config"]["mode"] != mode
            or manifest["value_head_trained"] is not False
        ):
            raise ValueError("require final protocol-57 arms without value-head training")
        contract = hashlib.sha256(
            json.dumps(
                [manifest["config"], manifest["inputs"], manifest["sources"]], sort_keys=True
            ).encode()
        ).hexdigest()
        if contract != manifest["contract_sha256"]:
            raise ValueError("manifest training contract differs")
        if (
            manifest["initial_checkpoint_sha256"] != ACTOR_SHA
            or sha(run / "initial.pt") != ACTOR_SHA
        ):
            raise ValueError("initial model differs from declared actor")
        final_sha = sha(run / "checkpoint.pt")
        if final_sha != manifest["checkpoint_sha256"] or final_sha != sha(run / "progress.pt"):
            raise ValueError("final checkpoint differs from completed progress")
        if manifest["progress_sha256"] != final_sha:
            raise ValueError("manifest final progress hash differs")
        if manifest["metric_sampled_rows"] != 64 * 1024 or manifest["policy_sampled_states"] != (
            32 * 1024 if name == "joint" else 0
        ):
            raise ValueError("sample counts differ from fixed step protocol")
        merge_pins(
            pins,
            {
                str(manifest_path): sha(manifest_path),
                str(run / "initial.pt"): ACTOR_SHA,
                str(run / "checkpoint.pt"): final_sha,
                str(run / "progress.pt"): final_sha,
                **manifest["inputs"],
                **manifest["sources"],
            },
        )
        for source, digest in manifest["sources"].items():
            path = Path(source)
            relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
            merge_pins(pins, {str(run / "source_snapshot" / relative): digest})
        check_pins(pins)
        initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["contract_sha256"] != contract:
            raise ValueError("checkpoint not bound to manifest contract")
        reports[name] = inspect_checkpoint(checkpoint, initial, mode=mode)
        reports[name]["checkpoint_sha256"] = final_sha
        manifests[name], checkpoints[name] = manifest, checkpoint
    a, b = (manifests[n] for n in RUNS)

    def stripped(m):
        return {k: v for k, v in m["config"].items() if k != "mode"}

    if stripped(a) != stripped(b) or a["inputs"] != b["inputs"] or a["sources"] != b["sources"]:
        raise ValueError("arms do not share configuration/data/sources")
    if a["metric_support"] != b["metric_support"]:
        raise ValueError("arms do not share metric support")
    ca, cb = (checkpoints[n] for n in RUNS)
    if ca["rng"] != cb["rng"] or not torch.equal(ca["torch_rng"], cb["torch_rng"]):
        raise ValueError("final RNG streams diverged")
    sources = [Path(__file__).resolve(), Path("scripts/audit_joint_policy_metric_panel.py")]
    merge_pins(pins, {str(p): sha(p) for p in sources})
    check_pins(pins)
    result = {
        "status": "passed",
        "inputs": pins,
        "arms": reports,
        "same_initial_tensors": True,
        "same_final_rng_states": True,
        "caveat": "Execution/provenance audit, not an independent model-quality result. "
        "Frozen value parameters do not imply unchanged value outputs after encoder updates.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.out, result)
    print(json.dumps({"status": "passed", "arms": reports}))


if __name__ == "__main__":
    main()
