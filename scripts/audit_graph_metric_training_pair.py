"""Final-only audit of protocol69: both arms train encoder and policy."""

import argparse
import json
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_joint_training_pair import inspect_checkpoint
from build_mcts_training_supervision import atomic, digest, sha
from train_graph_metric_mixture import CONFIG, INITIAL_SHA, RUNS


def inspect_arm(mode, manifest, checkpoint, initial):
    if (
        manifest["status"] != "completed"
        or manifest["config"] != {**CONFIG, "mode": mode}
        or manifest["completed_steps"] != CONFIG["steps"]
        or manifest["initial_checkpoint_sha256"] != INITIAL_SHA
        or manifest["value_head_trained"] is not False
    ):
        raise ValueError("completed training protocol differs")
    expected = digest(
        {"config": manifest["config"], "inputs": manifest["inputs"], "sources": manifest["sources"]}
    )
    if manifest["contract_sha256"] != expected or checkpoint["contract_sha256"] != expected:
        raise ValueError("checkpoint contract differs")
    for field, count in (
        ("metric_sampled_rows", 128),
        ("policy_sampled_states", 32),
        ("graph_sampled_rows", 64 if mode == "graph" else 0),
    ):
        if manifest[field] != count * CONFIG["steps"]:
            raise ValueError("sample accounting differs")
    # Both are joint-policy arms, unlike the older metric-only control.
    return inspect_checkpoint(checkpoint, initial, mode="joint", steps=CONFIG["steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite training-pair audit")
    pins, arms, loaded = {}, {}, {}
    for mode, directory in RUNS.items():
        paths = [
            directory / n for n in ("manifest.json", "initial.pt", "checkpoint.pt", "progress.pt")
        ]
        manifest = json.loads(paths[0].read_text())
        if (
            sha(paths[1]) != INITIAL_SHA
            or sha(paths[2]) != manifest.get("checkpoint_sha256")
            or sha(paths[3]) != manifest.get("progress_sha256")
            or sha(paths[2]) != sha(paths[3])
        ):
            raise ValueError("saved tensor SHA differs")
        merge_pins(pins, manifest["inputs"])
        merge_pins(pins, manifest["sources"])
        merge_pins(pins, {str(p): sha(p) for p in paths})
        initial = torch.load(paths[1], map_location="cpu", weights_only=False)
        checkpoint = torch.load(paths[2], map_location="cpu", weights_only=False)
        arms[mode] = {
            "checkpoint_sha256": sha(paths[2]),
            **inspect_arm(mode, manifest, checkpoint, initial),
        }
        loaded[mode] = manifest, checkpoint
    a, ca = loaded["graph"]
    b, cb = loaded["control"]
    if (
        a["inputs"] != b["inputs"]
        or a["sources"] != b["sources"]
        or a["architecture"] != b["architecture"]
        or a["metric_support"] != b["metric_support"]
        or ca["rng"] != cb["rng"]
        or not torch.equal(ca["torch_rng"], cb["torch_rng"])
    ):
        raise ValueError("paired inputs/sources/architecture/support/RNG differ")
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (Path(__file__).resolve(), Path("scripts/audit_joint_training_pair.py"))
        },
    )
    check_pins(pins)
    atomic(
        args.out,
        {
            "status": "passed",
            "inputs": pins,
            "arms": arms,
            "same_initial_checkpoint": INITIAL_SHA,
            "final_rng_aligned": True,
            "scope": "Implementation/completion audit, not scientific promotion. "
            "Both full-policy arms use supervised positive sets; no value targets. "
            "Sampled rows are not unique knots or MCTS evaluations.",
        },
    )
    print(json.dumps({"status": "passed", "arms": arms}), flush=True)


if __name__ == "__main__":
    main()
