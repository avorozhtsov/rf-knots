"""Frozen-parent native-MPS F-first continuation after v8's completed rung."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v8/step-275466.pt")
PARENT_SHA = "65204190bfe7c580c95e7459fce2daa0fc36b30cebfbf94cf6b06fd08af699f3"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_MANIFEST_SHA = "c2e223e2b4c50bf93d1ae506671d27a0362609bc4a59fdd559d1c1107780783e"
PROTOCOL = Path("research/368-r2-v9-short-storage-gated-continuation.md")
START_STEP = 275466


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT,
             PARENT, PARENT_MANIFEST, PROTOCOL, Path(__file__),
             Path("scripts/train_additive_root_only_r2_v8.py"),
             Path("scripts/train_additive_root_only_r2_v7.py"),
             Path("scripts/train_additive_root_only_r2_v6.py"),
             Path("scripts/train_additive_root_only_r2_v5.py"),
             Path("scripts/train_additive_root_only_r2_v4.py"),
             Path("scripts/train_additive_root_only_r2_v3.py"),
             Path("scripts/train_joint_f_r2_continuation_v1.py"),
             Path("scripts/train_joint_f_r2_r4_v1.py"),
             Path("scripts/train_joint_f_r2_r4_bounded_v2.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v3.py"),
             Path("scripts/mcts_tournament_signals_v1.py"),
             Path("scripts/tournament_term_sampler_v1.py"),
             Path("scripts/pretrained_raster_policy.py"),
             Path("scripts/mcts_training_supervision.py"))
    hashes = {str(path): base.sha(path) for path in paths}
    if (hashes[str(prior.BANK / "manifest.json")] != prior.BANK_SHA
            or hashes[str(prior.BANK_AUDIT)] != prior.BANK_AUDIT_SHA
            or hashes[str(PARENT)] != PARENT_SHA
            or hashes[str(PARENT_MANIFEST)] != PARENT_MANIFEST_SHA):
        raise ValueError("frozen bank/audit/parent SHA differs")
    parent = base.read(PARENT_MANIFEST)
    if (parent["optimizer_steps"] != START_STEP
            or parent["status"] != "checkpointed"
            or parent["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent["new_active_seconds"] < 3600
            or parent["contract"]["arm"] != "R2"
            or parent["contract"]["config"]["data_bank"] !=
            "additive-root-only-F-v3"):
        raise ValueError("parent checkpoint/contract differs")
    return hashes


def main():
    prior.PARENT = PARENT
    prior.PARENT_SHA = PARENT_SHA
    prior.START_STEP = START_STEP
    prior.PROTOCOL = PROTOCOL
    prior.inputs = inputs
    prior.main()


if __name__ == "__main__":
    main()
