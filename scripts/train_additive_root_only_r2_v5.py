"""Frozen-parent native-MPS F-first continuation after v4's completed rung."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v4/step-230662.pt")
PARENT_SHA = "89af09393db15060f063f1d636c9169bd15740db7cc27b3d5bbe855fa6bf8ca4"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_MANIFEST_SHA = "6724f7aca324a033bebbdb9d3054d22e084d8fdb802d4513558add477a8017e6"
PROTOCOL = Path("research/359-r2-f-continuation-v5-storage-gate.md")
START_STEP = 230662


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT,
             PARENT, PARENT_MANIFEST, PROTOCOL, Path(__file__),
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
            or parent["new_active_seconds"] < 21600
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
