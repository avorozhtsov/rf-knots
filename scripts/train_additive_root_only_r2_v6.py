"""Frozen-parent native-MPS F-first continuation after v5's completed rung."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v5/step-246407.pt")
PARENT_SHA = "12d0369be9cb1d54657637cb600a84cf8947d36130aaabcb85abd50294afa186"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_MANIFEST_SHA = "19814dc0cf6cb27cc8dea9b0eee7ef17ebe5c3e591d1176ba91046db19e705d8"
PROTOCOL = Path("research/361-r2-v6-low-disk-continuation.md")
START_STEP = 246407


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT,
             PARENT, PARENT_MANIFEST, PROTOCOL, Path(__file__),
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
            or parent["new_active_seconds"] < 7200
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
