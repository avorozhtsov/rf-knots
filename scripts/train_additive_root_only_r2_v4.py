"""Frozen-parent native-MPS F-first continuation of the audited R2 v3 bank."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v3/step-172421.pt")
PARENT_SHA = "b1ca2fb4ca4b0907a56b2b9372012c8a93549692b6a863aec9d1d1174c3d1e52"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_MANIFEST_SHA = "5a4bde678287750d30762124921525a2c29bb648af80340941c66899642ae84e"
PROTOCOL = Path("research/353-continuous-r2-f-mcts-rehearsal-v4.md")
START_STEP = 172421


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT,
             PARENT, PARENT_MANIFEST, PROTOCOL, Path(__file__),
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
