"""Frozen-parent native-MPS F-first continuation after v7's completed rung."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v7/step-265609.pt")
PARENT_SHA = "c105dad7409f7032582c170519bd35d1e14ffca4a9aee37e3ce65bdee0d446aa"
PARENT_MANIFEST = PARENT.parent / "manifest.json"
PARENT_MANIFEST_SHA = "c6555d864058a7c385cc1bada23fae616af7f6d134bf3cdf19fc1680ea1dfb4f"
PROTOCOL = Path("research/366-r2-v8-storage-bounded-continuation.md")
START_STEP = 265609


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT,
             PARENT, PARENT_MANIFEST, PROTOCOL, Path(__file__),
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
