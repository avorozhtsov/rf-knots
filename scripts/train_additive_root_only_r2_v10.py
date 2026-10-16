"""Native-MPS continuing R2 F training, sparse resumable checkpoints."""

from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_continuation_v1 as engine
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-root-only-r2-v9/step-280331.pt")
PARENT_SHA = "ecc2ad1884942ebf0f3b0fe312b9ea977a44a2eaf478fdfed6071af285a12588"
PARENT_MANIFEST_SHA = "8cf15b883d2857062cc7301924a0ec36ace845920789cb055383b1544aa7f4ee"
PROTOCOL = Path("research/375-continuous-r2-mcts-learning-contract-v1.md")
START_STEP = 280331
BLOCK = 1024


def inputs():
    paths = (prior.BANK / "manifest.json", prior.BANK_AUDIT, PARENT,
             PARENT.parent / "manifest.json", PROTOCOL, Path(__file__),
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
            or hashes[str(PARENT.parent / "manifest.json")] != PARENT_MANIFEST_SHA):
        raise ValueError("bank/audit/parent SHA differs")
    parent = base.read(PARENT.parent / "manifest.json")
    if (parent["optimizer_steps"] != START_STEP
            or parent["status"] != "checkpointed"
            or parent["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent["new_active_seconds"] < 1800
            or parent["contract"]["arm"] != "R2"):
        raise ValueError("completed v9 parent identity differs")
    return hashes


def main():
    engine.BLOCK = BLOCK
    prior.PARENT = PARENT
    prior.PARENT_SHA = PARENT_SHA
    prior.START_STEP = START_STEP
    prior.PROTOCOL = PROTOCOL
    prior.inputs = inputs
    base.CONFIG["continuation_mode"] = "continuous-until-explicit-or-resource-gate"
    prior.main()


if __name__ == "__main__":
    main()
