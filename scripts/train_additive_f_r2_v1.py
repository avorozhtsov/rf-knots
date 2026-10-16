"""Selected R2 continuation on the independently audited additive F bank."""

import argparse
from pathlib import Path

import train_joint_f_r2_continuation_v1 as previous
import train_joint_f_r2_r4_v1 as base

BANK = Path("artifacts/additive-f-bank-v1")
BANK_SHA = "3a65df6530d8fe62d9e297b204072db4c1a7ec437f131338c640dc28c23a182e"
BANK_AUDIT = Path("artifacts/additive-f-bank-audit-v1/report.json")
BANK_AUDIT_SHA = "47487fb089336eb424a34e79603fb510d61a0285868cea6a18d54b31ac7a2b57"
PARENT = Path("artifacts/joint-f-r2-continuation-v1/step-133506.pt")
PARENT_SHA = "4b11cb1078f14f64cfabd1d65c97826b8eb7fdefb16edec8d07d982c33d4cad2"
START_STEP = 133506
PROTOCOL = Path("research/320-additive-f-bank-and-new-source-gate-v1.md")


def bank_order(manifest):
    order = []
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{index:04d}.json"
        if entry["file"] != path.name or base.sha(path) != entry["sha256"]:
            raise ValueError(f"additive root SHA/order differs: {index}")
        if base.read(path)["groups"]:
            order.append(index)
    if (len(order) != 399 or sum(index < 1000 for index in order) != 392
            or sum(index >= 1000 for index in order) != 7):
        raise ValueError("additive bank F coverage differs")
    return order


def selected_group(order, step):
    old = [index for index in order if index < 1000]
    new = [index for index in order if index >= 1000]
    local = step - START_STEP
    if local < 0:
        raise ValueError("step precedes additive bank")
    if local % 5 == 0:
        root_index = new[(local // 5) % len(new)]
    else:
        root_index = old[(local - local // 5 - 1) % len(old)]
    row = base.read(BANK / f"root-{root_index:04d}.json")
    groups = row["groups"]
    group_index = (step // len(order)) % len(groups)
    return row, groups[group_index], group_index


def inputs():
    paths = (BANK / "manifest.json", BANK_AUDIT, PARENT,
             PARENT.parent / "manifest.json", PROTOCOL, Path(__file__),
             Path("scripts/train_joint_f_r2_continuation_v1.py"),
             Path("scripts/train_joint_f_r2_r4_v1.py"),
             Path("scripts/train_joint_f_r2_r4_bounded_v2.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v3.py"),
             Path("scripts/mcts_tournament_signals_v1.py"),
             Path("scripts/tournament_term_sampler_v1.py"),
             Path("scripts/pretrained_raster_policy.py"),
             Path("scripts/mcts_training_supervision.py"))
    hashes = {str(path): base.sha(path) for path in paths}
    if (hashes[str(BANK / "manifest.json")] != BANK_SHA
            or hashes[str(BANK_AUDIT)] != BANK_AUDIT_SHA
            or hashes[str(PARENT)] != PARENT_SHA):
        raise ValueError("frozen additive bank, audit or parent differs")
    parent = base.read(PARENT.parent / "manifest.json")
    if (parent["optimizer_steps"] != START_STEP
            or parent["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent["contract"]["arm"] != "R2"):
        raise ValueError("parent checkpoint/manifest differs")
    return hashes


def main():
    # Reuse the frozen numerical step/checkpoint implementation. The new
    # source, bank, sampler and protocol hashes are pinned in its contract.
    previous.PARENT = PARENT
    previous.PARENT_SHA = PARENT_SHA
    previous.PARENT_MANIFEST = PARENT.parent / "manifest.json"
    previous.START_STEP = START_STEP
    previous.PROTOCOL = PROTOCOL
    previous.inputs = inputs
    base.BANK = BANK
    base.bank_order = bank_order
    base.selected_group = selected_group
    base.CONFIG["data_bank"] = "additive-F-v1"
    base.CONFIG["old_root_draw_fraction"] = 0.8
    previous.main()


if __name__ == "__main__":
    main()
