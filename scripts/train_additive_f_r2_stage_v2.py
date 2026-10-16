"""Stage-aware R2 continuation on immutable additive F bank."""

from pathlib import Path

import train_additive_f_r2_v1 as first
import train_joint_f_r2_r4_v1 as base

PARENT = Path("artifacts/additive-f-r2-v1/step-133635.pt")
PARENT_SHA = "bbc34cd22cca7b0403be82a3c96910c0b2971a8ef9b564cc448ae131166c77fd"
START_STEP = 133635
PROTOCOL = Path("research/322-additive-r2-stage-aware-continuation-v2.md")
PATTERN = (0,) * 5 + (1,) * 7 + (2,) * 7 + (3,)
OLD_BY_STAGE = None
NEW_ORDER = None


def selected_group(order, step):
    if OLD_BY_STAGE is None or NEW_ORDER is None:
        raise ValueError("stage-aware order not initialized")
    local = step - START_STEP
    if local < 0:
        raise ValueError("step precedes v2 continuation")
    if local % 5 == 0:
        root_index = NEW_ORDER[(local // 5) % len(NEW_ORDER)]
    else:
        old_slot = local - local // 5 - 1
        pattern_position = old_slot % len(PATTERN)
        stage = PATTERN[pattern_position]
        within_cycle = sum(value == stage for value in PATTERN[:pattern_position])
        per_cycle = PATTERN.count(stage)
        ordinal = (old_slot // len(PATTERN)) * per_cycle + within_cycle
        rows = OLD_BY_STAGE[stage]
        root_index = rows[ordinal % len(rows)]
    row = base.read(first.BANK / f"root-{root_index:04d}.json")
    groups = row["groups"]
    group_index = (step // len(order)) % len(groups)
    return row, groups[group_index], group_index


def inputs():
    paths = (first.BANK / "manifest.json", first.BANK_AUDIT, PARENT,
             PARENT.parent / "manifest.json", PROTOCOL, Path(__file__),
             Path("scripts/train_additive_f_r2_v1.py"),
             Path("scripts/train_joint_f_r2_continuation_v1.py"),
             Path("scripts/train_joint_f_r2_r4_v1.py"),
             Path("scripts/train_joint_f_r2_r4_bounded_v2.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v3.py"),
             Path("scripts/mcts_tournament_signals_v1.py"),
             Path("scripts/tournament_term_sampler_v1.py"),
             Path("scripts/pretrained_raster_policy.py"),
             Path("scripts/mcts_training_supervision.py"))
    hashes = {str(path): base.sha(path) for path in paths}
    if (hashes[str(first.BANK / "manifest.json")] != first.BANK_SHA
            or hashes[str(first.BANK_AUDIT)] != first.BANK_AUDIT_SHA
            or hashes[str(PARENT)] != PARENT_SHA):
        raise ValueError("stage-aware bank/parent SHA differs")
    parent = base.read(PARENT.parent / "manifest.json")
    if (parent["optimizer_steps"] != START_STEP
            or parent["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent["contract"]["arm"] != "R2"
            or parent["contract"]["config"]["data_bank"] != "additive-F-v1"):
        raise ValueError("stage-aware parent manifest differs")
    return hashes


def initialize_order():
    global OLD_BY_STAGE, NEW_ORDER
    order = first.bank_order(base.read(first.BANK / "manifest.json"))
    old = [[] for _ in range(4)]
    new = []
    for index in order:
        row = base.read(first.BANK / f"root-{index:04d}.json")
        (new if index >= 1000 else old[row["stage"]]).append(index)
    if [len(rows) for rows in old] != [111, 194, 79, 8] or len(new) != 7:
        raise ValueError("stage-aware bank stratification differs")
    OLD_BY_STAGE, NEW_ORDER = old, new
    return order


def main():
    initialize_order()
    first.PARENT = PARENT
    first.PARENT_SHA = PARENT_SHA
    first.START_STEP = START_STEP
    first.PROTOCOL = PROTOCOL
    first.inputs = inputs
    first.selected_group = selected_group
    base.CONFIG["sampler"] = "additive-stage-aware-v2"
    base.CONFIG["old_root_stage_pattern"] = list(PATTERN)
    first.main()


if __name__ == "__main__":
    main()
