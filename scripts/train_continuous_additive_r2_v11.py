"""Generic immutable-bank continuation for R2, with preserved optimizer/RNG."""

import argparse
import os
import sys
from pathlib import Path

import train_additive_root_only_r2_v3 as prior
import train_joint_f_r2_continuation_v1 as engine
import train_joint_f_r2_r4_v1 as base

SPEC = None
SETTINGS = None
OLD_STAGES = None
ADDED = None
ORIGINAL_SETUP = prior.setup_training
PROTOCOL = Path("research/378-continuous-r2-first-online-bank-admission.md")


def bank_order(manifest):
    order = []
    for i, entry in enumerate(manifest["roots"]):
        path = prior.BANK / entry["file"]
        if entry["file"] != f"root-{i:04d}.json" or base.sha(path) != entry["sha256"]:
            raise ValueError("additive bank root SHA/order differs")
        if base.read(path)["groups"]:
            order.append(i)
    if len(order) != SETTINGS["F_roots"]:
        raise ValueError("additive F root accounting differs")
    return order


def initialize_order():
    global OLD_STAGES, ADDED
    order = bank_order(base.read(prior.BANK / "manifest.json"))
    old, new = [[] for _ in range(4)], []
    for i in order:
        row = base.read(prior.BANK / f"root-{i:04d}.json")
        (new if i >= SETTINGS["retained_shards"] else old[row["stage"]]).append(i)
    if sum(map(len, old)) != SETTINGS["retained_F_roots"] or not new or not all(old):
        raise ValueError("retained/new F sampling coverage differs")
    OLD_STAGES, ADDED = old, new
    return order


def selected_group(order, step):
    local = step - prior.START_STEP
    if local < 0 or OLD_STAGES is None:
        raise ValueError("sampler not initialized")
    if local % 10 == 0:
        index = ADDED[(local // 10) % len(ADDED)]
        cycle = local // (10 * len(ADDED))
    else:
        slot = local - local // 10 - 1
        pos = slot % len(prior.PATTERN)
        stage = prior.PATTERN[pos]
        ordinal = ((slot // len(prior.PATTERN)) * prior.PATTERN.count(stage)
                   + sum(value == stage for value in prior.PATTERN[:pos]))
        index = OLD_STAGES[stage][ordinal % len(OLD_STAGES[stage])]
        cycle = local // len(order)
    row = base.read(prior.BANK / f"root-{index:04d}.json")
    group_index = cycle % len(row["groups"])
    return row, row["groups"][group_index], group_index


def setup():
    ORIGINAL_SETUP()
    base.CONFIG["data_bank"] = "continuous-immutable-additive-F-v1"
    base.CONFIG["sampler"] = "retained-stage-90/added-root-only-10"
    base.CONFIG["continuation_mode"] = "continuous-until-explicit-or-resource-gate"


def inputs():
    paths = (SPEC, prior.BANK / "manifest.json", prior.BANK_AUDIT,
             prior.PARENT, prior.PARENT.parent / "manifest.json", PROTOCOL,
             Path(__file__), Path("scripts/run_continuous_additive_r2_v11.py"),
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
    expected = {str(prior.BANK / "manifest.json"): SETTINGS["bank_sha256"],
                str(prior.BANK_AUDIT): SETTINGS["audit_sha256"],
                str(prior.PARENT): SETTINGS["parent_sha256"],
                str(prior.PARENT.parent / "manifest.json"): SETTINGS["parent_manifest_sha256"]}
    if any(hashes[path] != expected_sha for path, expected_sha in expected.items()):
        raise ValueError("frozen additive bank/audit/parent SHA differs")
    parent = base.read(prior.PARENT.parent / "manifest.json")
    audit = base.read(prior.BANK_AUDIT)
    if (parent["status"] != "checkpointed" or parent["optimizer_steps"] != prior.START_STEP
            or parent["checkpoints"][-1]["sha256"] != SETTINGS["parent_sha256"]
            or audit["status"] != "passed" or audit["distinct_F_roots"] != SETTINGS["F_roots"]):
        raise ValueError("parent or audited bank identity differs")
    return hashes


def main():
    global SPEC, SETTINGS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", type=Path, default=os.environ.get("R2_CONTINUATION_SPEC"))
    args, remaining = parser.parse_known_args()
    if args.spec is None:
        raise ValueError("immutable start specification required")
    SPEC = Path(args.spec)
    SETTINGS = base.read(SPEC)
    sys.argv = [sys.argv[0], *remaining]
    prior.BANK = Path(SETTINGS["bank"])
    prior.BANK_AUDIT = Path(SETTINGS["audit"])
    prior.PARENT = Path(SETTINGS["parent"])
    prior.PARENT_SHA = SETTINGS["parent_sha256"]
    prior.START_STEP = SETTINGS["parent_step"]
    prior.PROTOCOL = PROTOCOL
    prior.inputs = inputs
    prior.bank_order = bank_order
    prior.initialize_order = initialize_order
    prior.selected_group = selected_group
    prior.setup_training = setup
    engine.BLOCK = 1024
    prior.main()


if __name__ == "__main__":
    main()
