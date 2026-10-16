"""Separate bounded, stage-balanced new TRAIN group MCTS source expansion."""

import argparse
import sys
import torch
from pathlib import Path

import run_continuous_train_mcts_v1 as engine
from build_mcts_training_supervision import sha

PROTOCOL = Path("research/387-residual-first-batch-one-root-independent-replication-v4.md")
SEED_BASE = 20277442
ACTOR = None
ORDER = None


def snapshot():
    if ACTOR is None or sha(Path(ACTOR["file"])) != ACTOR["sha256"]:
        raise ValueError("fixed replication actor SHA differs")
    if sha(Path(ACTOR["training_spec"])) != ACTOR["training_spec_sha256"]:
        raise ValueError("fixed replication model spec differs")
    saved = torch.load(ACTOR["file"], map_location="cpu", weights_only=False)
    if (saved["step"] != ACTOR["step"] or saved["contract_sha256"] != ACTOR["contract_sha256"]
            or len(saved["optimizer"]["state"]) != 71
            or not all(k in saved for k in ("rng", "torch_rng", "mps_rng"))):
        raise ValueError("fixed replication full state differs")
    return ACTOR


def stage_order(rows):
    buckets = [[i for i, row in enumerate(rows) if row["stage"] == stage]
               for stage in range(4)]
    if not all(buckets):
        raise ValueError("all four source stages required")
    return [bucket[k] for k in range(max(map(len, buckets)))
            for bucket in buckets if k < len(bucket)]


def task(position, n):
    if ORDER is None or len(ORDER) != n or position < 0:
        raise ValueError("frozen residual ordering missing")
    return (ORDER[(position // 4) % n], (position // 2) % 2,
            SEED_BASE + 2 * (position // (n * 4)) + position % 2)


def main():
    global ORDER, ACTOR
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stream-spec", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    spec = engine.read(args.stream_spec)
    ACTOR = spec["fixed_actor"]
    for path, expected in spec["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError("frozen source/test/protocol input differs")
    for key in ("roster", "roster_audit", "bank_manifest"):
        if sha(Path(spec[key])) != spec[key + "_sha256"]:
            raise ValueError("frozen new source input pin differs")
    engine.ROSTER = Path(spec["roster"])
    engine.ROSTER_AUDIT = Path(spec["roster_audit"])
    rows = engine.read(engine.ROSTER)["rows"]
    bank = Path(spec["bank_manifest"])
    seen = set()
    for entry in engine.read(bank)["roots"]:
        p = bank.parent / entry["file"]
        if sha(p) != entry["sha256"]:
            raise ValueError("retained bank row SHA differs")
        seen.add(tuple(engine.read(p)["query_group"]))
    parent = Path(spec["prior_roster"])
    if sha(parent) != spec["prior_roster_sha256"]:
        raise ValueError("prior candidate roster differs")
    seen.update(tuple(r["query_group"]) for r in engine.read(parent)["rows"])
    keys = [tuple(r["query_group"]) for r in rows]
    if len(rows) != 88 or len(set(keys)) != len(rows) or any(k in seen for k in keys):
        raise ValueError("new candidate uniqueness/novelty differs")
    ORDER = stage_order(rows)
    engine.task = task
    engine.snapshot = snapshot
    engine.PROTOCOL = PROTOCOL
    engine.AUDITOR = Path("scripts/audit_residual_train_mcts_v4.py")
    engine.CONFIG.update(stream_version=4, seed_base=SEED_BASE,
        global_task_offset=0, root_order="round-robin-stage-v1",
        roster=str(engine.ROSTER), roster_sha256=sha(engine.ROSTER),
        bank_manifest=str(bank), bank_manifest_sha256=sha(bank),
        prior_roster=str(parent), prior_roster_sha256=sha(parent),
        provider_source=str(Path(__file__)), provider_source_sha256=sha(Path(__file__)),
        teacher_provider_source="scripts/run_continuous_train_mcts_v2.py",
        teacher_provider_source_sha256=sha(Path("scripts/run_continuous_train_mcts_v2.py")),
        stream_spec=str(args.stream_spec), stream_spec_sha256=sha(args.stream_spec))
    sys.argv = [sys.argv[0], *remaining]
    engine.main()


if __name__ == "__main__":
    main()
