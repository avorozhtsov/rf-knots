"""Fail-closed, exact-list retention of completed v3 checkpoint history."""

import argparse
import json
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "artifacts/joint-f-r2-r4-smoothed-v4-migration.json"
LEDGER = ROOT / "artifacts/joint-f-r2-r4-smoothed-v3-retention.json"
PARENT_STEP = 31105
FIRST_V4_STEP = 31233


def state():
    migration = json.loads(MIGRATION.read_text())
    if migration["schema"] != "rf-knots-v3-to-v4-migration-v1":
        raise ValueError("unexpected migration schema")
    result = {"schema": "rf-knots-v3-checkpoint-retention-v1",
              "migration_sha256": sha(MIGRATION), "arms": {}}
    for arm in ("R2", "R4"):
        old_dir = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}"
        new_dir = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v4-{arm}"
        old_manifest = old_dir / "manifest.json"
        old = json.loads(old_manifest.read_text())
        new = json.loads((new_dir / "manifest.json").read_text())
        if (old["optimizer_steps"] != PARENT_STEP
                or old["checkpoints"][-1]["step"] != PARENT_STEP
                or new["optimizer_steps"] < FIRST_V4_STEP
                or new["checkpoints"][-1]["step"] != new["optimizer_steps"]
                or sha(new_dir / new["checkpoints"][-1]["file"])
                   != new["checkpoints"][-1]["sha256"]
                or migration["arms"][arm]["source_manifest_sha256"] != sha(old_manifest)
                or migration["arms"][arm]["source_checkpoint_sha256"]
                   != old["checkpoints"][-1]["sha256"]):
            raise ValueError(f"{arm}: migration or continuation evidence differs")
        listed = {entry["file"] for entry in old["checkpoints"]}
        if len(listed) != len(old["checkpoints"]):
            raise ValueError(f"{arm}: duplicate checkpoint names")
        on_disk = {path.name for path in old_dir.glob("step-*.pt")}
        if on_disk != listed:
            raise ValueError(f"{arm}: checkpoint inventory differs")
        keep, delete = [], []
        for entry in old["checkpoints"]:
            path = old_dir / entry["file"]
            if entry["file"] != f"step-{entry['step']:06d}.pt" or sha(path) != entry["sha256"]:
                raise ValueError(f"{arm}: checkpoint SHA/name differs at {entry['step']}")
            target = keep if (entry["step"] == 1 or entry["step"] % 1024 == 0
                              or entry["step"] == PARENT_STEP) else delete
            target.append({**entry, "bytes": path.stat().st_size})
        if keep[-1]["step"] != PARENT_STEP:
            raise ValueError(f"{arm}: parent checkpoint not retained")
        result["arms"][arm] = {"manifest_sha256": sha(old_manifest),
                               "old_contract_sha256": old["contract_sha256"],
                               "kept": keep, "deleted": delete,
                               "reclaim_bytes": sum(item["bytes"] for item in delete)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if LEDGER.exists():
        raise ValueError("retention ledger already exists; do not repeat deletion")
    result = state()
    result["status"] = "planned"
    print(json.dumps({"status": "planned", "plan_digest": digest(result),
                      "counts": {arm: {"keep": len(data["kept"]),
                                       "delete": len(data["deleted"]),
                                       "reclaim_bytes": data["reclaim_bytes"]}
                                 for arm, data in result["arms"].items()}}), flush=True)
    if not args.execute:
        return
    atomic(LEDGER, result)
    if state()["arms"] != result["arms"]:
        raise ValueError("pre-delete state changed")
    result["status"] = "deleting"
    atomic(LEDGER, result)
    for arm in ("R2", "R4"):
        directory = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}"
        for entry in result["arms"][arm]["deleted"]:
            path = directory / entry["file"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"{arm}: pre-unlink checkpoint changed: {entry['file']}")
            path.unlink()
    for arm in ("R2", "R4"):
        directory = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}"
        remaining = {path.name for path in directory.glob("step-*.pt")}
        expected = {entry["file"] for entry in result["arms"][arm]["kept"]}
        if remaining != expected or any(
            sha(directory / entry["file"]) != entry["sha256"]
            for entry in result["arms"][arm]["kept"]
        ):
            raise ValueError(f"{arm}: post-delete retention mismatch")
    result["status"] = "complete"
    atomic(LEDGER, result)
    print(json.dumps({"status": "complete", "ledger_sha256": sha(LEDGER)}), flush=True)


if __name__ == "__main__":
    main()
