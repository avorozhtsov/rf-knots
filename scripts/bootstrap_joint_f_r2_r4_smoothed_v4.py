"""Import a matched v3 boundary into a new sparse-checkpoint v4 contract."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from build_mcts_training_supervision import atomic, digest, sha


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "scripts/train_joint_f_r2_r4_smoothed_v3.py"
V4 = ROOT / "scripts/train_joint_f_r2_r4_smoothed_v4.py"
PROTOCOL = ROOT / "research/308-f-v4-storage-safe-continuation.md"
V3_SHA = "8fcede9f5d09266f37b7ed6c5589bd1a93177b0c94b82d1357c62dce77c96f6e"
V4_SHA = "07895f0d7dea6815dff8bccbd4f65e00454d9260ba84be76030a860b8e035f82"


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def main():
    if sha(V3) != V3_SHA or sha(V4) != V4_SHA:
        raise ValueError("trainer source SHA differs")
    old = {}
    for arm in ("R2", "R4"):
        directory = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v3-{arm}"
        data = json.loads((directory / "manifest.json").read_text())
        if data["contract"]["arm"] != arm or data["status"] != "checkpointed":
            raise ValueError(f"{arm}: v3 arm is not at a completed boundary")
        entry = data["checkpoints"][-1]
        if data["optimizer_steps"] != entry["step"]:
            raise ValueError(f"{arm}: uncommitted optimizer step")
        if sha(directory / entry["file"]) != entry["sha256"]:
            raise ValueError(f"{arm}: v3 checkpoint SHA differs")
        old[arm] = (directory, data, entry)
    if old["R2"][2]["step"] != old["R4"][2]["step"]:
        raise ValueError("v3 arms are not at equal steps")
    if any((ROOT / f"artifacts/joint-f-r2-r4-smoothed-v4-{arm}").exists()
           for arm in old):
        raise ValueError("v4 output already exists")

    report = {"schema": "rf-knots-v3-to-v4-migration-v1",
              "protocol_sha256": sha(PROTOCOL), "v3_source_sha256": V3_SHA,
              "v4_source_sha256": V4_SHA, "arms": {}}
    for arm, (old_dir, old_manifest, entry) in old.items():
        contract = copy.deepcopy(old_manifest["contract"])
        config = contract["config"]
        if config["checkpoint_interval"] != 16 or config["v3_source_sha256"] != V3_SHA:
            raise ValueError(f"{arm}: unexpected v3 training config")
        config["checkpoint_interval"] = 128
        config["v4_source_sha256"] = V4_SHA
        inputs = contract["inputs"]
        old_protocol = "research/306-f-smoothed-pairwise-repair-v1.md"
        if old_protocol not in inputs:
            raise ValueError(f"{arm}: original protocol pin missing")
        del inputs[old_protocol]
        inputs["research/308-f-v4-storage-safe-continuation.md"] = sha(PROTOCOL)
        new_contract_sha = digest(contract)
        source = old_dir / entry["file"]
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if saved["contract_sha256"] != old_manifest["contract_sha256"] or saved["step"] != entry["step"]:
            raise ValueError(f"{arm}: checkpoint identity differs")
        migrated = copy.deepcopy(saved)
        migrated["contract_sha256"] = new_contract_sha
        target_dir = ROOT / f"artifacts/joint-f-r2-r4-smoothed-v4-{arm}"
        target_dir.mkdir()
        target = target_dir / entry["file"]
        partial = target.with_suffix(".partial")
        torch.save(migrated, partial)
        partial.replace(target)
        reloaded = torch.load(target, map_location="cpu", weights_only=False)
        if not equal_state(migrated, reloaded):
            raise ValueError(f"{arm}: migrated payload changed during serialization")
        if not all(equal_state(saved[key], reloaded[key]) for key in saved
                   if key != "contract_sha256"):
            raise ValueError(f"{arm}: model, AdamW or RNG changed in migration")
        new_manifest = copy.deepcopy(old_manifest)
        new_manifest.update(contract=contract, contract_sha256=new_contract_sha,
                            checkpoints=[{"file": target.name, "sha256": sha(target),
                                          "step": entry["step"]}],
                            status="checkpointed", pid=None,
                            continuation_from={"manifest_sha256": sha(old_dir / "manifest.json"),
                                               "checkpoint_sha256": entry["sha256"],
                                               "step": entry["step"]})
        atomic(target_dir / "manifest.json", new_manifest)
        report["arms"][arm] = {"source_manifest_sha256": sha(old_dir / "manifest.json"),
                               "source_checkpoint_sha256": entry["sha256"],
                               "v4_checkpoint_sha256": sha(target),
                               "v4_contract_sha256": new_contract_sha,
                               "optimizer_steps": entry["step"],
                               "active_seconds": old_manifest["active_seconds"],
                               "cpu_seconds": old_manifest["cpu_seconds"]}
    report_path = ROOT / "artifacts/joint-f-r2-r4-smoothed-v4-migration.json"
    atomic(report_path, report)
    print(json.dumps({"report": str(report_path), "sha256": sha(report_path),
                      "steps": old["R2"][2]["step"]}), flush=True)


if __name__ == "__main__":
    main()
