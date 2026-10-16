"""Pinned current-optimizer teacher provider for the continuing MCTS engine."""

import argparse
import sys
from pathlib import Path

import torch
import run_continuous_train_mcts_v1 as engine
from build_mcts_training_supervision import sha

REFERENCE = Path("artifacts/continuous-r2-current-training.json")
PROTOCOL = Path("research/380-continuous-mcts-current-training-provider-v2.md")
SEED_BASE = 20277420
OFFSET = None


def snapshot():
    ref = engine.read(REFERENCE)
    out, spec_path = Path(ref["out"]), Path(ref["spec"])
    spec = engine.read(spec_path)
    if spec["out"] != str(out) or sha(spec_path) != ref["spec_sha256"]:
        raise ValueError("current optimizer start reference differs")
    manifest = engine.read(out / "manifest.json")
    if manifest["contract"]["arm"] != "R2" or manifest["contract"]["inputs"][str(spec_path)] != sha(spec_path):
        raise ValueError("current R2 contract/spec pin differs")
    checkpoint = manifest["checkpoints"][-1]
    path = (out / checkpoint["file"]).resolve()
    if sha(path) != checkpoint["sha256"]:
        raise ValueError("current full optimizer checkpoint SHA differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["step"] != checkpoint["step"]
            or saved["contract_sha256"] != manifest["contract_sha256"]
            or len(saved["optimizer"]["state"]) != 71
            or not all(k in saved for k in ("rng", "torch_rng", "mps_rng"))):
        raise ValueError("current full optimizer state identity differs")
    if sha(Path(spec["bank"]) / "manifest.json") != spec["bank_sha256"]:
        raise ValueError("current teacher bank pin differs")
    return {"file": str(path), "sha256": checkpoint["sha256"],
            "step": checkpoint["step"], "contract_sha256": saved["contract_sha256"],
            "training_spec": str(spec_path), "training_spec_sha256": sha(spec_path),
            "bank": spec["bank"], "bank_manifest_sha256": spec["bank_sha256"]}


def task(position, n):
    global_position = OFFSET + position
    return ((global_position // 4) % n, (global_position // 2) % 2,
            SEED_BASE + 2 * (global_position // (n * 4)) + global_position % 2)


def main():
    global OFFSET
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stream-spec", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    spec = engine.read(args.stream_spec)
    parent = Path(spec["parent_data"])
    parent_manifest = engine.read(parent / "manifest.json")
    if (sha(parent / "manifest.json") != spec["parent_manifest_sha256"]
            or parent_manifest["next_task"] != spec["global_task_offset"]
            or parent_manifest["next_task"] % 32
            or len(parent_manifest["audits"]) != parent_manifest["next_task"] // 32):
        raise ValueError("complete audited parent stream boundary differs")
    OFFSET = spec["global_task_offset"]
    engine.task = task
    engine.snapshot = snapshot
    engine.PROTOCOL = PROTOCOL
    engine.AUDITOR = Path("scripts/audit_continuous_mcts_batch_v2.py")
    engine.CONFIG.update(stream_version=2, seed_base=SEED_BASE,
        global_task_offset=OFFSET, provider_source=str(Path(__file__)),
        provider_source_sha256=sha(Path(__file__)),
        stream_spec=str(args.stream_spec), stream_spec_sha256=sha(args.stream_spec),
        parent_data=str(parent), parent_manifest_sha256=spec["parent_manifest_sha256"])
    sys.argv = [sys.argv[0], *remaining]
    engine.main()


if __name__ == "__main__":
    main()
