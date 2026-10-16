"""Atomically register a fully checkpointed research optimizer for CPU teachers."""

import argparse
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from train_joint_f_r2_r4_v1 import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/continuous-r2-current-training.json"))
    args = parser.parse_args()
    spec = read(args.spec)
    manifest = read(Path(spec["out"]) / "manifest.json")
    last = manifest["checkpoints"][-1]
    if (manifest["contract"]["arm"] != "R2"
            or manifest["contract"]["inputs"][str(args.spec)] != sha(args.spec)
            or sha(Path(spec["out"]) / last["file"]) != last["sha256"]
            or sha(Path(spec["bank"]) / "manifest.json") != spec["bank_sha256"]
            or sha(Path(spec["audit"])) != spec["audit_sha256"]):
        raise ValueError("current optimizer/spec/checkpoint/bank pin differs")
    atomic(args.out, {"out": spec["out"], "spec": str(args.spec),
                     "spec_sha256": sha(args.spec), "registered_step": last["step"],
                     "bank_manifest_sha256": spec["bank_sha256"]})
    print({"current_training": spec["out"], "step": last["step"],
           "reference_sha256": sha(args.out)}, flush=True)


if __name__ == "__main__":
    main()
