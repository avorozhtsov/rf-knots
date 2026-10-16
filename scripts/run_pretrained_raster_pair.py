"""Serial, exclusively locked launcher for the two protocol94 training arms."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from build_mcts_training_supervision import atomic
from check_qrun_common_witnesses import sha
from raster_pair_training import CONFIG, digest, inputs_and_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pins, sources = inputs_and_sources()
    contract = digest({"config": CONFIG, "inputs": pins, "sources": sources})
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    manifest = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "contract_sha256": contract,
            "config": CONFIG,
            "inputs": pins,
            "sources": sources,
            "completed_arms": [],
        }
    )
    if manifest["contract_sha256"] != contract:
        raise ValueError("launcher contract differs")
    manifest.update(status="running", pid=os.getpid())
    atomic(path, manifest)
    for arm in ("q254", "random"):
        output = args.out / arm
        command = [
            sys.executable,
            "-u",
            "scripts/train_pretrained_raster_pair.py",
            "--arm",
            arm,
            "--pgx-root",
            str(args.pgx_root),
            "--out",
            str(output),
        ]
        if output.exists():
            command.append("--resume")
        manifest.update(current_arm=arm, command=command)
        atomic(path, manifest)
        with (args.out / f"{arm}.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            manifest.update(status="failed", returncode=result.returncode)
            atomic(path, manifest)
            raise SystemExit(result.returncode)
        row = json.loads((output / "manifest.json").read_text())
        if (
            row["status"] != "completed"
            or sha(output / "checkpoint.pt") != row["checkpoint_sha256"]
        ):
            raise ValueError("child completion is not verified")
        if arm not in manifest["completed_arms"]:
            manifest["completed_arms"].append(arm)
        atomic(path, manifest)
    manifest.update(status="completed", current_arm=None)
    atomic(path, manifest)


if __name__ == "__main__":
    main()
