"""Serial launcher for the protocol69 pair; preserves failed/partial outputs."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from train_graph_metric_mixture import CONFIG, INITIAL_SHA, RUNS, inputs_and_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new pair output required, or explicit resume")
    pins, sources = inputs_and_sources()
    contract = {
        "inputs": pins,
        "sources": sources,
        "config": CONFIG,
        "initial_sha256": INITIAL_SHA,
        "runs": {k: str(v) for k, v in RUNS.items()},
        "launcher_sha256": sha(Path(__file__)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.out / "manifest.json"
    if args.resume:
        state = json.loads(path.read_text())
        if state["contract"] != contract:
            raise ValueError("pair resume contract differs")
    else:
        if any(p.exists() for p in RUNS.values()):
            raise ValueError("an arm already exists; inspect it before resuming")
        state = {"status": "training", "contract": contract, "completed": []}
    state["pid"] = os.getpid()
    for mode, directory in RUNS.items():
        existing = (
            json.loads((directory / "manifest.json").read_text())
            if (directory / "manifest.json").exists()
            else None
        )
        if existing and existing["status"] == "completed":
            if (
                existing["config"] != {**CONFIG, "mode": mode}
                or existing["initial_checkpoint_sha256"] != INITIAL_SHA
                or sha(directory / "checkpoint.pt") != existing["checkpoint_sha256"]
            ):
                raise ValueError("completed arm differs from pinned protocol")
        else:
            command = [
                sys.executable,
                "-u",
                "scripts/train_graph_metric_mixture.py",
                "--mode",
                mode,
                "--out",
                str(directory),
            ]
            if directory.exists():
                if not args.resume:
                    raise ValueError("partial arm requires explicit recovery")
                command.append("--resume")
            # The worker's exclusive writer lock rejects a still-live arm on recovery.
            with (args.out / f"{mode}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state.update(
                    status="training", current=mode, child_pid=child.pid, child_command=command
                )
                atomic(path, state)
                code = child.wait()
            if code:
                state.update(status="needs_attention", exit_code=code)
                atomic(path, state)
                raise SystemExit(code)
            existing = json.loads((directory / "manifest.json").read_text())
            if (
                existing["status"] != "completed"
                or sha(directory / "checkpoint.pt") != existing["checkpoint_sha256"]
            ):
                raise ValueError("worker exited without verified final checkpoint")
        record = {
            "mode": mode,
            "manifest_sha256": sha(directory / "manifest.json"),
            "checkpoint_sha256": existing["checkpoint_sha256"],
        }
        previous = [r for r in state["completed"] if r["mode"] == mode]
        if previous and previous != [record]:
            raise ValueError("completed pair prefix changed")
        if not previous:
            state["completed"].append(record)
        atomic(path, state)
    state.update(status="completed", current=None, child_pid=None)
    atomic(path, state)
    print(json.dumps({"status": "completed", "arms": len(state["completed"])}), flush=True)


if __name__ == "__main__":
    main()
