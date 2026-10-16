"""Continuing CPU canonical MCTS, frozen actor per32-condition audited batch."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M

ROSTER = Path("artifacts/continuous-train-roster-v1/roster.json")
ROSTER_AUDIT = Path("artifacts/continuous-train-roster-audit-v1/report.json")
TRAIN = Path("artifacts/additive-root-only-r2-v10")
PARENT = Path("artifacts/additive-root-only-r2-v9/step-280331.pt")
AUDITOR = Path("scripts/audit_continuous_mcts_batch_v1.py")
PROTOCOL = Path("research/376-continuous-train-mcts-data-stream-v1.md")
BATCH_SIZE = 32
SIMULATIONS = 16
CPU_LIMIT = 120
MIN_FREE = 12 * 1024**3
CONFIG = {"simulations": SIMULATIONS, "insertion_candidates": 4,
          "widening_batch": 4, "c_puct": 1.5, "temperature": 1.0}


def read(path):
    return json.loads(path.read_text())


def task(position, n):
    return ((position // 4) % n, (position // 2) % 2,
            20271420 + 2 * (position // (n * 4)) + position % 2)


def timeout(_signal, _frame):
    raise TimeoutError("per-condition CPU ceiling")


def snapshot():
    manifest = read(TRAIN / "manifest.json")
    entry = manifest["checkpoints"][-1]
    path = TRAIN / entry["file"]
    if sha(path) != entry["sha256"]:
        raise ValueError("teacher optimizer checkpoint SHA differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["step"] != entry["step"] or saved["contract_sha256"] != manifest["contract_sha256"]:
        raise ValueError("teacher snapshot identity differs")
    return {"file": str(path), "sha256": entry["sha256"],
            "step": entry["step"], "contract_sha256": saved["contract_sha256"]}


def policy(entry):
    path = Path(entry["file"])
    if sha(path) != entry["sha256"]:
        raise ValueError("batch teacher checkpoint changed")
    net, objective = load_actor()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["step"] != entry["step"] or saved["contract_sha256"] != entry["contract_sha256"]:
        raise ValueError("batch teacher step differs")
    net.encoder.graph_rounds = 2
    net.load_state_dict(saved["model"], strict=True)
    objective.load_state_dict(saved["objective"], strict=True)
    net.eval(); objective.eval()
    return net, objective


def audit_pending(args, state, manifest_path):
    """Reconcile completed batches after interruption at the audit boundary."""
    done = {entry["batch"] for entry in state["audits"]}
    for batch_id in range(state["next_task"] // BATCH_SIZE):
        if batch_id in done:
            continue
        batch = args.out / f"batch-{batch_id:06d}"
        batch.mkdir(exist_ok=True)
        first = batch_id * BATCH_SIZE
        immutable = {"schema": "rf-knots-continuous-MCTS-batch-v1",
                     "contract": state["contract"], "actor": state["batches"][str(batch_id)],
                     "batch_id": batch_id, "first_task": first,
                     "shards": state["shards"][first:first + BATCH_SIZE]}
        path = batch / "manifest.json"
        if path.exists() and read(path) != immutable:
            raise ValueError("immutable batch manifest differs")
        if not path.exists():
            atomic(path, immutable)
        report = batch / "audit" / "report.json"
        if not report.exists():
            subprocess.run([sys.executable, str(AUDITOR), "--data", str(args.out),
                            "--batch", str(batch_id), "--out", str(batch / "audit")], check=True)
        checked = read(report)
        if checked["partial"] or checked["conditions"] != BATCH_SIZE or checked["batch_manifest_sha256"] != sha(path):
            raise ValueError("completed batch audit identity differs")
        state["audits"].append({"batch": batch_id, "report": str(report), "sha256": sha(report)})
        atomic(manifest_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after < 0 or args.out.exists() != args.resume:
        raise ValueError("output/resume mismatch")
    pgx = args.pgx_root.resolve() / "src"
    if not (pgx / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench missing")
    sys.path.insert(0, str(pgx))
    torch.set_num_threads(1)
    roots = read(ROSTER)["rows"]
    audit = read(ROSTER_AUDIT)
    if (audit["status"] != "passed" or audit["roster_sha256"] != sha(ROSTER)
            or audit["roots_replayed"] != len(roots)):
        raise ValueError("independent roster audit differs")
    pins = (ROSTER, ROSTER_AUDIT, PROTOCOL, Path(__file__), AUDITOR,
            Path("scripts/build_continuous_train_roster_v1.py"),
            Path("scripts/bootstrap_cost_value_teacher_v1.py"),
            Path("scripts/resource_cost_search.py"),
            Path("scripts/audit_f_short_positive_r2_yield_v1.py"),
            Path("scripts/build_additive_root_only_f_bank_v2.py"),
            Path("scripts/mcts_tournament_signals_v1.py"),
            Path("scripts/mcts_training_supervision.py"),
            Path("scripts/screen_legacy_unselected_root_f_v1.py"),
            Path("scripts/screen_legacy_unselected_root_f_v2.py"),
            Path("scripts/build_f_curriculum_roster_v1.py"),
            *sorted(Path("src/rf_knots").glob("*.py")),
            *sorted((pgx / "pgx_mcts_bench").glob("*.py")))
    contract = {"schema": "rf-knots-continuous-TRAIN-MCTS-v1",
                "root_count": len(roots), "batch_size": BATCH_SIZE,
                "config": CONFIG, "task_cpu_limit": CPU_LIMIT,
                "gradient_scope": "exact_zero_spent_root_only",
                "inputs": {str(path): sha(path) for path in pins}}
    for path, expected in read(ROSTER)["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"roster source/input changed: {path}")
        contract["inputs"][path] = expected
    contract = json.loads(json.dumps(contract))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": digest(contract),
            "next_task": 0, "shards": [], "batches": {}, "audits": [],
            "cpu_seconds": 0.0, "status": "preparing"}
        if (state["contract"] != contract or state["contract_sha256"] != digest(contract)
                or state["next_task"] != len(state["shards"])):
            raise ValueError("continuing search contract differs")
        for i, entry in enumerate(state["shards"]):
            if entry["file"] != f"attempt-{i:06d}.json" or sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("committed search shard SHA differs")
        atomic(manifest_path, state)
        audit_pending(args, state, manifest_path)
        start = state["next_task"]
        active_batch = None
        while not args.stop_after or state["next_task"] - start < args.stop_after:
            if (args.out / "STOP_REQUEST").exists() or shutil.disk_usage(args.out).free < MIN_FREE:
                state.update(status="requested_or_storage_stop", pid=None)
                atomic(manifest_path, state)
                return
            for path, expected in contract["inputs"].items():
                if sha(Path(path)) != expected:
                    raise ValueError(f"pinned search input differs: {path}")
            position = state["next_task"]
            batch_id = position // BATCH_SIZE
            key = str(batch_id)
            if key not in state["batches"]:
                state["batches"][key] = snapshot()
                atomic(manifest_path, state)
            if active_batch != batch_id:
                net, objective = policy(state["batches"][key])
                active_batch = batch_id
            root_index, weight_index, seed = task(position, len(roots))
            root = roots[root_index]
            data = root["instances"][weight_index]
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            config = {**CONFIG, "seed": seed, "horizon": instance.cap_moves}
            state.update(status="collecting", pid=os.getpid(), current_batch=batch_id)
            atomic(manifest_path, state)
            started_cpu, started_wall = time.process_time(), time.monotonic()
            handler = signal.signal(signal.SIGPROF, timeout)
            signal.setitimer(signal.ITIMER_PROF, CPU_LIMIT)
            try:
                rows, counts = run_batch(
                    [{"condition": root_index, "panel_index": root["source_index"],
                      "instance": data}], net, objective, 0, config,
                    prior="distance", cached=True)
                row, censored = rows[0], False
            except TimeoutError:
                row, counts, censored = None, {}, True
            finally:
                signal.setitimer(signal.ITIMER_PROF, 0)
                signal.signal(signal.SIGPROF, handler)
            cpu = time.process_time() - started_cpu
            shard = args.out / f"attempt-{position:06d}.json"
            atomic(shard, {"position": position, "root_index": root_index,
                           "weight": instance.cc_weight, "seed": seed,
                           "batch_id": batch_id, "row": row, "counts": dict(counts),
                           "cpu_censored": censored, "cpu_seconds": cpu,
                           "wall_seconds": time.monotonic() - started_wall})
            state["shards"].append({"file": shard.name, "sha256": sha(shard)})
            state["next_task"] += 1
            state["cpu_seconds"] += cpu
            atomic(manifest_path, state)
            print(json.dumps({"next_task": state["next_task"], "batch": batch_id,
                              "root": root_index, "seed": seed, "cpu_seconds": cpu,
                              "solved": bool(row and row["solved"])}), flush=True)
            if state["next_task"] % BATCH_SIZE == 0:
                audit_pending(args, state, manifest_path)
        state.update(status="checkpointed", pid=None)
        atomic(manifest_path, state)


if __name__ == "__main__":
    main()
