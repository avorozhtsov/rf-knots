"""Resumable audited TRAIN F bank from tight-cap teacher and +16 hard roots."""

import argparse
import fcntl
import json
import os
from collections import Counter
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
OLD = Path("artifacts/qrun-rung1-onpolicy-teacher-v1")
OLD_AUDIT = Path("artifacts/qrun-rung1-onpolicy-teacher-audit-v2/report.json")
SLACK = Path("artifacts/f-hard-stage-move-slack-pilot-v1")
SLACK_AUDIT = Path("artifacts/f-hard-stage-move-slack-pilot-audit-v1/report.json")
PINS = {
    str(ROSTER): "268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506",
    str(OLD / "manifest.json"): "658c128c11fed26a639994c5d54af8ad31550f7f189be52c144cbb76c192f6a2",
    str(OLD_AUDIT): "ee1049c12bdaca9142d13c2a8759e6bba91fe3c10d34815e1ddff3e58e1bbb86",
    str(SLACK / "manifest.json"): "e4423cd705cc5c351e83b886fe941d63b652dd24666cd69feff8c0e92a6be926",
    str(SLACK_AUDIT): "8aecc3eaac6698b6f31645d1bb02e82612800d12cdf4075cf7f39e97bf70c8bc",
}
SLACK_ROOTS = (4, 8, 10, 21, 0, 1, 2, 3)
STAGE_START = (0, 140, 426, 713)


def read(path):
    return json.loads(path.read_text())


def paths_from_shard(path, expected_sha, instance, *, old):
    if sha(path) != expected_sha:
        raise ValueError(f"source shard SHA differs: {path}")
    row = read(path)
    result = row["result"]
    if len(result["episodes"]) != len(result["episodes_train_owned"]):
        raise ValueError(f"source ownership coverage differs: {path}")
    output = []
    for episode, admitted in zip(result["episodes"],
                                 result["episodes_train_owned"], strict=True):
        if not admitted or episode["continuation"]["stop_reason"] != "solved":
            continue
        actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
        witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                             instance.spec, actions)
        witness.verify()
        if (witness.crossing_changes > instance.cap_crossing_changes
                or witness.moves > instance.cap_moves):
            raise ValueError(f"source admitted path exceeds cap: {path}")
        output.append((actions, witness))
    if len(output) != result["eligible_solved_episodes"]:
        raise ValueError(f"source admitted path count differs: {path}")
    return row, output


def make_group(root, root_index, weight_index, source, old_manifest, slack_manifest):
    weight = (10, 1000)[weight_index]
    stage = root["stage"]
    base = root["instances"][weight_index]
    instance_dict = ({**base, "cap_moves": base["cap_moves"] + 16}
                     if source == "slack16" else base)
    instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
    reference = UnknotWitness.from_dict(root["witness"])
    reference.verify()
    if ((reference.start.word, reference.start.strands)
            != (instance.word, instance.strands)
            or reference.crossing_changes > instance.cap_crossing_changes
            or reference.moves > instance.cap_moves):
        raise ValueError("reference replay/start/cap differs")
    paths = {tuple(step.action for step in reference.steps): reference}
    source_shards = []
    if source == "tight":
        position = root_index * 2 + weight_index
        entry = old_manifest["shards"][position]
        path = OLD / f"condition-{position:04d}.json"
        if entry["file"] != path.name:
            raise ValueError("tight source ordering differs")
        row, found = paths_from_shard(path, entry["sha256"], instance, old=True)
        if ((row["task"], row["root_index"], row["weight"], row["stage"])
                != (position, root_index, weight, stage)):
            raise ValueError("tight task identity differs")
        source_shards.append({"file": str(path), "sha256": entry["sha256"]})
        for actions, witness in found:
            paths.setdefault(actions, witness)
    else:
        if stage != 3:
            raise ValueError("slack source applies only to stage 3")
        local_index = root_index - STAGE_START[3]
        if local_index not in SLACK_ROOTS:
            raise ValueError("slack root not in frozen sample")
        for position, task in enumerate(slack_manifest["contract"]["tasks"]):
            if task[0] != local_index or task[1] != weight_index or task[3] != 16:
                continue
            entry = slack_manifest["shards"][position]
            path = SLACK / f"condition-{position:04d}.json"
            if entry["file"] != path.name:
                raise ValueError("slack source ordering differs")
            row, found = paths_from_shard(path, entry["sha256"], instance, old=False)
            if ((row["task"], row["root_index"], row["weight"], row["move_slack"],
                 row["seed"], row["simulations"])
                    != (position, local_index, weight, 16, task[2], 128)):
                raise ValueError("slack task identity differs")
            source_shards.append({"file": str(path), "sha256": entry["sha256"]})
            for actions, witness in found:
                paths.setdefault(actions, witness)
        if len(source_shards) != 2:
            raise ValueError("slack source seed coverage differs")
    witnesses = list(paths.values())
    traces = verified_success_traces(instance, witnesses)
    signals = _groups(traces, "F", seed=20261006, k=4,
                      near_margin=2, gap_scale=weight, gap_cap=4)
    if not signals or not any(signal["coefficients"] for signal in signals):
        return None
    return {"root_index": root_index, "stage": stage, "weight": weight,
            "source": source, "instance": instance_dict,
            "source_shards": source_shards,
            "signal_terms": sum(len(signal["coefficients"]) for signal in signals),
            "positive_paths": signals[0]["positive_paths"],
            "negative_paths": signals[0]["negative_paths"],
            "paths": [{"actions": list(actions),
                       "cc": witness.crossing_changes, "moves": witness.moves}
                      for actions, witness in paths.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output without --resume; existing output with --resume")
    for path, expected in PINS.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"pinned source differs: {path}")
    roots = [row for stage in read(ROSTER)["stages"] for row in stage]
    old_manifest, slack_manifest = read(OLD / "manifest.json"), read(SLACK / "manifest.json")
    if (len(roots) != 1000 or old_manifest["status"] != "completed"
            or old_manifest["next_task"] != 2000
            or slack_manifest["status"] != "completed"
            or slack_manifest["next_task"] != 64):
        raise ValueError("source collection coverage differs")
    inputs = {**PINS, str(Path(__file__).resolve()): sha(Path(__file__)),
              "research/300-joint-f-bank-and-r2-r4-start-gate-v1.md":
              sha(Path("research/300-joint-f-bank-and-r2-r4-start-gate-v1.md")),
              "scripts/mcts_tournament_signals_v1.py":
              sha(Path("scripts/mcts_tournament_signals_v1.py"))}
    contract = {"schema": "rf-knots-joint-F-bank-v1", "inputs": inputs,
                "root_count": 1000, "slack_roots": list(SLACK_ROOTS),
                "F": {"near_margin": 2, "max_positives": 3, "k": 4,
                      "gap_cap": 4, "seed": 20261006}}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "next_root": 0, "roots": []}
        if (state["contract"] != contract or state["contract_sha256"] != contract_sha
                or state["next_root"] != len(state["roots"])):
            raise ValueError("bank resume contract/coverage differs")
        for position, entry in enumerate(state["roots"]):
            path = args.out / f"root-{position:04d}.json"
            if entry["file"] != path.name or sha(path) != entry["sha256"]:
                raise ValueError(f"saved bank root differs: {position}")
        end = min(1000, state["next_root"] + (args.stop_after or 1000))
        state.update(status="building", pid=os.getpid())
        atomic(manifest_path, state)
        for position in range(state["next_root"], end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned bank input changed")
            root = roots[position]
            groups = [group for weight_index in range(2)
                      if (group := make_group(root, position, weight_index, "tight",
                                              old_manifest, slack_manifest)) is not None]
            if position >= STAGE_START[3] and position - STAGE_START[3] in SLACK_ROOTS:
                groups.extend(group for weight_index in range(2)
                              if (group := make_group(root, position, weight_index,
                                                      "slack16", old_manifest,
                                                      slack_manifest)) is not None)
            payload = {"root_index": position, "stage": root["stage"],
                       "query_group": root["query_group"], "groups": groups}
            path = args.out / f"root-{position:04d}.json"
            atomic(path, payload)
            state["roots"].append({"file": path.name, "sha256": sha(path)})
            state.update(next_root=position + 1, pid=os.getpid())
            atomic(manifest_path, state)
            if (position + 1) % 16 == 0 or groups:
                print(json.dumps({"next_root": position + 1,
                                  "F_groups": len(groups)}), flush=True)
        if state["next_root"] == 1000:
            counter = Counter()
            for entry in state["roots"]:
                row = read(args.out / entry["file"])
                if row["groups"]:
                    counter["distinct_roots"] += 1
                for group in row["groups"]:
                    counter["groups"] += 1
                    counter["paths"] += len(group["paths"])
                    counter[f"stage{group['stage']}_L{group['weight']}_{group['source']}"] += 1
            state.update(status="completed", pid=None, summary=dict(counter))
            atomic(manifest_path, state)
            print(json.dumps({"completed": True, "summary": dict(counter),
                              "manifest_sha256": sha(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
