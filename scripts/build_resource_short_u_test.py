"""Score-blind protocol136 orchestration and replay for the new135 short-u test."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from raster_pair_training import CORPUS, PANELS

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import canonical, invariants, owner, verify_record

CONFIG = dict(seed=20261072, attempts=10000, max_strands=33, min_depth=24, max_depth=96)
PROTOCOL = Path("research/136-resource-units-fresh-short-u-test.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires explicit resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pins, groups, training, historical = partition_inputs()
    excluded = set(training | historical)
    for path in [CORPUS, *PANELS]:
        data = json.loads(path.read_text())
        excluded.update(
            (r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in data["validation"]
        )
        merge_pins(pins, {str(path): sha(path)})
    for path in (
        Path(__file__).resolve(),
        PROTOCOL,
        Path("scripts/build_constructive_short_u_panel.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    contract = dict(config=CONFIG, inputs=pins)
    path = args.out / "manifest.json"
    state = (
        json.loads(path.read_text())
        if args.resume
        else dict(contract=contract, status="preparing", contract_sha256=digest(contract))
    )
    if state["contract"] != contract:
        raise ValueError("panel resume contract differs")
    check_pins(pins)
    if state["status"] == "completed":
        if sha(args.out / "generator" / "panel.json") != state["panel_sha256"]:
            raise ValueError("completed panel changed")
        return
    state.update(pid=os.getpid(), status="preparing")
    atomic(path, state)
    registry = dict(
        schema="query-group-exclusion-registry-not-labelled-examples",
        validation=[
            dict(left_fingerprint=list(g), right_fingerprint=[1, 0]) for g in sorted(excluded)
        ],
        size_shift=[],
    )
    registry_path = args.out / "excluded-groups.json"
    if registry_path.exists():
        if json.loads(registry_path.read_text()) != registry:
            raise ValueError("exclusion registry changed")
    else:
        atomic(registry_path, registry)
    output = args.out / "generator"
    command = [
        sys.executable,
        "-u",
        "scripts/build_constructive_short_u_panel.py",
        "--out",
        str(output),
        "--protocol",
        str(PROTOCOL),
        "--exclude-panel",
        str(registry_path),
    ]
    for key, value in CONFIG.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if output.exists():
        command.append("--resume")
    state.update(
        status="generating",
        command=command,
        excluded_group_count=len(excluded),
        registry_sha256=sha(registry_path),
    )
    atomic(path, state)
    with (args.out / "generation.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(status="failed", returncode=result.returncode)
        atomic(path, state)
        raise RuntimeError("panel generator failed; see generation.log")
    state.update(status="auditing")
    atomic(path, state)
    panel = json.loads((output / "panel.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    if (
        progress["status"] != "completed"
        or progress["next_index"] != CONFIG["attempts"]
        or sum(progress["counts"].values()) != CONFIG["attempts"]
        or {k: panel["contract"][k] for k in CONFIG} != CONFIG
    ):
        raise ValueError("fixed attempt/config coverage differs")
    counts, seen = Counter(), set()
    selected_groups = set()
    for row, lifted in zip(panel["validation"], panel["size_shift"], strict=True):
        witness = verify_record(row)
        verify_record(lifted)
        regenerated = conjugate_witness(
            CONFIG["seed"],
            row["index"],
            **{k: CONFIG[k] for k in ("max_strands", "min_depth", "max_depth")},
        )
        if regenerated.to_dict() != row["witness"]:
            raise ValueError("selected witness differs from deterministic construction")
        fp = invariants(witness.start.word, witness.start.strands)
        if fp[0] != determinant(witness.start.word, witness.start.strands):
            raise ValueError("independent determinant differs")
        group = fp[0], abs(fp[1])
        key = canonical(witness.start.word, witness.start.strands)
        if group in excluded or owner(fp) != "test" or key in seen:
            raise ValueError("query exclusion, ownership or uniqueness failed")
        seen.add(key)
        selected_groups.add(group)
        counts[str(witness.crossing_changes)] += 1
    check_pins(pins)
    check_pins(panel["contract"]["inputs"])
    adequate = len(seen) == 64 and counts == {"1": 32, "2": 32} and len(selected_groups) >= 8
    state.update(
        status="completed",
        evaluation_ready=adequate,
        selected_roots=len(seen),
        classes=dict(counts),
        selected_groups=sorted(selected_groups),
        training_groups={k: len(v) for k, v in groups.items()},
        panel_sha256=sha(output / "panel.json"),
        progress_sha256=sha(output / "progress.json"),
        scope="RF-lineage query separation; Q254 pretraining exposure not proven absent",
    )
    atomic(path, state)
    print(
        json.dumps(
            {k: state[k] for k in ("status", "evaluation_ready", "selected_roots", "classes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
