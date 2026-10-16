"""Protocol150 score-blind short-u acquisition and post-exit panel consumer."""

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
from long_general_training_data import load_long_source

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner, verify_record

CONFIG = dict(seed=20261078, attempts=10000, max_strands=33, min_depth=24, max_depth=96)
LIFT = dict(strands=34, length=385)
PROTOCOL = Path("research/150-long-mixture-fresh-short-u-test.md")
SCHEMA = "rf-knots-long-mixture-short-u-acquisition-v1"


def group(fp):
    return fp[0], abs(fp[1])


def registry_groups(corpus):
    excluded = set(map(tuple, corpus["excluded_query_groups"]))
    for groups in corpus["training_groups"].values():
        excluded.update(map(tuple, groups))
    for row in corpus["records"]["train"]:
        excluded.update(map(group, row["fingerprints"]))
    for split in ("validation", "test"):
        excluded.update(group(r["left_fingerprint"]) for r in corpus["records"][split])
    return excluded


def inputs(long_data):
    _, _, pins = load_long_source(long_data)
    corpus = json.loads((long_data / "corpus.json").read_text())
    excluded = registry_groups(corpus)
    for path in (
        Path(__file__).resolve(),
        PROTOCOL,
        Path("scripts/build_constructive_short_u_panel.py"),
        Path("scripts/long_general_training_data.py"),
        Path("tests/test_long_general_training_data.py"),
        Path("tests/test_long_mixture_short_u_test.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    return pins, excluded


def lifted_record(record):
    witness = verify_record(record)
    enlarged, _ = lift(witness, LIFT["strands"], LIFT["length"])
    return {
        **record,
        "evaluation_view": "size_shift",
        "witness": enlarged.to_dict(),
        "pair_prefix_length": enlarged.moves,
        "interval": {
            **record["interval"],
            "left": enlarged.start.to_dict(),
            "witness_semantic_length": enlarged.moves,
        },
    }


def audit_records(panel, excluded):
    if len(panel["validation"]) != len(panel["size_shift"]):
        raise ValueError("original/lift coverage differs")
    seen, groups, counts, bound_classes = set(), Counter(), Counter(), Counter()
    lengths, strands = [], []
    for row, enlarged in zip(panel["validation"], panel["size_shift"], strict=True):
        witness = verify_record(row)
        lifted = verify_record(enlarged)
        regenerated = conjugate_witness(
            CONFIG["seed"],
            row["index"],
            **{k: CONFIG[k] for k in ("max_strands", "min_depth", "max_depth")},
        )
        if witness.to_dict() != regenerated.to_dict() or enlarged != lifted_record(row):
            raise ValueError("selected deterministic witness/lift differs")
        if row["seed"] != CONFIG["seed"] or row["split"] != "test":
            raise ValueError("selected seed/split differs")
        if type(row["index"]) is not int or not 0 <= row["index"] < CONFIG["attempts"]:
            raise ValueError("selected attempt outside contract")
        fp = invariants(witness.start.word, witness.start.strands)
        lfp = invariants(lifted.start.word, lifted.start.strands)
        if (
            any(
                determinant(w.start.word, w.start.strands) != f[0]
                for w, f in ((witness, fp), (lifted, lfp))
            )
            or fp != lfp
        ):
            raise ValueError("independent original/lift determinant differs")
        g, key = group(fp), canonical(witness.start.word, witness.start.strands)
        if g in excluded or owner(fp) != "test" or key in seen or fp[0] == 1:
            raise ValueError("query exclusion/owner/uniqueness differs")
        if groups[g] >= 4:
            raise ValueError("selected invariant-group cap exceeded")
        if len(lifted.start.word) < LIFT["length"] or lifted.start.strands < LIFT["strands"]:
            raise ValueError("new size-shift bounds not reached")
        seen.add(key)
        groups[g] += 1
        counts[str(witness.crossing_changes)] += 1
        lo = max(row["interval"]["lower"], int(fp[0] != 1))
        hi = row["interval"]["upper"]
        if lo > hi:
            raise ValueError("certified short-u interval inconsistent")
        bound_classes[f"{lo}..{hi}"] += 1
        lengths.append([len(witness.start.word), len(lifted.start.word)])
        strands.append([witness.start.strands, lifted.start.strands])
    adequate = len(seen) == 64 and counts == {"1": 32, "2": 32} and len(groups) >= 8
    return dict(
        evaluation_ready=adequate,
        selected_roots=len(seen),
        classes=dict(counts),
        selected_groups=[list(g) for g in sorted(groups)],
        certified_bounds=dict(bound_classes),
        endpoint_lengths=lengths,
        endpoint_strands=strands,
    )


def generator_unlocked(directory):
    if directory.exists():
        with (directory / ".runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "original generator is still active; do not duplicate"
                ) from error


def verify_completed(root, long_data):
    pins, excluded = inputs(long_data)
    manifest_path, panel_path = root / "manifest.json", root / "panel.json"
    state, panel = (json.loads(p.read_text()) for p in (manifest_path, panel_path))
    expected_contract = dict(config=CONFIG, lift=LIFT, inputs=pins)
    if (
        state["schema"] != SCHEMA
        or state["status"] != "completed"
        or state["contract"] != expected_contract
        or state["contract_sha256"] != digest(expected_contract)
        or state["panel_sha256"] != sha(panel_path)
        or panel["acquisition_contract"] != expected_contract
    ):
        raise ValueError("completed acquisition contract differs")
    generated_path = root / "generator" / "panel.json"
    progress_path = root / "generator" / "progress.json"
    generated, progress = (json.loads(p.read_text()) for p in (generated_path, progress_path))
    if (
        sha(generated_path) != state["generated_panel_sha256"]
        or sha(progress_path) != state["progress_sha256"]
        or progress["status"] != "completed"
        or progress["next_index"] != CONFIG["attempts"]
        or sum(progress["counts"].values()) != CONFIG["attempts"]
        or {k: generated["contract"][k] for k in CONFIG} != CONFIG
        or generated["contract"] != progress["contract"]
        or generated["schema"] != "rf-knots-gordian-pair-panel-v1"
        or generated["pair_kind"] != "short-u"
        or generated["partition"] != "test"
        or generated["construction_complete"] is not True
        or panel["validation"] != generated["validation"]
        or panel["contract"] != generated["contract"]
        or panel["lift"] != LIFT
    ):
        raise ValueError("generator attempt/input/original coverage differs")
    registry_path = root / "excluded-groups.json"
    if json.loads(registry_path.read_text()) != registry(excluded):
        raise ValueError("query exclusion registry differs")
    if generated["contract"]["inputs"].get(str(registry_path)) != sha(registry_path):
        raise ValueError("generator did not bind the exclusion registry")
    merge_pins(pins, generated["contract"]["inputs"])
    merge_pins(
        pins, {str(p): sha(p) for p in (manifest_path, panel_path, generated_path, progress_path)}
    )
    check_pins(pins)
    report = audit_records(panel, excluded)
    if report != state["selected"] or panel["complete"] != report["evaluation_ready"]:
        raise ValueError("completed selected certificate report differs")
    check_pins(pins)
    return dict(
        status="passed",
        **report,
        panel=str(panel_path),
        panel_sha256=sha(panel_path),
        inputs=pins,
        attempts=CONFIG["attempts"],
        model_forwards=0,
        optimizer_steps=0,
        scope="Synthetic RF-lineage query-disjoint short-u roots with full original/lift "
        "replay. Familiar unknot reference; full Q254 exposure not established. "
        "Rejected attempt invariants not regenerated by this consumer.",
    )


def registry(excluded):
    return dict(
        schema="query-group-exclusion-registry-not-labelled-examples",
        validation=[
            dict(left_fingerprint=list(g), right_fingerprint=[1, 0]) for g in sorted(excluded)
        ],
        size_shift=[],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        if args.resume:
            raise ValueError("audit-only is not a resume")
        target = args.out / "consumer-audit.json"
        if target.exists() or target.with_suffix(".partial").exists():
            raise ValueError("consumer output must be new")
        report = verify_completed(args.out, args.long_data)
        atomic(target, report)
        print(
            json.dumps({k: report[k] for k in ("status", "evaluation_ready", "selected_roots")}),
            flush=True,
        )
        return
    # Do not create an apparent run if the parent data/audit is still unavailable.
    pins, excluded = inputs(args.long_data)
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "manifest.json"
    contract = dict(config=CONFIG, lift=LIFT, inputs=pins)
    state = (
        json.loads(manifest_path.read_text())
        if args.resume
        else dict(
            schema=SCHEMA, contract=contract, contract_sha256=digest(contract), status="preparing"
        )
    )
    if state["contract"] != contract or state["contract_sha256"] != digest(contract):
        raise ValueError("resume acquisition contract differs")
    if state["status"] == "completed":
        verify_completed(args.out, args.long_data)
        print("completed acquisition verified; no generator restart", flush=True)
        return
    if state["status"] not in ("preparing", "generating", "failed", "auditing"):
        raise ValueError("unknown acquisition state")
    generator = args.out / "generator"
    generator_unlocked(generator)
    state.update(pid=os.getpid(), status="preparing")
    atomic(manifest_path, state)
    registry_path = args.out / "excluded-groups.json"
    if registry_path.exists():
        if json.loads(registry_path.read_text()) != registry(excluded):
            raise ValueError("resume exclusion registry differs")
    else:
        atomic(registry_path, registry(excluded))
    command = [
        sys.executable,
        "-u",
        "scripts/build_constructive_short_u_panel.py",
        "--out",
        str(generator),
        "--protocol",
        str(PROTOCOL),
        "--exclude-panel",
        str(registry_path),
    ]
    for key, value in CONFIG.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if generator.exists():
        command.append("--resume")
    state.update(status="generating", command=command, excluded_groups=len(excluded))
    atomic(manifest_path, state)
    with (args.out / "generation.log").open("a") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(status="failed", returncode=result.returncode)
        atomic(manifest_path, state)
        raise RuntimeError("short-u generator failed; preserve output and inspect its log")
    state.update(status="auditing")
    atomic(manifest_path, state)
    generated_path = generator / "panel.json"
    progress_path = generator / "progress.json"
    generated = json.loads(generated_path.read_text())
    progress = json.loads(progress_path.read_text())
    if (
        progress["status"] != "completed"
        or progress["next_index"] != CONFIG["attempts"]
        or sum(progress["counts"].values()) != CONFIG["attempts"]
        or {k: generated["contract"][k] for k in CONFIG} != CONFIG
        or generated["contract"] != progress["contract"]
    ):
        raise ValueError("generator fixed budget/config differs")
    panel = {
        **generated,
        "size_shift": [lifted_record(r) for r in generated["validation"]],
        "acquisition_contract": contract,
        "lift": LIFT,
    }
    report = audit_records(panel, excluded)
    panel["complete"] = report["evaluation_ready"]
    check_pins(pins)
    check_pins(generated["contract"]["inputs"])
    target = args.out / "panel.json"
    if target.exists():
        if json.loads(target.read_text()) != panel:
            raise ValueError("existing final panel differs; preserve for diagnosis")
    else:
        atomic(target, panel)
    state.update(
        status="completed",
        selected=report,
        panel_sha256=sha(target),
        generated_panel_sha256=sha(generated_path),
        progress_sha256=sha(progress_path),
    )
    atomic(manifest_path, state)
    print(
        json.dumps({k: report[k] for k in ("evaluation_ready", "selected_roots", "classes")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
