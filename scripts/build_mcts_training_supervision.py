"""Prepare audited set-valued policy and explicit-unknot metric supervision.

This is data preparation, not a training run or a generalization result.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mcts_training_supervision import success_supervision

from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import canonical, invariants, owner, path_fingerprints
from rf_knots.evidence import UnknotWitness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def short_u_views(witness, witness_id, *, maximum_cc=2):
    """Endpoint is explicitly the empty one-braid, never an arbitrary pair endpoint."""
    witness.verify()
    states = [witness.start] + [step.after for step in witness.steps]
    remaining = witness.crossing_changes
    fp = invariants(witness.start.word, witness.start.strands)
    views = []
    for depth, state in enumerate(states[:-1]):
        if depth and witness.steps[depth - 1].action.kind == "CROSSING_CHANGE":
            remaining -= 1
            fp = invariants(state.word, state.strands)
        if owner(fp) != "train":
            raise ValueError("non-TRAIN state in metric suffix")
        lower = max(abs(fp[1]) // 2, int(fp[0] != 1))
        if lower > remaining:
            raise ValueError("certified lower bound exceeds suffix witness upper bound")
        if remaining <= maximum_cc:
            views.append(
                {
                    "word": list(state.word),
                    "strands": state.strands,
                    "right_word": [],
                    "right_strands": 1,
                    "fingerprint": list(fp),
                    "lower": lower,
                    "upper": remaining,
                    "remaining_moves": witness.moves - depth,
                    "witness_id": witness_id,
                    "suffix_index": depth,
                }
            )
    return views


def condition_witnesses(instance, root, row, audit_row, heldout):
    """References/eligible successes only; relative groups have a stricter gate."""
    if row["condition"] != audit_row["condition"]:
        raise ValueError("condition audit mismatch")
    witnesses = [UnknotWitness.from_dict(root["reference_witness"])]
    for episode in row["episodes"]:
        if episode["eligible"] and episode["continuation"]["stop_reason"] == "solved":
            actions = tuple(episode["tree_prefix"] + episode["continuation"]["actions"])
            witnesses.append(
                UnknotWitness.from_actions(instance.word, instance.strands, instance.spec, actions)
            )
    if audit_row["best_witness_train_owned"]:
        witnesses.append(UnknotWitness.from_dict(row["witness"]))
    for witness in witnesses:
        if any(
            owner(fp) != "train" or (fp[0], abs(fp[1])) in heldout
            for fp in path_fingerprints(witness)
        ):
            raise ValueError("heldout state in a proposed training witness")
    return witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite supervision")
    audit = json.loads(args.audit.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    if audit["status"] != "passed" or manifest["status"] != "completed":
        raise ValueError("require completed collection and passed audit")
    manifest_name = str(args.data / "manifest.json")
    if audit["inputs"].get(manifest_name) != sha(Path(manifest_name)):
        raise ValueError("audit does not cover this collection")
    pins = {
        **audit["inputs"],
        str(args.audit): sha(args.audit),
        str(Path(__file__).resolve()): sha(Path(__file__)),
        "scripts/mcts_training_supervision.py": sha(Path("scripts/mcts_training_supervision.py")),
        "research/55-mcts-success-supervision-contract.md": sha(
            Path("research/55-mcts-success-supervision-contract.md")
        ),
    }
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("audited/preparation input changed")
    roots = json.loads((args.data / "roots.json").read_text())
    heldout = set(map(tuple, manifest["heldout_query_groups"]))
    gates = {r["condition"]: r for r in audit["conditions"]}
    assert set(gates) == set(range(len(roots)))
    args.out.mkdir(parents=True)
    report = {
        "schema": "rf-knots-mcts-success-supervision-v1",
        "status": "preparing",
        "pid": os.getpid(),
        "inputs": pins,
        "conditions": [],
        "counts": {},
    }
    atomic(args.out / "manifest.json", report)
    witnesses_by_id, metric_views, counts = {}, {}, Counter()
    for shard in manifest["shards"]:
        for row in json.loads((args.data / shard["file"]).read_text()):
            index = row["condition"]
            root, gate = roots[index], gates[index]
            i = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
            witnesses = condition_witnesses(i, root, row, gate, heldout)
            targets = success_supervision(i, witnesses)
            if not gate["all_episodes_train_owned"]:
                targets["preferences"] = []
            for section in ("positives", "preferences"):
                for item in targets[section]:
                    item["state"] = asdict(item["state"])
            target_path = args.out / f"targets-{index:05d}.json"
            atomic(
                target_path,
                {
                    "condition": index,
                    "instance": root["instance"],
                    "base_root_index": root["base_root_index"],
                    "corpus_index": root["corpus_index"],
                    **targets,
                },
            )
            counts["conditions"] += 1
            counts["set_valued_states"] += len(targets["positives"])
            counts["multi_positive_states"] += sum(
                len(r["positive_actions"]) > 1 for r in targets["positives"]
            )
            counts["preference_conditions"] += bool(targets["preferences"])
            counts["preference_terms"] += len(targets["preferences"])
            counts["conditions_with_alternative_paths"] += targets["unique_success_paths"] > 1
            report["conditions"].append(
                {
                    "condition": index,
                    "file": target_path.name,
                    "sha256": sha(target_path),
                    "unique_success_paths": targets["unique_success_paths"],
                    "distinct_success_costs": targets["distinct_success_costs"],
                }
            )
            for witness in witnesses:
                payload = witness.to_dict()
                witness_id = digest(payload)
                if witness_id in witnesses_by_id:
                    continue
                witnesses_by_id[witness_id] = payload
                for view in short_u_views(witness, witness_id):
                    key = canonical(tuple(view["word"]), view["strands"])
                    old = metric_views.get(key)
                    if old is not None and old["lower"] != view["lower"]:
                        raise ValueError("same state has inconsistent exact invariant bound")
                    if old is None or (view["upper"], view["remaining_moves"]) < (
                        old["upper"],
                        old["remaining_moves"],
                    ):
                        metric_views[key] = view
        report["counts"] = dict(counts)
        atomic(args.out / "manifest.json", report)
        print(json.dumps(dict(counts)), flush=True)
    if counts["conditions"] != len(roots):
        raise ValueError("incomplete supervision coverage")
    metric = {
        "right_endpoint": {"word": [], "strands": 1},
        "views": list(metric_views.values()),
        "witnesses": witnesses_by_id,
    }
    atomic(args.out / "short-u-views.json", metric)
    counts["metric_views"] = len(metric_views)
    counts["metric_witnesses"] = len(witnesses_by_id)
    report.update(
        status="completed",
        counts=dict(counts),
        metric_sha256=sha(args.out / "short-u-views.json"),
        metric_bound_counts=dict(
            Counter(f"{v['lower']}:{v['upper']}" for v in metric_views.values())
        ),
    )
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("audited/preparation input changed during preparation")
    atomic(args.out / "manifest.json", report)


if __name__ == "__main__":
    main()
