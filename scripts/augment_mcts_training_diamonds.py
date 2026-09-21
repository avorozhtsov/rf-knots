"""Add proof-certified action alternatives without changing metric targets or costs."""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from build_mcts_training_supervision import atomic, condition_witnesses, digest, sha
from mcts_training_diamonds import reorder_reduce_prefix, verified_diamond

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, REDUCE
from rf_knots.cost_mcts_rollout import advance
from rf_knots.embedding_replay_corpus import invariants
from rf_knots.evidence import SemanticAction


def state_from_dict(row):
    return M.SearchState(tuple(row["word"]), row["strands"], row["used_cc"], row["used_moves"])


def add_alternatives(instance, data, witnesses, source_registry, heldout):
    positives = {state_from_dict(r["state"]): r for r in data["positives"]}
    certificates, seen_paths = [], set()
    for witness in witnesses:
        source_id = digest(witness.to_dict())
        if source_registry.get(source_id) != witness.to_dict():
            raise ValueError("source witness is absent from pinned base registry")
        actions = tuple(s.action.to_flat(instance.spec) for s in witness.steps)
        if actions in seen_paths:
            continue
        seen_paths.add(actions)
        states = [M.SearchState(instance.word, instance.strands, 0, 0)]
        fps = [invariants(instance.word, instance.strands)]
        for action in actions:
            states.append(advance(instance, states[-1], action))
            fps.append(
                invariants(states[-1].word, states[-1].strands)
                if instance.spec.decode(action)[0] == CROSSING_CHANGE
                else fps[-1]
            )
        for offset in range(len(actions)):
            state = states[offset]
            target = positives[state]
            cost = (
                instance.cc_weight * (witness.crossing_changes - state.used_cc)
                + witness.moves
                - state.used_moves
            )
            if cost != target["known_suffix_cost"]:
                continue

            def record(
                first,
                method,
                span,
                details=None,
                *,
                target=target,
                source_id=source_id,
                offset=offset,
            ):
                if first in target["positive_actions"]:
                    return
                target["positive_actions"] = sorted(set(target["positive_actions"]) | {first})
                target["actions"] = sorted(set(target["actions"]) | {first})
                certificates.append(
                    {
                        "condition": data["condition"],
                        "witness_id": source_id,
                        "suffix_index": offset,
                        "span": span,
                        "method": method,
                        "first_action": SemanticAction.from_flat(instance.spec, first).to_dict(),
                        **(details or {}),
                    }
                )

            # Different raw action positions can lead to exactly the same next
            # state. Certify all such choices already present in the sparse pool.
            for action in tuple(target["actions"]):
                if action not in target["positive_actions"]:
                    if advance(instance, state, action) == states[offset + 1]:
                        record(action, "same-successor", 1)
            if offset + 1 == len(actions):
                continue
            d = verified_diamond(
                instance, state, actions[offset], actions[offset + 1], fps[offset], fps[offset + 2]
            )
            if (
                d is not None
                and (d["intermediate_fingerprint"][0], abs(d["intermediate_fingerprint"][1]))
                not in heldout
            ):
                record(
                    d["actions"][0],
                    "adjacent-diamond",
                    2,
                    {
                        "alternate_actions": [
                            SemanticAction.from_flat(instance.spec, a).to_dict()
                            for a in d["actions"]
                        ],
                        "intermediate_fingerprint": list(d["intermediate_fingerprint"]),
                    },
                )
            if instance.spec.decode(actions[offset])[0] != REDUCE:
                continue
            for span in range(2, len(actions) - offset + 1):
                if instance.spec.decode(actions[offset + span - 1])[0] != REDUCE:
                    break
                alternate = reorder_reduce_prefix(
                    instance, states, actions, offset, span, fps[offset]
                )
                if alternate is not None:
                    record(alternate[0], "independent-reduce-prefix", span)
    for row in data["preferences"]:
        row["actions"] = positives[state_from_dict(row["state"])]["actions"]
    return certificates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite augmented supervision")
    base = json.loads((args.base / "manifest.json").read_text())
    audit = json.loads(args.audit.read_text())
    collection = json.loads((args.data / "manifest.json").read_text())
    if (
        base["status"] != "completed"
        or audit["status"] != "passed"
        or collection["status"] != "completed"
    ):
        raise ValueError("require completed, audited base data")
    if base["inputs"].get(str(args.audit)) != sha(args.audit) or audit["inputs"].get(
        str(args.data / "manifest.json")
    ) != sha(args.data / "manifest.json"):
        raise ValueError("audit is not linked to the specified base/collection")
    pins = {
        **base["inputs"],
        str(args.base / "manifest.json"): sha(args.base / "manifest.json"),
        str(args.audit): sha(args.audit),
        str(Path(__file__).resolve()): sha(Path(__file__)),
        "scripts/mcts_training_diamonds.py": sha(Path("scripts/mcts_training_diamonds.py")),
        "research/56-verified-witness-order-alternatives.md": sha(
            Path("research/56-verified-witness-order-alternatives.md")
        ),
        str(args.base / "short-u-views.json"): base["metric_sha256"],
    }
    for item in base["conditions"]:
        pins[str(args.base / item["file"])] = item["sha256"]
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("base/audit/source input hash changed")
    roots = json.loads((args.data / "roots.json").read_text())
    rows = {
        r["condition"]: r
        for shard in collection["shards"]
        for r in json.loads((args.data / shard["file"]).read_text())
    }
    gates = {r["condition"]: r for r in audit["conditions"]}
    heldout = set(map(tuple, collection["heldout_query_groups"]))
    registry = json.loads((args.base / "short-u-views.json").read_text())["witnesses"]
    args.out.mkdir(parents=True)
    for name in pins:
        p = Path(name)
        if p.suffix in (".py", ".md"):
            relative = p.resolve().relative_to(Path.cwd())
            target = args.out / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    report = {
        **base,
        "status": "augmenting",
        "pid": os.getpid(),
        "inputs": pins,
        "conditions": [],
        "certificates": [],
        "augmentation_counts": {},
        "unique_success_paths_excludes_diamonds": True,
    }
    atomic(args.out / "manifest.json", report)
    counts = Counter()
    for item in base["conditions"]:
        index = item["condition"]
        data = json.loads((args.base / item["file"]).read_text())
        instance = M.Instance(**{**data["instance"], "word": tuple(data["instance"]["word"])})
        witnesses = condition_witnesses(instance, roots[index], rows[index], gates[index], heldout)
        certificates = add_alternatives(instance, data, witnesses, registry, heldout)
        target_path, cert_path = (
            args.out / item["file"],
            args.out / f"certificates-{index:05d}.json",
        )
        atomic(target_path, data)
        atomic(cert_path, certificates)
        report["conditions"].append({**item, "sha256": sha(target_path)})
        report["certificates"].append(
            {
                "condition": index,
                "file": cert_path.name,
                "sha256": sha(cert_path),
                "count": len(certificates),
            }
        )
        counts["conditions"] += 1
        counts["certificates"] += len(certificates)
        counts["multi_positive_states"] += sum(
            len(r["positive_actions"]) > 1 for r in data["positives"]
        )
        counts["positive_action_slots"] += sum(
            len(r["positive_actions"]) for r in data["positives"]
        )
        counts["maximum_positive_set"] = max(
            counts["maximum_positive_set"],
            max(len(r["positive_actions"]) for r in data["positives"]),
        )
        report["augmentation_counts"] = dict(counts)
        if (index + 1) % 8 == 0:
            atomic(args.out / "manifest.json", report)
            print(json.dumps(dict(counts)), flush=True)
    shutil.copy2(args.base / "short-u-views.json", args.out / "short-u-views.json")
    assert sha(args.out / "short-u-views.json") == base["metric_sha256"]
    report["counts"] = {**base["counts"], "multi_positive_states": counts["multi_positive_states"]}
    if any(sha(Path(p)) != h for p, h in pins.items()):
        raise ValueError("base/audit/source input changed during augmentation")
    report["status"] = "completed"
    atomic(args.out / "manifest.json", report)


if __name__ == "__main__":
    main()
