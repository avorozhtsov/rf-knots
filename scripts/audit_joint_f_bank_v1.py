"""Independent SHA, replay, source and exact F-rule audit of joint TRAIN bank."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness

BANK = Path("artifacts/joint-f-bank-v2")
ROSTER = Path("artifacts/f-curriculum-roster-v3/roster.json")
MANIFEST_SHA = "6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12"
CONTRACT_SHA = "0cb4647e48463aca272dcc932223873db24772e2a00bbf2465dc83b8a15f8dc8"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new audit output must not exist")
    manifest_path = BANK / "manifest.json"
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("bank manifest SHA differs")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["status"] != "completed" or manifest["next_root"] != 1000
            or len(manifest["roots"]) != 1000
            or manifest["contract_sha256"] != CONTRACT_SHA):
        raise ValueError("bank contract/coverage differs")
    for path, expected in manifest["contract"]["inputs"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bank pinned input differs: {path}")
    roots = [root for stage in json.loads(ROSTER.read_text())["stages"] for root in stage]
    if len(roots) != 1000:
        raise ValueError("roster coverage differs")
    counter = Counter()
    distinct = set()
    seen_groups = set()
    for position, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{position:04d}.json"
        if entry["file"] != path.name or sha(path) != entry["sha256"]:
            raise ValueError(f"bank root SHA/order differs: {position}")
        row = json.loads(path.read_text())
        root = roots[position]
        if (row["root_index"] != position or row["stage"] != root["stage"]
                or row["query_group"] != root["query_group"]):
            raise ValueError(f"bank root identity differs: {position}")
        if row["groups"]:
            distinct.add(position)
        for group in row["groups"]:
            weight = group["weight"]
            weight_index = 0 if weight == 10 else 1 if weight == 1000 else -1
            if weight_index < 0 or group["root_index"] != position:
                raise ValueError(f"group identity differs: {position}")
            source = group["source"]
            base = root["instances"][weight_index]
            expected_instance = ({**base, "cap_moves": base["cap_moves"] + 16}
                                 if source == "slack16" else base)
            if (source not in ("tight", "slack16")
                    or group["instance"] != expected_instance
                    or (source == "slack16" and root["stage"] != 3)):
                raise ValueError(f"group resources/source differ: {position}")
            key = (position, weight, source)
            if key in seen_groups:
                raise ValueError(f"duplicate group: {key}")
            seen_groups.add(key)
            for shard in group["source_shards"]:
                if sha(Path(shard["file"])) != shard["sha256"]:
                    raise ValueError(f"source shard SHA differs: {key}")
            if len(group["source_shards"]) != (2 if source == "slack16" else 1):
                raise ValueError(f"source shard coverage differs: {key}")
            instance = M.Instance(**{**group["instance"],
                                     "word": tuple(group["instance"]["word"])})
            witnesses = []
            seen_actions = set()
            for item in group["paths"]:
                actions = tuple(item["actions"])
                if actions in seen_actions:
                    raise ValueError(f"duplicate path: {key}")
                seen_actions.add(actions)
                witness = UnknotWitness.from_actions(instance.word, instance.strands,
                                                     instance.spec, actions)
                witness.verify()
                if ((witness.crossing_changes, witness.moves)
                        != (item["cc"], item["moves"])
                        or witness.crossing_changes > instance.cap_crossing_changes
                        or witness.moves > instance.cap_moves):
                    raise ValueError(f"path replay/caps differ: {key}")
                witnesses.append(witness)
            reference = UnknotWitness.from_dict(root["witness"])
            reference.verify()
            if tuple(step.action.to_flat(instance.spec) for step in reference.steps) not in seen_actions:
                raise ValueError(f"reference missing: {key}")
            traces = verified_success_traces(instance, witnesses)
            signals = _groups(traces, "F", seed=20261006, k=4,
                              near_margin=2, gap_scale=weight, gap_cap=4)
            if (len(signals) != 1
                    or len(signals[0]["coefficients"]) != group["signal_terms"]
                    or signals[0]["positive_paths"] != group["positive_paths"]
                    or signals[0]["negative_paths"] != group["negative_paths"]):
                raise ValueError(f"exact F signal differs: {key}")
            counter["groups"] += 1
            counter["paths"] += len(witnesses)
            counter[f"stage{root['stage']}_L{weight}_{source}"] += 1
    counter["distinct_roots"] = len(distinct)
    if dict(counter) != manifest["summary"]:
        raise ValueError("bank summary differs from independent replay")
    report = {"schema": "rf-knots-joint-F-bank-audit-v1",
              "manifest_sha256": MANIFEST_SHA,
              "source_sha256": sha(Path(__file__)),
              "summary": dict(counter)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"report_sha256": sha(args.out), "summary": dict(counter)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
