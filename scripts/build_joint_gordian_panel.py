"""Prospective general-pair complement to short-u TEST; fixed protocol 60."""

import argparse
import fcntl
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from build_common_block_gordian_panel import EXTRA, adequate, category
from build_fresh_group_panel import EXCLUSIONS, atomic_json, excluded_groups, group, sha
from common_block_gordian import make_pair

from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner

PROTOCOL = Path("research/60-prospective-joint-general-gordian-test.md")


def admission_reason(row, excluded, seen, groups, classes):
    a, b = row["left_fingerprint"], row["right_fingerprint"]
    left = row["interval"]["left"]
    key = canonical(tuple(left["word"]), left["strands"])
    g, c = group(a), category(row)
    if min(abs(a[1]), abs(b[1])) // 2 < 3:
        return "insufficient_endpoint_u_lower"
    if owner(a) != "test" or owner(b) == "quarantine":
        return "other_partition"
    if g in excluded:
        return "previous_group"
    if key in seen:
        return "duplicate"
    if groups[g, c] >= 4:
        return "group_cap"
    if classes[c] >= 32:
        return "class_cap"
    return "accepted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    names = (*EXCLUSIONS, *EXTRA, "embedding-common-block-gordian-supplement-v1")
    panels = [Path("artifacts") / name / "panel.json" for name in dict.fromkeys(names)]
    sources = [
        Path(__file__).resolve().relative_to(Path.cwd()),
        PROTOCOL,
        Path("scripts/build_common_block_gordian_panel.py"),
        Path("scripts/common_block_gordian.py"),
        Path("scripts/build_fresh_group_panel.py"),
        *sorted(Path("src/rf_knots").glob("*.py")),
    ]
    contract = {
        "seed": 20261025,
        "attempts": 2048,
        "family": "common-positive-block",
        "inputs": {str(p): sha(p) for p in panels + sources},
    }
    if args.out.exists() != args.resume:
        raise ValueError("existing output requires resume; new output cannot resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    excluded = excluded_groups([json.loads(p.read_text()) for p in panels])
    progress_path = args.out / "progress.json"
    previous = json.loads(progress_path.read_text()) if args.resume else None
    if previous and previous["contract"] != contract:
        raise ValueError("resume construction contract changed")
    if previous and previous["status"] == "completed":
        if sha(args.out / "panel.json") != previous["panel_sha256"]:
            raise ValueError("completed panel changed")
        print("already completed")
        return
    if not previous:
        for p in sources:
            destination = args.out / "source_snapshot" / p
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destination)
        atomic_json(args.out / "excluded_groups.json", sorted(excluded))
    elif json.loads((args.out / "excluded_groups.json").read_text()) != list(
        map(list, sorted(excluded))
    ):
        raise ValueError("saved exclusion set changed")
    records = previous["records"] if previous else []
    index = previous["next_index"] if previous else 0
    counts = Counter(previous["counts"] if previous else {})
    groups, classes, seen = Counter(), Counter(), set()
    for row in records:
        witness = verify_evaluation_pair(row)
        if admission_reason(row, excluded, seen, groups, classes) != "accepted":
            raise ValueError("committed record violates selection contract")
        groups[group(row["left_fingerprint"]), category(row)] += 1
        classes[category(row)] += 1
        seen.add(canonical(witness.start.word, witness.start.strands))

    def save(status, **extra):
        if any(sha(Path(p)) != h for p, h in contract["inputs"].items()):
            raise ValueError("frozen construction input changed")
        report = {
            "status": status,
            "pid": os.getpid(),
            "contract": contract,
            "next_index": index,
            "records": records,
            "counts": dict(counts),
            "classes": dict(classes),
            "groups": len({g for g, _ in groups}),
            **extra,
        }
        atomic_json(progress_path, report)
        print(
            json.dumps(
                {k: report[k] for k in ("status", "next_index", "counts", "classes", "groups")}
            ),
            flush=True,
        )

    save("building")
    while index < contract["attempts"]:
        witness, prefix, construction = make_pair(contract["seed"], index)
        endpoint = witness.steps[prefix - 1].after
        a = invariants(witness.start.word, witness.start.strands)
        b = invariants(endpoint.word, endpoint.strands)
        upper = sum(s.action.kind == "CROSSING_CHANGE" for s in witness.steps[:prefix])
        row = {
            "split": "test",
            "seed": contract["seed"],
            "index": index,
            "construction": construction,
            "left_fingerprint": list(a),
            "right_fingerprint": list(b),
            "witness": witness.to_dict(),
            "pair_prefix_length": prefix,
            "interval": {
                "metric": "cc",
                "left": witness.start.to_dict(),
                "right": endpoint.to_dict(),
                "lower": abs(a[1] - b[1]) // 2,
                "upper": upper,
                "witness_semantic_length": prefix,
            },
        }
        reason = admission_reason(row, excluded, seen, groups, classes)
        if reason == "accepted":
            verify_evaluation_pair(row)
            records.append(row)
            groups[group(a), category(row)] += 1
            classes[category(row)] += 1
            seen.add(canonical(witness.start.word, witness.start.strands))
        counts[reason] += 1
        index += 1
        if index % 16 == 0:
            save("building")
    save("writing_panel")
    large = []
    for row in records:
        witness, prefix = lift(verify_evaluation_pair(row), 33, 512)
        expanded = {
            **row,
            "evaluation_view": "size_shift",
            "witness": witness.to_dict(),
            "pair_prefix_length": row["pair_prefix_length"] + prefix,
            "interval": {
                **row["interval"],
                "left": witness.start.to_dict(),
                "witness_semantic_length": row["pair_prefix_length"] + prefix,
            },
        }
        verify_evaluation_pair(expanded)
        large.append(expanded)
    panel = {
        "schema": "rf-knots-gordian-pair-panel-v1",
        "pair_kind": "general-gordian",
        "partition": "test",
        "complete": adequate(records),
        "construction_complete": True,
        "contract": contract,
        "validation": records,
        "size_shift": large,
        "groups": len({g for g, _ in groups}),
        "classes": dict(classes),
        "caveat": "Fresh query groups in a known synthetic family, not all knot types. "
        "Both endpoint signature bounds u>=3; lifts are correlated representations.",
    }
    panel_path = args.out / "panel.json"
    if panel_path.exists() and json.loads(panel_path.read_text()) != panel:
        raise ValueError("uncommitted final panel differs; preserving it")
    if not panel_path.exists():
        atomic_json(panel_path, panel)
    save("completed", panel_sha256=sha(panel_path))


if __name__ == "__main__":
    main()
