"""Fixed, score-blind supplementary family; research/44 (not a repaired old panel)."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil

from build_fresh_group_panel import EXCLUSIONS, atomic_json, excluded_groups, group, sha
from common_block_gordian import make_pair
from rf_knots.embedding_evaluation import verify_evaluation_pair
from rf_knots.embedding_replay_corpus import canonical, invariants, lift, owner

PROTOCOL = Path("research/44-common-block-gordian-supplement.md")
EXTRA = (
    "embedding-fresh-group-test-v1", "embedding-general-gordian-test-v1",
    "embedding-constructive-short-u-test-v1", "embedding-clean-paired-gordian-test-v1",
    "embedding-clean-paired-short-u-test-v1", "embedding-diverse-metric-gordian-test-v1",
    "embedding-diverse-metric-short-u-test-v1",
)


def category(row):
    p = row["interval"]
    return str(p["lower"]) if p["lower"] == p["upper"] else "interval"


def adequate(records):
    classes = Counter(category(r) for r in records)
    return (len(records) >= 32 and len({group(r["left_fingerprint"]) for r in records}) >= 8
            and all(classes[str(c)] >= 4 for c in range(3)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    panels = [Path("artifacts") / name / "panel.json" for name in (*EXCLUSIONS, *EXTRA)]
    code = [Path(__file__).resolve().relative_to(Path.cwd()), PROTOCOL,
            Path("scripts/common_block_gordian.py"), Path("scripts/build_fresh_group_panel.py"),
            *sorted(Path("src/rf_knots").rglob("*.py"))]
    contract = {"seed": 20261021, "attempts": 1024, "family": "common-positive-block",
                "inputs": {str(path): sha(path) for path in panels + code}}
    args.out.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    excluded = excluded_groups([json.loads(path.read_text()) for path in panels])
    previous = json.loads((args.out / "progress.json").read_text()) if args.resume else None
    if previous and previous["contract"] != contract:
        raise ValueError("resume input/source contract changed")
    if previous and previous["status"] == "completed":
        print("already completed")
        return
    if not previous:
        for source in code:
            target = args.out / "source_snapshot" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(args.out / "excluded_groups.json", sorted(excluded))
    records = previous["records"] if previous else []
    index = previous["next_index"] if previous else 0
    counts = Counter(previous["counts"] if previous else {})
    groups, classes, seen = Counter(), Counter(), set()
    for row in records:
        w = verify_evaluation_pair(row)
        groups[group(row["left_fingerprint"]), category(row)] += 1
        classes[category(row)] += 1
        seen.add(canonical(w.start.word, w.start.strands))

    def save(status):
        if any(sha(Path(name)) != digest for name, digest in contract["inputs"].items()):
            raise ValueError("input changed during supplementary generation")
        report = {"status": status, "pid": os.getpid(), "contract": contract,
                  "next_index": index, "records": records, "counts": dict(counts),
                  "classes": dict(classes), "groups": len({g for g, _ in groups})}
        atomic_json(args.out / "progress.json", report)
        print(json.dumps({k: report[k] for k in ("status", "next_index", "counts", "classes", "groups")}), flush=True)

    save("building")
    while index < contract["attempts"]:
        w, prefix, construction = make_pair(contract["seed"], index)
        endpoint = w.steps[prefix - 1].after
        a, b = invariants(w.start.word, w.start.strands), invariants(endpoint.word, endpoint.strands)
        upper = sum(s.action.kind == "CROSSING_CHANGE" for s in w.steps[:prefix])
        row = {"split": "test", "seed": contract["seed"], "index": index,
               "construction": construction, "left_fingerprint": list(a), "right_fingerprint": list(b),
               "witness": w.to_dict(), "pair_prefix_length": prefix,
               "interval": {"metric": "cc", "left": w.start.to_dict(), "right": endpoint.to_dict(),
                            "lower": abs(a[1] - b[1]) // 2, "upper": upper,
                            "witness_semantic_length": prefix}}
        g, c = group(a), category(row)
        key = canonical(w.start.word, w.start.strands)
        reason = ("insufficient_endpoint_u_lower" if min(abs(a[1]), abs(b[1])) // 2 < 3
                  else "other_partition" if owner(a) != "test" or owner(b) == "quarantine"
                  else "previous_group" if g in excluded else "duplicate" if key in seen
                  else "group_cap" if groups[g, c] >= 4 else "class_cap" if classes[c] >= 32
                  else "accepted")
        if reason == "accepted":
            verify_evaluation_pair(row)
            records.append(row); groups[g, c] += 1; classes[c] += 1; seen.add(key)
        counts[reason] += 1
        index += 1
        if index % 16 == 0:
            save("building")
    save("writing_panel")
    large = []
    for row in records:
        w, prefix = lift(verify_evaluation_pair(row), 33, 512)
        expanded = {**row, "evaluation_view": "size_shift", "witness": w.to_dict(),
                    "pair_prefix_length": row["pair_prefix_length"] + prefix,
                    "interval": {**row["interval"], "left": w.start.to_dict(),
                                 "witness_semantic_length": row["pair_prefix_length"] + prefix}}
        verify_evaluation_pair(expanded)
        large.append(expanded)
    atomic_json(args.out / "panel.json", {"schema": "rf-knots-gordian-pair-panel-v1",
        "pair_kind": "general-gordian", "partition": "test", "complete": adequate(records),
        "construction_complete": True, "contract": contract, "validation": records,
        "size_shift": large, "groups": len({g for g, _ in groups}), "classes": dict(classes),
        "caveat": "Supplementary synthetic common-block family; not the failed positive-seed panel. "
                  "Both endpoints have signature lower bound u>=3. Correlated views, groups not knot IDs."})
    save("completed")


if __name__ == "__main__":
    main()
