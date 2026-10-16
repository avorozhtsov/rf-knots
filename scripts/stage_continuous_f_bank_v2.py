"""Immutable additive snapshots retaining every previously audited F root."""

import argparse
import json
import shutil
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--old-audit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable output already exists")
    old_sha = sha(args.old_bank / "manifest.json")
    previous_audit = read(args.old_audit)
    if previous_audit["status"] != "passed" or previous_audit["manifest_sha256"] != old_sha:
        raise ValueError("previous independently audited bank identity differs")
    report = read(args.audit)
    candidates = args.audit.parent / "root-f-candidates.json"
    if report["partial"] or report["conditions"] != 32 or sha(candidates) != report["candidate_rows_sha256"]:
        raise ValueError("full independent MCTS batch audit missing")
    old = read(args.old_bank / "manifest.json")
    seen, entries, old_f, groups = set(), [], 0, 0
    args.out.mkdir(parents=True)
    for index, entry in enumerate(old["roots"]):
        source = args.old_bank / entry["file"]
        if sha(source) != entry["sha256"] or entry["file"] != f"root-{index:04d}.json":
            raise ValueError("old row SHA/order differs")
        row = read(source)
        seen.add(tuple(row["query_group"]))
        old_f += bool(row["groups"]); groups += len(row["groups"])
        shutil.copyfile(source, args.out / entry["file"])
        entries.append(entry)
    if old_f != previous_audit["distinct_F_roots"]:
        raise ValueError("previous F root accounting differs")
    added = 0
    for candidate in read(candidates):
        key = tuple(candidate["query_group"])
        if key in seen:
            continue
        seen.add(key)
        index = len(entries)
        row = {**candidate, "root_index": index}
        row.pop("source_root_index", None)
        for group in row["groups"]:
            group["root_index"] = index
        target = args.out / f"root-{index:04d}.json"
        atomic(target, row)
        entries.append({"file": target.name, "sha256": sha(target)})
        added += 1; groups += len(row["groups"])
    if not added:
        raise ValueError("batch contains no unseen F root group")
    contract = {"schema": "rf-knots-continuous-additive-F-bank-v2",
               "old_bank": str(args.old_bank), "old_manifest_sha256": old_sha,
               "old_audit": str(args.old_audit), "old_audit_sha256": sha(args.old_audit),
               "old_shard_count": len(old["roots"]), "new_F_roots": added,
               "batch_audit": str(args.audit), "batch_audit_sha256": sha(args.audit),
               "candidate_rows_sha256": sha(candidates), "builder_sha256": sha(Path(__file__))}
    summary = {"retained_F_roots": old_f, "new_F_roots": added,
               "total_F_roots": old_f + added, "F_groups": groups}
    atomic(args.out / "manifest.json", {"status": "completed", "roots": entries,
        "contract": contract, "contract_sha256": digest(contract), "summary": summary})
    print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
