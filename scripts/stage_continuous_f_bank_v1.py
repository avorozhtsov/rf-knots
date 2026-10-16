"""Create an immutable additive bank from a completed audited MCTS batch."""

import argparse
import json
import shutil
from pathlib import Path

from build_mcts_training_supervision import atomic, digest, sha

OLD = Path("artifacts/additive-root-only-f-bank-v3")
OLD_SHA = "e1129f31cacb538a9b3a8824f8f552dabdd80803d7fd95c5feb0fb1aaa449c33"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or sha(OLD / "manifest.json") != OLD_SHA:
        raise ValueError("bank output exists or retained bank changed")
    report = read(args.audit)
    candidates = args.audit.parent / "root-f-candidates.json"
    if (report["partial"] or report["conditions"] != 32
            or sha(candidates) != report["candidate_rows_sha256"]):
        raise ValueError("full independent batch audit missing")
    retained = read(OLD / "manifest.json")
    seen, entries = set(), []
    args.out.mkdir(parents=True)
    for index, entry in enumerate(retained["roots"]):
        source = OLD / entry["file"]
        if sha(source) != entry["sha256"]:
            raise ValueError("old bank row SHA differs")
        row = read(source)
        seen.add(tuple(row["query_group"]))
        shutil.copyfile(source, args.out / entry["file"])
        entries.append(entry)
    new = 0
    for candidate in read(candidates):
        identity = tuple(candidate["query_group"])
        if identity in seen:
            continue
        seen.add(identity)
        index = len(entries)
        row = {**candidate, "root_index": index}
        row.pop("source_root_index", None)
        for group in row["groups"]:
            group["root_index"] = index
        path = args.out / f"root-{index:04d}.json"
        atomic(path, row)
        entries.append({"file": path.name, "sha256": sha(path)})
        new += 1
    if new == 0:
        raise ValueError("batch has no new distinct F roots")
    contract = {"schema": "rf-knots-continuous-additive-F-bank-v1",
        "old_bank": str(OLD), "old_manifest_sha256": OLD_SHA,
        "old_shard_count": len(retained["roots"]),
        "batch_audit": str(args.audit), "batch_audit_sha256": sha(args.audit),
        "candidate_rows_sha256": sha(candidates), "new_F_roots": new,
        "builder_sha256": sha(Path(__file__))}
    summary = {"retained_F_roots": 402, "new_F_roots": new,
               "total_F_roots": 402 + new,
               "F_groups": sum(len(read(args.out / e["file"])["groups"]) for e in entries)}
    atomic(args.out / "manifest.json", {"status": "completed", "roots": entries,
        "contract": contract, "contract_sha256": digest(contract), "summary": summary})
    print(json.dumps({"manifest_sha256": sha(args.out / "manifest.json"), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
