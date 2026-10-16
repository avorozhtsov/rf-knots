"""Fail-closed, exact-manifest retention of completed R2 v4 checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path

RUN = Path("artifacts/additive-root-only-r2-v4")
MANIFEST = RUN / "manifest.json"
JOURNAL = Path("artifacts/additive-root-only-r2-v4-retention-v1.json")
FINAL = "step-230662.pt"
FINAL_SHA = "89af09393db15060f063f1d636c9169bd15740db7cc27b3d5bbe855fa6bf8ca4"
PROTOCOL = Path("research/371-completed-r2-v4-retention-v1.md")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def update(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if JOURNAL.exists() or JOURNAL.with_suffix(JOURNAL.suffix + ".tmp").exists():
        raise ValueError("retention journal already exists; do not repeat")
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["checkpoints"]
    if (manifest["status"] != "checkpointed"
            or manifest["optimizer_steps"] != 230662
            or manifest["new_active_seconds"] < 21600
            or len(entries) != 911
            or entries[-1] != {"file": FINAL, "sha256": FINAL_SHA, "step": 230662}):
        raise ValueError("completed v4 identity differs")
    retained, deleted = [], []
    for index, entry in enumerate(entries):
        name = entry["file"]
        if name != f"step-{entry['step']}.pt" or Path(name).name != name:
            raise ValueError("checkpoint name/path differs")
        path = RUN / name
        info = path.lstat()
        if not path.is_file() or path.is_symlink() or info.st_nlink != 1:
            raise ValueError(f"not a unique regular file: {path}")
        if sha(path) != entry["sha256"]:
            raise ValueError(f"checkpoint SHA differs: {path}")
        row = {"file": str(path), "step": entry["step"],
               "sha256": entry["sha256"], "bytes": info.st_size}
        (retained if index == 0 or index == len(entries) - 1 or index % 29 == 0
         else deleted).append(row)
    if FINAL not in {Path(row["file"]).name for row in retained}:
        raise ValueError("downstream parent not retained")
    report = {"schema": "rf-knots-completed-r2-v4-retention-v1",
              "status": "planned", "manifest_sha256": sha(MANIFEST),
              "protocol_sha256": sha(PROTOCOL),
              "script_sha256": sha(Path(__file__)),
              "retained": retained, "deleted": deleted,
              "deleted_count": len(deleted),
              "deleted_bytes": sum(row["bytes"] for row in deleted)}
    if not args.execute:
        print(json.dumps({"status": "dry_run", "retained": len(retained),
                          "delete": len(deleted),
                          "bytes": report["deleted_bytes"]}, sort_keys=True))
        return
    write_new(JOURNAL, report)
    for row in deleted:
        (RUN / Path(row["file"]).name).unlink()
    for row in retained:
        if sha(Path(row["file"])) != row["sha256"]:
            raise ValueError(f"retained SHA differs: {row['file']}")
    report["status"] = "completed"
    update(JOURNAL, report)
    print(json.dumps({"status": report["status"],
                      "retained": len(retained), "deleted": len(deleted),
                      "bytes": report["deleted_bytes"],
                      "journal_sha256": sha(JOURNAL)}, sort_keys=True))


if __name__ == "__main__":
    main()
