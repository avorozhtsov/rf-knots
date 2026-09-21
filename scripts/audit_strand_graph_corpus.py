"""Independent read/replay/split audit with a second determinant implementation."""

import argparse
import hashlib
import json
from pathlib import Path

from rf_knots.embedding_replay_corpus import canonical, path_fingerprints, verify_record
from rf_knots.invariants import determinant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("audit output already exists; refusing to overwrite")
    payload = json.loads(args.corpus.read_text())
    groups, roots, dimensions = {}, {}, {}
    train_path_groups = set()
    for split in ("train", "validation", "size_shift"):
        groups[split], roots[split], dimensions[split] = set(), set(), []
        for row in payload[split]:
            witness = verify_record(row)
            det, sig = row["left_fingerprint"]
            groups[split].add((det, abs(sig)))
            roots[split].add(canonical(witness.start.word, witness.start.strands))
            dimensions[split].append((len(witness.start.word), witness.start.strands))
            if split == "train":
                train_path_groups.update((d, abs(s)) for d, s in path_fingerprints(witness))
    assert not groups["validation"] & train_path_groups
    assert not roots["train"] & roots["validation"]
    train_max_length = max(length for length, _ in dimensions["train"])
    train_max_strands = max(n for _, n in dimensions["train"])
    assert all(length > train_max_length and n > train_max_strands
               for length, n in dimensions["size_shift"])
    checked = []
    # Every held-out fingerprint plus 16 deterministically sampled train roots.
    represented = set()
    sample = payload["train"][:: max(1, len(payload["train"]) // 16)]
    for row in payload["validation"]:
        key = tuple(row["left_fingerprint"])
        if key not in represented:
            represented.add(key)
            sample.append(row)
    for row in sample:
        state = row["interval"]["left"]
        actual = determinant(tuple(state["word"]), state["strands"])
        assert actual == row["left_fingerprint"][0]
        checked.append({"split": row["split"], "index": row["index"], "determinant": actual})
    result = {
        "status": "passed",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "replayed_records": sum(len(payload[k]) for k in groups),
        "unique_roots": {k: len(v) for k, v in roots.items()},
        "train_path_fingerprint_groups": len(train_path_groups),
        "heldout_query_groups": len(groups["validation"]),
        "heldout_train_intersection": [],
        "size_ranges": {
            k: {
                "length": [min(length for length, _ in v), max(length for length, _ in v)],
                "strands": [min(n for _, n in v), max(n for _, n in v)],
            }
            for k, v in dimensions.items()
        },
        "independent_burau_determinant_checks": checked,
        "caveat": "fingerprints partition types conservatively; they do not identify knots. "
        "Reference endpoints and post-crossing validation suffixes may be known.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps({k: v for k, v in result.items() if k != "independent_burau_determinant_checks"})
    )


if __name__ == "__main__":
    main()
