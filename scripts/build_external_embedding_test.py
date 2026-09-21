"""Read-only import of pre-existing proof-pool witnesses into an independent test.

Take ALL replayable Artin roots with <=2 CC in the frozen hash-test partition.
Source and rejected rows are retained; no model scores select this cohort.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from rf_knots.embedding_replay_corpus import invariants, lift, owner, verify_record
from rf_knots.evidence import BraidState, SemanticAction, UnknotWitness, WitnessStep


def word(text):
    return tuple(int(g) for g in text.split(",") if g)


def replay_rows(rows):
    rows = sorted(rows, key=lambda r: int(r["step_index"]))
    if [int(r["step_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("nonconsecutive proof steps")
    first = rows[0]
    start = BraidState(word(first["start_word"]), int(first["start_strands"]))
    steps = tuple(
        WitnessStep(
            SemanticAction(
                r["kind"], int(r["position"] or 0), int(r["generator"] or 1), int(r["sign"] or 1)
            ),
            BraidState(word(r["after_word"]), int(r["after_strands"])),
        )
        for r in rows
    )
    states = [start] + [s.after for s in steps]
    witness = UnknotWitness(
        start,
        max(1, max(len(s.word) for s in states)),
        max(2, max(s.strands for s in states)),
        steps,
    )
    witness.verify()
    for row in rows:
        if any(
            row[key] != first[key] for key in ("pool_sha256", "evidence_id", "representation_id")
        ):
            raise ValueError("source provenance changes within proof")
        if (word(row["start_word"]), int(row["start_strands"])) != (start.word, start.strands):
            raise ValueError("source start changes within proof")
        if witness.crossing_changes != int(row["declared_cc"]) or witness.moves != int(
            row["declared_moves"]
        ):
            raise ValueError("source proof count mismatch")
        for weight in (10, 1000):
            if weight * witness.crossing_changes + witness.moves != int(row[f"l{weight}"]):
                raise ValueError("source cost mismatch")
    return witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    grouped = defaultdict(list)
    with args.source.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["schema"] != "unknotdb-rf-best-witness-steps-v0":
                raise ValueError("unsupported source schema")
            grouped[row["representation_id"]].append(row)
    records, rejected = [], []
    for identifier, rows in sorted(grouped.items()):
        if int(rows[0]["declared_cc"]) > 2:
            continue
        try:
            witness = replay_rows(rows)
            fp = invariants(witness.start.word, witness.start.strands)
            if owner(fp) != "test":
                continue
            row = {
                "split": "test",
                "source_representation_id": identifier,
                "source_pool_sha256": rows[0]["pool_sha256"],
                "source_evidence_id": rows[0]["evidence_id"],
                "pair_prefix_length": witness.moves,
                "left_fingerprint": list(fp),
                "right_fingerprint": [1, 0],
                "interval": {
                    "metric": "cc",
                    "left": witness.start.to_dict(),
                    "right": witness.steps[-1].after.to_dict(),
                    "lower": abs(fp[1]) // 2,
                    "upper": witness.crossing_changes,
                    "witness_semantic_length": witness.moves,
                },
                "witness": witness.to_dict(),
            }
            verify_record(row)
            records.append(row)
        except (ValueError, KeyError, AssertionError) as exc:
            rejected.append({"id": identifier, "error": str(exc)})
    large = []
    for row in records:
        w0 = verify_record(row)
        w, _ = lift(w0, max(9, w0.start.strands + 2), max(60, len(w0.start.word) + 16))
        lifted = {
            **row,
            "evaluation_view": "size_shift",
            "pair_prefix_length": w.moves,
            "witness": w.to_dict(),
            "interval": {
                **row["interval"],
                "left": w.start.to_dict(),
                "witness_semantic_length": w.moves,
            },
        }
        verify_record(lifted)
        large.append(lifted)
    panel = {
        "schema": "rf-knots-short-u-panel-v1",
        "partition": "test",
        "complete": True,
        "selection": "all replayable Artin source roots with <=2CC and owner=test; no model scores",
        "source_sha256": source_sha,
        "source": str(args.source),
        "source_root_count": len(grouped),
        "validation": records,
        "size_shift": large,
        "replay_rejects": rejected,
        "fingerprint_groups": len(
            {(r["left_fingerprint"][0], abs(r["left_fingerprint"][1])) for r in records}
        ),
        "caveat": (
            "Historical successfully solved cohort, not a random population; "
            "same-type representations are correlated."
        ),
    }
    shutil.copyfile(args.source, args.out / "source.tsv")
    if hashlib.sha256((args.out / "source.tsv").read_bytes()).hexdigest() != source_sha:
        raise ValueError("source changed during import")
    shutil.copyfile(__file__, args.out / "builder.py")
    (args.out / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in panel.items() if k not in ("validation", "size_shift")}, default=str
        )
    )


if __name__ == "__main__":
    main()
