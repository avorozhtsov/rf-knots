"""Freeze a small, pre-outcome DEV search panel for nine-arm screening.

This is historical validation-origin material, not TEST150. It is deliberately
short/medium length and therefore cannot establish long-diagram utility.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from evaluate_embedding_cost_search import query_instance
from rf_knots.embedding_replay_corpus import owner, verify_record


CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
TRAIN = Path("artifacts/diverse-cost-mcts-training-v1/manifest.json")
SEED = 20261003
PER_BAND = 8


def group(fingerprint):
    return (fingerprint[0], abs(fingerprint[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("DEV panel already exists; never overwrite")
    corpus = json.loads(CORPUS.read_text())
    train = json.loads(TRAIN.read_text())
    if train["status"] != "completed" or corpus["schema"] != "rf-knots-replay-fingerprint-corpus-v1":
        raise ValueError("source corpus or TRAIN roster is not complete")
    excluded = {tuple(row) for row in train["heldout_query_groups"]}
    selected, seen = [], set()
    for partition in ("validation", "size_shift"):
        ranked = sorted(
            corpus[partition],
            key=lambda row: hashlib.sha256(
                f"{SEED}:{partition}:{row['index']}".encode()
            ).hexdigest(),
        )
        count = 0
        for row in ranked:
            query = group(row["left_fingerprint"])
            if query in seen:
                continue
            if row["split"] != partition or owner(tuple(row["left_fingerprint"])) != "validation":
                raise ValueError("DEV ownership differs")
            witness = verify_record(row)
            base, fits = query_instance(row["index"], witness, witness_caps=True)
            if not fits or query not in excluded:
                raise ValueError("DEV witness or historical exclusion differs")
            selected.append(
                {
                    "partition": partition,
                    "corpus_index": row["index"],
                    "query_group": list(query),
                    "word_length": len(witness.start.word),
                    "reference_cc": witness.crossing_changes,
                    "reference_moves": witness.moves,
                    "witness": witness.to_dict(),
                    "instances": [
                        replace(base, cc_weight=weight, uid=f"tournament-dev-{partition}-{row['index']}-{weight}").to_dict()
                        for weight in (10, 1000)
                    ],
                }
            )
            seen.add(query)
            count += 1
            if count == PER_BAND:
                break
        if count != PER_BAND:
            raise ValueError("not enough distinct DEV query groups")
    payload = {
        "schema": "rf-knots-tournament-dev-screen-v1",
        "role": "pre-outcome historical validation search screen; not long-diagram evidence or TEST",
        "selection": {"seed": SEED, "per_partition": PER_BAND, "weights": [10, 1000]},
        "inputs": {str(CORPUS): sha(CORPUS), str(TRAIN): sha(TRAIN), str(Path(__file__).resolve()): sha(Path(__file__))},
        "rows": selected,
    }
    args.out.mkdir(parents=True)
    atomic(args.out / "panel.json", payload)
    print(json.dumps({"rows": len(selected), "conditions": len(selected) * 2,
                      "word_lengths": [min(r["word_length"] for r in selected), max(r["word_length"] for r in selected)],
                      "panel_sha256": sha(args.out / "panel.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
