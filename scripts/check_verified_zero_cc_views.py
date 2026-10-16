"""Protocol123 engineering gate on fixed TRAIN pairs; no neural computation."""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from check_qrun_common_witnesses import check_pins, sha, write_new
from raster_pair_training import digest
from verified_zero_cc_views import make_view, seed_for, verify_view

from rf_knots.embedding_augmentation import AugmentedRepresentation, ZeroCCOperation
from rf_knots.embedding_supervision import EmbeddingRepresentation

GRADIENTS = Path("artifacts/raster-task-gradient-diagnostic-v1")
GRADIENT_REPORT_SHA = "f4911c95002d5db7cc076e07b633b7324bf9f9db06142a77592e391da494fdb2"
SEED, ROUNDS = 20261061, (1, 4, 16)


def source(row, side):
    return EmbeddingRepresentation(
        -1, tuple(row[f"{side}_word"]), row[f"{side}_strands"], None, "train"
    )


def decoded_view(payload):
    return AugmentedRepresentation(
        tuple(payload["word"]),
        payload["strands"],
        tuple(ZeroCCOperation(**op) for op in payload["operations"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rp = GRADIENTS / "report.json"
    if sha(rp) != GRADIENT_REPORT_SHA:
        raise ValueError("different upstream diagnostic")
    upstream = json.loads(rp.read_text())
    ip, sp = GRADIENTS / "inputs.json", GRADIENTS / "selection.json"
    if sha(ip) != upstream["inputs_sha256"] or sha(sp) != upstream["selection_sha256"]:
        raise ValueError("changed upstream source/selection")
    pins = dict(json.loads(ip.read_text())["inputs"])
    for path in (
        rp,
        ip,
        sp,
        Path(__file__).resolve(),
        Path("scripts/verified_zero_cc_views.py"),
        Path("tests/test_verified_zero_cc_views.py"),
        Path("research/123-verified-zero-cc-view-gate.md"),
    ):
        pins[str(path)] = sha(path)
    check_pins(pins)
    rows = json.loads(sp.read_text())["metrics"]
    if len(rows) != 32:
        raise ValueError("require the fixed32 TRAIN rows")
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            inputs=pins,
            seed=SEED,
            rounds=ROUNDS,
            pid=os.getpid(),
            source="fixed TRAIN draw16 from protocol118",
        ),
    )
    variants = []
    for index, row in enumerate(rows):
        for rounds in ROUNDS:
            record = dict(
                parent_index=index,
                rounds=rounds,
                parent_row_sha256=digest(row),
                lower=row["spec"]["distance_lower"],
                upper=row["spec"]["distance_upper"],
                parent_witness_semantic_length=row["witness_semantic_length"],
                seeds={},
            )
            for side in ("left", "right"):
                seed = seed_for(SEED, index, side, rounds)
                record["seeds"][side] = seed
                record[side] = asdict(make_view(source(row, side), seed, rounds))
            variants.append(record)
        if (index + 1) % 8 == 0:
            print(
                json.dumps(dict(parent_pairs=index + 1, generated_pairs=len(variants))), flush=True
            )
    path = args.out / "variants.json"
    write_new(path, variants)
    saved = json.loads(path.read_text())
    if len(saved) != 96 or [(r["parent_index"], r["rounds"]) for r in saved] != [
        (i, n) for i in range(32) for n in ROUNDS
    ]:
        raise ValueError("saved variant coverage differs")
    operations, unique = Counter(), set()
    by_round = {n: dict(lengths=[], strands=[], operations=0) for n in ROUNDS}
    certificates = anchor_certificates = 0
    for record in saved:
        parent = rows[record["parent_index"]]
        if (
            record["parent_row_sha256"] != digest(parent)
            or record["lower"] != parent["spec"]["distance_lower"]
            or record["upper"] != parent["spec"]["distance_upper"]
            or record["parent_witness_semantic_length"] != parent["witness_semantic_length"]
        ):
            raise ValueError("parent identity, interval or base priority changed")
        for side in ("left", "right"):
            base, view = source(parent, side), decoded_view(record[side])
            expected_seed = seed_for(SEED, record["parent_index"], side, record["rounds"])
            if record["seeds"][side] != expected_seed:
                raise ValueError("changed view seed")
            verify_view(base, view)
            if (view.word, view.strands) == (base.word, base.strands):
                raise ValueError("view failed to change representation")
            certificates += 1
            anchor_certificates += base.strands == 1 and not base.word
            operations.update(op.kind for op in view.operations)
            unique.add((view.strands, view.word))
            group = by_round[record["rounds"]]
            group["lengths"].append(len(view.word))
            group["strands"].append(view.strands)
            group["operations"] += len(view.operations)
    summary = {
        n: dict(
            endpoints=len(v["lengths"]),
            word_length_range=[min(v["lengths"]), max(v["lengths"])],
            strand_range=[min(v["strands"]), max(v["strands"])],
            operations=v["operations"],
        )
        for n, v in by_round.items()
    }
    check_pins(pins)
    write_new(
        args.out / "report.json",
        dict(
            status="passed",
            protocol=123,
            inputs_sha256=sha(args.out / "inputs.json"),
            variants_sha256=sha(path),
            pair_variants=len(saved),
            endpoint_certificates=certificates,
            canonical_anchor_certificates=anchor_certificates,
            unique_raw_endpoints=len(unique),
            operations=dict(operations),
            by_round=summary,
            all_saved_certificates_replayed=True,
            inherited_intervals_unchanged=True,
            parent_priority_metadata_unchanged=True,
            neural_forwards=0,
            optimizer_steps=0,
            policy_targets_generated=False,
            scope="Zero-CC macro certificates only; not native MCTS move costs "
            "or proof of improved learning.",
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed",
                endpoint_certificates=certificates,
                report_sha256=sha(args.out / "report.json"),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
