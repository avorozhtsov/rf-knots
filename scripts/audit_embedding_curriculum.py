#!/usr/bin/env python3
"""Build and replay-audit a virtual million-pair embedding curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
from pathlib import Path

from rf_knots.embedding_augmentation import (
    MillionPairCurriculum,
    augment_markov_zero_cc,
    augment_zero_cc,
    procedural_cc_pair,
    random_absent_knot_braid,
    reverse_markov_zero_cc,
    reverse_zero_cc,
)
from rf_knots.embedding_supervision import (
    DistanceInterval,
    UnknotDBEmbeddingSidecar,
)

SCHEMA = "rf-knots-embedding-curriculum-v0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def ukb0(word: tuple[int, ...], strands: int) -> bytes:
    return (
        b"UKB0\x00\x00"
        + struct.pack("<H", strands)
        + struct.pack("<I", len(word))
        + (struct.pack(f"<{len(word)}h", *word) if word else b"")
    )


def load_geodesic(path: Path) -> list[DistanceInterval]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = [
        DistanceInterval("cc", int(left), int(right), distance, distance, length, "train")
        for left, right, distance, length in connection.execute(
            "SELECT left_representation,right_representation,distance,"
            "witness_semantic_length FROM geodesic_pairs ORDER BY "
            "left_representation,right_representation"
        )
    ]
    connection.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--geodesic", type=Path, required=True)
    parser.add_argument("--hard-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--audit-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    hard = sqlite3.connect(f"file:{args.hard_cases}?mode=ro", uri=True)
    protected = {
        int(row[0])
        for row in hard.execute(
            "SELECT DISTINCT representation_id FROM hard_case_representations"
        )
    }
    hard.close()
    raw = sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True)
    representation_ids = tuple(
        int(row[0])
        for row in raw.execute(
            "SELECT representation_id FROM representations WHERE split='train' "
            "ORDER BY representation_id"
        )
        if int(row[0]) not in protected
    )
    existing_representation_sha256 = frozenset(
        bytes(row[0]) for row in raw.execute("SELECT sha256 FROM representations")
    )
    raw.close()

    with UnknotDBEmbeddingSidecar(args.pairs) as sidecar:
        natural = tuple(
            pair
            for pair in sidecar.pairs("train")
            if pair.left_representation not in protected
            and pair.right_representation not in protected
        )
        geodesic = tuple(load_geodesic(args.geodesic))
        edge_interior_ids = set(
            sidecar.representation_ids_by_domain(
                split="train", edge_interior=True, stopping_point=False
            )
        )
        edge_interior = tuple(
            pair
            for pair in natural
            if pair.metric == "cc"
            and (
                pair.left_representation in edge_interior_ids
                or pair.right_representation in edge_interior_ids
            )
        )
        mirror_orbit = tuple(
            pair
            for pair in natural
            if pair.provenance == "mirror_orbit_task_symmetry"
        )
        exact_cc1 = tuple(
            pair
            for pair in natural + geodesic
            if pair.metric == "cc" and pair.exact and pair.lower == 1
        )
        exact_cc2 = tuple(
            pair
            for pair in geodesic
            if pair.metric == "cc" and pair.exact and pair.lower == 2
        )
        curriculum = MillionPairCurriculum(
            representation_ids=representation_ids,
            edge_interior=edge_interior,
            mirror_orbit=mirror_orbit,
            exact_cc1=exact_cc1,
            exact_cc2=exact_cc2,
            natural=natural,
            existing_representation_sha256=existing_representation_sha256,
            size=args.size,
            seed=args.seed,
        )
        counts = {}
        descriptor_digest = hashlib.sha256()
        for index in range(len(curriculum)):
            spec = curriculum.spec(index)
            counts[spec.stratum] = counts.get(spec.stratum, 0) + 1
            descriptor_digest.update(
                (
                    f"{spec.index}:{spec.stratum}:{spec.left_representation}:"
                    f"{spec.right_representation}:{spec.metric}:{spec.distance_lower}:"
                    f"{spec.distance_upper}:{spec.left_seed}:{spec.right_seed}\n"
                ).encode()
            )

        audited = 0
        max_augmented_length = 0
        stride = max(len(curriculum) // max(args.audit_samples, 1), 1)
        procedural_absence_checks = 0
        procedural_present_roots = 0
        raw_pairs = sqlite3.connect(f"file:{args.pairs}?mode=ro", uri=True)
        for index in range(0, len(curriculum), stride):
            if audited >= args.audit_samples:
                break
            spec = curriculum.spec(index)
            if spec.stratum == "natural_replay":
                continue
            if spec.stratum == "hard_procedural_same_knot_cc0":
                simple = index % 2 == 0
                root = random_absent_knot_braid(
                    spec.left_seed,
                    simple=simple,
                    existing_sha256=existing_representation_sha256,
                )
                left_variant = augment_markov_zero_cc(
                    root, spec.left_seed, simple=simple
                )
                right_variant = augment_markov_zero_cc(
                    root, spec.right_seed, simple=simple
                )
                if reverse_markov_zero_cc(left_variant) != (root.word, root.strands):
                    raise ValueError(f"left Markov augmentation {index} failed replay")
                if reverse_markov_zero_cc(right_variant) != (root.word, root.strands):
                    raise ValueError(f"right Markov augmentation {index} failed replay")
                max_augmented_length = max(
                    max_augmented_length,
                    len(left_variant.word),
                    len(right_variant.word),
                )
                root_sha = hashlib.sha256(ukb0(root.word, root.strands)).digest()
                procedural_present_roots += int(
                    raw_pairs.execute(
                        "SELECT 1 FROM representations WHERE sha256=?", (root_sha,)
                    ).fetchone()
                    is not None
                )
                procedural_absence_checks += 1
                audited += 1
                continue
            if spec.stratum == "hard_procedural_cc_upper":
                simple = index % 2 == 0
                root = random_absent_knot_braid(
                    spec.left_seed,
                    simple=simple,
                    existing_sha256=existing_representation_sha256,
                )
                root, target, cc_count = procedural_cc_pair(
                    spec.left_seed, simple=simple, root=root
                )
                changed_word, changed_strands = reverse_markov_zero_cc(target)
                differences = sum(
                    left == -right and abs(left) == abs(right)
                    for left, right in zip(root.word, changed_word, strict=True)
                )
                if changed_strands != root.strands or differences != cc_count:
                    raise ValueError(f"procedural CC witness {index} failed replay")
                max_augmented_length = max(max_augmented_length, len(target.word))
                root_sha = hashlib.sha256(ukb0(root.word, root.strands)).digest()
                procedural_present_roots += int(
                    raw_pairs.execute(
                        "SELECT 1 FROM representations WHERE sha256=?", (root_sha,)
                    ).fetchone()
                    is not None
                )
                procedural_absence_checks += 1
                audited += 1
                continue
            left = sidecar.representation(spec.left_representation)
            augmented_left = augment_zero_cc(left, spec.left_seed)
            if reverse_zero_cc(augmented_left) != left.word:
                raise ValueError(f"left augmentation {index} failed inverse replay")
            max_augmented_length = max(max_augmented_length, len(augmented_left.word))
            if spec.stratum != "hard_cc0_detour":
                right = sidecar.representation(spec.right_representation)
                augmented_right = augment_zero_cc(right, spec.right_seed)
                if reverse_zero_cc(augmented_right) != right.word:
                    raise ValueError(f"right augmentation {index} failed inverse replay")
                max_augmented_length = max(
                    max_augmented_length, len(augmented_right.word)
                )
            audited += 1
        raw_pairs.close()

    manifest = {
        "schema": SCHEMA,
        "status": "validated_virtual_corpus_not_trained",
        "pairs_sidecar": str(args.pairs.resolve()),
        "pairs_sidecar_sha256": sha256(args.pairs),
        "geodesic_sidecar": str(args.geodesic.resolve()),
        "geodesic_sidecar_sha256": sha256(args.geodesic),
        "hard_cases_sidecar": str(args.hard_cases.resolve()),
        "hard_cases_sidecar_sha256": sha256(args.hard_cases),
        "seed": args.seed,
        "virtual_pairs": len(curriculum),
        "stream_contract": {
            "finite_reference_cycle": len(curriculum),
            "training_iterator": "unbounded deterministic IterableDataset",
            "generation_key": "seed,epoch,stream_generation,base_index",
            "stopping_condition": "trainer max_steps/examples_seen/wall_time",
        },
        "strata": counts,
        "hard_fraction": sum(
            count for name, count in counts.items() if name.startswith("hard_")
        )
        / len(curriculum),
        "source_pools": {
            "representations": len(representation_ids),
            "natural_pairs": len(natural),
            "edge_interior_pairs": len(edge_interior),
            "mirror_orbit_pairs": len(mirror_orbit),
            "exact_cc1_pairs": len(exact_cc1),
            "exact_cc2_pairs": len(exact_cc2),
        },
        "augmentation_contract": {
            "cyclic_origin_shift": "exact CC=0 and RM=0",
            "inverse_pair_insertions": [6, 20],
            "pair_distance": "preserved under independent zero-CC endpoint transforms",
        },
        "descriptor_stream_sha256": descriptor_digest.hexdigest(),
        "inverse_replay_audit_samples": audited,
        "inverse_replay_failures": 0,
        "procedural_absence_checks": procedural_absence_checks,
        "procedural_roots_found_in_sidecar": procedural_present_roots,
        "max_audited_word_length": max_augmented_length,
        "planned_embedding_dimensions": [32, 64, 128, 256],
    }
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
