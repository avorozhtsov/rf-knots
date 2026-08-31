"""Construct the pinned virtual embedding curriculum from UnknotDB sidecars."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rf_knots.embedding_augmentation import MillionPairCurriculum
from rf_knots.embedding_gordian import load_gordian_pairs, protected_knot_ids
from rf_knots.embedding_supervision import (
    DistanceInterval,
    UnknotDBEmbeddingSidecar,
)
from rf_knots.embedding_validation import load_training_quarantine


def _geodesic_pairs(path: Path) -> tuple[DistanceInterval, ...]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = tuple(
        DistanceInterval(
            "cc", int(left), int(right), distance, distance, length, "train", "geodesic"
        )
        for left, right, distance, length in connection.execute(
            "SELECT left_representation,right_representation,distance,"
            "witness_semantic_length FROM geodesic_pairs ORDER BY "
            "left_representation,right_representation"
        )
    )
    connection.close()
    return rows


def load_million_pair_curriculum(
    pairs_path: str | Path,
    geodesic_path: str | Path,
    hard_cases_path: str | Path,
    gordian_path: str | Path | None = None,
    validation_protocol_path: str | Path | None = None,
    *,
    size: int = 1_000_000,
    seed: int = 20260829,
) -> tuple[MillionPairCurriculum, dict[str, int]]:
    """Load pools without admitting protected representations into training."""
    pairs_path = Path(pairs_path)
    geodesic_path = Path(geodesic_path)
    hard_cases_path = Path(hard_cases_path)
    hard = sqlite3.connect(f"file:{hard_cases_path}?mode=ro", uri=True)
    protected = {
        int(row[0])
        for row in hard.execute(
            "SELECT DISTINCT representation_id FROM hard_case_representations"
        )
    }
    hard.close()
    protected_knots = protected_knot_ids(gordian_path) if gordian_path else frozenset()
    quarantine = (
        load_training_quarantine(validation_protocol_path)
        if validation_protocol_path
        else None
    )
    excluded_representations = protected | (
        set(quarantine.representation_ids) if quarantine else set()
    )
    excluded_knots = protected_knots | (
        quarantine.knot_ids if quarantine else frozenset()
    )
    raw = sqlite3.connect(f"file:{pairs_path}?mode=ro", uri=True)
    representation_rows = {
        int(row[0]): (row[1], str(row[2]))
        for row in raw.execute(
            "SELECT representation_id,knot_id,split FROM representations"
        )
    }
    representation_knots = {
        representation_id: row[0]
        for representation_id, row in representation_rows.items()
    }
    representation_ids = tuple(
        representation_id
        for representation_id, (knot_id, split) in sorted(representation_rows.items())
        if split == "train"
        and representation_id not in excluded_representations
        and knot_id not in excluded_knots
    )
    existing_representation_sha256 = frozenset(
        bytes(row[0]) for row in raw.execute("SELECT sha256 FROM representations")
    )
    raw.close()

    with UnknotDBEmbeddingSidecar(pairs_path) as sidecar:
        natural = tuple(
            pair
            for pair in sidecar.pairs("train")
            if pair.left_representation not in excluded_representations
            and pair.right_representation not in excluded_representations
            and sidecar.representation(pair.left_representation).knot_id
            not in excluded_knots
            and sidecar.representation(pair.right_representation).knot_id
            not in excluded_knots
        )
        edge_interior_ids = set(
            sidecar.representation_ids_by_domain(
                split="train", edge_interior=True, stopping_point=False
            )
        )
    geodesic = tuple(
        pair
        for pair in _geodesic_pairs(geodesic_path)
        if pair.left_representation not in excluded_representations
        and pair.right_representation not in excluded_representations
        and representation_knots.get(pair.left_representation) not in excluded_knots
        and representation_knots.get(pair.right_representation) not in excluded_knots
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
    gordian = (
        tuple(
            pair
            for pair in load_gordian_pairs(gordian_path, "train")
            if pair.left.canonical_knot_id not in excluded_knots
            and pair.right.canonical_knot_id not in excluded_knots
            and not excluded_representations.intersection(
                pair.left.representation_ids + pair.right.representation_ids
            )
        )
        if gordian_path
        else ()
    )
    curriculum = MillionPairCurriculum(
        representation_ids=representation_ids,
        edge_interior=edge_interior,
        mirror_orbit=mirror_orbit,
        exact_cc1=exact_cc1,
        exact_cc2=exact_cc2,
        gordian=gordian,
        natural=natural,
        existing_representation_sha256=existing_representation_sha256,
        size=size,
        seed=seed,
    )
    return curriculum, {
        "representations": len(representation_ids),
        "natural_pairs": len(natural),
        "edge_interior_pairs": len(edge_interior),
        "mirror_orbit_pairs": len(mirror_orbit),
        "exact_cc1_pairs": len(exact_cc1),
        "exact_cc2_pairs": len(exact_cc2),
        "gordian_pairs": len(gordian),
        "gordian_protected_knot_ids": len(protected_knots),
        "protected_representations": len(protected),
        "quarantine_knot_ids": len(quarantine.knot_ids) if quarantine else 0,
        "quarantine_representations": (
            len(quarantine.representation_ids) if quarantine else 0
        ),
    }
