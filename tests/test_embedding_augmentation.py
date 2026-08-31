from rf_knots.embedding_augmentation import (
    MillionPairCurriculum,
    augment_markov_zero_cc,
    augment_zero_cc,
    collate_embedding_pairs,
    random_knot_braid,
    reverse_markov_zero_cc,
    reverse_zero_cc,
    scramble_zero_cc,
)
from rf_knots.embedding_supervision import (
    DistanceInterval,
    EmbeddingRepresentation,
)


def _pair(distance: int) -> DistanceInterval:
    return DistanceInterval("cc", 1, 2, distance, distance, 7, "train")


def test_zero_cc_augmentation_has_exact_replayable_inverse() -> None:
    source = EmbeddingRepresentation(7, (1, -2, 1), 3, None, "train")
    augmented = augment_zero_cc(source, 42, minimum_insertions=6, maximum_insertions=6)
    assert augmented.semantic_length == 6
    assert len(augmented.word) == len(source.word) + 12
    assert reverse_zero_cc(augmented) == source.word


def test_virtual_corpus_is_exactly_eighty_percent_hard() -> None:
    curriculum = MillionPairCurriculum(
        representation_ids=(1, 2, 3),
        edge_interior=(_pair(0),),
        mirror_orbit=(_pair(0),),
        exact_cc1=(_pair(1),),
        exact_cc2=(_pair(2),),
        natural=(_pair(0),),
        existing_representation_sha256=frozenset(),
        size=100,
    )
    strata = [curriculum.spec(index).stratum for index in range(len(curriculum))]
    assert sum(value.startswith("hard_") for value in strata) == 80
    assert strata.count("hard_edge_interior") == 15
    assert strata.count("hard_cc1_detour") == 7
    assert strata.count("hard_cc1_long_detour") == 8
    assert strata.count("hard_cc2_detour") == 5
    assert strata.count("hard_cc2_long_detour") == 5
    assert strata.count("hard_mirror_orbit") == 10
    assert strata.count("hard_zero_cc_orbit") == 10
    assert strata.count("hard_procedural_same_knot_cc0") == 10
    assert strata.count("hard_procedural_cc_upper") == 10
    assert strata.count("natural_replay") == 20


def test_virtual_corpus_is_deterministic() -> None:
    curriculum = MillionPairCurriculum(
        representation_ids=(1,),
        edge_interior=(_pair(0),),
        mirror_orbit=(_pair(0),),
        exact_cc1=(_pair(1),),
        exact_cc2=(_pair(2),),
        natural=(_pair(0),),
        existing_representation_sha256=frozenset(),
    )
    assert curriculum.spec(12345) == curriculum.spec(12345)


def test_collate_embedding_pairs_pads_variable_words(tmp_path) -> None:
    # Exercise the tensor contract without coupling this unit test to SQLite.
    from rf_knots.embedding_augmentation import (
        MaterializedSyntheticPair,
        SyntheticPairSpec,
    )

    spec = SyntheticPairSpec(0, "hard_zero_cc_orbit", "cc", 1, 1, 0, 0, 0, 4, 5, 1)
    batch = collate_embedding_pairs(
        [MaterializedSyntheticPair(spec, "cc", (1,), (1, -1, 2), 3, 3, 12)]
    )
    assert batch["left_word"].shape == (1, 1)
    assert batch["right_word"].shape == (1, 3)
    assert batch["witness_semantic_length"].tolist() == [12]


def test_procedural_markov_variant_has_replayable_inverse() -> None:
    root = random_knot_braid(17, simple=False)
    variant = augment_markov_zero_cc(root, 23, simple=False)
    assert reverse_markov_zero_cc(variant) == (root.word, root.strands)
    assert variant.word != root.word


def test_catalogue_scramble_depth_zero_to_six_replays() -> None:
    root = EmbeddingRepresentation(7, (1, -2, 1), 3, "knot:3_1", "train")
    for depth in range(7):
        variant = scramble_zero_cc(root, 100 + depth, depth, mirror=True)
        assert len(variant.operations) == depth
        assert reverse_markov_zero_cc(variant) == ((-1, 2, -1), 3)
