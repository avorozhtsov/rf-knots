import itertools
import random

import pytest

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, INSERT, ActionSpec
from rf_knots.lazy_actions import LazyActionCatalogue, factorized_action_candidates


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("allow", [False, True])
def test_complete_catalogue_matches_reference(cyclic, allow):
    rng = random.Random(23)
    cases = [((), 1), ((), 2), ((1,), 2)]
    for n in range(2, 6):
        largest = n if cyclic and n >= 3 else n - 1
        for length in range(9):
            cases.append(
                (tuple(rng.choice((-1, 1)) * rng.randint(1, largest) for _ in range(length)), n)
            )
    for word, n in cases:
        for growth in (0, 1, 2):
            spec = ActionSpec(max(1, len(word) + growth), n + 1, cyclic)
            cat = LazyActionCatalogue(spec, word, n, allow)
            actual = list(cat.materialize(0, len(cat)))
            assert len({a for a, _, _ in actual}) == len(cat)
            assert sorted(actual) == sorted(reference.successors(spec, word, n, allow))
            assert all(reference.is_legal(spec, word, n, a, allow) for a, _, _ in actual)
            for chunk in (1, 3, 17):
                sliced = list(
                    itertools.chain.from_iterable(
                        cat.materialize(start, chunk) for start in range(0, len(cat), chunk)
                    )
                )
                assert sliced == actual


def test_large_action_ids_do_not_wrap_and_prefix_is_sparse():
    spec = ActionSpec(100_000, 100_000)
    assert spec.num_actions > 2**31
    for g in (1, 99_999):
        for sign in (-1, 1):
            a = spec.encode(INSERT, 99_999, g, sign)
            assert spec.decode(a) == (INSERT, 99_999, g, sign)
    a = spec.encode(CROSSING_CHANGE, 99_999)
    assert spec.decode(a) == (CROSSING_CHANGE, 99_999, 1, 1)
    cat = LazyActionCatalogue(spec, (99_999, 99_998, -99_999), 100_000, True)
    assert len(cat.ordinary) < 20
    assert cat.insertions == 2 * 3 * 99_999
    assert len(list(cat.materialize(0, 8))) == 8
    for i in (0, 1, len(cat) - 1):
        assert reference.is_legal(spec, cat.word, cat.strands, cat.action_at(i), True)
    with pytest.raises(IndexError):
        cat.action_at(len(cat))
    with pytest.raises(ValueError):
        list(cat.materialize(-1, 1))


def test_factorized_proposal_is_small_and_exact_for_large_strand_caps():
    spec = ActionSpec(100_000, 100_000)
    word = (1, -2, 1)
    kinds = [0.0] * 9
    positions = [[-100.0] * len(word) for _ in range(9)]
    kinds[3] = 10.0
    positions[3][1] = 20.0
    generators = [-100.0] * (100_000 - 1)
    generators[99_998] = 30.0
    result = factorized_action_candidates(
        spec,
        word,
        100_000,
        kind_scores=kinds,
        position_scores=positions,
        generator_scores=generators,
        top_positions=2,
        top_generators=3,
    )
    assert len(result) <= 2 * 2 * 3 * 2
    actions = [action for action, _ in result]
    assert len(actions) == len(set(actions))
    assert all(reference.is_legal(spec, word, 100_000, action, False) for action in actions)
    assert any(spec.decode(action)[:3] == (3, 1, 99_999) for action in actions)


def test_factorized_proposal_does_not_turn_unopened_actions_into_negatives():
    spec = ActionSpec(16, 8)
    word = (1, 2, 1)
    positions = [[0.0] * len(word) for _ in range(9)]
    result = factorized_action_candidates(
        spec,
        word,
        8,
        kind_scores=[0.0] * 9,
        position_scores=positions,
        generator_scores=[0.0] * 7,
        top_positions=1,
        top_generators=1,
    )
    assert result
    # The helper only returns proposed legal actions; it exposes no complement
    # set and therefore cannot fabricate a negative label for the rest.
    assert all(reference.is_legal(spec, word, 8, action, False) for action, _ in result)
