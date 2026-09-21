from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_normalization import fast_normalize_global_view, normalize_global_view


def test_replay_and_mirror_chirality():
    for sign in (1, -1):
        word = tuple(sign * g for g in (1, 1, 1, 2, 3, -3, 3, 4, 5))
        result = normalize_global_view(word, 6)
        state, strands = word, 6
        for action in result.actions:
            assert result.spec.decode(action)[0] != CROSSING_CHANGE
            assert reference.is_legal(result.spec, state, strands, action, False)
            state, strands = reference.apply(result.spec, state, strands, action)
        assert (state, strands) == (result.word, result.strands)
        assert result.word == (sign, sign, sign) and result.strands == 2


def test_global_normalization_is_not_an_unknot_oracle():
    result = normalize_global_view((1, -2, 1, -2), 3)
    assert result.word == (1, -2, 1, -2)
    assert result.strands == 3


def test_seam_reduction_and_idempotence():
    result = normalize_global_view((2, 1, 1, 1, -2), 3)
    assert result.word == (1, 1, 1)
    # Unused strands of a link must not be silently removed.
    assert result.strands == 3
    again = normalize_global_view(result.word, result.strands)
    assert not again.actions


def test_fast_matches_replayed_reference_random_and_exhaustive():
    import itertools
    import random

    rng = random.Random(239)
    cases = [
        (word, 3)
        for length in range(7)
        for word in itertools.product((1, -1, 2, -2), repeat=length)
    ]
    cases += [
        (tuple(rng.choice((-1, 1)) * rng.randrange(1, n) for _ in range(length)), n)
        for n in range(2, 12)
        for length in range(100)
    ]
    for word, n in cases:
        slow = normalize_global_view(word, n)
        assert fast_normalize_global_view(word, n) == (slow.word, slow.strands)


def test_fast_no_strand_sized_allocation_and_deep_stabilization():
    assert fast_normalize_global_view((1, 1, 1), 10**12) == ((1, 1, 1), 10**12)
    assert fast_normalize_global_view((1, 1, 1) + tuple(range(2, 20002)), 20002) == ((1, 1, 1), 2)
    assert fast_normalize_global_view((1, -1) * 20000, 2) == ((), 2)
