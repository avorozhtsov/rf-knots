"""Differential tests: the JAX kernels must agree with the Python reference.

Index arithmetic on compacted arrays is the likeliest place for a silent bug, and
a silent bug here produces beautiful, meaningless learning curves. So every move
and every mask bit is compared against a list-based implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rf_knots import braid, reference
from rf_knots.actions import BRAID, COMMUTE, CROSSING_CHANGE, INSERT, PASS, REDUCE, ActionSpec

SPEC = ActionSpec(max_len=12, max_strands=4)


def to_array(word: tuple[int, ...]) -> jnp.ndarray:
    padded = list(word) + [0] * (SPEC.max_len - len(word))
    return jnp.asarray(padded, dtype=jnp.int32)


def from_array(array) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(array) if int(x) != 0)


def random_reachable_states(count: int, seed: int = 0) -> list[tuple[tuple[int, ...], int]]:
    """States reachable from the unknot by uniform random legal type-preserving moves."""
    rng = np.random.default_rng(seed)
    states = []
    word: tuple[int, ...] = ()
    n = 1
    while len(states) < count:
        legal = [
            action
            for action in reference.legal_actions(SPEC, word, n)
            if SPEC.decode(action)[0] != PASS
        ]
        if not legal:
            word, n = (), 1
            continue
        word, n = reference.apply(SPEC, word, n, int(rng.choice(legal)))
        states.append((word, n))
        if rng.random() < 0.05:  # restart occasionally to keep words short
            word, n = (), 1
    return states


STATES = random_reachable_states(120)


def test_word_length_and_triviality() -> None:
    assert int(braid.word_length(to_array(()))) == 0
    assert int(braid.word_length(to_array((1, -1, 2)))) == 3
    assert bool(braid.is_trivial(to_array(()), jnp.int32(1)))
    assert not bool(braid.is_trivial(to_array(()), jnp.int32(2)))
    assert not bool(braid.is_trivial(to_array((1,)), jnp.int32(2)))


@pytest.mark.parametrize("word,n", STATES[:40])
def test_mask_matches_reference(word: tuple[int, ...], n: int) -> None:
    got = np.asarray(
        braid.legal_action_mask(SPEC, to_array(word), jnp.int32(n), False), dtype=bool
    )
    expected = np.array(
        [reference.is_legal(SPEC, word, n, a, False) for a in range(SPEC.num_actions)]
    )
    mismatch = np.flatnonzero(got != expected)
    assert not len(mismatch), (
        f"{reference.format_word(word, n)}: mask disagrees on "
        + ", ".join(SPEC.describe(int(a)) for a in mismatch[:5])
    )


def test_crossing_change_mask_is_gated() -> None:
    word, n = (1, 2, -2), 3
    off = np.asarray(braid.legal_action_mask(SPEC, to_array(word), jnp.int32(n), False))
    on = np.asarray(braid.legal_action_mask(SPEC, to_array(word), jnp.int32(n), True))
    start = SPEC.start_of(CROSSING_CHANGE)
    assert not off[start:].any()
    assert on[start:].sum() == len(word)
    assert (off[:start] == on[:start]).all()


def test_mask_is_never_empty_and_pass_is_a_last_resort() -> None:
    """Pgx requires a non-empty mask; PASS is the fallback and nothing more.

    Left always-legal, PASS is a strictly dominated instant-forfeit: the
    Scrambler ends the game on ply 1 and hands the Simplifier a trivial win. At
    any realistic `max_len` some other move is always available, so PASS should
    never actually appear.
    """
    pass_action = SPEC.start_of(PASS)
    for word, n in STATES:
        mask = np.asarray(braid.legal_action_mask(SPEC, to_array(word), jnp.int32(n), False))
        assert mask.any()
        others = mask.copy()
        others[pass_action] = False
        assert mask[pass_action] == (not others.any())
        assert others.any(), f"no move other than PASS from {reference.format_word(word, n)}"


def test_apply_matches_reference_on_every_legal_action() -> None:
    apply_jit = jax.jit(braid.apply_action, static_argnums=0)
    checked = 0
    for word, n in STATES[:30]:
        for action in reference.legal_actions(SPEC, word, n, allow_crossing=True):
            got_word, got_n = apply_jit(SPEC, to_array(word), jnp.int32(n), jnp.int32(action))
            expected = reference.apply(SPEC, word, n, action)
            assert (from_array(got_word), int(got_n)) == expected, (
                f"{SPEC.describe(action)} on {reference.format_word(word, n)}"
            )
            # the compaction invariant must survive every move
            raw = np.asarray(got_word)
            filled = int((raw != 0).sum())
            assert (raw[:filled] != 0).all() and (raw[filled:] == 0).all(), (
                f"{SPEC.describe(action)} left a hole in the word: {raw}"
            )
            checked += 1
    assert checked > 500


def test_moves_preserve_the_number_of_components() -> None:
    for word, n in STATES:
        assert reference.num_components(word, n) == 1


def _wraps_the_seam(kind: int, position: int, length: int) -> bool:
    """Does this move span the storage seam (i.e. use position 0 from the end)?"""
    if kind in (REDUCE, COMMUTE):
        return position == length - 1
    if kind == BRAID:
        return position >= length - 2
    return False


def test_interior_braid_moves_preserve_the_group_element_exactly() -> None:
    """Away from the seam a tier-1 move is still an identity in B_n."""
    group_kinds = {REDUCE, COMMUTE, BRAID, INSERT}
    checked = 0
    for word, n in STATES[:30]:
        for action in reference.legal_actions(SPEC, word, n):
            kind, position, _, _ = SPEC.decode(action)
            if kind not in group_kinds or _wraps_the_seam(kind, position, len(word)):
                continue
            new_word, new_n = reference.apply(SPEC, word, n, action)
            assert new_n == n
            assert reference.equal_in_braid_group(word, new_word, n), (
                f"{SPEC.describe(action)} changed the element of B_{n}"
            )
            checked += 1
    assert checked > 100


def test_seam_moves_are_rotate_then_interior_move() -> None:
    """The soundness argument for cyclic moves, checked on every seam move.

    Rotation is conjugation -- a Markov move that preserves the closure -- and an
    interior tier-1 move preserves the braid group element outright. So a move
    across the seam must land on the same necklace as rotating it into the
    interior first. That is exactly what the cyclic representation buys, and it is
    why the two explicit rotation actions could be deleted.
    """
    group_kinds = {REDUCE, COMMUTE, BRAID}
    checked = 0
    for word, n in STATES:
        for action in reference.legal_actions(SPEC, word, n):
            kind, position, _, _ = SPEC.decode(action)
            if kind not in group_kinds or not _wraps_the_seam(kind, position, len(word)):
                continue
            new_word, new_n = reference.apply(SPEC, word, n, action)
            assert new_n == n
            long_way = reference.seam_move_via_rotation(SPEC, word, n, kind, position)
            assert new_word in reference.rotations(long_way), (
                f"{SPEC.describe(action)} on {reference.format_word(word, n)}: "
                f"{new_word} is not a rotation of {long_way}"
            )
            # exponent sum is a conjugacy invariant and is preserved by tier 1
            assert reference.writhe(new_word) == reference.writhe(word)
            assert reference.num_components(new_word, new_n) == 1
            checked += 1
    assert checked > 20, f"only {checked} seam moves seen; the test is not exercising them"


def test_a_seam_move_equals_rotate_then_move() -> None:
    """The decomposition that makes the seam moves sound, checked directly."""
    word, n = (1, 2, 2, -1), 3
    length = len(word)
    seam = SPEC.encode(REDUCE, position=length - 1)  # word[3] == -word[0]
    assert reference.is_legal(SPEC, word, n, seam, False)
    cyclic_result, _ = reference.apply(SPEC, word, n, seam)

    rotated = word[1:] + word[:1]  # conjugation: (2, 2, -1, 1)
    linear = SPEC.encode(REDUCE, position=len(rotated) - 2)
    linear_result, _ = reference.apply(SPEC, rotated, n, linear)
    assert cyclic_result in reference.rotations(linear_result)


def test_jax_components_matches_reference() -> None:
    for word, n in STATES[:40]:
        got = int(braid.num_components(to_array(word), jnp.int32(n), SPEC.max_strands))
        assert got == reference.num_components(word, n)


def test_writhe_matches_reference() -> None:
    for word, _ in STATES[:40]:
        assert int(braid.writhe(to_array(word))) == reference.writhe(word)


def test_decode_matches_python_decode() -> None:
    decode = jax.jit(braid.decode, static_argnums=0)
    for action in range(SPEC.num_actions):
        kind, position, letter = decode(SPEC, jnp.int32(action))
        expected_kind, expected_position, generator, sign = SPEC.decode(action)
        assert int(kind) == expected_kind
        assert int(position) == expected_position
        if expected_kind == INSERT:
            assert int(letter) == sign * generator
