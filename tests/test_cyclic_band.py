from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rf_knots import braid, reference
from rf_knots.actions import (
    BRAID,
    COMMUTE,
    INSERT,
    STABILIZE_POS,
    ActionSpec,
)
from rf_knots.config import BraidConfig
from rf_knots.env import BraidUnknot

SPEC = ActionSpec(max_len=32, max_strands=6, cyclic_band_generators=True)


def _array(word: tuple[int, ...]) -> jnp.ndarray:
    return jnp.asarray(word + (0,) * (SPEC.max_len - len(word)), dtype=jnp.int32)


def test_bstar_adds_one_generator_and_compiles_seam_to_ordinary_braid() -> None:
    assert SPEC.num_generators == SPEC.max_strands
    compiled = reference.compile_cyclic_bands((4, -4), 4)
    assert compiled == (3, 2, 1, -2, -3, 3, 2, -1, -2, -3)
    assert reference.equal_in_braid_group(compiled, (), 4)


def test_cyclic_local_relations_are_exact_after_artin_compilation() -> None:
    for strands in range(3, 7):
        seam = strands
        for neighbour in (1, strands - 1):
            left = reference.compile_cyclic_bands((seam, neighbour, seam), strands)
            right = reference.compile_cyclic_bands((neighbour, seam, neighbour), strands)
            assert reference.equal_in_braid_group(left, right, strands)
        for remote in range(2, strands - 1):
            left = reference.compile_cyclic_bands((seam, remote), strands)
            right = reference.compile_cyclic_bands((remote, seam), strands)
            assert reference.equal_in_braid_group(left, right, strands)


def test_jax_and_reference_agree_on_seam_insert_reduce_and_braid() -> None:
    n = 4
    cases = [
        ((), SPEC.encode(INSERT, position=0, generator=n, sign=1)),
        ((n, 1, n), SPEC.encode(BRAID, position=0)),
        ((n, 2), SPEC.encode(COMMUTE, position=0)),
    ]
    for word, action in cases:
        assert reference.is_legal(SPEC, word, n, action, allow_crossing=True)
        expected = reference.apply(SPEC, word, n, action)
        got_word, got_n = braid.apply_action(SPEC, _array(word), jnp.int32(n), action)
        got = tuple(int(value) for value in np.asarray(got_word) if int(value))
        assert (got, int(got_n)) == expected


def test_strand_growth_waits_until_seam_letters_are_removed() -> None:
    state_word = (4, 1, -4)
    stabilize = SPEC.encode(STABILIZE_POS)
    assert not reference.is_legal(SPEC, state_word, 4, stabilize, allow_crossing=True)
    mask = braid.legal_action_mask(SPEC, _array(state_word), jnp.int32(4), True)
    assert not bool(mask[stabilize])


def test_two_strand_seam_duplicate_is_not_an_extra_action() -> None:
    seam_insert = SPEC.encode(INSERT, position=0, generator=2, sign=1)
    ordinary_insert = SPEC.encode(INSERT, position=0, generator=1, sign=1)
    word = _array(())
    mask = braid.legal_action_mask(SPEC, word, jnp.int32(2), False)
    assert bool(mask[ordinary_insert])
    assert not bool(mask[seam_insert])


def test_permutation_and_components_equal_compiled_artin_word() -> None:
    word = (4, 1, -2, 4, 3)
    compiled = reference.compile_cyclic_bands(word, 4)
    assert reference.permutation(word, 4, True) == reference.permutation(compiled, 4)
    assert reference.num_components(word, 4, True) == reference.num_components(compiled, 4)


def test_environment_exposes_seam_planes_and_actions() -> None:
    config = BraidConfig(
        max_len=32,
        max_strands=5,
        scramble_budget=2,
        simplify_budget=8,
        allow_crossing_change=True,
        cyclic_band_generators=True,
    )
    env = BraidUnknot(config)
    state = env.init_from_word([4, 1, -4], 4)
    assert env.spec.num_generators == 5
    assert state.observation.shape == (32, 2 * 5 + 10)
    assert bool(state.observation[0, 3])  # + seam generator plane
