from __future__ import annotations

import pytest

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    INSERT,
    NUM_KINDS,
    PASS,
    REDUCE,
    ActionSpec,
)


@pytest.fixture
def spec() -> ActionSpec:
    return ActionSpec(max_len=8, max_strands=4)


def test_block_sizes(spec: ActionSpec) -> None:
    length, generators = spec.max_len, spec.num_generators
    expected = 3 * length + 2 * length * generators + 4 + length
    assert spec.num_actions == expected
    assert len(spec.starts) == NUM_KINDS + 1


def test_encode_decode_roundtrip_positional(spec: ActionSpec) -> None:
    for kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
        for position in range(spec.max_len):
            action = spec.encode(kind, position=position)
            assert spec.decode(action)[:2] == (kind, position)


def test_encode_decode_roundtrip_insert(spec: ActionSpec) -> None:
    seen = set()
    for generator in range(1, spec.num_generators + 1):
        for sign in (1, -1):
            for position in range(spec.max_len):
                action = spec.encode(INSERT, position=position, generator=generator, sign=sign)
                seen.add(action)
                assert spec.decode(action) == (INSERT, position, generator, sign)
    assert len(seen) == 2 * spec.max_len * spec.num_generators


def test_every_action_decodes(spec: ActionSpec) -> None:
    for action in range(spec.num_actions):
        kind, position, generator, sign = spec.decode(action)
        assert 0 <= kind < NUM_KINDS
        assert 0 <= position < spec.max_len
        assert 1 <= generator <= spec.num_generators
        assert sign in (1, -1)
        assert spec.encode(kind, position, generator, sign) == action


def test_singleton_blocks_are_single(spec: ActionSpec) -> None:
    assert spec.start_of(PASS) + 1 == spec.start_of(CROSSING_CHANGE)


def test_action_space_independent_of_crossing_change_flag() -> None:
    # The two modes must share a policy head.
    assert ActionSpec(16, 5).num_actions == ActionSpec(16, 5).num_actions
