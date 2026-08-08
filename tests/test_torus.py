"""The raster is only useful if it is lossless and if packing is a braid identity.

Both are checked against something independent: decoding is checked by round trip,
and packing is checked against the Artin representation, which is faithful and so
decides equality in `B_n` rather than testing a proxy for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_knots.reference import equal_in_braid_group
from rf_knots.torus import (
    EDGE_CHANNELS,
    RASTER_CHANNELS,
    channels,
    pack_layers,
    packed_word,
    raster,
    word_from_raster,
)

WORDS: list[tuple[tuple[int, ...], int]] = [
    ((), 1),
    ((1,), 2),
    ((1, 1, 1), 2),
    ((1, -2, 1, -2), 3),
    ((1, 3, 2, -1, 3, -2), 4),
    ((1, 2, 3, 4, -1, -2, -3, -4, 2, 4), 5),
    ((2, 4, 6, 1, 3, 5, 7, -2, -4, -6), 8),
]


@pytest.mark.parametrize(("word", "strands"), WORDS)
@pytest.mark.parametrize("pad_mode", ["identity", "zero"])
@pytest.mark.parametrize("pack", [False, True])
def test_the_raster_round_trips_back_to_the_word(word, strands, pad_mode, pack):
    planes = raster(word, strands, max_strands=8, rows=24, pack=pack, pad_mode=pad_mode)
    assert planes.shape == (24, 8, RASTER_CHANNELS)
    recovered = word_from_raster(planes, strands)
    if pack:
        # Packing reorders commuting letters, so the recovered word is the packed
        # one -- and that one is equal in the group, which is the next test.
        assert recovered == packed_word(word, strands)
    else:
        assert recovered == word


@pytest.mark.parametrize(("word", "strands"), WORDS)
def test_packing_preserves_the_braid_group_element(word, strands):
    assert equal_in_braid_group(packed_word(word, strands), word, strands)


@pytest.mark.parametrize(("word", "strands"), WORDS)
def test_packing_never_lengthens_and_usually_shortens(word, strands):
    layers = pack_layers(word, strands)
    assert sum(len(layer) for layer in layers) == len([x for x in word if x])
    assert len(layers) <= len([x for x in word if x])


def test_far_commuting_letters_share_a_row_and_adjacent_ones_do_not():
    assert pack_layers((1, 3), 4) == ((1, 3),)
    assert pack_layers((1, 2), 4) == ((1,), (2,))
    assert pack_layers((1, 3, 2), 4) == ((1, 3), (2,))
    # sigma_1 sigma_1 does not commute with itself: order within a strand matters.
    assert pack_layers((1, 1), 3) == ((1,), (1,))


def test_the_input_width_does_not_depend_on_the_strand_count():
    """The claim the whole representation exists to make."""
    narrow = raster((1, -1), 2, max_strands=2, rows=8)
    wide = raster((1, -1), 2, max_strands=16, rows=8)
    assert narrow.shape[2] == wide.shape[2] == channels()
    assert channels(edges=True) == RASTER_CHANNELS + EDGE_CHANNELS
    # The occupied part of the picture is bit-identical; only the canvas grew.
    assert np.array_equal(narrow, wide[:, :2])


def test_a_crossing_is_written_twice_from_both_sides():
    positive = raster((2,), 4, rows=1)[0]
    assert tuple(positive[1, :3]) == (0.0, 1.0, 1.0)  # 011: right, over
    assert tuple(positive[2, :3]) == (1.0, 0.0, 0.0)  # 100: left, under
    negative = raster((-2,), 4, rows=1)[0]
    assert tuple(negative[1, :3]) == (0.0, 0.0, 1.0)  # 001: right, under
    assert tuple(negative[2, :3]) == (1.0, 1.0, 0.0)  # 110: left, over
    assert tuple(positive[3, :3]) == (0.0, 1.0, 0.0)  # 010: uninvolved, straight
    assert tuple(positive[0, :3]) == (0.0, 1.0, 0.0)


def test_identity_padding_is_a_picture_of_the_same_knot():
    """Every padded row is a legal braid picture, so padding adds no new symbols."""
    planes = raster((1, 2), 3, rows=6, pad_mode="identity")
    assert (planes[2:, :3, 1] == 1.0).all()  # straight
    assert (planes[2:, :3, 3] == 1.0).all()  # and present
    assert word_from_raster(planes, 3) == (1, 2)
    blank = raster((1, 2), 3, rows=6, pad_mode="zero")
    assert (blank[2:] == 0.0).all()


def test_edge_channels_mark_both_boundaries():
    planes = raster((1,), 3, max_strands=6, rows=2, edges=True)
    first, last = planes[..., RASTER_CHANNELS], planes[..., RASTER_CHANNELS + 1]
    assert first[0].tolist() == [1, 0, 0, 0, 0, 0]
    assert last[0].tolist() == [0, 0, 1, 0, 0, 0]  # strand 2 is the last active one


@pytest.mark.parametrize(
    ("word", "strands", "message"),
    [
        ((3,), 3, "invalid"),
        ((1,), 1, "invalid"),
    ],
)
def test_generators_outside_the_braid_are_refused(word, strands, message):
    with pytest.raises(ValueError, match=message):
        raster(word, strands, rows=4)


def test_a_word_that_does_not_fit_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="do not fit"):
        raster((1, 1, 1, 1), 2, rows=2)


def test_decoding_rejects_a_half_crossing_with_no_partner():
    planes = raster((1,), 3, rows=1)
    planes[0, 1, :3] = (0.0, 1.0, 0.0)  # erase the other half
    with pytest.raises(ValueError, match="no partner"):
        word_from_raster(planes, 3)


def test_decoding_rejects_halves_that_disagree_on_sign():
    planes = raster((1,), 3, rows=1)
    planes[0, 1, :3] = (1.0, 1.0, 0.0)  # left-over, as if the letter were negative
    with pytest.raises(ValueError, match="disagree"):
        word_from_raster(planes, 3)


def test_packing_a_random_word_is_always_a_braid_identity():
    """The property, on words a hand-written list would not think to include."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        strands = int(rng.integers(2, 7))
        length = int(rng.integers(0, 16))
        word = tuple(
            int(rng.choice([-1, 1])) * int(rng.integers(1, strands))
            for _ in range(length)
        )
        assert equal_in_braid_group(packed_word(word, strands), word, strands)
        planes = raster(word, strands, max_strands=8, rows=24, pack=True)
        assert word_from_raster(planes, strands) == packed_word(word, strands)
