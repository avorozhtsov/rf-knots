from __future__ import annotations

import pytest

from rf_knots.render import braid_ascii, braid_svg, word_label


def test_word_label() -> None:
    assert word_label((), 1) == "B1: e"
    assert word_label((1, -2, 1), 3) == "B3: s1 s2^-1 s1"


def test_ascii_has_one_row_per_letter_and_marks_over_under() -> None:
    art = braid_ascii((1, -1), 2).splitlines()
    assert len(art) == 2
    assert "X" in art[0] and "x" not in art[0]
    assert "x" in art[1] and "X" not in art[1]
    assert all(len(row) == 2 * 2 - 1 for row in art)


def test_ascii_places_the_crossing_between_the_right_strands() -> None:
    art = braid_ascii((2,), 4).splitlines()[0]
    # strands at columns 0, 2, 4, 6; sigma_2 exchanges strands 2 and 3
    assert art[0] == "|"
    assert art[3] == "X"
    assert art[6] == "|"


def test_svg_is_well_formed_and_scales_with_the_word() -> None:
    short = braid_svg((1,), 2)
    long = braid_svg((1, 1, 1, 1), 2)
    for svg in (short, long):
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<svg") == 1
    assert len(long) > len(short)


def test_svg_draws_a_gap_for_the_under_strand() -> None:
    """Over/under must be visible: a crossing change is a swap of exactly that."""
    svg = braid_svg((1,), 2, closure=False)
    # three paths for one crossing: two halves of the under-strand, one over
    assert svg.count("<path") == 3


def test_svg_handles_the_empty_braid() -> None:
    svg = braid_svg((), 1)
    assert svg.startswith("<svg")


def test_rejects_letters_outside_the_braid_group() -> None:
    with pytest.raises(ValueError):
        braid_svg((3,), 3)
    with pytest.raises(ValueError):
        braid_ascii((3,), 3)
