"""The programmable generator replaces the learned proposer.

Eight seeds showed the trained Scrambler is indistinguishable from a random one,
so difficulty now comes from an explicit grade -- crossing number of a source
knot, plus scramble depth -- rather than from an adversary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rf_knots.config import BraidConfig
from rf_knots.generator import (
    GradedGenerator,
    torus_braid,
    torus_crossing_number,
    torus_sources,
)
from rf_knots.reference import num_components


def test_torus_braid_formula() -> None:
    assert torus_braid(2, 3) == ((1, 1, 1), 2)  # trefoil
    assert torus_braid(2, 5) == ((1,) * 5, 2)  # 5_1
    assert torus_braid(3, 4) == ((1, 2) * 4, 3)  # 8_19


def test_torus_closure_is_a_knot_exactly_when_coprime() -> None:
    for p in range(2, 6):
        for q in range(2, 8):
            word, strands = torus_braid(p, q)
            components = num_components(word, strands)
            assert components == math.gcd(p, q), f"T({p},{q})"


def test_crossing_numbers_are_the_known_ones() -> None:
    assert torus_crossing_number(2, 3) == 3  # trefoil
    assert torus_crossing_number(2, 5) == 5  # 5_1
    assert torus_crossing_number(2, 7) == 7  # 7_1
    assert torus_crossing_number(3, 4) == 8  # 8_19
    assert torus_crossing_number(3, 5) == 10  # 10_124


def test_sources_are_knots_ordered_by_crossing_number() -> None:
    sources = torus_sources(max_strands=5, max_crossings=8)
    assert sources[0].name == "unknot"
    assert [s.crossing_number for s in sources] == sorted(
        s.crossing_number for s in sources
    )
    for source in sources:
        assert num_components(source.word, source.strands) == 1, source.name
        assert source.canonical_length == len(source.word)


def test_grade_orders_the_whole_space() -> None:
    config = BraidConfig(max_len=32, max_strands=5, scramble_budget=6, simplify_budget=24)
    generator = GradedGenerator(config, max_crossings=5, crossing_weight=5)
    levels = generator.levels(max_scramble=6)
    scores = [5 * source.crossing_number + moves for source, moves in levels]
    assert scores == sorted(scores)
    # the easiest level is the unknot with no scrambling; the hardest is the
    # largest crossing number fully scrambled
    assert levels[0][0].name == "unknot" and levels[0][1] == 0
    assert levels[-1][0].crossing_number == max(s.crossing_number for s in generator.sources)


def test_scrambling_preserves_the_knot() -> None:
    """The property the whole design rests on, now for knots and not just the unknot."""
    config = BraidConfig(max_len=48, max_strands=5, scramble_budget=6, simplify_budget=24)
    generator = GradedGenerator(config, max_crossings=8)
    rng = np.random.default_rng(0)
    for source in generator.sources:
        for moves in (0, 3, 8):
            instance = generator.generate(source, moves, rng)
            assert num_components(instance.word, instance.strands) == 1
            assert instance.crossing_number == source.crossing_number
            assert instance.canonical_length == len(source.word)
            assert len(instance.word) <= config.max_len


def test_complexity_is_a_single_orderable_number() -> None:
    config = BraidConfig(max_len=48, max_strands=5, scramble_budget=6, simplify_budget=24)
    generator = GradedGenerator(config, max_crossings=5)
    rng = np.random.default_rng(0)
    trefoil = next(s for s in generator.sources if s.crossing_number == 3)
    unknot = generator.sources[0]
    easy = generator.generate(unknot, 4, rng)
    hard = generator.generate(trefoil, 1, rng)
    assert easy.complexity(5) < hard.complexity(5)


def test_generator_respects_the_word_capacity() -> None:
    tiny = BraidConfig(max_len=8, max_strands=4, scramble_budget=2, simplify_budget=8)
    generator = GradedGenerator(tiny, max_crossings=10)
    for source in generator.sources:
        assert source.canonical_length + 2 <= tiny.max_len
    with pytest.raises(ValueError):
        torus_braid(0, 3)


def test_unknotting_numbers_are_the_proved_values() -> None:
    """u(T(p,q)) = (p-1)(q-1)/2 -- Milnor conjecture, Kronheimer-Mrowka 1993.

    Exact ground truth for the quantity the project is ultimately about, on a
    family where it is a theorem rather than a search result.
    """
    from rf_knots.generator import torus_unknotting_number

    assert torus_unknotting_number(2, 3) == 1  # trefoil
    assert torus_unknotting_number(2, 5) == 2  # 5_1
    assert torus_unknotting_number(2, 7) == 3  # 7_1
    assert torus_unknotting_number(3, 4) == 3  # 8_19
    assert torus_unknotting_number(3, 5) == 4  # 10_124
    assert torus_unknotting_number(4, 5) == 6
    with pytest.raises(ValueError):
        torus_unknotting_number(2, 4)  # a link


def test_sources_carry_their_unknotting_number() -> None:
    sources = torus_sources(max_strands=5, max_crossings=8)
    known = {"unknot": 0, "T(2,3)": 1, "T(2,5)": 2, "T(2,7)": 3, "T(3,4)": 3}
    for source in sources:
        if source.name in known:
            assert source.unknotting_number == known[source.name], source.name
    # u is monotone in the crossing number across this family
    ordered = sorted(sources, key=lambda s: s.crossing_number)
    assert [s.unknotting_number for s in ordered] == sorted(
        s.unknotting_number for s in ordered
    )


# -- positive braids -----------------------------------------------------------


def test_positive_braid_formula_reproduces_every_torus_knot() -> None:
    """`u = (c - s + 1) / 2` has to agree with `(p-1)(q-1)/2` on the overlap.

    Torus knots are positive braids, so the two theorems must give the same
    number for every one of them. If they ever disagree, one of the derivations
    is wrong and the whole positive-braid family is untrustworthy.
    """
    from rf_knots.generator import positive_braid_unknotting_number, torus_sources

    checked = 0
    for source in torus_sources(5, 16):
        if not source.word:
            continue
        assert positive_braid_unknotting_number(
            len(source.word), source.strands
        ) == source.unknotting_number, source.name
        checked += 1
    assert checked >= 10, checked


def test_positive_braid_sources_close_to_knots_with_integer_genus() -> None:
    from rf_knots.generator import positive_braid_sources
    from rf_knots.reference import num_components

    sources = positive_braid_sources(5, 14, per_grade=3, seed=0)
    assert sources, "no positive braids generated"
    for source in sources:
        assert all(letter > 0 for letter in source.word), source.name
        assert max(abs(x) for x in source.word) < source.strands, source.name
        # One component is the whole filter: the closure must be a knot.
        assert num_components(list(source.word), source.strands) == 1, source.name
        # Every generator has to appear, or the permutation cannot be an s-cycle.
        assert {abs(x) for x in source.word} == set(range(1, source.strands))
        # Euler characteristic of the braid's own Seifert surface: s disks, c
        # bands, one boundary component, so 2 - 2g - 1 = s - c.
        genus = (source.crossing_number - source.strands + 1) / 2
        assert genus == int(genus), source.name
        assert source.unknotting_number == int(genus), source.name
        assert source.unknotting_number >= 1, "u = 0 is the unknot, already a source"


def test_positive_braid_names_are_stable_across_processes() -> None:
    """Every worker rebuilds the generator and the ladder looks stages up by
    name, so two workers disagreeing would train and evaluate on different
    knots without any error."""
    from rf_knots.generator import positive_braid_sources

    first = positive_braid_sources(5, 12, per_grade=3, seed=7)
    second = positive_braid_sources(5, 12, per_grade=3, seed=7)
    assert [(s.name, s.word) for s in first] == [(s.name, s.word) for s in second]
    assert len({s.name for s in first}) == len(first), "names must be unique"

    other = positive_braid_sources(5, 12, per_grade=3, seed=8)
    assert [s.word for s in first] != [s.word for s in other], "seed must matter"


def test_a_positive_braid_with_u_zero_really_is_the_unknot() -> None:
    """The one part of the formula that can be checked computationally.

    `u = 0` predicts the closure is already the unknot, and `bfs_unknot` decides
    that with type-preserving moves only -- no crossing changes, no appeal to the
    Milnor conjecture. The `u > 0` direction cannot be checked this way, since a
    BFS failure is also what a depth limit looks like.
    """
    from rf_knots.actions import ActionSpec
    from rf_knots.generator import positive_braid_sources
    from rf_knots.reference import bfs_unknot

    trivial = positive_braid_sources(4, 5, per_grade=2, seed=3, min_unknotting=0)
    trivial = [s for s in trivial if s.unknotting_number == 0]
    assert trivial, "expected some u = 0 positive braids to check"
    spec = ActionSpec(max_len=32, max_strands=5)
    for source in trivial:
        solution = bfs_unknot(spec, source.word, source.strands, max_depth=6)
        assert solution is not None, f"{source.name} {source.word} claims u=0"


# -- random mixed-sign knots ---------------------------------------------------


def test_noncanonical_random_knots_are_knots_and_carry_no_label() -> None:
    """The point of this family is the absence of structure.

    Torus knots and positive braids are the knots we can label, and every one of
    them is fibred, chiral, positive-signature, with u = g3 = g4. An agent can
    master all of them without learning anything general. These have mixed signs
    and no theorem, so `u` is explicitly unknown rather than quietly wrong.
    """
    from rf_knots.generator import UNKNOWN_UNKNOTTING, random_braid_sources
    from rf_knots.reference import free_reduce, num_components

    sources = random_braid_sources(5, (10, 12), per_grade=2, seed=4)
    assert sources
    for source in sources:
        assert source.unknotting_number == UNKNOWN_UNKNOTTING
        assert num_components(list(source.word), source.strands) == 1, source.name
        # mixed signs: a family of all-positive words would be positive braids
        # again, with the structure this set exists to avoid
        assert min(source.word) < 0, source.name
        # already free-reduced, so the crossing count is not a lie
        assert tuple(free_reduce(list(source.word))) == source.word
        assert len(source.word) == source.crossing_number
        assert source.strands >= 3, "on two strands a reduced word is sigma_1^c"


def test_canonical_scheduled_random_knots_receive_word_validated_exact_u() -> None:
    from rf_knots.generator import UNKNOWN_UNKNOTTING, random_braid_sources
    from rf_knots.knot_table import scheduled_unknotting_number

    sources = random_braid_sources(
        5,
        (10, 12, 14, 16, 18),
        per_grade=1,
        seed=0,
    )
    by_name = {source.name: source for source in sources}
    expected = {
        "R(3,10)#0": 1,
        "R(5,10)#0": 1,
        "R(3,12)#0": 1,
        "R(5,12)#0": 2,
        "R(3,14)#0": 2,
        "R(5,14)#0": 2,
        "R(3,16)#0": 4,
        "R(5,16)#0": 1,
        "R(3,18)#0": 2,
        "R(5,18)#0": 1,
    }
    assert {name: by_name[name].unknotting_number for name in expected} == expected
    source = by_name["R(3,18)#0"]
    assert scheduled_unknotting_number(source.name, source.word, source.strands) == 2
    assert scheduled_unknotting_number(source.name, source.word[::-1], source.strands) is None
    assert UNKNOWN_UNKNOTTING not in expected.values()


def test_random_knots_are_deterministic_and_seed_dependent() -> None:
    """Every worker rebuilds the generator and the ladder resolves rungs by name;
    two workers disagreeing about what `R(3,12)#0` is would train and evaluate on
    different knots with nothing to raise an error."""
    from rf_knots.generator import random_braid_sources

    first = random_braid_sources(5, (10,), per_grade=2, seed=4)
    again = random_braid_sources(5, (10,), per_grade=2, seed=4)
    other = random_braid_sources(5, (10,), per_grade=2, seed=5)
    assert [(s.name, s.word) for s in first] == [(s.name, s.word) for s in again]
    assert [s.word for s in first] != [s.word for s in other]
    assert len({s.name for s in first}) == len(first)
