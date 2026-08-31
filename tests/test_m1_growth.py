"""The growth-gap measurement, checked against independent oracles.

The claim this module exists to make -- *growth is required for the optimum on
this instance* -- is exactly the kind of claim a capped search can fake, so these
tests are about the places it could:

* the shrink-only move set really is shrink-only, checked by application rather
  than by reading the filter;
* a capped shrink search is never recorded as a gap, because `c_shrink` is only a
  lower bound when the search exhausted;
* whatever `c_shrink` claims replays as a growth-free completion of exactly that
  cost;
* a `known_upper` handed in by a caller is replayed and checked before anything
  is concluded from it;
* the certified floor is sound on real words, and a floor above the budget is
  reported as a proof rather than as a cap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))

import m1_growth as growth
import m1_growth_run
import m1_target as target

from rf_knots import reference
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import (
    UNKNOT_SEED,
    InterleavedConfig,
    generate_many,
    torus_seed,
)

SMALL = InterleavedConfig(
    crossing_changes=2, walk_length=4, target_strands=3,
    target_length=8, max_len=16, max_strands=4,
)


class _Args:
    families = None


def instances(count=4, config=SMALL, seed=11, seed_representation=UNKNOT_SEED):
    return list(
        generate_many(
            config, seed=seed, count=count, seed_representation=seed_representation
        )
    )


# -- the shrink-only move set --------------------------------------------------


def test_the_shrink_move_set_never_lengthens_a_word():
    """Checked by applying every move it offers, not by reading its filter."""
    spec = SMALL.spec()
    call = growth.shrink_successors(spec)
    seen = 0
    for instance in instances(count=3):
        for action, next_word, next_strands in call(instance.word, instance.strands):
            assert len(next_word) <= len(instance.word)
            assert next_strands <= instance.strands
            assert spec.decode(action)[0] not in growth.GROWTH_KINDS
            seen += 1
    assert seen > 20


def test_the_shrink_move_set_is_the_shared_one_minus_growth():
    spec = SMALL.spec()
    call = growth.shrink_successors(spec)
    dropped_any = 0
    for instance in instances(count=2):
        everything = reference.successors(
            spec, instance.word, instance.strands, allow_crossing=True
        )
        kept = call(instance.word, instance.strands)
        dropped = [
            row for row in everything
            if spec.decode(row[0])[0] in growth.GROWTH_KINDS
        ]
        assert len(kept) + len(dropped) == len(everything)
        assert set(kept) <= set(everything)
        dropped_any += len(dropped)
    assert dropped_any > 0, "no state offered a growth move, so nothing was tested"


# -- the certified floor -------------------------------------------------------


def test_the_certified_floor_never_exceeds_a_witness_that_exists():
    """Soundness, against witnesses this repository built and replayed."""
    seed = torus_seed(3, ActionSpec(24, 5))
    config = InterleavedConfig(
        crossing_changes=2, walk_length=4, target_strands=3,
        target_length=10, max_len=24, max_strands=5,
    )
    checked = 0
    for instance in instances(count=5, config=config, seed_representation=seed):
        floor = target.certified_change_floor(instance.word, instance.strands)
        assert floor <= instance.certified_upper_bound
        checked += 1
    assert checked >= 4


def test_a_floor_above_the_budget_is_a_proof_and_not_a_cap():
    """`T(2,7)` has certified `u >= 3`, so no two-change completion exists."""
    spec = ActionSpec(16, 4)
    word = (1,) * 7
    floor = target.certified_change_floor(word, 2)
    assert floor >= 3
    found = target.shortest_completion(
        spec, word, 2, 2, growth=2, node_cap=50_000, objective="lex"
    )
    assert found.moves is None
    assert found.complete is True
    assert found.reason == "certified-floor-exceeds-budget"
    assert found.expanded == 0


# -- the gap itself ------------------------------------------------------------


def test_a_capped_shrink_search_is_undetermined_and_never_a_gap():
    spec = SMALL.spec()
    instance = instances(count=1)[0]
    row = growth.measure_growth_gap(
        spec, instance.word, instance.strands, shrink_node_cap=1, full_node_cap=1
    )
    assert row.verdict == "undetermined-shrink-capped"
    assert row.proven is False
    assert row.accepted is False
    assert row.gap is None


def test_the_certified_verdict_needs_no_search_at_all():
    """`L == c_shrink` settles the instance, because `c_full >= u >= L`."""
    spec = ActionSpec(20, 4)
    row = growth.measure_growth_gap(spec, (1,) * 5, 2, growth=2)
    assert row.c_shrink == 2
    assert row.certified_floor == 2
    assert row.verdict == "no-gap-certified"
    assert row.proven is True
    assert row.gap == 0
    assert row.full_expanded == 0


def test_the_shrink_optimum_agrees_with_a_shrink_only_replay():
    """Whatever `c_shrink` claims, a shrink-only path of that cost must replay."""
    spec = SMALL.spec()
    checked = 0
    for instance in instances(count=3):
        found = growth.shrink_optimum(
            spec, instance.word, instance.strands, change_cap=6, node_cap=60_000
        )
        if found.moves is None:
            continue
        path = target.completion_path(
            spec,
            instance.word,
            instance.strands,
            found.crossing_changes,
            found.moves,
            len(instance.word),
            node_cap=60_000,
        )
        assert path is not None
        witness = UnknotWitness.from_actions(
            instance.word, instance.strands, spec, path
        )
        assert witness.crossing_changes == found.crossing_changes
        assert growth.counts_growth(spec, path) == 0
        checked += 1
    assert checked >= 2


def test_a_known_upper_bound_is_replayed_before_it_is_believed():
    """A caller's claim is checked by replay; a wrong one raises, never counts.

    `sigma_1^5` has `c_shrink = 2`, so a *claimed* one-change completion would be
    a gap -- and the actions handed in here are a real two-change completion
    mislabelled.  The measurement replays them and refuses.
    """
    spec = ActionSpec(20, 4)
    word = (1,) * 5
    honest = growth.shrink_optimum(spec, word, 2, change_cap=4, node_cap=60_000)
    path = target.completion_path(
        spec, word, 2, honest.crossing_changes, honest.moves, len(word),
        node_cap=60_000,
    )
    assert path is not None
    with pytest.raises(AssertionError, match="against a claimed 1"):
        growth.measure_growth_gap(
            spec, word, 2, growth=2, change_floor=0,
            known_upper=1, known_upper_actions=path,
        )


def test_an_accepted_row_always_carries_growth_and_a_gap():
    """The invariant, asserted over whatever a small sweep happens to produce."""
    spec = SMALL.spec()
    seen = 0
    for instance in instances(count=6):
        row = growth.measure_growth_gap(
            spec, instance.word, instance.strands, growth=2,
            shrink_node_cap=40_000, full_node_cap=40_000,
        )
        assert row.verdict in growth.VERDICTS
        assert (row.gap is None) or row.proven
        if row.accepted:
            assert row.gap >= 1
            assert row.c_full < row.c_shrink
            if row.witness_replayed:
                assert row.witness_growth_moves > 0
                witness = UnknotWitness.from_actions(
                    row.word, row.strands, spec, row.witness_actions
                )
                assert witness.crossing_changes == row.c_full
        seen += 1
    assert seen >= 5


def test_the_gap_row_round_trips_to_json_safe_types():
    spec = SMALL.spec()
    instance = instances(count=1)[0]
    row = growth.measure_growth_gap(spec, instance.word, instance.strands)
    payload = json.loads(json.dumps(row.to_dict()))
    assert payload["verdict"] in growth.VERDICTS
    assert payload["proven"] in (True, False)


# -- the candidate families ----------------------------------------------------


def test_every_family_uses_the_shipped_dials_and_names_its_intent():
    for family in m1_growth_run.families(_Args()):
        assert family.intent, f"{family.name} does not say what it is for"
        assert isinstance(family.config, InterleavedConfig)
        provenance = family.provenance()
        assert provenance["config"]["crossing_changes"] >= 0
        assert provenance["seed"]


def test_a_shrink_family_walks_towards_a_shorter_word_than_its_seed():
    """The one respect in which these differ from the corpus the gate was built
    on: the drift target is below the seed length, so `REDUCE` carries the
    `toward_target` multiplier and the reversed witness has to grow."""

    class _Chosen(_Args):
        families = ["shrink-T(2,11)", "shrink-P(3,g=4)"]

    families = m1_growth_run.families(_Chosen())
    assert len(families) == 2
    for family in families:
        assert family.config.target_length < len(family.seed.word)
        assert family.config.weights.reduce > family.config.weights.insert


def test_measuring_a_gap_never_disturbs_the_generator():
    """`m1_growth` reads the generator and never reconfigures it in place."""
    spec = SMALL.spec()
    words = [(instance.word, instance.strands) for instance in instances(count=2, seed=3)]
    for word, strands in words:
        growth.measure_growth_gap(
            spec, word, strands, shrink_node_cap=2_000, full_node_cap=2_000
        )
    after = [(instance.word, instance.strands) for instance in instances(count=2, seed=3)]
    assert words == after
