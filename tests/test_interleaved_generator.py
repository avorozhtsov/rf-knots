"""The interleaved backward generator emits knots that carry their own proof.

Every test here is really one test asked in different ways: the reversed
construction must be a *replayable* path in the shared action space, not merely
a construction that happened to run.  `UnknotWitness.verify` is the arbiter, and
where a second opinion is cheap enough `UnknotWitness.from_states` supplies one
that knows nothing about this module's table of inverses.

The bound is `u <= crossing_changes` and `crossing_changes <= k`.  Both
inequalities are asserted; neither is ever asserted as an equality, because
neither is one.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    KIND_NAMES,
    REDUCE,
    STABILIZE_POS,
    ActionSpec,
)
from rf_knots.evidence import BraidState, LowerBoundClaim, UnknotWitness
from rf_knots.interleaved_generator import (
    BRITTENHAM_HERMILLER_B5,
    ISOTOPY_KINDS,
    UNKNOT_SEED,
    GordianInterval,
    InterleavedConfig,
    LipschitzValues,
    SeedRepresentation,
    WalkWeights,
    _legal_kinds,
    erase_state_loops,
    generate,
    generate_many,
    gordian_interval,
    gordian_lower_bound,
    inverse_action,
    invertible_successors,
    load_quarantine,
    positive_braid_seed,
    torus_seed,
)
from rf_knots.reference import num_components

SMALL = InterleavedConfig(
    crossing_changes=3,
    walk_length=8,
    target_strands=3,
    target_length=10,
    max_len=18,
    max_strands=4,
)

TARGET_SHAPE = InterleavedConfig(
    crossing_changes=5,
    walk_length=24,
    target_strands=4,
    target_length=32,
    max_len=64,
    max_strands=6,
)


def _states(witness: UnknotWitness) -> list[BraidState]:
    return [witness.start, *(step.after for step in witness.steps)]


# -- the contract --------------------------------------------------------------


@pytest.mark.parametrize("index", range(6))
def test_every_emitted_witness_verifies(index: int) -> None:
    instance = generate(TARGET_SHAPE, seed=4242, index=index)
    assert instance.witness.verify() == BraidState((), 1)
    assert instance.witness.start == BraidState(instance.word, instance.strands)


def test_witness_holds_exactly_k_changes_when_the_walk_is_long() -> None:
    """With `l` large enough to leave the neighbourhood, nothing cancels."""
    for index in range(12):
        instance = generate(TARGET_SHAPE, seed=7, index=index)
        assert instance.erased_crossing_changes == 0
        assert instance.certified_upper_bound == instance.crossing_change_budget == 5


def test_the_bound_is_never_larger_than_k() -> None:
    """The one inequality that must hold for every configuration."""
    for walk_length in (0, 1, 2, 5, 13, 30):
        config = InterleavedConfig(
            crossing_changes=5,
            walk_length=walk_length,
            target_strands=3,
            target_length=12,
            max_len=32,
            max_strands=4,
        )
        for index in range(8):
            instance = generate(config, seed=1234, index=index)
            instance.witness.verify()
            assert instance.certified_upper_bound <= instance.crossing_change_budget
            # The accounting closes: what is missing was erased, or never placed.
            assert (
                instance.crossing_change_budget
                - instance.certified_upper_bound
                - instance.erased_crossing_changes
                - instance.skipped_crossing_changes
            ) == 0


def test_cancelling_changes_lower_the_bound_instead_of_being_hidden() -> None:
    """`l = 0` puts every crossing change on the same site, and they cancel.

    A crossing change is its own inverse, so `k` consecutive changes at one
    position are `k mod 2` changes.  Loop erasure notices -- the braid revisits a
    state it has already been in -- and emits the shorter witness.  The claim
    that comes out is `u <= 1`, which is true; a witness with the redundant pairs
    left in would claim `u <= 5`, which is also true but weaker, and would put a
    five-change label on a one-change knot.
    """
    config = InterleavedConfig(
        crossing_changes=5,
        walk_length=0,
        target_strands=3,
        target_length=8,
        max_len=24,
        max_strands=4,
    )
    seen = set()
    for index in range(10):
        instance = generate(config, seed=31, index=index)
        instance.witness.verify()
        assert instance.certified_upper_bound < instance.crossing_change_budget
        assert instance.erased_crossing_changes > 0
        seen.add(instance.certified_upper_bound)
    assert seen == {1}


def test_erased_witness_is_still_a_witness_for_the_emitted_word() -> None:
    """A short walk in a small box revisits states constantly; erase and check.

    `l = 2` inside `max_len = 8` closes a loop on nearly every instance, which
    is the case where the erased path has to be re-derived rather than sliced --
    the surviving actions are a subsequence of the original, not a prefix.
    """
    config = InterleavedConfig(
        crossing_changes=6,
        walk_length=2,
        target_strands=2,
        target_length=4,
        max_len=8,
        max_strands=3,
    )
    erased = [
        instance
        for instance in (generate(config, seed=99, index=i) for i in range(20))
        if instance.erased_steps
    ]
    assert len(erased) > 10, "expected loops in a box this small"
    for instance in erased:
        assert instance.witness.start == BraidState(instance.word, instance.strands)
        assert instance.witness.verify() == BraidState((), 1)
        # Erasure is idempotent: what survives visits no state twice.
        states = _states(instance.witness)
        assert len(set(states)) == len(states)


def test_every_state_on_the_witness_is_a_knot_inside_the_action_spec() -> None:
    for index in range(6):
        instance = generate(TARGET_SHAPE, seed=808, index=index)
        for state in _states(instance.witness):
            assert num_components(state.word, state.strands) == 1
            assert len(state.word) <= TARGET_SHAPE.max_len
            assert state.strands <= TARGET_SHAPE.max_strands


def test_the_interleaved_shape_survives_into_the_witness() -> None:
    """Crossing changes separated by long isotopy runs -- §4.1's whole point."""
    instance = generate(TARGET_SHAPE, seed=17, index=0)
    steps = instance.witness_crossing_change_steps
    assert len(steps) == 5
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    # `l = 24` isotopy moves between changes, minus whatever the caps refused.
    assert all(gap > 10 for gap in gaps), gaps
    kinds = {step.action.kind for step in instance.witness.steps}
    assert KIND_NAMES[CROSSING_CHANGE] in kinds
    assert kinds - {KIND_NAMES[CROSSING_CHANGE]}, "the witness is all changes"


def test_settle_walk_buries_the_last_change() -> None:
    """Without it the first move of the reversed witness is a crossing change."""
    plain = generate(SMALL, seed=5, index=0)
    assert plain.witness.steps[0].action.kind == KIND_NAMES[CROSSING_CHANGE]

    settled = generate(replace(SMALL, settle_walk_length=8), seed=5, index=0)
    assert settled.witness.steps[0].action.kind != KIND_NAMES[CROSSING_CHANGE]
    assert settled.certified_upper_bound == SMALL.crossing_changes


# -- determinism ---------------------------------------------------------------


def test_a_seed_pins_the_stream() -> None:
    first = [(x.word, x.strands) for x in generate_many(SMALL, seed=2024, count=6)]
    again = [(x.word, x.strands) for x in generate_many(SMALL, seed=2024, count=6)]
    other = [(x.word, x.strands) for x in generate_many(SMALL, seed=2025, count=6)]
    assert first == again
    assert first != other


def test_the_stream_is_resumable_at_any_offset() -> None:
    whole = [x.word for x in generate_many(SMALL, seed=11, count=8)]
    tail = [x.word for x in generate_many(SMALL, seed=11, count=4, start=4)]
    assert whole[4:] == tail


def test_indices_are_independent_not_sequential() -> None:
    """`generate(seed, i)` must not depend on `generate(seed, i - 1)` having run."""
    direct = generate(SMALL, seed=3, index=5)
    after_others = None
    for index in range(6):
        after_others = generate(SMALL, seed=3, index=index)
    assert after_others is not None
    assert after_others.word == direct.word


# -- inverses: the part that is easy to get wrong -------------------------------


def test_a_seam_reduction_has_no_inverse_in_the_move_set() -> None:
    """`REDUCE` across the seam lands on the right necklace and the wrong list."""
    spec = ActionSpec(max_len=12, max_strands=3)
    word = (-1, 2, -2, 1)  # word[3] == -word[0]: the cancelling pair straddles
    action = spec.encode(REDUCE, position=3)
    from rf_knots import reference

    assert reference.is_legal(spec, word, 3, action, True)
    after, strands = reference.apply(spec, word, 3, action)
    assert after == (2, -2)  # the correct necklace, rotated
    assert inverse_action(spec, word, 3, action, after, strands) is None


def test_a_tail_reduction_has_no_inverse_either() -> None:
    """`INSERT` splices *before* index `p`, so it can never rebuild a tail."""
    spec = ActionSpec(max_len=12, max_strands=3)
    word = (1, 2, -2)
    action = spec.encode(REDUCE, position=1)
    after, strands = (1,), 3
    assert inverse_action(spec, word, 3, action, after, strands) is None


def test_an_interior_destabilisation_has_no_inverse() -> None:
    """`STABILIZE` appends, so only a trailing top letter can be put back."""
    spec = ActionSpec(max_len=12, max_strands=4)
    word = (2, 1)  # the lone sigma_2 is first, not last
    action = spec.encode(DESTABILIZE)
    from rf_knots import reference

    assert reference.is_legal(spec, word, 3, action, True)
    after, strands = reference.apply(spec, word, 3, action)
    assert (after, strands) == ((1,), 2)
    assert inverse_action(spec, word, 3, action, after, strands) is None

    trailing = (1, 2)
    after, strands = reference.apply(spec, trailing, 3, action)
    back = inverse_action(spec, trailing, 3, action, after, strands)
    assert back == spec.encode(STABILIZE_POS)


def test_the_involutions_are_their_own_inverses() -> None:
    spec = ActionSpec(max_len=12, max_strands=4)
    from rf_knots import reference

    cases = [
        ((1, 3), 4, spec.encode(COMMUTE, position=0)),
        ((1, 2, 1), 3, spec.encode(BRAID, position=0)),
        ((1, 1, 1), 2, spec.encode(CROSSING_CHANGE, position=1)),
    ]
    for word, strands, action in cases:
        assert reference.is_legal(spec, word, strands, action, True)
        after, new_strands = reference.apply(spec, word, strands, action)
        assert inverse_action(spec, word, strands, action, after, new_strands) == action


def test_insert_is_undone_by_reduce_at_the_same_position() -> None:
    spec = ActionSpec(max_len=12, max_strands=4)
    from rf_knots import reference

    for position in range(3):
        action = spec.encode(INSERT, position=position, generator=2, sign=-1)
        word, strands = (1, 2, 1), 3
        after, new_strands = reference.apply(spec, word, strands, action)
        assert inverse_action(spec, word, strands, action, after, new_strands) == (
            spec.encode(REDUCE, position=position)
        )


@pytest.mark.parametrize("cyclic", [False, True])
def test_candidate_enumeration_agrees_with_the_reference_move_set(cyclic: bool) -> None:
    """`_legal_kinds` is a hand-written copy of `reference.is_legal`; pin it.

    The fast enumerator exists because `reference.successors` materialises the
    whole `INSERT` block -- thousands of words per step.  A copy of a legality
    predicate is a place for the two to drift apart, so it is differentially
    tested against `invertible_successors`, which is the reference enumerator
    with this module's inverse filter bolted on.
    """
    rng = np.random.default_rng(0 if not cyclic else 1)
    spec = ActionSpec(max_len=14, max_strands=5, cyclic_band_generators=cyclic)
    positional = (REDUCE, COMMUTE, BRAID)
    for _ in range(1500):
        strands = int(rng.integers(1, 6))
        largest = strands if cyclic else strands - 1
        length = 0 if largest < 1 else int(rng.integers(0, 13))
        word = tuple(
            int(rng.integers(1, largest + 1)) * (1 if rng.integers(2) else -1)
            for _ in range(length)
        )
        expected: dict[int, set] = {}
        for action, _back, _word, _strands in invertible_successors(spec, word, strands):
            kind, position, _generator, _sign = spec.decode(action)
            expected.setdefault(kind, set()).add(
                position if kind in positional else True
            )
        candidates = _legal_kinds(spec, word, strands)
        assert set(candidates) <= set(ISOTOPY_KINDS), "PASS or a change leaked in"
        actual = {
            kind: (set(payload) if isinstance(payload, list) else {True})
            for kind, payload in candidates.items()
        }
        assert actual == expected, (cyclic, strands, word)


# -- the independent oracle ----------------------------------------------------


def test_from_states_recovers_the_same_path_without_the_inverse_table() -> None:
    """A second opinion that knows nothing about `inverse_action`.

    `UnknotWitness.from_states` searches the whole action space for *some* legal
    action realising each transition, so agreement here means the reversed path
    is legal for reasons independent of the case analysis in this module.  It
    picks the lowest-numbered match, which is often a different action of the
    same effect, so the states are compared and not the action names.

    It is also roughly eleven times slower than generating the instance on this
    small configuration, and worse on a wide one -- the action space it scans
    grows with `max_len * max_strands`.  That is why it is here and not in
    `generate`.
    """
    spec = SMALL.spec()
    for index in range(3):
        instance = generate(SMALL, seed=64, index=index)
        translated = UnknotWitness.from_states(_states(instance.witness), spec)
        assert _states(translated) == _states(instance.witness)
        assert translated.crossing_changes == instance.certified_upper_bound


# -- loop erasure --------------------------------------------------------------


def test_erase_state_loops_removes_a_closed_sub_walk() -> None:
    states = [("a", 1), ("b", 1), ("c", 1), ("b", 1), ("d", 1)]
    actions = [10, 11, 12, 13]
    kept_actions, kept_states = erase_state_loops(actions, states)
    assert kept_states == [("a", 1), ("b", 1), ("d", 1)]
    assert kept_actions == [10, 13]


def test_erase_state_loops_keeps_a_simple_path_intact() -> None:
    states = [("a", 1), ("b", 1), ("c", 1)]
    actions = [1, 2]
    assert erase_state_loops(actions, states) == ([1, 2], states)


def test_erase_state_loops_can_empty_the_path() -> None:
    states = [("a", 1), ("b", 1), ("a", 1)]
    assert erase_state_loops([1, 2], states) == ([], [("a", 1)])


def test_loop_erasure_can_be_switched_off() -> None:
    config = InterleavedConfig(
        crossing_changes=5,
        walk_length=0,
        target_strands=3,
        target_length=8,
        max_len=24,
        max_strands=4,
        erase_loops=False,
    )
    instance = generate(config, seed=31, index=0)
    instance.witness.verify()
    # Still true, still verified -- just a weaker statement about the same knot.
    assert instance.certified_upper_bound == 5
    assert instance.erased_crossing_changes == 0


# -- the ActionSpec boundary ---------------------------------------------------


def test_a_word_cap_that_forbids_growth_does_not_crash() -> None:
    config = InterleavedConfig(
        crossing_changes=3,
        walk_length=6,
        target_strands=2,
        target_length=2,
        max_len=3,
        max_strands=2,
    )
    for index in range(8):
        instance = generate(config, seed=6, index=index)
        instance.witness.verify()
        assert instance.certified_upper_bound <= 3
        for state in _states(instance.witness):
            assert len(state.word) <= 3
            assert state.strands <= 2


def test_a_one_strand_cap_leaves_nothing_to_do() -> None:
    """`max_strands = 1` admits only the empty word, so nothing can be built."""
    config = InterleavedConfig(
        crossing_changes=3,
        walk_length=6,
        target_strands=1,
        target_length=1,
        max_len=1,
        max_strands=1,
    )
    instance = generate(config, seed=6, index=0)
    assert instance.is_trivial
    assert instance.skipped_crossing_changes == 3
    assert instance.certified_upper_bound == 0
    assert instance.witness.verify() == BraidState((), 1)
    with pytest.raises(RuntimeError, match="consecutive instances rejected"):
        list(generate_many(config, seed=6, count=1))


def test_zero_changes_is_the_unknot_and_says_so() -> None:
    config = InterleavedConfig(
        crossing_changes=0,
        walk_length=10,
        target_strands=3,
        target_length=10,
        max_len=32,
        max_strands=4,
    )
    instance = generate(config, seed=6, index=0)
    assert instance.is_trivial
    assert instance.certified_upper_bound == 0


def test_the_strand_target_is_reached_and_held() -> None:
    """Stabilisation is a one-way ratchet; the walk must not be one too."""
    for index in range(10):
        instance = generate(TARGET_SHAPE, seed=55, index=index)
        assert instance.strands == TARGET_SHAPE.target_strands


def test_a_strand_band_admits_markov_churn() -> None:
    config = InterleavedConfig(
        crossing_changes=4,
        walk_length=16,
        target_strands=3,
        target_length=16,
        max_len=32,
        max_strands=5,
        strand_band=1,
    )
    widths = {generate(config, seed=8, index=i).strands for i in range(12)}
    assert widths <= {3, 4}
    assert 4 in widths


def test_configuration_rejects_impossible_dials() -> None:
    with pytest.raises(ValueError, match="crossing_changes"):
        InterleavedConfig(crossing_changes=-1)
    with pytest.raises(ValueError, match="target_strands"):
        InterleavedConfig(target_strands=9, max_strands=6)
    with pytest.raises(ValueError, match="target_length"):
        InterleavedConfig(target_length=100, max_len=64)
    with pytest.raises(ValueError, match="walk lengths"):
        InterleavedConfig(walk_length=-2)


# -- coverage across the dials -------------------------------------------------


@pytest.mark.parametrize(
    ("crossing_changes", "walk_length", "strands", "target_length", "max_len"),
    [
        (1, 4, 2, 6, 16),
        (2, 12, 3, 14, 24),
        (3, 20, 4, 24, 40),
        (5, 24, 4, 32, 64),
        (7, 10, 5, 40, 64),
    ],
)
def test_the_dials_move_the_output(
    crossing_changes: int,
    walk_length: int,
    strands: int,
    target_length: int,
    max_len: int,
) -> None:
    config = InterleavedConfig(
        crossing_changes=crossing_changes,
        walk_length=walk_length,
        target_strands=strands,
        target_length=target_length,
        max_len=max_len,
        max_strands=strands + 2,
    )
    for index in range(3):
        instance = generate(config, seed=90, index=index)
        assert instance.witness.verify() == BraidState((), 1)
        assert instance.certified_upper_bound <= crossing_changes
        assert instance.strands <= strands + 2
        assert len(instance.word) <= max_len


def test_band_generators_are_supported_end_to_end() -> None:
    config = InterleavedConfig(
        crossing_changes=4,
        walk_length=12,
        target_strands=4,
        target_length=16,
        max_len=32,
        max_strands=5,
        cyclic_band_generators=True,
    )
    for index in range(4):
        instance = generate(config, seed=12, index=index)
        assert instance.witness.cyclic_band_generators is True
        assert instance.witness.verify() == BraidState((), 1, True)
        assert instance.certified_upper_bound <= 4


def test_weights_are_a_dial_not_a_hardcoded_table() -> None:
    """Two weightings, one seed, different knots -- and both still verified."""
    config = InterleavedConfig(
        crossing_changes=3,
        walk_length=16,
        target_strands=3,
        target_length=16,
        max_len=32,
        max_strands=4,
    )
    shuffly = generate(
        replace(config, weights=WalkWeights(commute=20.0, braid=20.0)), seed=3, index=0
    )
    inserty = generate(
        replace(config, weights=WalkWeights(commute=0.2, braid=0.2, insert=20.0)),
        seed=3,
        index=0,
    )
    assert shuffly.word != inserty.word
    for instance in (shuffly, inserty):
        assert instance.witness.verify() == BraidState((), 1)


def test_an_all_zero_weighting_falls_back_rather_than_dividing_by_zero() -> None:
    """Every dial at zero is a user error, not a crash: sample uniformly."""
    weights = WalkWeights(
        reduce=0.0, commute=0.0, braid=0.0, insert=0.0, destabilize=0.0, stabilize=0.0
    )
    config = InterleavedConfig(
        crossing_changes=2,
        walk_length=10,
        target_strands=2,
        target_length=4,
        max_len=16,
        max_strands=3,
        weights=weights,
    )
    instance = generate(config, seed=3, index=0)
    assert instance.witness.verify() == BraidState((), 1)
    assert instance.certified_upper_bound <= 2


# -- the shard writer ----------------------------------------------------------


def _shard_writer():
    """Import `scripts/write_interleaved_shards.py` the way the other tests do."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "write_interleaved_shards.py"
    spec = importlib.util.spec_from_file_location("write_interleaved_shards", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shard_rows_replay_from_disk(tmp_path, monkeypatch) -> None:
    """The written row is evidence, so reading it back must re-verify it.

    `EvidenceStore.records()` re-hashes every row and re-replays every witness,
    so a passing read is a statement about the file and not about this process.
    """
    from rf_knots.evidence import EvidenceStore

    writer = _shard_writer()
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "write_interleaved_shards.py",
            "--output-dir", str(output),
            "--count", "6",
            "--shard-size", "4",
            "-k", "2",
            "-l", "6",
            "--target-strands", "3",
            "--target-length", "10",
            "--max-len", "24",
            "--max-strands", "4",
            "--seed", "5150",
        ],
    )
    writer.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == writer.MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["examples"] == 6
    assert manifest["label"] == "u <= metrics.certified_upper_bound"
    assert manifest["generator"]["crossing_changes"] == 2
    assert manifest["generator"]["walk_length"] == 6
    assert manifest["config_sha256"]
    assert [row["count"] for row in manifest["shards"]] == [4, 2]

    rows = []
    for shard in manifest["shards"]:
        path = output / shard["path"]
        assert writer.sha256_file(path) == shard["sha256"]
        rows.extend(EvidenceStore(path).records())
    assert len(rows) == 6
    for record in rows:
        assert record.solver == writer.SOLVER
        assert record.outcome == writer.OUTCOME
        assert record.witness is not None
        assert record.certified_upper_bound <= 2
        assert record.metrics["exact_unknotting_number"] is None
        assert record.metrics["certified_lower_bound"] is None
        assert record.search_budget["seed"] == 5150
        assert record.witness.verify() == BraidState((), 1)


def test_shard_writer_resumes_where_it_stopped(tmp_path, monkeypatch) -> None:
    writer = _shard_writer()
    output = tmp_path / "shards"
    common = [
        "--output-dir", str(output),
        "--shard-size", "3",
        "-k", "2", "-l", "6",
        "--target-strands", "3", "--target-length", "10",
        "--max-len", "24", "--max-strands", "4",
        "--seed", "99",
    ]
    monkeypatch.setattr("sys.argv", ["w", *common, "--count", "3"])
    writer.main()
    monkeypatch.setattr("sys.argv", ["w", *common, "--count", "6", "--resume"])
    writer.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["examples"] == 6
    assert [row["first_stream_index"] for row in manifest["shards"]] == [0, 3]

    monkeypatch.setattr(
        "sys.argv", ["w", *common, "--count", "9", "--resume", "-l", "7"]
    )
    with pytest.raises(SystemExit, match="different dials"):
        writer.main()


def test_shard_writer_records_the_accept_filter(tmp_path, monkeypatch) -> None:
    """`--min-certified-lower-bound` must show up in the manifest, not just act.

    Without it the corpus labels trefoils `u <= 5`: at `k = 5, l = 24` the
    Murasugi bound is 0 or 1 on three quarters of the instances.  A shard that
    was filtered and one that was not are different datasets and the manifest
    has to say which it is.
    """
    pytest.importorskip("spherogram")
    from rf_knots.evidence import EvidenceStore

    writer = _shard_writer()
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "w",
            "--output-dir", str(output),
            "--count", "3", "--shard-size", "3",
            "-k", "3", "-l", "10",
            "--target-strands", "3", "--target-length", "12",
            "--max-len", "24", "--max-strands", "4",
            "--min-certified-lower-bound", "1",
            "--attempts-per-instance", "60",
            "--seed", "404",
        ],
    )
    writer.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["min_certified_lower_bound"] == 1
    assert manifest["lower_bounds"] is True
    assert manifest["rejected"] >= 0
    for record in EvidenceStore(output / "shard-000000.jsonl").records():
        assert record.certified_lower_bound is not None
        assert record.certified_lower_bound >= 1
        assert record.certified_upper_bound <= 3
        # The one claim the corpus is allowed to make, and only when they meet.
        if record.metrics["exact_unknotting_number"] is not None:
            assert record.certified_lower_bound == record.certified_upper_bound


# -- seeds: §4.1b ---------------------------------------------------------------

SEED_SPEC = ActionSpec(max_len=80, max_strands=6)

SEEDED = InterleavedConfig(
    crossing_changes=3,
    walk_length=24,
    target_strands=4,
    target_length=32,
    max_len=80,
    max_strands=6,
)


@pytest.mark.parametrize("m", range(8))
def test_a_torus_seed_carries_a_program_that_meets_its_own_theorem(m: int) -> None:
    """`T(2,2m+1)`: `u = m` from Milnor, `u <= m` from a program that replays."""
    seed = torus_seed(m, SEED_SPEC)
    assert seed.word == (1,) * (2 * m + 1)
    assert seed.strands == 2
    assert seed.certified_lower_bound == m
    assert seed.verify_program(SEED_SPEC) == m
    assert seed.is_exact and seed.replay_complete


def test_the_torus_lower_bound_is_a_theorem_not_a_table_row() -> None:
    seed = torus_seed(7, SEED_SPEC)
    claim = seed.lower_claims[0]
    assert claim.value == 7
    assert claim.source == "rf_knots.generator.torus_unknotting_number"
    assert "Kronheimer" in (claim.citation or "")


@pytest.mark.parametrize(("strands", "genus"), [(2, 3), (3, 4), (4, 5), (5, 6)])
def test_a_positive_braid_seed_is_positive_and_its_program_is_its_genus(
    strands: int, genus: int
) -> None:
    """The theorem needs positivity and one component; both are re-checked here."""
    seed = positive_braid_seed(SEED_SPEC, strands=strands, genus=genus, seed=17)
    assert all(letter > 0 for letter in seed.word)
    assert num_components(seed.word, seed.strands) == 1
    assert seed.strands == strands
    assert (len(seed.word) - strands + 1) // 2 == genus
    assert seed.certified_lower_bound == genus
    assert seed.verify_program(SEED_SPEC) == genus
    assert seed.is_exact


def test_positive_braid_seeds_differ_between_rng_seeds() -> None:
    words = {
        positive_braid_seed(SEED_SPEC, strands=4, genus=5, seed=index).word
        for index in range(6)
    }
    assert len(words) > 1


def test_a_seed_whose_closure_is_a_link_is_refused() -> None:
    with pytest.raises(ValueError, match="link"):
        SeedRepresentation(name="bad", family="test", word=(1, 1), strands=2)


def test_a_seed_cannot_claim_an_upper_bound_below_its_own_lower_bound() -> None:
    with pytest.raises(ValueError, match="below its own certified"):
        SeedRepresentation(
            name="bad",
            family="test",
            word=(1, 1, 1),
            strands=2,
            lower_claims=(LowerBoundClaim(1, "signature", "test"),),
            program=(),
            u_upper=0,
        )


def test_a_seed_carrying_a_program_must_state_its_upper_bound() -> None:
    with pytest.raises(ValueError, match="must state its u_upper"):
        SeedRepresentation(
            name="bad", family="test", word=(1, 1, 1), strands=2, program=()
        )


def test_a_seed_program_is_portable_across_action_specs() -> None:
    """The program is stored semantically, so a wider spec replays it unchanged."""
    seed = torus_seed(5, ActionSpec(max_len=32, max_strands=4))
    assert seed.verify_program(ActionSpec(max_len=128, max_strands=9)) == 5


def test_a_seed_that_does_not_fit_the_action_spec_is_refused() -> None:
    seed = torus_seed(9, ActionSpec(max_len=64, max_strands=6))
    small = InterleavedConfig(
        crossing_changes=1,
        walk_length=2,
        target_strands=2,
        target_length=4,
        max_len=8,
        max_strands=3,
    )
    with pytest.raises(ValueError, match="does not fit the action spec"):
        generate(small, seed=1, index=0, seed_representation=seed)


# -- seeded generation ----------------------------------------------------------


def test_the_unknot_seeded_path_is_bit_for_bit_what_it_was() -> None:
    """§4.1b is an extension, not a replacement: `L = 0` is the old behaviour."""
    for index in range(8):
        old = generate(TARGET_SHAPE, seed=99, index=index)
        new = generate(
            TARGET_SHAPE, seed=99, index=index, seed_representation=UNKNOT_SEED
        )
        assert (old.word, old.strands) == (new.word, new.strands)
        assert old.certified_upper_bound == new.certified_upper_bound
        assert old.witness.to_dict() == new.witness.to_dict()
        assert old.u_lower_inherited == 0
        assert old.u_upper == old.certified_upper_bound
        assert old.replay_complete


@pytest.mark.parametrize("index", range(6))
def test_a_seeded_witness_replays_and_passes_through_the_seed(index: int) -> None:
    seed = torus_seed(7, SEED_SPEC)
    instance = generate(SEEDED, seed=2024, index=index, seed_representation=seed)
    assert instance.witness.verify() == BraidState((), 1)
    joint = instance.witness.steps[len(instance.reverse_actions) - 1].after
    assert (joint.word, joint.strands) == (seed.word, seed.strands)
    assert instance.witness.crossing_changes == instance.d_upper + seed.u_upper


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_the_inherited_lower_bound_is_the_seed_bound_minus_the_walk(k: int) -> None:
    """§4.1b(3), and no invariant of `R'` is computed to obtain it."""
    seed = torus_seed(7, SEED_SPEC)
    config = replace(SEEDED, crossing_changes=k)
    for index in range(6):
        instance = generate(config, seed=5, index=index, seed_representation=seed)
        assert instance.d_upper <= k
        assert instance.u_lower_inherited == max(0, 7 - instance.d_upper)
        assert instance.u_upper == instance.d_upper + 7


def test_the_inherited_bound_degrades_to_nothing_when_k_exceeds_L() -> None:
    """The one-sidedness §4.1b's fourth hazard warns about, asserted."""
    seed = torus_seed(2, SEED_SPEC)
    config = replace(SEEDED, crossing_changes=5)
    instance = generate(config, seed=5, index=0, seed_representation=seed)
    assert instance.u_lower_inherited == 0


def test_a_seed_without_a_program_still_certifies_the_pair() -> None:
    """Catalogue-composed: no witness, but the walk is replayed all the same."""
    seed = SeedRepresentation(
        name="catalogue",
        family="test",
        word=(1,) * 7,
        strands=2,
        lower_claims=(LowerBoundClaim(3, "signature", "test"),),
        program=None,
        u_upper=None,
    )
    instance = generate(SEEDED, seed=8, index=0, seed_representation=seed)
    assert instance.witness is None
    assert not instance.replay_complete
    assert instance.witness_provenance == "catalogue-composed"
    assert instance.u_upper is None
    assert instance.certified_upper_bound is None
    assert instance.u_lower_inherited == max(0, 3 - instance.d_upper)
    assert instance.instance_id.startswith("braid:")
    assert instance.state == BraidState(instance.word, instance.strands)


def test_the_walk_from_a_seed_never_holds_more_than_k_changes() -> None:
    seed = positive_braid_seed(SEED_SPEC, strands=4, genus=5, seed=3)
    for k in (0, 1, 3, 5):
        config = replace(SEEDED, crossing_changes=k)
        for index in range(4):
            instance = generate(config, seed=7, index=index, seed_representation=seed)
            instance.witness.verify()
            assert instance.d_upper <= k
            assert instance.certified_upper_bound == instance.d_upper + 5


def test_a_zero_change_walk_from_a_seed_is_a_pure_isotopy_pair() -> None:
    seed = torus_seed(5, SEED_SPEC)
    config = replace(SEEDED, crossing_changes=0, settle_walk_length=20)
    instance = generate(config, seed=4, index=0, seed_representation=seed)
    assert instance.d_upper == 0
    assert instance.u_lower_inherited == 5
    assert instance.witness.verify() == BraidState((), 1)


def test_generate_many_threads_the_seed_through() -> None:
    seed = torus_seed(7, SEED_SPEC)
    emitted = list(
        generate_many(SEEDED, seed=31, count=5, seed_representation=seed)
    )
    assert len(emitted) == 5
    assert all(item.seed_representation.name == "T(2,15)" for item in emitted)
    assert all(item.u_lower_inherited >= 4 for item in emitted)


def test_the_seeded_metrics_row_names_its_provenance() -> None:
    seed = torus_seed(7, SEED_SPEC)
    metrics = generate(
        SEEDED, seed=1, index=0, seed_representation=seed
    ).metrics()
    assert metrics["seed_name"] == "T(2,15)"
    assert metrics["seed_family"] == "torus-2q"
    assert metrics["seed_certified_lower_bound"] == 7
    assert metrics["witness_provenance"] == "replay-complete"
    assert json.loads(json.dumps(metrics)) == metrics


# -- Gordian intervals ----------------------------------------------------------


def test_gordian_lower_bound_uses_the_signed_invariants() -> None:
    """`|sigma_1/2 - sigma_2/2|`, not `||sigma_1|/2 - |sigma_2|/2|`.

    A knot and its mirror sit at `+3` and `-3`; the signed difference is 6 and
    the unsigned one is 0.  Using the unsigned form would report distance zero
    between two knots six crossing changes apart.
    """
    left = LipschitzValues(signature_half=3, tau=3)
    right = LipschitzValues(signature_half=-3, tau=-3)
    assert gordian_lower_bound(left, right) == (6, "signature")


def test_gordian_lower_bound_says_so_when_it_has_nothing() -> None:
    assert gordian_lower_bound(LipschitzValues(), LipschitzValues()) == (0, "none")


def test_gordian_lower_bound_takes_the_strongest_available_invariant() -> None:
    left = LipschitzValues(signature_half=1, tau=5)
    right = LipschitzValues(signature_half=0, tau=0)
    assert gordian_lower_bound(left, right) == (5, "ozsvath-szabo-tau")


def test_a_gordian_interval_is_two_sided_and_never_empty() -> None:
    seed = torus_seed(7, SEED_SPEC)
    instance = generate(SEEDED, seed=6, index=0, seed_representation=seed)
    interval = gordian_interval(
        instance,
        left=LipschitzValues(signature_half=-5, tau=5),
        right=LipschitzValues(signature_half=-7, tau=7),
    )
    assert interval.metric == "cc"
    assert interval.upper == instance.d_upper
    assert 0 <= interval.lower <= interval.upper
    assert interval.right_word == seed.word
    assert interval.witness_semantic_length == len(instance.reverse_actions)
    assert json.loads(json.dumps(interval.to_dict()))["lower"] == interval.lower


def test_an_empty_gordian_interval_is_refused_rather_than_stored() -> None:
    with pytest.raises(ValueError, match="empty"):
        GordianInterval(
            metric="cc",
            left_word=(1,),
            left_strands=2,
            right_word=(),
            right_strands=1,
            lower=3,
            upper=1,
            witness_semantic_length=1,
        )


def test_gordian_intervals_load_into_the_metric_objective() -> None:
    """The rows are the shape `MetricEmbeddingObjective` already consumes."""
    torch = pytest.importorskip("torch")
    from rf_knots.embedding_metric import MetricBatch, MetricEmbeddingObjective

    seed = torus_seed(7, SEED_SPEC)
    rows = [
        gordian_interval(
            generate(SEEDED, seed=12, index=index, seed_representation=seed),
            left=LipschitzValues(signature_half=-5, tau=5),
            right=LipschitzValues(signature_half=-7, tau=7),
        )
        for index in range(4)
    ]
    batch = MetricBatch(
        metric=tuple(row.metric for row in rows),
        lower=torch.tensor([float(row.lower) for row in rows]),
        upper=torch.tensor([float(row.upper) for row in rows]),
        witness_semantic_length=torch.tensor(
            [float(row.witness_semantic_length) for row in rows]
        ),
    )
    left = {"knot": torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)}
    right = {"knot": torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)}
    loss, diagnostics = MetricEmbeddingObjective()(left, right, batch)
    assert loss.isfinite()
    assert "cc_loss" in diagnostics


# -- quarantine -----------------------------------------------------------------


def test_the_quarantine_holds_the_b5_word_and_its_masked_states() -> None:
    quarantine = load_quarantine(unknotdb_root=None, b5_radius=3)
    assert quarantine.screen_word(BRITTENHAM_HERMILLER_B5, 5) is not None
    masked = list(BRITTENHAM_HERMILLER_B5)
    masked[0] = -masked[0]
    masked[7] = -masked[7]
    assert quarantine.screen_word(tuple(masked), 5) is not None
    # A different strand count is a different representation and is not screened
    # by the word test; that is what the name and fingerprint screens are for.
    assert quarantine.screen_word(BRITTENHAM_HERMILLER_B5, 6) is None


def test_the_quarantine_rejects_the_target_fingerprint() -> None:
    quarantine = load_quarantine(unknotdb_root=None, b5_radius=1)
    assert quarantine.screen_fingerprint(49, 0) == "quarantined-fingerprint"
    assert quarantine.screen_fingerprint(49, 4) is None
    assert quarantine.screen_fingerprint(75, 0) is None


def test_the_quarantine_rejects_the_named_chain_knots_and_the_composite() -> None:
    quarantine = load_quarantine(unknotdb_root=None, b5_radius=1)
    assert quarantine.screen_name("14a_18636") == "quarantined-name"
    assert quarantine.screen_name("15n_81556") == "quarantined-name"
    assert quarantine.screen_name("12n_412") == "quarantined-name"
    assert quarantine.screen_name(None, ("7_1", "m(7_1)")) == "quarantined-composite"
    assert quarantine.screen_name("7_1") is None
    assert quarantine.screen_name(None, ("3_1", "3_1")) is None


def test_the_seed_families_pass_the_screen() -> None:
    quarantine = load_quarantine(unknotdb_root=None, b5_radius=2)
    assert quarantine.screen_seed(torus_seed(7, SEED_SPEC)) is None
    assert (
        quarantine.screen_seed(positive_braid_seed(SEED_SPEC, strands=4, genus=5))
        is None
    )
    assert quarantine.screen_seed(UNKNOT_SEED) is None


def test_the_quarantine_states_what_it_does_not_guarantee() -> None:
    """The `k`-ball closure is not computed and the manifest has to say so."""
    provenance = load_quarantine(unknotdb_root=None, b5_radius=1).provenance()
    assert "not computed" in provenance["k_ball_closure"]
    assert provenance["representations"] >= 21
    assert "7_1 # m(7_1)" in provenance["names"]
    assert json.loads(json.dumps(provenance))


# -- the bound the seed promised, computed again --------------------------------


def test_the_computed_bound_never_falls_below_the_inherited_one() -> None:
    """The one assertion that would break §4.1b(3) if it failed."""
    pytest.importorskip("spherogram", reason="certified bounds need the 'bounds' extra")
    from rf_knots.lower_bounds import computed_claims
    from rf_knots.reference import free_reduce

    seed = torus_seed(7, SEED_SPEC)
    config = replace(SEEDED, crossing_changes=3)
    for index in range(6):
        instance = generate(config, seed=77, index=index, seed_representation=seed)
        claims = computed_claims(free_reduce(instance.word), instance.strands)
        computed = max((claim.value for claim in claims), default=0)
        assert computed >= instance.u_lower_inherited


def test_shard_writer_seeds_from_a_torus_knot_and_says_so(tmp_path, monkeypatch) -> None:
    """§4.1b end to end: the row inherits a bound the walk never computed."""
    from rf_knots.evidence import EvidenceStore

    writer = _shard_writer()
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "w",
            "--output-dir", str(output),
            "--count", "4", "--shard-size", "4",
            "-k", "3", "-l", "12",
            "--target-strands", "4", "--target-length", "28",
            "--max-len", "80", "--max-strands", "6",
            "--seed-family", "torus", "--seed-strength", "7",
            "--gordian-pairs",
            "--seed", "606",
        ],
    )
    writer.main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed_representation"]["name"] == "T(2,15)"
    assert manifest["seed_representation"]["certified_lower_bound"] == 7
    assert manifest["witness_provenance"] == "replay-complete"
    assert manifest["quarantine"]["screened"] is True
    assert "not computed" in manifest["quarantine"]["k_ball_closure"]
    assert manifest["quarantine_rejections"] == {}

    records = EvidenceStore(output / manifest["shards"][0]["path"]).records()
    assert len(records) == 4
    for record in records:
        assert record.witness.verify() == BraidState((), 1)
        assert record.metrics["seed_name"] == "T(2,15)"
        assert record.metrics["u_lower_inherited"] == max(
            0, 7 - record.metrics["d_upper"]
        )
        # `u <= d + 7` and `u >= 7 - d`: both sides come out of the same walk.
        assert record.certified_upper_bound == record.metrics["u_upper"]
        assert record.certified_lower_bound >= record.metrics["u_lower_inherited"]
        methods = {claim.method for claim in record.lower_bounds}
        assert "inherited-lipschitz" in methods

    intervals = [
        json.loads(line)
        for line in (output / "gordian-intervals.jsonl").read_text().splitlines()
    ]
    assert len(intervals) == 4
    assert all(row["right"]["word"] == [1] * 15 for row in intervals)
    assert all(row["lower"] <= row["upper"] <= 3 for row in intervals)


def test_shard_writer_seeds_from_a_positive_braid(tmp_path, monkeypatch) -> None:
    writer = _shard_writer()
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "w",
            "--output-dir", str(output),
            "--count", "3", "--shard-size", "3",
            "-k", "2", "-l", "10",
            "--target-strands", "4", "--target-length", "24",
            "--max-len", "80", "--max-strands", "6",
            "--seed-family", "positive-braid", "--seed-strength", "5",
            "--braid-seed-strands", "4",
            "--seed", "707",
        ],
    )
    writer.main()
    manifest = json.loads((output / "manifest.json").read_text())
    seed = manifest["seed_representation"]
    assert seed["family"] == "positive-braid"
    assert seed["certified_lower_bound"] == 5
    assert seed["u_upper"] == 5
    assert all(letter > 0 for letter in seed["word"])
    assert seed["lower_bounds"][0]["method"] == "positive-braid-genus"


def test_shard_writer_refuses_a_quarantined_seed(tmp_path, monkeypatch) -> None:
    """§4.1b moves the quarantine to seed selection; this is it firing."""
    writer = _shard_writer()

    def quarantined(_root):
        from rf_knots.interleaved_generator import (
            BRITTENHAM_HERMILLER_B5,
            Quarantine,
            torus_seed,
        )

        seed = torus_seed(7, ActionSpec(max_len=80, max_strands=6))
        del BRITTENHAM_HERMILLER_B5
        return Quarantine(words=frozenset({(seed.word, seed.strands)}))

    monkeypatch.setattr(writer, "load_quarantine", quarantined)
    monkeypatch.setattr(
        "sys.argv",
        [
            "w",
            "--output-dir", str(tmp_path / "shards"),
            "--count", "1", "--shard-size", "1",
            "-k", "1", "-l", "4",
            "--target-strands", "4", "--target-length", "20",
            "--max-len", "80", "--max-strands", "6",
            "--seed-family", "torus", "--seed-strength", "7",
        ],
    )
    with pytest.raises(SystemExit, match="quarantined"):
        writer.main()


def test_the_unseeded_shard_writer_is_unchanged(tmp_path, monkeypatch) -> None:
    """No new flag, no new behaviour: the §4.1 corpus is the same corpus."""
    from rf_knots.evidence import EvidenceStore

    writer = _shard_writer()
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        [
            "w",
            "--output-dir", str(output),
            "--count", "4", "--shard-size", "4",
            "-k", "2", "-l", "6",
            "--target-strands", "3", "--target-length", "10",
            "--max-len", "24", "--max-strands", "4",
            "--seed", "5150",
        ],
    )
    writer.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed_representation"]["family"] == "unknot"
    assert manifest["seed_representation"]["certified_lower_bound"] == 0
    for record in EvidenceStore(output / "shard-000000.jsonl").records():
        assert record.metrics["u_lower_inherited"] == 0
        assert record.metrics["u_upper"] == record.certified_upper_bound
        assert not any(
            claim.method == "inherited-lipschitz" for claim in record.lower_bounds
        )


# -- the M3 measurement ---------------------------------------------------------


def _measurement():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "measure_seeded_generator.py"
    spec = importlib.util.spec_from_file_location("measure_seeded_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_measurement_reports_both_bounds_and_never_only_one(
    tmp_path, monkeypatch
) -> None:
    """The whole point of the grid is the comparison, so both columns must exist."""
    pytest.importorskip("spherogram")
    measure = _measurement()
    output = tmp_path / "m3"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m",
            "--count", "5",
            "--strengths", "0", "7",
            "--depths", "3",
            "--families", "unknot", "torus",
            "-l", "12",
            "--target-strands", "4", "--target-length", "24",
            "--max-len", "80", "--max-strands", "6",
            "--output-dir", str(output),
            "--unknotdb", str(tmp_path / "absent"),
            "--seed", "31337",
        ],
    )
    measure.main()

    report = json.loads((output / "report.json").read_text())
    assert report["schema"] == measure.SCHEMA
    assert report["total_emitted"] == 10
    assert report["truncated"] == []
    assert report["seeds_rejected"] == []
    assert "not computed" in report["quarantine"]["k_ball_closure"]

    cells = {row["family"]: row for row in report["cells"]}
    assert set(cells) == {"unknot", "torus-2q"}
    assert cells["unknot"]["inherited_lower_bound"]["max"] == 0
    assert cells["torus-2q"]["inherited_lower_bound"]["min"] >= 4
    for row in report["cells"]:
        # The comparison the grid exists for: no instance may come back with a
        # computed bound below the one it inherited.
        assert row["achieved_minus_inherited"]["min"] >= 0
        assert row["replay_complete_fraction"] == 1.0
        assert set(row["acceptance"]) == {">=2", ">=3", ">=4", ">=5"}

    intervals = [
        json.loads(line)
        for line in (output / "gordian-intervals.jsonl").read_text().splitlines()
    ]
    assert len(intervals) == 10
    assert all(row["lower"] <= row["upper"] <= 3 for row in intervals)

    body = (output / "report.md").read_text()
    assert "## 0. Verdict" in body
    assert "k`-ball closure" in body


def test_the_measurement_projects_before_it_sweeps(tmp_path, monkeypatch, capsys) -> None:
    """`docs/lessons.md`: project the cost of a sweep before starting it."""
    measure = _measurement()
    monkeypatch.setattr(
        "sys.argv",
        [
            "m",
            "--project", "--project-count", "2", "--count", "1000",
            "--strengths", "0", "3",
            "--depths", "1",
            "--families", "unknot", "torus",
            "-l", "6",
            "--target-strands", "3", "--target-length", "12",
            "--max-len", "32", "--max-strands", "4",
            "--output-dir", str(tmp_path / "m3"),
            "--unknotdb", str(tmp_path / "absent"),
        ],
    )
    measure.main()
    projection = json.loads(capsys.readouterr().out)["projection"]
    assert projection["cells"] == 2
    assert projection["projected_instances"] == 2000
    assert projection["projected_seconds"] > 0
    assert not (tmp_path / "m3").exists()
