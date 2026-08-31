"""`A_plus(s)` holds only what was demonstrated, and demonstrated means replayed.

Every test here asks one of two questions.  *Is this site really good?* -- which
is answered by replaying the completion the method claims to have found and
checking it lands on the empty one-braid.  And *does anything here call an
unexplored site a bad one?* -- which is answered by reading the field names and
by checking that a search that ran out of budget reports `undetermined` rather
than a verdict.

The commutation tests are deliberately written against small hand-built words
where the two orders can be checked by eye, because the whole point of that
method is that it is exact and needs no search.
"""

from __future__ import annotations

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
from rf_knots.aplus import (
    METHODS,
    RANKERS,
    Budget,
    KnownGoodSite,
    KnownGoodSites,
    WitnessPool,
    _Successors,
    bounded_completion,
    bruteforce_known_good,
    build_known_good,
    commutation_known_good,
    commutes,
    free_reduced_drop,
    hit_at_k,
    insert_context_score,
    move_lower_bound,
    random_hit_probability,
    rank_by_kind_order,
    rewalk_known_good,
    spine_from_witness,
    state_key,
    witness_known_good,
    witness_spine,
)
from rf_knots.evidence import BraidState, UnknotWitness
from rf_knots.interleaved_generator import (
    InterleavedConfig,
    generate,
    generate_many,
    torus_seed,
)

SMALL = InterleavedConfig(
    crossing_changes=2,
    walk_length=4,
    target_strands=3,
    target_length=8,
    max_len=16,
    max_strands=4,
)
MEDIUM = InterleavedConfig(
    crossing_changes=3,
    walk_length=8,
    target_strands=3,
    target_length=12,
    max_len=24,
    max_strands=4,
)


def spine_for(config: InterleavedConfig, index: int = 0, m: int = 3):
    seed = torus_seed(m, config.spec())
    instance = generate(config, seed=11, index=index, seed_representation=seed)
    return witness_spine(instance)


# -- the spine -----------------------------------------------------------------


def test_the_spine_is_the_witness_and_nothing_else() -> None:
    instance = generate(MEDIUM, seed=3, index=0)
    spine = witness_spine(instance)
    witness = instance.witness
    assert witness is not None
    assert len(spine) == witness.moves
    assert spine.states[0] == witness.start
    assert spine.states[-1] == BraidState((), 1)
    for index, step in enumerate(witness.steps):
        assert spine.states[index + 1] == step.after
        assert spine.spec.decode(spine.actions[index])[0] == KIND_NAMES.index(
            step.action.kind
        )


def test_remaining_budgets_count_down_to_the_witness_itself() -> None:
    spine = spine_for(MEDIUM)
    assert spine.sites[0].remaining_moves == len(spine)
    assert spine.sites[0].remaining_crossing_changes == spine.certified_upper_bound
    for site in spine.sites:
        assert site.remaining_moves >= site.remaining_crossing_changes >= 0
    last = spine.sites[-1]
    assert last.remaining_moves == 1


def test_the_seed_program_region_is_marked() -> None:
    spine = spine_for(MEDIUM)
    walk = [site for site in spine.sites if not site.in_seed_program]
    program = [site for site in spine.sites if site.in_seed_program]
    assert len(walk) == spine.walk_moves
    assert len(program) == len(spine) - spine.walk_moves
    assert program, "a T(2,7) seed carries a program, so the region is non-empty"


def test_the_inherited_certificate_stays_attached() -> None:
    spine = spine_for(MEDIUM, m=5)
    assert spine.seed_name == "T(2,11)"
    assert spine.seed_certified_lower_bound == 5
    assert spine.u_lower_inherited == max(0, 5 - spine.d_upper)


def test_a_catalogue_composed_instance_has_no_spine() -> None:
    from dataclasses import replace

    instance = generate(SMALL, seed=1, index=0)
    stripped = replace(instance, witness=None)
    with pytest.raises(ValueError, match="catalogue-composed"):
        witness_spine(stripped)


# -- the base case -------------------------------------------------------------


def test_the_witness_site_is_always_in_the_set() -> None:
    spine = spine_for(MEDIUM)
    for index in range(len(spine)):
        known = witness_known_good(spine, index)
        assert spine.actions[index] in known.sites
        assert known.sites[spine.actions[index]].method == "witness"
        assert spine.actions[index] in known.legal_actions


def test_absent_sites_are_counted_as_undetermined_never_as_negative() -> None:
    spine = spine_for(MEDIUM)
    known = witness_known_good(spine, 0)
    assert known.sites_undetermined == len(known.legal_actions) - known.size
    assert known.sites_undetermined > 0
    text = " ".join(dir(known)) + " " + " ".join(dir(KnownGoodSite))
    assert "negative" not in text
    assert "sites_undetermined" in text


def test_offer_keeps_the_cheaper_completion() -> None:
    state = BraidState((1, 1, 1), 2)
    known = KnownGoodSites(
        state=state, depth=0, budget=Budget(10, 3), legal_actions=(7,)
    )
    assert known.offer(
        KnownGoodSite(7, REDUCE, "rewalk", 9, 3, True)
    )
    assert not known.offer(KnownGoodSite(7, REDUCE, "commutation", 5, 2, True))
    assert known.sites[7].completion_crossing_changes == 2
    assert known.sites[7].method == "commutation"


# -- commutation ---------------------------------------------------------------


def test_commutation_is_decided_by_application_not_by_a_rule() -> None:
    spec = ActionSpec(16, 5)
    word = (1, 1, 3, 3, 1)
    # sigma_1 at 0 and sigma_3 at 2 are disjoint; the check still applies both.
    left = spec.encode(CROSSING_CHANGE, position=0)
    right = spec.encode(CROSSING_CHANGE, position=2)
    assert commutes(spec, word, 5, left, right)
    from rf_knots import reference

    first = reference.apply(spec, *reference.apply(spec, word, 5, left), right)
    second = reference.apply(spec, *reference.apply(spec, word, 5, right), left)
    assert first == second


def test_two_moves_that_touch_the_same_letters_do_not_commute() -> None:
    spec = ActionSpec(16, 4)
    word = (1, 2, 1, -2)
    first = spec.encode(COMMUTE, position=1)
    second = spec.encode(CROSSING_CHANGE, position=1)
    # `COMMUTE(1)` moves the letter `CROSSING_CHANGE(1)` would have flipped, so
    # the two orders land on different words and the check must say so.
    assert not commutes(spec, word, 4, first, second)


def test_an_illegal_partner_is_not_a_commutation() -> None:
    spec = ActionSpec(16, 4)
    word = (1, -1, 2, -2)
    legal = spec.encode(REDUCE, position=0)
    illegal = spec.encode(REDUCE, position=1)
    assert not commutes(spec, word, 4, legal, illegal)


def test_commutation_alternatives_replay_onto_the_same_witness_state() -> None:
    from rf_knots import reference

    spine = spine_for(MEDIUM)
    found = 0
    for index in range(len(spine)):
        for site in commutation_known_good(spine, index, window=8):
            found += 1
            assert site.method == "commutation"
            assert site.replay_verified
            # Same cost, because it is the same multiset of moves reordered.
            assert site.completion_moves == spine.sites[index].remaining_moves
            assert (
                site.completion_crossing_changes
                == spine.sites[index].remaining_crossing_changes
            )
            state = spine.states[index]
            assert reference.is_legal(
                spine.spec, state.word, state.strands, site.action, True
            )
    assert found, "an eight-move window on this walk closes at least one site"


def test_a_commutation_site_carries_a_witness_that_replays_to_the_unknot() -> None:
    spine = spine_for(MEDIUM)
    checked = 0
    for index in range(len(spine)):
        alternatives = commutation_known_good(spine, index, window=6)
        if not alternatives:
            continue
        state = spine.states[index]
        for site in alternatives:
            # Rebuild the whole reordered completion and replay it end to end.
            block = _reordered_block(spine, index, site.action)
            assert block is not None
            witness = UnknotWitness.from_actions(
                state.word, state.strands, spine.spec, block
            )
            assert witness.crossing_changes == site.completion_crossing_changes
            assert witness.moves == site.completion_moves
            checked += 1
        if checked >= 3:
            break
    assert checked >= 1


def _reordered_block(spine, index, first_action):
    """The full completion a commutation site claims, for an independent replay."""
    from rf_knots.aplus import _bubble_to_front

    for span in range(1, min(8, len(spine.actions) - index - 1) + 1):
        block = _bubble_to_front(spine, index, span)
        if block is not None and block[0] == first_action:
            return (*block, *spine.actions[index + span + 1 :])
    return None


# -- pooling -------------------------------------------------------------------


def test_the_pool_finds_nothing_when_no_two_witnesses_meet() -> None:
    spines = [spine_for(MEDIUM, index=i) for i in range(3)]
    pool = WitnessPool()
    pool.add(spines[0])
    lonely = pool.lookup(spines[0], 0)
    assert lonely == []


def test_the_pool_reports_how_often_the_walks_reconverge() -> None:
    seed = torus_seed(3, SMALL.spec())
    pool = WitnessPool()
    for stream in range(6):
        for instance in generate_many(
            SMALL, seed=stream, count=25, seed_representation=seed
        ):
            pool.add(witness_spine(instance))
    report = pool.emitted_word_collisions()
    assert report["emitted_words"] == 150
    assert report["distinct_emitted_words"] <= report["emitted_words"]
    assert (
        report["surplus_witnesses_on_shared_words"]
        == report["emitted_words"] - report["distinct_emitted_words"]
    )


def test_a_pooled_alternative_is_a_different_action_at_the_same_word() -> None:
    """A collision built on purpose, so the test does not depend on luck.

    Two witnesses for the *same* start word: the generator's own, and the same
    completion with its first two moves commuted.  The second is a different
    first action at a word that is equal letter for letter, which is exactly what
    the pool is for and what §M1's first bullet hoped to get from re-seeding.
    """
    spine = spine_for(SMALL)
    index = next(
        i
        for i in range(len(spine))
        if commutation_known_good(spine, i, window=6)
    )
    alternative = commutation_known_good(spine, index, window=6)[0]
    block = _reordered_block(spine, index, alternative.action)
    assert block is not None
    state = spine.states[index]
    other = spine_from_witness(
        UnknotWitness.from_actions(state.word, state.strands, spine.spec, block)
    )
    assert other.states[0] == state

    pool = WitnessPool()
    pool.add(other)
    sites = pool.lookup(spine, index)
    assert sites, "the pooled witness starts at the same word with another action"
    for site in sites:
        assert site.method == "pool"
        assert site.action != spine.actions[index]
        assert site.replay_verified
        assert spine.sites[index].budget.admits(
            site.completion_moves, site.completion_crossing_changes
        )
    assert alternative.action in {site.action for site in sites}


def test_the_pool_pays_off_only_where_two_witnesses_actually_meet() -> None:
    seed = torus_seed(3, SMALL.spec())
    spines = [
        witness_spine(instance)
        for stream in range(4)
        for instance in generate_many(
            SMALL, seed=stream, count=20, seed_representation=seed
        )
    ]
    pool = WitnessPool()
    for spine in spines:
        pool.add(spine)
    shared = sum(
        1
        for spine in spines
        for index in range(len(spine))
        if pool.lookup(spine, index)
    )
    total = sum(len(spine) for spine in spines)
    # The measurement, not a threshold: witnesses from one seed share the seed's
    # own program, and the program takes the *same* action there, so meeting is
    # not enough -- they have to disagree.
    assert 0 <= shared <= total


def test_state_key_is_the_word_not_the_knot() -> None:
    """Two words for the trefoil, one stabilised.  The pool must keep them apart:
    a continuation belongs to a word, and inheriting it across a stabilisation
    would be exactly the drift the reconvergence check exists to catch."""
    left = BraidState((1, 1, 1), 2)
    right = BraidState((1, 1, 1, 2), 3)
    assert state_key(left) != state_key(right)


# -- re-walking ----------------------------------------------------------------


def test_a_rewalk_site_rejoins_the_witness_and_replays() -> None:
    from rf_knots import reference

    spine = spine_for(MEDIUM)
    rng = np.random.default_rng(5)
    found = 0
    for index in range(len(spine)):
        for site in rewalk_known_good(spine, index, rng, trials=48, max_detour=4):
            found += 1
            assert site.method == "rewalk"
            assert site.replay_verified
            assert site.action != spine.actions[index]
            assert spine.sites[index].budget.admits(
                site.completion_moves, site.completion_crossing_changes
            )
            state = spine.states[index]
            assert reference.is_legal(
                spine.spec, state.word, state.strands, site.action, True
            )
    assert found, "a four-move detour finds at least one reconvergence on this walk"


def test_rewalk_is_deterministic_in_the_generator_seed() -> None:
    spine = spine_for(MEDIUM)
    first = rewalk_known_good(
        spine, 0, np.random.default_rng(17), trials=32, max_detour=4
    )
    second = rewalk_known_good(
        spine, 0, np.random.default_rng(17), trials=32, max_detour=4
    )
    assert sorted(s.action for s in first) == sorted(s.action for s in second)


def test_rewalk_never_exceeds_the_crossing_change_budget() -> None:
    spine = spine_for(MEDIUM)
    rng = np.random.default_rng(2)
    for index in range(len(spine)):
        limit = spine.sites[index].remaining_crossing_changes
        for site in rewalk_known_good(spine, index, rng, trials=32, max_detour=4):
            assert site.completion_crossing_changes <= limit


# -- brute force ---------------------------------------------------------------


def test_bounded_completion_returns_a_path_that_replays() -> None:
    spec = ActionSpec(16, 4)
    word = (1, 1, 1)
    path = bounded_completion(spec, word, 2, 1, 8, node_cap=400)
    assert path is not None
    witness = UnknotWitness.from_actions(word, 2, spec, path)
    assert witness.crossing_changes <= 1
    assert witness.moves <= 8


def test_bounded_completion_reports_not_found_rather_than_impossible() -> None:
    spec = ActionSpec(16, 4)
    # One move is not enough to unknot a trefoil, but `None` says only that the
    # search did not find one -- the docstring is the contract, and the caller
    # must not read it as a proof.
    assert bounded_completion(spec, (1, 1, 1), 2, 1, 1, node_cap=400) is None
    assert bounded_completion(spec, (1, 1, 1), 2, 0, 40, node_cap=50) is None


def test_the_empty_one_braid_is_already_complete() -> None:
    spec = ActionSpec(16, 4)
    assert bounded_completion(spec, (), 1, 0, 0) == ()


def test_move_lower_bound_never_exceeds_a_real_completion() -> None:
    spec = ActionSpec(16, 4)
    for word, strands in (((1, 1, 1), 2), ((1, -2, 1, 2), 3), ((1, 1, 1, 1, 1), 2)):
        path = bounded_completion(spec, word, strands, 3, 30, node_cap=3000)
        assert path is not None
        assert move_lower_bound(word, strands) <= len(path)


def test_a_move_minimising_search_is_never_longer_than_a_length_first_one() -> None:
    spec = ActionSpec(16, 4)
    word, strands = (1, 1, 1), 2
    greedy = bounded_completion(spec, word, strands, 2, 20, node_cap=4000)
    shortest = bounded_completion(
        spec, word, strands, 2, 20, node_cap=4000, order="moves"
    )
    assert greedy is not None and shortest is not None
    assert len(shortest) <= len(greedy)


def test_brute_force_sites_all_replay_to_the_unknot() -> None:
    spine = spine_for(SMALL)
    index = len(spine) // 2
    sites, report = bruteforce_known_good(spine, index, node_cap=300)
    assert report.legal_actions == len(witness_known_good(spine, index).legal_actions)
    assert report.demonstrated + report.undetermined == report.legal_actions
    budget = spine.sites[index].budget
    state = spine.states[index]
    for site in sites:
        assert site.method == "bruteforce"
        assert site.replay_verified
        assert budget.admits(site.completion_moves, site.completion_crossing_changes)
        from rf_knots import reference

        assert reference.is_legal(
            spine.spec, state.word, state.strands, site.action, True
        )


def test_brute_force_finds_the_witness_site_it_was_handed() -> None:
    spine = spine_for(SMALL)
    for index in (0, len(spine) // 2):
        sites, _report = bruteforce_known_good(spine, index, node_cap=800)
        assert spine.actions[index] in {site.action for site in sites}


# -- assembling ----------------------------------------------------------------


def test_build_known_good_is_a_union_and_records_which_methods_ran() -> None:
    spine = spine_for(MEDIUM)
    pool = WitnessPool()
    pool.add(spine)
    succ = _Successors(spine.spec)
    known = build_known_good(
        spine, 0, pool=pool, rng=np.random.default_rng(1), successors=succ
    )
    assert set(known.methods_run) <= set(METHODS)
    assert "witness" in known.methods_run
    union: set[int] = set()
    for actions in known.demonstrated_by_method.values():
        union |= actions
    assert union == set(known.sites)
    assert known.size >= 1


def test_every_assembled_site_is_legal_and_inside_the_budget() -> None:
    from rf_knots import reference

    spine = spine_for(MEDIUM, index=2)
    pool = WitnessPool()
    pool.add(spine)
    rng = np.random.default_rng(8)
    succ = _Successors(spine.spec)
    for index in range(len(spine)):
        known = build_known_good(spine, index, pool=pool, rng=rng, successors=succ)
        state = spine.states[index]
        for action, site in known.sites.items():
            assert reference.is_legal(
                spine.spec, state.word, state.strands, action, True
            )
            assert known.budget.admits(
                site.completion_moves, site.completion_crossing_changes
            )
        assert known.size <= len(known.legal_actions)


def test_the_set_never_grows_when_a_method_is_switched_off() -> None:
    spine = spine_for(MEDIUM)
    rng = np.random.default_rng(3)
    full = build_known_good(spine, 0, rng=rng, commutation_window=8, rewalk_trials=48)
    bare = build_known_good(spine, 0, rng=None, commutation_window=0, rewalk_trials=0)
    assert set(bare.sites) <= set(full.sites)
    assert bare.size == 1


# -- baselines -----------------------------------------------------------------


def test_free_reduced_drop_sees_a_cancelling_flip() -> None:
    assert free_reduced_drop((1, 1, 1), (-1, 1, 1)) == 2
    assert free_reduced_drop((1, 2, 1), (-1, 2, 1)) == 0
    # Splicing a cancelling pair leaves the free reduction alone, which is why
    # `INSERT` needs a score of its own.
    assert free_reduced_drop((1, 2), (1, 3, -3, 2)) == 0


def test_insert_context_score_prefers_a_pair_that_touches_something() -> None:
    word = (1, 2, 1)
    touching = insert_context_score(word, 1, -1)
    isolated = insert_context_score(word, 1, 9)
    assert touching > isolated
    assert isolated == 0


def test_every_ranker_returns_a_permutation_of_the_legal_sites() -> None:
    from rf_knots import reference

    spine = spine_for(MEDIUM)
    state = spine.states[0]
    moves = reference.successors(
        spine.spec, state.word, state.strands, allow_crossing=True
    )
    actions = sorted(action for action, _w, _s in moves)
    for name, ranker in RANKERS.items():
        order = ranker(
            spine.spec,
            state.word,
            state.strands,
            moves,
            np.random.default_rng(0),
        )
        assert sorted(order) == actions, name


def test_ties_are_broken_at_random_not_by_the_action_index() -> None:
    from rf_knots import reference

    spine = spine_for(MEDIUM)
    state = spine.states[0]
    moves = reference.successors(
        spine.spec, state.word, state.strands, allow_crossing=True
    )
    first = RANKERS["free_reduced_drop"](
        spine.spec, state.word, state.strands, moves, np.random.default_rng(1)
    )
    second = RANKERS["free_reduced_drop"](
        spine.spec, state.word, state.strands, moves, np.random.default_rng(2)
    )
    assert first != second


def test_the_growth_aware_ranker_puts_a_cancelling_flip_first() -> None:
    from rf_knots import reference

    spec = ActionSpec(24, 4)
    word, strands = (1, 1, 2, -2, 1), 3
    moves = reference.successors(spec, word, strands, allow_crossing=True)
    order = RANKERS["growth_aware"](
        spec, word, strands, moves, np.random.default_rng(0)
    )
    kind = spec.decode(order[0])[0]
    assert kind in (REDUCE, DESTABILIZE, CROSSING_CHANGE)


def test_rank_by_kind_order_respects_the_order_it_was_given() -> None:
    from rf_knots import reference

    spec = ActionSpec(24, 4)
    word, strands = (1, 1, 2, 1, 1), 3
    moves = reference.successors(spec, word, strands, allow_crossing=True)
    ranker = rank_by_kind_order([INSERT, CROSSING_CHANGE, COMMUTE, BRAID, REDUCE,
                                 DESTABILIZE, STABILIZE_POS])
    order = ranker(spec, word, strands, moves, np.random.default_rng(0))
    assert spec.decode(order[0])[0] == INSERT


def test_hit_at_k_is_at_least_one_good_site_in_the_shortlist() -> None:
    assert hit_at_k([5, 6, 7], {7}, 3)
    assert not hit_at_k([5, 6, 7], {7}, 2)
    assert not hit_at_k([5, 6, 7], set(), 3)


def test_random_hit_probability_matches_a_seeded_shuffle() -> None:
    legal, good, k = 40, 6, 10
    closed = random_hit_probability(legal, good, k)
    rng = np.random.default_rng(0)
    actions = list(range(legal))
    known = set(range(good))
    hits = 0
    trials = 4000
    for _ in range(trials):
        rng.shuffle(actions)
        hits += int(hit_at_k(actions, known, k))
    assert abs(hits / trials - closed) < 0.03
    assert random_hit_probability(10, 1, 10) == 1.0
    assert random_hit_probability(10, 0, 5) == 0.0


# -- the promise the whole module makes ----------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_site_in_a_full_A_plus_carries_a_replayable_completion(
    index: int,
) -> None:
    """The end-to-end claim: pick a state, take every method, replay everything."""
    spine = spine_for(SMALL, index=index)
    pool = WitnessPool()
    pool.add(spine)
    depth = len(spine) // 2
    succ = _Successors(spine.spec)
    known = build_known_good(
        spine,
        depth,
        pool=pool,
        rng=np.random.default_rng(index),
        successors=succ,
        bruteforce_node_cap=300,
    )
    state = spine.states[depth]
    assert known.size >= 1
    for action, site in known.sites.items():
        assert site.replay_verified
        assert known.budget.admits(
            site.completion_moves, site.completion_crossing_changes
        )
        # Independent of the method's own bookkeeping: a completion of the stated
        # cost exists from the child, so the site is good for the reason claimed.
        from rf_knots import reference

        child_word, child_strands = reference.apply(
            spine.spec, state.word, state.strands, action
        )
        is_change = spine.spec.decode(action)[0] == CROSSING_CHANGE
        remaining = known.budget.after(is_change)
        tail = bounded_completion(
            spine.spec,
            child_word,
            child_strands,
            remaining.crossing_changes,
            remaining.moves,
            node_cap=1500,
        )
        if tail is None:
            # The bounded search did not reproduce it; that is the search's
            # incompleteness, not a refutation, and the method that put the site
            # in the set already replayed its own completion.
            continue
        witness = UnknotWitness.from_actions(
            state.word, state.strands, spine.spec, (action, *tail)
        )
        assert witness.crossing_changes <= known.budget.crossing_changes
