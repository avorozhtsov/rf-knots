"""The M1 gate's harness, checked against independent oracles.

The point of note 22 §M1's revised gate is that the *harness* is what fell last
time, so these tests are about the harness rather than about any ranker:

* `move_bound` is admissible and consistent -- if it is not, A* may test the goal
  on pop and return a completion that is not shortest, and the whole target is
  silently wrong;
* `shortest_completion`'s optimal-first-action set agrees, site for site, with a
  layer-synchronous breadth-first oracle written a different way;
* every site the target claims replays through `UnknotWitness.from_actions` at
  exactly the claimed length;
* the `INSERT`-block restriction is real -- `kind_prior` is provably constant
  inside it, which is why it is the gate's baseline;
* ties are broken by the harness, so no ranker can earn credit from the order
  `reference.successors` happens to emit;
* the identity split never puts one identity on both sides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))

import m1_head
import m1_target as target

from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, INSERT, ActionSpec
from rf_knots.aplus import RANKERS, witness_spine
from rf_knots.evidence import UnknotWitness
from rf_knots.interleaved_generator import (
    UNKNOT_SEED,
    InterleavedConfig,
    generate_many,
    torus_seed,
)

# -- oracles -------------------------------------------------------------------


def bfs_shortest(spec, word, strands, crossing_change_budget, growth=3, max_depth=9):
    """Layer-synchronous BFS: the shortest length and every optimal first action.

    Deliberately written a different way from `shortest_completion` -- no
    heuristic, no priority queue, no bitmask -- so that agreement between the two
    is evidence rather than a tautology.
    """
    length_cap = min(len(word) + growth, spec.max_len)
    if not word and strands == 1:
        return 0, frozenset()
    frontier = {(word, strands, 0): frozenset()}
    seen = {(word, strands, 0)}
    for depth in range(max_depth):
        nxt: dict[tuple, frozenset] = {}
        for (current, active, used), firsts in frontier.items():
            for action, next_word, next_strands in reference.successors(
                spec, current, active, allow_crossing=True
            ):
                if len(next_word) > length_cap:
                    continue
                changes = used + int(spec.decode(action)[0] == CROSSING_CHANGE)
                if changes > crossing_change_budget:
                    continue
                key = (next_word, next_strands, changes)
                if key in seen:
                    continue
                carried = frozenset({action}) if depth == 0 else firsts
                nxt[key] = nxt.get(key, frozenset()) | carried
        goals = [value for key, value in nxt.items() if not key[0] and key[1] == 1]
        if goals:
            return depth + 1, frozenset().union(*goals)
        seen |= set(nxt)
        frontier = nxt
    return None, frozenset()


def small_states(count=4, depths=4):
    """A handful of real generated states small enough for the BFS oracle."""
    config = InterleavedConfig(
        crossing_changes=2, walk_length=3, target_strands=3,
        target_length=6, max_len=14, max_strands=4,
    )
    spec = config.spec()
    out = []
    for instance in generate_many(config, seed=5, count=count):
        spine = witness_spine(instance)
        total = len(spine)
        picked = sorted({int(round(f * (total - 1))) for f in np.linspace(0, 0.8, depths)})
        for index in picked:
            out.append((spec, spine, index))
    return out


# -- the heuristic -------------------------------------------------------------


def test_move_bound_dominates_the_repository_bound():
    from rf_knots.aplus import move_lower_bound

    for length in range(0, 30):
        for strands in range(1, 9):
            assert target.move_bound(length, strands) >= move_lower_bound(
                (0,) * length, strands
            )


def test_move_bound_is_consistent_across_every_legal_move():
    spec = ActionSpec(16, 5)
    seen = 0
    for spec_, spine, index in small_states(count=3, depths=5):
        state = spine.states[index]
        here = target.move_bound(len(state.word), state.strands)
        for _action, next_word, next_strands in reference.successors(
            spec_, state.word, state.strands, allow_crossing=True
        ):
            there = target.move_bound(len(next_word), next_strands)
            assert abs(here - there) <= 1
            seen += 1
    assert seen > 100
    assert target.move_bound(0, 1) == 0
    assert spec.max_len == 16


def test_move_bound_never_exceeds_a_real_completion():
    """Admissibility, checked against witnesses that actually exist."""
    for _spec, spine, index in small_states(count=4, depths=5):
        state = spine.states[index]
        bound = target.move_bound(len(state.word), state.strands)
        assert bound <= spine.sites[index].remaining_moves


# -- the target ----------------------------------------------------------------


def test_shortest_completion_agrees_with_a_breadth_first_oracle():
    checked = 0
    for spec, spine, index in small_states(count=4, depths=4):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, growth=3, node_cap=200_000,
        )
        moves, firsts = bfs_shortest(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, growth=3, max_depth=8,
        )
        if moves is None:
            continue
        assert found.moves == moves, (site.state.word, site.state.strands)
        assert set(found.first_actions) == set(firsts)
        checked += 1
    assert checked >= 8, "the oracle comparison never ran on enough states"


def test_target_sites_are_legal_and_never_more_than_the_legal_set():
    for spec, spine, index in small_states(count=2, depths=3):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        assert set(found.first_actions) <= set(found.legal_actions)
        for action in found.first_actions:
            assert reference.is_legal(
                spec, site.state.word, site.state.strands, action, True
            )


def test_every_claimed_site_replays_at_exactly_the_claimed_length():
    replayed = 0
    for spec, spine, index in small_states(count=3, depths=4):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None:
            continue
        exhibited = target.exhibit_target(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, found,
        )
        assert set(exhibited) == set(found.first_actions)
        for action, moves in exhibited.items():
            assert moves == found.moves
            assert action in found.legal_actions
        replayed += len(exhibited)
    assert replayed > 10


def test_the_shortest_completion_never_exceeds_the_witness_that_exists():
    """A witness is a completion, so `m*` can only be shorter -- or the cap bit."""
    for spec, spine, index in small_states(count=3, depths=4):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, growth=8, node_cap=200_000,
        )
        if found.moves is None:
            continue
        assert found.moves <= site.remaining_moves


def test_a_cap_hit_is_undetermined_and_never_an_empty_target():
    spec, spine, index = small_states(count=1, depths=2)[0]
    site = spine.sites[index]
    found = target.shortest_completion(
        spec, site.state.word, site.state.strands,
        site.remaining_crossing_changes, node_cap=1,
    )
    assert found.moves is None
    assert found.first_actions == ()
    assert found.reason == "node-cap"
    assert found.complete is False


def test_the_target_is_far_tighter_than_feasibility():
    """The measured reason the old gate fell, reproduced as an assertion."""
    tight = loose = states = 0
    for spec, spine, index in small_states(count=3, depths=4):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None:
            continue
        slack = target.shortest_completion  # documented alias, not called
        assert slack is target.shortest_completion
        tight += len(found.first_actions)
        loose += len(found.legal_actions)
        states += 1
    assert states >= 6
    assert tight / loose < 0.35, "the shortest target is not tight; the gate is vacuous"


# -- the ranker interface ------------------------------------------------------


def _gate_state(spec, spine, index):
    site = spine.sites[index]
    return target.GateState(
        spec=spec, word=site.state.word, strands=site.state.strands,
        remaining_crossing_changes=site.remaining_crossing_changes,
        remaining_moves=site.remaining_moves, config="test", seed_name="0_1",
        instance_id=spine.instance_id, identity="k", depth=index,
        region="seed-program" if site.in_seed_program else "walk",
    )


def test_kind_prior_is_constant_inside_the_insert_block():
    """Why `kind_prior` is the gate's baseline: there it *is* random ordering."""
    for spec, spine, index in small_states(count=2, depths=3):
        state = _gate_state(spec, spine, index)
        sites = target.sites_of(spec, state.word, state.strands)
        inserts = [site for site in sites if site.kind == INSERT]
        if len(inserts) < 2:
            continue
        scores = target.kind_prior(state, inserts)
        assert len(set(scores)) == 1


def test_ties_are_broken_by_the_harness_not_by_the_action_index():
    scores = [0.0] * 40
    first = {target.order_by_scores(scores, np.random.default_rng(seed))[0]
             for seed in range(60)}
    assert len(first) > 20, "an all-equal score is collapsing onto one index"
    ordered = target.order_by_scores([3.0, 1.0, 2.0], np.random.default_rng(0))
    assert ordered == [0, 2, 1]


def test_the_baselines_reproduce_the_aplus_orderings_where_they_are_decided():
    """Same heuristics, one interface.  Only the tie-breaking differs."""
    spec, spine, index = small_states(count=1, depths=2)[0]
    state = _gate_state(spec, spine, index)
    sites = target.sites_of(spec, state.word, state.strands)
    moves = [(site.action, site.word, site.strands) for site in sites]
    for name in ("kind_prior", "free_reduced_drop", "growth_aware"):
        scores = dict(
            zip(
                [site.action for site in sites],
                target.BASELINES[name](state, sites),
                strict=True,
            )
        )
        order = RANKERS[name](
            spec, state.word, state.strands, moves, np.random.default_rng(3)
        )
        for earlier, later in zip(order, order[1:], strict=False):
            if scores[earlier] != scores[later]:
                assert scores[earlier] > scores[later], name


def test_uniform_random_matches_the_closed_form_chance():
    rng = np.random.default_rng(11)
    total, positives, trials = 30, 4, 4000
    hits = 0
    for _ in range(trials):
        order = target.order_by_scores([0.0] * total, rng)
        hits += target.recall_at(order, set(range(positives)), 5)
    assert abs(hits / trials - target.chance_at(total, positives, 5)) < 0.03


def test_chance_at_edges():
    assert target.chance_at(10, 0, 3) == 0.0
    assert target.chance_at(10, 10, 3) == 1.0
    assert target.chance_at(3, 1, 5) == 1.0
    assert 0.0 < target.chance_at(100, 1, 1) < 0.02


# -- scoring -------------------------------------------------------------------


def _scored(spec, spine, index, rankers):
    site = spine.sites[index]
    state = _gate_state(spec, spine, index)
    sites = target.sites_of(spec, state.word, state.strands)
    found = target.shortest_completion(
        spec, state.word, state.strands, site.remaining_crossing_changes,
        node_cap=100_000,
    )
    if found.moves is None:
        return None
    return target.score_state(
        state, sites, found.first_actions, found.moves, rankers,
        np.random.default_rng(1),
    )


def test_a_perfect_ranker_scores_one_and_an_adversary_scores_zero():
    for spec, spine, index in small_states(count=2, depths=3):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None or not found.first_actions:
            continue
        good = set(found.first_actions)

        def oracle(state, sites, good=good):
            return [1.0 if s.action in good else 0.0 for s in sites]

        def adversary(state, sites, good=good):
            return [-1.0 if s.action in good else 0.0 for s in sites]

        row = _scored(spec, spine, index, {"oracle": oracle, "adversary": adversary})
        assert row is not None
        assert row.hits["oracle"]["all@1"] == 1
        assert row.hits["adversary"]["all@1"] == 0
        if row.target_insert and row.insert_legal > len(row.target_insert):
            assert row.hits["oracle"]["insert@1"] == 1
            assert row.hits["adversary"]["insert@1"] == 0


def test_the_shortest_completion_target_contains_no_insert():
    """The finding the gate rests on: `INSERT` is never optimal in this move set.

    `moves = (L + n - 1)/2 + 2i + 2s + c` exactly, so an insertion costs two
    moves nothing pays back.  If this test ever fails the identity has been
    broken by an edit to the move set, and §M1's `INSERT` gate may be live again.
    """
    inserts = states = 0
    for spec, spine, index in small_states(count=4, depths=5):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None:
            continue
        states += 1
        inserts += sum(spec.decode(a)[0] == INSERT for a in found.first_actions)
    assert states >= 8
    assert inserts == 0


def test_the_move_count_identity_holds_on_real_witness_suffixes():
    from rf_knots.actions import BRAID, COMMUTE, STABILIZE_NEG, STABILIZE_POS

    checked = 0
    for spec, spine, index in small_states(count=3, depths=5):
        actions = spine.actions[index:]
        state = spine.states[index]
        i = sum(spec.decode(a)[0] == INSERT for a in actions)
        s = sum(spec.decode(a)[0] in (STABILIZE_POS, STABILIZE_NEG) for a in actions)
        c = sum(
            spec.decode(a)[0] in (COMMUTE, BRAID, CROSSING_CHANGE) for a in actions
        )
        predicted = (len(state.word) + state.strands - 1) / 2 + 2 * i + 2 * s + c
        assert predicted == len(actions)
        checked += 1
    assert checked >= 10


def test_every_insert_site_has_an_exhibited_completion_at_shortest_plus_two():
    """Why the block is flat, shown rather than argued.  No search is used."""
    exhibited = 0
    for spec, spine, index in small_states(count=3, depths=4):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None:
            continue
        path = target.completion_path(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, found.moves,
            min(len(site.state.word) + found.growth, spec.max_len),
        )
        if path is None:
            continue
        for action in found.legal_actions:
            if spec.decode(action)[0] != INSERT:
                continue
            actions = target.undo_completion(
                spec, site.state.word, site.state.strands, action, path
            )
            witness = UnknotWitness.from_actions(
                site.state.word, site.state.strands, spec, actions
            )
            assert witness.moves == found.moves + 2
            exhibited += 1
    assert exhibited > 50


def test_the_committed_block_target_is_still_flat():
    """Forbidding the undo does not rescue the block: it ties at `m* + 2`."""
    flat = measured = 0
    for spec, spine, index in small_states(count=2, depths=3):
        site = spine.sites[index]
        found = target.shortest_completion(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, node_cap=100_000,
        )
        if found.moves is None:
            continue
        block = [a for a in found.legal_actions if spec.decode(a)[0] == INSERT]
        if len(block) < 2:
            continue
        result = target.block_target(
            spec, site.state.word, site.state.strands,
            site.remaining_crossing_changes, block, found.moves,
            node_cap=60_000, exhibit=False,
        )
        if result.best_moves is None:
            continue
        measured += 1
        assert result.best_moves >= found.moves + 2
        flat += int(len(result.target) == len(block))
    assert measured >= 3
    assert flat >= 1, "the committed block split everywhere; re-read the identity"


def test_undo_completion_refuses_a_non_insert():
    spec = ActionSpec(16, 4)
    with pytest.raises(ValueError, match="only an INSERT"):
        target.undo_completion(spec, (1, -1), 2, spec.encode(CROSSING_CHANGE), ())


def test_summarise_and_bootstrap_agree_with_the_rows():
    rows = []
    for spec, spine, index in small_states(count=3, depths=4):
        row = _scored(spec, spine, index, dict(target.BASELINES))
        if row is not None:
            rows.append(row)
    assert rows
    table = target.summarise(rows, "all@1", ["kind_prior", "uniform_random"])
    assert table["states"] == len(rows)
    manual = np.mean([row.hits["kind_prior"]["all@1"] for row in rows])
    assert abs(table["kind_prior"] - manual) < 1e-6
    interval = target.cluster_bootstrap(
        rows, "all@1", "kind_prior", "uniform_random", draws=200
    )
    assert interval["ci95_low"] <= interval["observed_delta"] <= interval["ci95_high"]


def test_summarise_on_no_rows_says_nothing_rather_than_zero():
    assert target.summarise([], "all@1", ["kind_prior"]) == {"states": 0}
    assert target.cluster_bootstrap([], "all@1", "a", "b") == {"draws": 0}


# -- the split -----------------------------------------------------------------


def _instance_payload(config, seeds, count=6, seed=3):
    out = []
    for representation in seeds:
        for instance in generate_many(
            config, seed=seed, count=count, seed_representation=representation
        ):
            spine = witness_spine(instance)
            out.append(
                (
                    spine.instance_id,
                    instance.word,
                    instance.strands,
                    [(state.word, state.strands) for state in spine.states],
                )
            )
    return out


def test_identity_key_is_an_invariant_of_the_closed_braid():
    """Conjugation and Markov stabilisation must not change the key."""
    word = (1, 2, -1, 2)
    base = target.identity_key(word, 3)
    rotated = word[1:] + word[:1]
    assert target.identity_key(rotated, 3) == base
    assert target.identity_key(word + (3,), 4) == base


def test_the_split_never_puts_one_identity_on_both_sides():
    config = InterleavedConfig(
        crossing_changes=2, walk_length=3, target_strands=3,
        target_length=6, max_len=14, max_strands=4,
    )
    spec = config.spec()
    payload = _instance_payload(config, [UNKNOT_SEED, torus_seed(1, spec)])
    assignment, report = assign = target.assign_split(payload, 0.4, seed=7)
    assert report.train_instances + report.eval_instances == len(payload)
    sides: dict[str, set[str]] = {}
    for instance_id, word, strands, _states in payload:
        sides.setdefault(target.identity_key(word, strands), set()).add(
            assignment[instance_id]
        )
    for key, values in sides.items():
        assert len(values) == 1, f"identity {key} straddles the split"
    assert assign[1] is report


def test_the_split_counts_shared_representations_without_merging_on_them():
    """Merging on a shared state collapses the corpus; the count is reported."""
    shared = ((1, -1, 2), 3)
    payload = [
        ("a", (1, 2, -1, 2), 3, [shared, ((), 1)]),
        ("b", (1, 2, -1, 2, 3, -3), 4, [shared, ((), 1)]),
    ]
    assignment, report = target.assign_split(payload, 0.5, seed=1)
    assert report.merged_by_state_collision >= 1
    assert set(assignment.values()) <= {"train", "eval"}
    assert report.largest_group >= 1


def test_every_witness_shares_the_unknot_so_merging_would_collapse():
    config = InterleavedConfig(
        crossing_changes=2, walk_length=3, target_strands=3,
        target_length=6, max_len=14, max_strands=4,
    )
    payload = _instance_payload(config, [UNKNOT_SEED], count=6)
    tails = [states[-1] for _id, _w, _s, states in payload]
    assert len(set(tails)) == 1, "the shared tail this split has to survive"
    _assignment, report = target.assign_split(payload, 0.4, seed=2)
    assert report.identities >= 1
    assert report.train_instances + report.eval_instances == len(payload)


def test_the_split_report_states_its_own_limits():
    payload = [("a", (1, 2, -1, 2), 3, [((1,), 2)])]
    _assignment, report = target.assign_split(payload, 0.5, seed=1)
    text = report.to_dict()
    assert text["guarantees"] and text["not_guaranteed"]
    joined = " ".join(text["not_guaranteed"])
    assert "unknot" in joined and "coarser" in joined


# -- the head adapter ----------------------------------------------------------


def test_the_head_adapter_refuses_rather_than_guessing():
    probe = m1_head.status()
    assert probe.to_dict()["ready"] == probe.ready
    if not probe.ready:
        with pytest.raises(RuntimeError, match="not available yet"):
            m1_head.build_ranker(None)


def test_the_head_adapter_names_the_four_site_fields():
    assert m1_head.SITE_FIELDS == ("kind", "position", "generator", "sign")


def test_the_head_adapter_never_takes_the_encoder_default_geometry():
    """§6.4: the constructor default is `legacy`, which is the wrong geometry."""
    import inspect

    from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder

    parameters = inspect.signature(ScalableBraidEmbeddingEncoder.__init__).parameters
    assert parameters["positional_geometry"].default == "legacy"
    assert m1_head.GEOMETRY == "necklace"
    assert m1_head.CYCLIC_BAND_GENERATORS is False
    assert m1_head.status().to_dict()["positional_geometry"] == "necklace"


def test_the_head_adapter_scores_every_site_once():
    probe = m1_head.status()
    if not probe.ready:
        pytest.skip(probe.detail)
    rank = m1_head.build_ranker(None)
    spec, spine, index = small_states(count=1, depths=2)[0]
    state = _gate_state(spec, spine, index)
    sites = target.sites_of(spec, state.word, state.strands)
    scores = rank(state, sites)
    assert len(scores) == len(sites)
    assert all(np.isfinite(value) for value in scores)
    inserts = [site for site in sites if site.kind == INSERT]
    assert len(rank(state, inserts)) == len(inserts)


def test_a_checkpoint_from_another_geometry_is_refused(tmp_path):
    probe = m1_head.status()
    if not probe.ready:
        pytest.skip(probe.detail)
    import torch

    path = tmp_path / "legacy.pt"
    torch.save({"positional_geometry": "legacy"}, path)
    with pytest.raises(RuntimeError, match="trained under positional_geometry"):
        m1_head.build_ranker(str(path), geometry="necklace")


def test_site_features_rejects_a_row_that_cannot_reproduce_its_action():
    spec, spine, index = small_states(count=1, depths=2)[0]
    state = spine.states[index]
    sites = target.sites_of(spec, state.word, state.strands)
    rows = m1_head.site_features(spec, sites)
    assert all(len(row) == 4 for row in rows)

    class Broken:
        def __init__(self, site):
            self.kind, self.position = site.kind, site.position
            self.generator, self.sign = site.generator, -site.sign
            self.action = site.action

    inserts = [s for s in sites if s.kind == INSERT]
    if not inserts:
        pytest.skip("no INSERT at this state")
    with pytest.raises(AssertionError, match="encodes to"):
        m1_head.site_features(spec, [Broken(inserts[0])])


# -- the driver ----------------------------------------------------------------


def test_the_driver_renders_a_report_from_a_tiny_sweep(tmp_path):
    import m1_gate

    argv = sys.argv
    sys.argv = [
        "m1_gate.py", "--out", str(tmp_path), "--configs", "C-small",
        "--states-per-instance", "3", "--bootstrap", "50", "--seconds", "240",
    ]
    try:
        m1_gate.main()
    finally:
        sys.argv = argv
    payload = __import__("json").loads((tmp_path / "report.json").read_text())
    assert payload["schema"] == "rf-knots-m1-revised-gate-v0"
    assert payload["gate"]["baseline"] == "kind_prior"
    assert payload["evidence_tiers"]["tier3_attested"]["consulted"] is False
    assert payload["evidence_tiers"]["tier2_exact_closure"]["merged_into_the_gate"] is False
    text = (tmp_path / "report.md").read_text()
    assert "whole state, recall@1 -- the gate" in text
    assert "Not guaranteed." in text
    assert "moves = (L + n - 1) / 2 + 2i + 2s + c" in text
    assert payload["insert_block_is_not_a_gate"][
        "insert_sites_in_the_shortest_completion_target"
    ] == 0


def test_the_driver_scores_the_insert_block_only_on_states_that_have_one():
    import m1_gate

    class Row:
        def __init__(self, insert_legal, target_insert):
            self.insert_legal = insert_legal
            self.target_insert = target_insert

    rows = [Row(0, ()), Row(5, ()), Row(1, (7,)), Row(5, (7,))]
    kept = m1_gate.insert_block_rows(rows)
    assert kept == [rows[3]]


def test_the_transfer_config_reaches_a_strand_count_training_cannot():
    import m1_gate

    assert m1_gate.CONFIGS["W-wide"]["role"] == "transfer"
    assert m1_gate.CONFIGS["W-wide"]["config"].target_strands > (
        m1_gate.CONFIGS["A-main"]["config"].max_strands
    )
    for name, entry in m1_gate.CONFIGS.items():
        if entry["role"] == "train-family":
            assert entry["config"].max_strands <= 4, name


def test_witness_replay_is_the_standard_the_target_is_held_to():
    """`exhibit_target` must go through `UnknotWitness`, not a private replay."""
    spec, spine, index = small_states(count=1, depths=2)[0]
    site = spine.sites[index]
    found = target.shortest_completion(
        spec, site.state.word, site.state.strands,
        site.remaining_crossing_changes, node_cap=100_000,
    )
    if found.moves is None:
        pytest.skip("no completion under the cap")
    path = target.completion_path(
        spec, site.state.word, site.state.strands,
        site.remaining_crossing_changes, found.moves,
        min(len(site.state.word) + found.growth, spec.max_len),
    )
    assert path is not None and len(path) == found.moves
    witness = UnknotWitness.from_actions(
        site.state.word, site.state.strands, spec, path
    )
    assert witness.moves == found.moves
    assert witness.crossing_changes <= site.remaining_crossing_changes


# -- the site tensor's four columns --------------------------------------------


def test_a_site_row_round_trips_to_its_own_flat_action_index():
    """`INSERT(p, g, +1)` and `INSERT(p, g, -1)` are different actions."""
    for spec, spine, index in small_states(count=2, depths=3):
        state = spine.states[index]
        sites = target.sites_of(spec, state.word, state.strands)
        for site in sites:
            assert spec.encode(
                site.kind, position=site.position,
                generator=site.generator, sign=site.sign,
            ) == site.action
        rows = m1_head.site_features(spec, sites)
        assert all(len(row) == 4 for row in rows)
        assert {row[3] for row in rows} <= {1, -1}


def test_dropping_the_sign_column_would_merge_distinct_insert_sites():
    """The failure a three-column site tensor would cause, made explicit."""
    spec = ActionSpec(16, 5)
    positive = spec.encode(INSERT, position=2, generator=3, sign=1)
    negative = spec.encode(INSERT, position=2, generator=3, sign=-1)
    assert positive != negative
    assert spec.decode(positive)[:3] == spec.decode(negative)[:3]
    assert spec.decode(positive)[3] == 1 and spec.decode(negative)[3] == -1


def test_the_harness_and_aplus_agree_on_what_a_site_is():
    """`aplus.py` stores flat action indices; the gate's sites must be the same."""
    from rf_knots.aplus import _Successors, build_known_good

    for spec, spine, index in small_states(count=2, depths=3):
        state = spine.states[index]
        succ = _Successors(spec)
        known = build_known_good(spine, index, rewalk_trials=0, successors=succ)
        sites = target.sites_of(spec, state.word, state.strands, succ)
        assert {site.action for site in sites} == set(known.legal_actions)
        assert set(known.sites) <= {site.action for site in sites}


def test_sites_of_rejects_a_spec_that_cannot_round_trip():
    spec, spine, index = small_states(count=1, depths=2)[0]
    state = spine.states[index]

    class Broken:
        max_len = spec.max_len
        max_strands = spec.max_strands
        cyclic_band_generators = spec.cyclic_band_generators

        def decode(self, action):
            return spec.decode(action)

        def encode(self, kind, position=0, generator=1, sign=1):
            return spec.encode(kind, position=position, generator=generator, sign=1)

        def describe(self, action):
            return spec.describe(action)

    def successors(word, strands):
        return reference.successors(spec, word, strands, allow_crossing=True)

    inserts = any(
        spec.decode(action)[0] == INSERT and spec.decode(action)[3] < 0
        for action, _w, _s in successors(state.word, state.strands)
    )
    if not inserts:
        pytest.skip("no negative-sign INSERT at this state")
    with pytest.raises(AssertionError, match="re-encodes"):
        target.sites_of(Broken(), state.word, state.strands, successors)
