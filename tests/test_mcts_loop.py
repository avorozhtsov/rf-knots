"""Tests for the first MCTS training loop.

The load-bearing one is `test_one_state_dict_runs_the_loop_at_two_sizes`: the whole
architecture line exists so that a single set of weights searches a `(16, 4)` braid
and a `(512, 24)` braid, and note 22 §M1 asks for that to be *asserted* rather than
assumed.  Nothing else here will fail if that property regresses silently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, ActionSpec
from rf_knots.embedding_encoder import ScalableBraidEmbeddingEncoder
from rf_knots.evidence import UnknotWitness

# A genuine 3-braid with `u = 1` and a 4-braid, both small enough to search in a test.
TREFOIL = ((1, 1, 1), 2)
FIGURE_EIGHT = ((1, -2, 1, -2), 3)


def test_transposition_preserves_shorter_and_incomparable_paths():
    state = M.SearchState((1, 1, 1), 2, 1, 10)
    seen = {}
    assert M.remember_cost(seen, state)
    assert M.remember_cost(seen, M.SearchState(state.word, 2, 1, 5))
    assert seen[state.key] == [(1, 5)]
    assert M.remember_cost(seen, M.SearchState(state.word, 2, 0, 8))
    assert not M.remember_cost(seen, M.SearchState(state.word, 2, 1, 9))
    assert set(seen[state.key]) == {(1, 5), (0, 8)}
    assert not M.dominated_cost(seen[state.key], 2, 3)


def test_tree_does_not_prune_same_cc_cheaper_move_successor():
    instance = M.Instance("regression", (1, 1, 1), 2, 2, cap_moves=32)
    tree = M.Tree(instance, seed=0)
    source = M.SearchState(instance.word, instance.strands, 0, 0)
    actions, children, _ = tree.candidates(source)
    changed = next(c for c in children if c.used_cc == 1)
    tree.seen[changed.key] = [(0, 20)]
    _, after, _ = tree.candidates(source)
    assert any(c.key == changed.key for c in after)
    tree.seen[changed.key] = [(0, 1)]
    _, after, _ = tree.candidates(source)
    assert not any(c.key == changed.key for c in after)


def small_instance(word=TREFOIL[0], strands=TREFOIL[1], **kwargs) -> M.Instance:
    return M.Instance(
        name="small",
        word=tuple(word),
        strands=strands,
        cap_crossing_changes=kwargs.pop("cap_crossing_changes", 1),
        cap_moves=kwargs.pop("cap_moves", 12),
        **kwargs,
    )


def big_instance() -> M.Instance:
    """A 21-letter 12-strand word searched at the `(512, 24)` spec of §M0.4."""
    word = tuple([1, 2, -3, 4, -5, 6, 7, -8, 9, 10, 11, -1, 2, 3, -4, 5, 6, -7, 8, 9, 10])
    strands = 12
    return M.Instance(
        name="big",
        word=word,
        strands=strands,
        cap_crossing_changes=2,
        cap_moves=6,
        growth_len=512 - len(word),
        growth_strands=24 - strands,
    )


# ------------------------------------------------------------ size independence


def test_one_state_dict_runs_the_loop_at_two_sizes():
    torch.manual_seed(0)
    net = M.SitePolicyValueNet()
    before = net.parameter_report()

    small = small_instance()
    big = big_instance()
    assert (small.spec.max_len, small.spec.max_strands) == (5, 3)
    assert (big.spec.max_len, big.spec.max_strands) == (512, 24)
    # The two action spaces are wildly different sizes; the network is not.
    assert big.spec.num_actions > 40 * small.spec.num_actions

    state = {k: v.clone() for k, v in net.state_dict().items()}
    evaluator = M.NetworkEvaluator(net, micro_batch=4)
    small_results = M.run_searches([small], evaluator, simulations=16, seeds=(0,))
    big_results = M.run_searches([big], evaluator, simulations=8, seeds=(0,))

    assert small_results[0].expansions > 0
    assert big_results[0].expansions > 0
    assert net.parameter_report() == before
    assert net.parameter_report()["site_head"] == 49_833
    for key, value in net.state_dict().items():
        assert torch.equal(state[key], value), f"search mutated {key}"

    # And the same `state_dict` loads into a fresh network and keeps the shape.
    fresh = M.SitePolicyValueNet()
    fresh.load_state_dict(state)
    assert fresh.parameter_report() == before


def test_site_head_width_is_independent_of_the_action_space():
    torch.manual_seed(0)
    net = M.SitePolicyValueNet()
    counts = set()
    for growth in (2, 64, 480):
        instance = small_instance(growth_len=growth, growth_strands=growth // 8 + 1)
        state = M.SearchState(instance.word, instance.strands, 0, 0)
        actions = tuple(a for a, _ in M.successors_of(state, instance, instance.spec))
        batch = M.build_batch([(instance, state, actions)])
        with torch.no_grad():
            logits, value, effective = net(
                batch["word"], batch["lengths"], batch["strands"],
                batch["sites"], batch["site_mask"], batch["budgets"],
            )
        assert logits.shape[1] == len(actions)
        assert torch.isfinite(value).all()
        assert bool(effective.any())
        counts.add(net.parameter_report()["site_head"])
    assert counts == {49_833}


def test_the_loop_constructs_the_equivariant_geometry():
    """The default geometry is the legacy one, and it is *not* shift-equivariant."""
    net = M.SitePolicyValueNet()
    assert net.encoder.positional_geometry == "necklace"

    word = torch.tensor([[1, -2, 1, 1, -2, -2]], dtype=torch.long)
    lengths = torch.tensor([word.shape[1]])
    strands = torch.tensor([3])
    rolled = torch.roll(word, shifts=2, dims=1)

    deltas = {}
    for geometry in ("necklace", "legacy"):
        torch.manual_seed(0)
        encoder = ScalableBraidEmbeddingEncoder(positional_geometry=geometry).eval()
        with torch.no_grad():
            a = encoder(word, lengths, strands)["knot"]
            b = encoder(rolled, lengths, strands)["knot"]
        deltas[geometry] = float((a - b).abs().max())

    # Conjugation is a no-op on the knot, so the `knot` embedding must not move.
    assert deltas["necklace"] < 1e-5
    assert deltas["legacy"] > 100 * deltas["necklace"]


# ------------------------------------------------------------------- the search


def test_every_claimed_witness_replays():
    instance = small_instance(cap_moves=16)
    results = M.run_searches([instance], M.UniformEvaluator(), simulations=400, seeds=(0, 1))
    claimed, verified = M.verify_all(results)
    assert claimed >= 1
    assert claimed == verified
    for result in results:
        if result.solved:
            terminal = result.witness.verify()
            assert (terminal.word, terminal.strands) == ((), 1)


def test_the_search_respects_the_crossing_change_cap():
    instance = small_instance(cap_crossing_changes=0, cap_moves=16)
    results = M.run_searches([instance], M.UniformEvaluator(), simulations=200, seeds=(0,))
    # A trefoil is not the unknot, so with no crossing change there is no witness.
    assert not results[0].solved
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    kinds = {instance.spec.decode(a)[0] for a, _ in M.successors_of(state, instance, instance.spec)}
    assert CROSSING_CHANGE not in kinds


def test_both_root_rules_search_and_stay_inside_the_budget():
    instance = small_instance(cap_moves=16)
    for rule in ("puct", "gumbel"):
        results = M.run_searches(
            [instance], M.UniformEvaluator(), simulations=64, seeds=(0,), root_selection=rule
        )
        assert results[0].expansions <= 64
        assert results[0].simulations <= 64


def test_classical_beam_spends_the_same_currency():
    instance = small_instance(cap_moves=16)
    result = M.classical_beam(instance, simulations=50, seed=0)
    assert result.expansions <= 50
    if result.solved:
        result.witness.verify()


# ---------------------------------------------------------------- supervision


def test_witness_samples_reencode_actions_into_the_local_spec():
    """A witness carries its own `ActionSpec`; the sample must not carry its ids."""
    spec = ActionSpec(24, 5)
    word, strands = (1, 1, 1), 2
    actions = [spec.encode(CROSSING_CHANGE, 0), spec.encode(0, 0)]
    witness = UnknotWitness.from_actions(word, strands, spec, actions + [spec.encode(4)])
    samples = M.samples_from_witness(witness, name="probe")
    assert samples
    for sample in samples:
        assert set(sample.positives) <= set(sample.actions)
        for action in sample.positives:
            # Legal in the sample's own, much smaller, spec.
            assert action < sample.instance.spec.num_actions


def test_aplus_never_pushes_an_unexplored_site_down():
    """§5.3: absent sites are unexplored, not negative -- checked on the gradient."""
    torch.manual_seed(0)
    net = M.SitePolicyValueNet()
    instance = small_instance()
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    actions = tuple(a for a, _ in M.successors_of(state, instance, instance.spec))
    M.PolicySample(
        instance=instance,
        state=state,
        actions=actions,
        visits=np.zeros(len(actions)),
        positives=(actions[0],),
        value_target=None,
    )
    batch = M.build_batch([(instance, state, actions)])
    logits, _, effective = net(
        batch["word"], batch["lengths"], batch["strands"],
        batch["sites"], batch["site_mask"], batch["budgets"],
    )
    logits = logits.detach().requires_grad_(True)
    positive = torch.zeros_like(effective)
    positive[0, 0] = True
    loss = M.set_valued_loss(logits, effective, positive & effective)
    loss.backward()
    grad = logits.grad[0]
    probability = M.SiteHead.masked_log_policy(logits.detach(), effective)[0].exp()
    # An unexplored site's gradient is exactly its own share of the normaliser --
    # the same term every positive also carries -- and never anything extra.
    for index in range(1, len(actions)):
        assert grad[index] == pytest.approx(float(probability[index]), abs=2e-6)
    assert float(grad[0]) < 0.0  # the demonstrated site is the only one pushed up
    assert float(grad.sum()) == pytest.approx(0.0, abs=1e-5)
    # And the loss has no way to express a negative at all.
    with pytest.raises(TypeError):
        M.set_valued_loss(logits, effective, positive.float())


def test_training_moves_the_prior_toward_a_demonstrated_site():
    torch.manual_seed(0)
    net = M.SitePolicyValueNet()
    spec = ActionSpec(24, 5)
    witness = UnknotWitness.from_actions(
        (1, 1, 1), 2, spec,
        [spec.encode(CROSSING_CHANGE, 0), spec.encode(0, 0), spec.encode(4)],
    )
    samples = M.samples_from_witness(witness, name="probe")
    assert samples
    _, first = M.training_loss(net, samples)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    M.train_on(net, samples, optimizer, batch_size=8, epochs=12, seed=0)
    _, last = M.training_loss(net, samples)
    assert last.aplus < first.aplus
    assert first.aplus_rows == len(samples)


def test_samples_from_tree_label_the_witness_action_at_its_own_depth():
    instance = small_instance(cap_moves=16)
    results = M.run_searches([instance], M.UniformEvaluator(), simulations=400, seeds=(0,))
    solved = [r for r in results if r.solved]
    if not solved:
        pytest.skip("uniform search did not solve this instance at 400 simulations")
    samples = M.samples_from_tree(solved[0])
    assert samples
    labelled = [s for s in samples if s.positives]
    assert labelled
    for sample in labelled:
        assert set(sample.positives) <= set(sample.actions)
        # The labelled action is the one the replayed witness takes here, so
        # applying it must land on the next witness state.
        action = sample.positives[0]
        children = dict(M.successors_of(sample.state, sample.instance, sample.instance.spec))
        assert action in children


def test_curriculum_instances_carry_the_banks_own_answers():
    rows = [
        {
            "name": "11a_266",
            "word": [-1, 2, -1, -3, 2, -1, 2, -3, 2, -3, 2],
            "strands": 4,
            "crossings": 11,
            "certified_unknotting_lower_bound": 1,
            "certified_unknotting_upper_bound": 1,
        }
    ]
    instances = M.instances_from_rows(rows, source="bank")
    assert len(instances) == 1
    assert instances[0].cap_crossing_changes == 1
    assert instances[0].uid.startswith("bank:000:")
    assert instances[0].spec.max_len == 13
