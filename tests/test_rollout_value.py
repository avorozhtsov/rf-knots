import pytest
import torch

from rf_knots import mcts_loop as M
from rf_knots.rollout_value import expanded_resource_model, rollout_candidates, verify_episode


def test_resource_migration_preserves_logits_and_value():
    architecture = dict(width=16, depth=2, embedding_dim=8, encoder_kind="strand_graph")
    old = M.SitePolicyValueNet(**architecture).double().eval()
    new, _ = expanded_resource_model({"architecture": architecture, "model": old.state_dict()})
    # Copy via float32 migration is tested at its actual float32 precision.
    old = old.float()
    new.eval()
    problem = M.Instance("t", (1, 1, 1), 2, 1)
    state = M.SearchState(problem.word, 2, 0, 0)
    actions = rollout_candidates(problem, state)
    a = M.build_batch([(problem, state, actions)])
    b = M.build_batch([(problem, state, actions)], budget_dim=5)
    with torch.no_grad():
        x = old(**a)
        y = new(**b)
    for left, right in zip(x, y, strict=True):
        torch.testing.assert_close(left, right)
    assert b["budgets"].tolist() == [[32, 1, 2, 1, 1000]]
    assert all(p.requires_grad for p in new.parameters())


def test_proposals_are_deterministic_sparse_and_unique():
    problem = M.Instance("large", (1, 2, 3), 4, 1)
    state = M.SearchState(problem.word, 4, 0, 0)
    a = rollout_candidates(problem, state)
    assert a == rollout_candidates(problem, state)
    assert len(a) == len(set(a))
    assert len(a) <= 4 * len(problem.word) + 3 + 16


def test_unfinished_rollout_is_not_a_negative():
    problem = M.Instance("t", (1, 1, 1), 2, 1)
    with pytest.raises(ValueError, match="incomplete"):
        verify_episode({"actions": [], "success": False, "cc": 0, "moves": 0}, problem)


def test_horizon_failure_is_valid_despite_a_known_solution():
    problem = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=0)
    verify_episode({"actions": [], "success": False, "cc": 0, "moves": 0}, problem)


def test_one_winner_nine_losers_cancel_shared_prefix_exactly():
    from rf_knots.actions import CROSSING_CHANGE, DESTABILIZE, INSERT, REDUCE
    from rf_knots.rollout_value import relative_terms

    problem = M.Instance("t", (1, 1, 1), 2, 1, cap_moves=8)
    s = problem.spec
    cc, reduce, destab = s.encode(CROSSING_CHANGE, 0), s.encode(REDUCE, 0), s.encode(DESTABILIZE)
    good = {"actions": [cc, reduce, destab], "success": True, "cc": 1, "moves": 3, "eligible": True}
    bad = {**good, "actions": [cc, s.encode(INSERT, 0, 1, 1), reduce, reduce, destab], "moves": 5}
    terms = relative_terms([good] + [bad] * 9, problem)
    assert terms
    assert all(state.used_moves > 0 for state, _, _ in terms)
    assert relative_terms([good] * 10, problem) == []
    assert relative_terms([good, {**bad, "eligible": False}], problem) == []


def test_relative_clip_limits_already_improved_actions():
    from rf_knots.rollout_value import clipped_relative_loss

    old = torch.tensor([-3.0, -3.0])
    current = torch.tensor([-1.0, -5.0], requires_grad=True)
    loss = clipped_relative_loss(current, old, torch.tensor([1.0, -1.0]))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros(2))
