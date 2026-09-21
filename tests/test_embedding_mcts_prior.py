import numpy as np
import pytest

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, REDUCE
from rf_knots.embedding_mcts_prior import EmbeddingDistancePrior


def test_exact_raw_action_application_and_budget_metadata():
    instance = M.Instance("shifted", (1, 1, -1, 1, 1), 2, 2, cap_moves=12)
    state = M.SearchState(instance.word, 2, 1, 3)
    actions = [instance.spec.encode(CROSSING_CHANGE, 4), instance.spec.encode(REDUCE, 1)]
    seen = []

    def distance(states):
        seen.extend(states)
        return [2, 0]

    evaluator = EmbeddingDistancePrior(distance)
    result = evaluator.evaluate([(instance, state, actions)])[0]
    for a, child in zip(actions, seen, strict=True):
        assert child.key == reference.apply(instance.spec, state.word, state.strands, a)
        assert child.used_moves == 4
    assert [s.used_cc for s in seen] == [2, 1]
    assert result.priors[1] / result.priors[0] == pytest.approx(np.exp(2))
    assert result.value == 0 and evaluator.action_scores == 2 and evaluator.requests == 1


def test_batch_boundaries_and_empty_requests():
    i = M.Instance("trefoil", (1, 1, 1), 2, 1)
    s = M.SearchState(i.word, 2, 0, 0)
    actions = [i.spec.encode(CROSSING_CHANGE, p) for p in range(3)]
    e = EmbeddingDistancePrior(lambda states: [0, 1, 2, 3])
    out = e.evaluate([(i, s, actions), (i, s, []), (i, s, actions[:1])])
    assert [len(r.priors) for r in out] == [3, 0, 1]
    assert out[0].priors.sum() == pytest.approx(1)
    assert out[2].priors.tolist() == [1]
    assert e.evaluate([]) == []


@pytest.mark.parametrize("scores", [[float("nan")], [-1], [], [float("inf")]])
def test_invalid_distances_rejected(scores):
    i = M.Instance("t", (1, 1, 1), 2, 1)
    e = EmbeddingDistancePrior(lambda states: scores)
    with pytest.raises(ValueError, match="distance"):
        e.evaluate([(i, M.SearchState(i.word, 2, 0, 0), [i.spec.encode(CROSSING_CHANGE, 0)])])


def test_large_gaps_retain_nonzero_prior():
    i = M.Instance("t", (1, 1, 1), 2, 1)
    e = EmbeddingDistancePrior(lambda states: [0, 1e308], temperature=1e-10)
    out = e.evaluate([(i, M.SearchState(i.word, 2, 0, 0),
                       [i.spec.encode(CROSSING_CHANGE, 0), i.spec.encode(CROSSING_CHANGE, 1)])])[0]
    assert np.isfinite(out.priors).all() and (out.priors > 0).all()


def test_illegal_action_and_spent_move_budget_rejected_before_scoring():
    i = M.Instance("t", (1, 1, 1), 2, 0, cap_moves=2)
    e = EmbeddingDistancePrior(lambda states: pytest.fail("must not score illegal action"))
    a = i.spec.encode(CROSSING_CHANGE, 0)
    with pytest.raises(ValueError, match="illegal"):
        e.evaluate([(i, M.SearchState(i.word, 2, 0, 0), [a])])
    with pytest.raises(ValueError, match="budget"):
        e.evaluate([(i, M.SearchState(i.word, 2, 0, 2), [a])])


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        EmbeddingDistancePrior(lambda s: [], temperature=temperature)
