import pytest

from rf_knots import mcts_loop as M, reference
from rf_knots.actions import ActionSpec, CROSSING_CHANGE, REDUCE, DESTABILIZE
from rf_knots.evidence import UnknotWitness


def test_sparse_witness_keeps_good_actions_without_successor_enumeration(monkeypatch):
    spec = ActionSpec(16, 5)
    witness = UnknotWitness.from_actions((1, 1, 1), 2, spec, [
        spec.encode(CROSSING_CHANGE, 0), spec.encode(REDUCE, 0), spec.encode(DESTABILIZE),
    ])
    full = M.samples_from_witness(witness, name='full', include_commutation=False)

    def forbidden(*args, **kwargs):
        pytest.fail('sparse witness path must not enumerate successor words')

    monkeypatch.setattr(M, 'successors_of', forbidden)
    sparse = M.samples_from_witness(witness, name='sparse', candidate_limit=2,
                                    include_commutation=False)
    assert len(sparse) == len(full)
    for a, b in zip(sparse, full, strict=True):
        assert a.positives == b.positives
        assert set(a.positives) <= set(a.actions) <= set(b.actions)
        assert len(a.actions) <= 2 + len(a.positives)
        assert all(reference.is_legal(a.instance.spec, a.state.word, a.state.strands,
                                     action, a.instance.cap_crossing_changes > 0)
                   for action in a.actions)
