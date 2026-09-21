import pytest

from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants


@pytest.mark.parametrize("index", range(24))
def test_generated_conjugates_have_exactly_replayed_short_witness(index):
    w = conjugate_witness(20261010, index)
    w.verify()
    assert w.crossing_changes == 1 + (index // 6) % 2
    assert w.steps[-1].after.word == ()
    assert w.steps[-1].after.strands == 1
    det, sig = invariants(w.start.word, w.start.strands)
    assert det % 2 == 1
    assert abs(sig) // 2 <= w.crossing_changes
    assert conjugate_witness(20261010, index) == w


@pytest.mark.parametrize("index",[9,19,29,39])
def test_larger_constructive_range_keeps_exact_witness(index):
    w=conjugate_witness(20261012,index,max_strands=12,min_depth=8,max_depth=32)
    w.verify()
    assert w.start.strands==12
    assert w.crossing_changes==1+(index//10)%2
    assert abs(invariants(w.start.word,w.start.strands)[1])//2<=w.crossing_changes


@pytest.mark.parametrize("kwargs",[{"max_strands":2},{"min_depth":-1},{"min_depth":8,"max_depth":4}])
def test_invalid_constructor_size_is_rejected(kwargs):
    with pytest.raises(ValueError):conjugate_witness(1,0,**kwargs)
