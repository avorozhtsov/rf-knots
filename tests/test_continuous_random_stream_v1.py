import pytest
from continuous_random_stream_v1 import seed_for,SEED,checkpins
def test_distinct_bounded_seeds():
    s=[seed_for(b,o) for b in range(100) for o in range(32)]
    assert len(set(s))==3200 and s[0]==SEED
def test_bad_pin_rejected(tmp_path):
    p=tmp_path/'a';p.write_text('evidence')
    with pytest.raises(ValueError):checkpins({str(p):'0'*64})
def test_good_pin(tmp_path):
    from build_mcts_training_supervision import sha
    p=tmp_path/'a';p.write_text('evidence');checkpins({str(p):sha(p)})
