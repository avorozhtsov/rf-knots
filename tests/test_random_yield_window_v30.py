import pytest
from learning_pipeline_v30 import window_gate

@pytest.mark.parametrize('solved,cpu,expected',[(0,0,False),(7,1,False),(8,2400,True),
    (8,2400.01,False),(16,1000,True)])
def test_gate(solved,cpu,expected):
    assert window_gate(solved,cpu)==expected

def test_low_batch_not_fatal_or_removed():
    assert window_gate(sum([1,3,2,2]),sum([128.8,100,100,100]))
    assert not window_gate(sum([1,1,2,3]),100)
