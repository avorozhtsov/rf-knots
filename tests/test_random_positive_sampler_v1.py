import pytest
from random_positive_sampler_v1 import source_ordinal
def test_exact_source_balance():
    a=[source_ordinal(351232+i)[0] for i in range(1040)]
    assert a.count('random')==a.count('retained')==520
def test_source_phase_resume():
    for n in range(130):
        assert source_ordinal(351232+2*n)==('retained',n)
        assert source_ordinal(351233+2*n)==('random',n)
def test_negative_origin_rejected():
    with pytest.raises(ValueError):source_ordinal(351231)
