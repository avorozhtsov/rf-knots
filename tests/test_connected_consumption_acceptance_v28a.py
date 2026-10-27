import pytest
from check_connected_learning_live_v28a import consumption_gate


def test_ready_without_actual_new_draw_fails():
    with pytest.raises(AssertionError,match='no NEW random'):
        consumption_gate(dict(retained=5,random=5,new_random=0,distinct_new_path_ids=[]),10)


def test_wrong_ratio_fails():
    with pytest.raises(AssertionError,match='SOURCE ratio'):
        consumption_gate(dict(retained=9,random=1,new_random=1,distinct_new_path_ids=['new']),10)


def test_advancing_exact_source_counts_pass():
    consumption_gate(dict(retained=5,random=6,new_random=2,distinct_new_path_ids=['new']),11)
