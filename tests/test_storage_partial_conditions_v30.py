import pytest
from recover_random_window_277_280_v30 import condition_names,missing_conditions

def test_exact_missing_seven(tmp_path):
    names=condition_names()
    for n in names[:9]:(tmp_path/n).write_text('immutable saved condition')
    assert missing_conditions(tmp_path)==names[9:]
    assert len(names)==len(set(names))==16

def test_complete_is_never_researched(tmp_path):
    for n in condition_names():(tmp_path/n).write_text('saved')
    assert missing_conditions(tmp_path)==[]

def test_unknown_condition_failclose(tmp_path):
    (tmp_path/'search-7-10-1.json').write_text('unknown')
    with pytest.raises(AssertionError):missing_conditions(tmp_path)
