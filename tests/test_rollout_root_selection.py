import importlib.util
from pathlib import Path

import pytest

from rf_knots.mcts_loop import Instance


@pytest.fixture
def collector():
    path=Path(__file__).parents[1]/"scripts/collect_rollout_value.py"
    spec=importlib.util.spec_from_file_location("collect_rollout_value",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_paired_costs_have_identical_root_and_caps_but_separate_groups(collector):
    root=Instance("t",(1,1,1),2,1,cap_moves=9,uid="rollout-3",cc_weight=1000)
    paired=collector.condition_training_root(root,3,paired=True)
    assert [p.cc_weight for p in paired]==[10,1000]
    assert len({p.uid for p in paired})==2
    rows=[{k:v for k,v in p.to_dict().items() if k not in ("cc_weight","uid")} for p in paired]
    assert rows[0]==rows[1]
    assert collector.condition_training_root(root,3)==[root]


def test_size_stratification_includes_long_roots_without_outcome_selection(collector):
    views=[{"fingerprint":[3,-2],"remaining_moves":m,"word":[1]*n,"strands":s}
           for m,n,s in [(40,24,3),(48,60,9),(96,128,17),(130,160,19)]]
    index={"views":views}
    chosen=collector.choose_training_views(index,3,42,maximum_moves=128,stratify_size=True)
    assert chosen==[0,1,2]
    assert collector.choose_training_views(index,2,42)==[0,1]
    with pytest.raises(ValueError,match="insufficient"):
        collector.choose_training_views(index,4,42,maximum_moves=128,stratify_size=True)


def test_selection_is_deterministic_without_duplicates(collector):
    views=[{"fingerprint":[3 if i%2 else 5,(-1)**i*2],"remaining_moves":10+i%40,
            "word":[1]*(20+i%90),"strands":2+i%16} for i in range(300)]
    a=collector.choose_training_views({"views":views},128,101,maximum_moves=128,stratify_size=True)
    assert len(a)==len(set(a))==128
    assert a==collector.choose_training_views({"views":views},128,101,maximum_moves=128,stratify_size=True)


@pytest.mark.parametrize("count,limit",[(0,64),(1,0),(-1,64)])
def test_invalid_selection_fails(collector,count,limit):
    with pytest.raises(ValueError):collector.choose_training_views({"views":[]},count,0,maximum_moves=limit)
