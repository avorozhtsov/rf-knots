from audit_random_representation_mcts_v1 import rank_paths


def test_equal_cost_and_singleton_no_fake_negative():
    assert rank_paths([{'cost':1}])==[]
    assert rank_paths([{'cost':1},{'cost':1}])==[]


def test_two_costs_balanced_signed_mass():
    rows=rank_paths([{'cost':10},{'cost':20}])
    assert [(r['sign'],r['weight']) for r in rows]==[(1,.5),(-1,.5)]


def test_quantile_ties_and_sign_mass():
    rows=rank_paths([{'cost':c} for c in [1,1,2,3,4,5]])
    assert sum(r['weight'] for r in rows if r['sign']==1)==.5
    assert sum(r['weight'] for r in rows if r['sign']==-1)==.5
    assert all(r['cost'] in [1,5] for r in rows)
