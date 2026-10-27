from audit_random_representation_mcts_v1 import rank_paths


def test_empty_singleton_and_equal_cost_no_sign():
    for paths in ([],[{'cost':1}],[{'cost':1},{'cost':1}]):
        assert rank_paths(paths)==[]


def test_two_distinct_costs_balanced():
    rows=rank_paths([{'cost':10},{'cost':20}])
    assert [(r['sign'],r['weight']) for r in rows]==[(1,.5),(-1,.5)]


def test_inclusive_eighty_percent_ties_and_mass():
    rows=rank_paths([{'cost':c} for c in [1,1,2,3,4,5]])
    assert [(r['cost'],r['sign']) for r in rows]==[(1,1),(1,1),(4,-1),(5,-1)]
    for sign in [-1,1]:assert sum(r['weight'] for r in rows if r['sign']==sign)==.5


def test_no_manufactured_missing_sign():
    assert rank_paths([{'cost':1}]*5+[{'cost':2}])==[]
