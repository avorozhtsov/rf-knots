from random_view_positive_sampler_v2 import view_slot
def test_each_view_and_objective_visited():
    for groups in [1,4,7]:
        for views in [1,2,5]:
            slots=[view_slot(n*groups,groups,views) for n in range(2*views)]
            assert set(slots)=={(v,w) for v in range(views) for w in (10,1000)}
def test_views_not_path_count_weighted():
    assert [view_slot(n*4,4,3)[0] for n in range(6)]==[0,1,2,0,1,2]
