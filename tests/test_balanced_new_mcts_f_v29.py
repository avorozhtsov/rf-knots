"""Pure budget/clock/metadata tests; no live optimizer or artifact mutation."""
import math
import pytest
import torch
import balanced_new_mcts_f_v29 as b


@pytest.mark.parametrize('offset,w',[(0,.2),(1999,.2),(2000,.35),(3999,.35),(4000,.5),(10000,.5)])
def test_warmup_uses_absolute_preserved_origin(offset,w):
    assert b.mcts_weight(100+offset,100)==w


@pytest.mark.parametrize('w',[.2,.35,.5])
@pytest.mark.parametrize('f,m,ema',[(.001,20,.002),(3.,.00001,2.),(1.,1.,1.)])
def test_exact_budget_with_bounded_amplification(f,m,ema,w):
    fs,ms,target=b.scales(f,m,w,ema)
    assert math.isclose(m*ms/(f*fs+m*ms),w,abs_tol=1e-12)
    assert fs<=1 and ms<=10 and target<=min(f,ema)


@pytest.mark.parametrize('f,m,ema',[(0,1,1),(1,0,1),(1,float('nan'),1),(1,1,float('inf'))])
def test_bad_component_fails_instead_of_one_sided_training(f,m,ema):
    with pytest.raises(ValueError):b.scales(f,m,.5,ema)


def test_full_vector_norm_includes_heads_not_just_encoder():
    assert b.full_norm([torch.tensor([3.]),None,torch.tensor([4.])])==5.


def test_no_old_auxiliary_or_rehearsal_allowed():
    m=dict(F_loss=1,positive_loss=1,F_raw_full_norm=1,MCTS_raw_full_norm=1,
        F_effective_full_norm=.5,MCTS_effective_full_norm=.5,combined_grad_norm=1,
        balance_ema_f=1,actual_MCTS_norm_share=.5,target_MCTS_norm_share=.5,
        aux_loss=0,rehearsal_encoder_grad_norm=0,real_AdamW_steps_this_update=1)
    b.numerical_gate(m)
    with pytest.raises(ValueError,match='auxiliary'):b.numerical_gate({**m,'aux_loss':.01})


def test_fresh_sources_filter_historical_paths_and_keep_phase(monkeypatch):
    row=lambda ident:dict(path_id=ident,instance=dict(word=[1],strands=2),observations=[{'shard':'x'}])
    groups=[((1,0),[(((1,),2),[row('old'),row('fresh')])])]
    monkeypatch.setattr(b.sampler,'load_bank',lambda bank:groups)
    s=dict(balance_origin_step=100,parent_step=100,positive_bank='m',random_positive_bank='r',
        fresh_retained_ids=['fresh'],fresh_random_ids=['fresh'])
    b.initialize(s)
    assert all(r['path_id']=='fresh' for g in b.sampler.SOURCES.values() for _,views in g for _,rs in views for r in rs)


def test_v29_round_handoff_carries_ema_but_not_round_draw_counts(monkeypatch):
    monkeypatch.setattr(b.sampler,'load_bank',lambda bank:[((1,0),[(((1,),2),[{'path_id':'fresh'}])])])
    s=dict(balance_origin_step=100,parent_step=5000,positive_bank='m',random_positive_bank='r',
        fresh_retained_ids=['fresh'],fresh_random_ids=['fresh'])
    b.initialize(s,dict(step=5000,metrics={'balance_ema_f':.02}))
    assert b.EMA_F==.02 and b.COUNTS['random']==0 and b.mcts_weight(5000,b.ORIGIN)==.5


def test_uid_and_L_weight_are_metadata_but_braid_and_caps_are_not():
    from audit_fresh_main_eligibility_v29 import semantic_instance
    a=dict(uid='root-10',cc_weight=10,word=[1,1,1],strands=2,cap_moves=8)
    assert semantic_instance(a)==semantic_instance({**a,'uid':'root-1000','cc_weight':1000})
    assert semantic_instance(a)!=semantic_instance({**a,'cap_moves':9})
    assert semantic_instance(a)!=semantic_instance({**a,'word':[1,1]})
