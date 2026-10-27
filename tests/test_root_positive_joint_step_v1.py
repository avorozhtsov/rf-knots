import pytest
import torch
from root_positive_joint_step_v1 import DeferredOptimizer,capped_scale


def test_defer_then_one_real_optimizer_update():
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.01);p.grad=torch.tensor(1.)
    d=DeferredOptimizer(opt);d.step();assert float(p)==1. and len(opt.state)==0
    opt.step();assert len(opt.state)==1 and float(opt.state[p]['step'])==1
    with pytest.raises(ValueError):d.step()


def test_positive_cap_never_scales_up_and_zero_is_skipped():
    assert capped_scale(1.,0.)==0
    assert capped_scale(1.,.001)==1
    assert capped_scale(1.,10.)==.005
    for f,p in [(1.,10.),(.001,3.),(3.,.01)]:assert capped_scale(f,p)*p<=.05*f+1e-15


def test_nonfinite_or_vanishing_F_rejected():
    for f,p in [(0.,1.),(1e-13,1.),(float('nan'),1.),(1.,float('inf'))]:
        with pytest.raises(ValueError):capped_scale(f,p)
