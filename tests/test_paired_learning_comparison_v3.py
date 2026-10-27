import numpy as np
from analyze_learning_comparison_v3 import bootstrap
from paired_learning_comparison_v3 import ready_thresholds
import paired_learning_comparison_v3 as runner
from pathlib import Path

def test_gate_requires_both_thresholds():
    assert not ready_thresholds(172799,100000)
    assert not ready_thresholds(172800,99999)
    assert ready_thresholds(172800,100000)

def test_cluster_bootstrap_null():
    x=bootstrap(np.zeros(32));assert x['delta']==0 and x['CI95']==[0,0]

def test_cluster_bootstrap_improvement():
    x=bootstrap(np.ones(32));assert x['delta']==1 and x['CI95']==[1,1]

def test_bootstrap_deterministic():
    assert bootstrap(np.arange(32)/32)==bootstrap(np.arange(32)/32)

def test_inflight_batch_and_old_episodes_not_counted(monkeypatch):
    records={
        'data/manifest.json':dict(audits=[dict(batch=0,report='r0',sha256='h'),dict(batch=1,report='r1',sha256='h')]),
        'r1':dict(partial=False,conditions=32,batch_manifest_sha256='h',episodes_replayed=3),
        'data/batch-000001/manifest.json':dict(shards=[dict(file='new.json',sha256='h')]),
        'data/new.json':dict(row=dict(episodes=[1,2,3]))}
    monkeypatch.setattr(runner,'read',lambda p:records[str(p)])
    monkeypatch.setattr(runner,'sha',lambda p:'h')
    assert runner.new_episodes(dict(main_search='data',first_new_task=17))==(3,32)

def test_no_network_execution_before_both_gates(monkeypatch):
    import sys
    monkeypatch.setattr(sys,'argv',['runner','--run-if-ready'])
    monkeypatch.setattr(runner,'gate',lambda:dict(ready=False))
    def forbidden(*args):raise AssertionError('premature network evaluation')
    monkeypatch.setattr(runner,'freeze_candidate',forbidden)
    runner.main()

def test_duplicate_audits_cannot_inflate_episode_gate(monkeypatch):
    import pytest
    records={
        'data/manifest.json':dict(audits=[dict(batch=0,report='r',sha256='h')]*2),
        'r':dict(partial=False,conditions=32,batch_manifest_sha256='h',episodes_replayed=1),
        'data/batch-000000/manifest.json':dict(shards=[dict(file='one',sha256='h')]),
        'data/one':dict(row=dict(episodes=[1]))}
    monkeypatch.setattr(runner,'read',lambda p:records[str(p)])
    monkeypatch.setattr(runner,'sha',lambda p:'h')
    with pytest.raises(AssertionError,match='duplicate audited batch'):
        runner.new_episodes(dict(main_search='data',first_new_task=0))

def test_CPU_interrupt_preserves_completed_incumbent_only(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace as N
    import rf_knots.cost_mcts_rollout as search
    import pretrained_raster_policy as prior
    import audit_cost_mcts_continuations as replay
    @dataclass
    class Continuation:
        dummy:int=1
    class Witness:
        crossing_changes=0
        moves=1
        def verify(self):pass
        def to_dict(self):return dict(mock_complete_witness=True)
    class Tree:
        def __init__(self,*args,**kwargs):
            self.seed=1;self.finished=False;self.simulations=0
            self.continuations=[];self.best_witness=None
        def select_leaf(self):return N(state=N(key=((1,),2)),actions=(1,),widening=False)
        def prepare_pending(self,p):pass
        def apply(self,p,v,r):
            self.simulations+=1;self.continuations.append((1,(),r));self.best_witness=Witness()
    class Evaluator:
        def __init__(self,*args,**kwargs):self.n=0
        def evaluate(self,x):
            self.n+=1
            if self.n==2:raise TimeoutError('synthetic incomplete simulation')
            return [None]
    class Policy:
        def __init__(self,*args,**kwargs):pass
        def run(self,*args,**kwargs):return [Continuation()]
    monkeypatch.setattr(search,'ContinuationCostTree',Tree)
    monkeypatch.setattr(search,'BatchedPolicyContinuation',Policy)
    monkeypatch.setattr(prior,'RawKnotDistance',lambda *a,**k:N(evaluations=0))
    monkeypatch.setattr(prior,'RasterPolicyEvaluator',lambda *a,**k:N(calls=0))
    monkeypatch.setattr(prior,'RasterDistancePrior',Evaluator)
    monkeypatch.setattr(replay,'audit_episode',lambda *a,**k:None)
    monkeypatch.setattr(runner.signal,'setitimer',lambda *a,**k:None)
    import verified_reduction_search_v28 as verified
    monkeypatch.setattr(verified,'ContinuationCostTree',Tree)
    monkeypatch.setattr(verified,'BatchedPolicyContinuation',Policy)
    monkeypatch.setattr(verified,'RawKnotDistance',lambda *a,**k:N(evaluations=0))
    monkeypatch.setattr(verified,'RasterPolicyEvaluator',lambda *a,**k:N(calls=0))
    monkeypatch.setattr(verified,'RasterDistancePrior',Evaluator)
    monkeypatch.setattr(verified,'automatic_action',lambda *a:None)
    r=runner.timed_search(N(cap_moves=2),None,None,1)
    assert r['cpu_censored'] and r['solved'] and r['actual_simulations']==1
    assert len(r['episodes'])==1 and r['CC']==0 and r['moves']==1

from freeze_learning_comparison_v3 import substantial, state_id
from rf_knots.embedding_normalization import fast_normalize_global_view

def test_raw_long_trivial_tail_is_not_substantial():
    word=tuple([1,-1]*12+[1,2])
    assert len(word)>=20
    assert not substantial(word,3)

def test_nontrivial_long_normalized_state_remains_protected():
    word=tuple([1,1]*12)
    assert substantial(word,3)
    assert state_id(word,3)==state_id(tuple([2,-2])+word,3)
