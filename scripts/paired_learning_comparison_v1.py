"""Gated one-shot equal-CPU comparison. No evaluation before48activeh AND100k audited episodes."""
import argparse
import fcntl
import hashlib
import json
import math
import shutil
import signal
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read,policy

STUDY=Path('artifacts/paired-learning-48h100k-v1')
PROTOCOL=Path('artifacts/paired-learning-48h100k-v1-protocol.json')

def ready_thresholds(active,episodes):
    return active>=172800 and episodes>=100000

def training_delta(baseline,run):
    total=0.;visited=set()
    while True:
        assert str(run) not in visited;visited.add(str(run))
        m=read(run/'manifest.json');specfile=Path(next(k for k in m['contract']['inputs'] if 'start-spec-' in k))
        s=read(specfile);assert sha(specfile)==m['contract']['inputs'][str(specfile)]
        if str(run)==baseline['run']:
            assert m['optimizer_steps']>=baseline['step']
            return total+m['new_active_seconds']-baseline['checkpointed_active_origin_seconds']
        total+=m['new_active_seconds'];parent=Path(s['parent'])
        assert sha(parent)==s['parent_sha256']
        pm=read(parent.parent/'manifest.json')
        assert pm['checkpoints'][-1]['sha256']==s['parent_sha256']
        assert sha(parent.parent/'manifest.json')==s['parent_manifest_sha256']
        run=parent.parent

def new_episodes(baseline):
    data=Path(baseline['main_search']);m=read(data/'manifest.json')
    first=((baseline['first_new_task']+31)//32)*32 # discard the entire in-flight batch
    count=0;conditions=0
    for a in m['audits']:
        if a['batch']*32<first:continue
        p=Path(a['report']);assert sha(p)==a['sha256'];r=read(p)
        assert not r['partial'] and r['conditions']==32
        bp=data/f"batch-{a['batch']:06d}"/'manifest.json';assert sha(bp)==r['batch_manifest_sha256']
        batch=read(bp);actual=0
        for e in batch['shards']:
            sp=data/e['file'];assert sha(sp)==e['sha256'];s=read(sp)
            if s.get('row'):actual+=len(s['row']['episodes'])
        assert actual==r['episodes_replayed'];count+=actual;conditions+=32
    return count,conditions

def gate():
    lock=read(STUDY/'study.json')
    for f,h in lock['inputs'].items():assert sha(Path(f))==h,f
    b=read(STUDY/'baseline.json');ref=read(Path('artifacts/continuous-r2-current-training.json'))
    active=training_delta(b,Path(ref['out']));episodes,conditions=new_episodes(b)
    return dict(ready=ready_thresholds(active,episodes),new_checkpointed_active_seconds=active,
                new_audited_MCTS_episodes=episodes,new_audited_conditions=conditions,current_run=ref['out'])

def freeze_candidate(g):
    path=STUDY/'candidate.json'
    if path.exists():return read(path)
    assert g['ready'];run=Path(g['current_run']);m=read(run/'manifest.json');e=m['checkpoints'][-1]
    cp=run/e['file'];assert sha(cp)==e['sha256']
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    assert saved['step']==e['step'] and len(saved['optimizer']['state'])==71
    assert all(k in saved for k in ['rng','torch_rng','mps_rng'])
    for key in ['model','objective','auxiliary_head']:
        assert all(torch.isfinite(v).all() for v in saved[key].values())
    shutil.copy2(cp,STUDY/'candidate.pt');assert sha(STUDY/'candidate.pt')==e['sha256']
    payload=dict(checkpoint=str(STUDY/'candidate.pt'),checkpoint_sha256=e['sha256'],step=e['step'],
                 contract_sha256=m['contract_sha256'],source_checkpoint=str(cp),gates=g)
    atomic(path,payload);return payload

def timed_search(instance,net,obj,seed):
    # Same canonical select/prepare/continuation/apply operations as cost_mcts_rollout.
    # CPU interrupt discards only the unfinished simulation, not previous incumbents.
    from rf_knots import mcts_loop as M
    from rf_knots.cost_mcts_rollout import ContinuationCostTree,BatchedPolicyContinuation
    from pretrained_raster_policy import RasterDistancePrior,RasterPolicyEvaluator,RawKnotDistance
    from audit_cost_mcts_continuations import audit_episode
    cap=4096;torch.manual_seed(seed)
    tree=ContinuationCostTree(instance,seed=seed*9973,budget=cap,widening_batch=4,c_puct=1.5)
    heuristic=RawKnotDistance(net,obj,microbatch=2)
    evaluator=RasterDistancePrior(heuristic,temperature=1.)
    neural=RasterPolicyEvaluator(net,microbatch=2)
    continuation=BatchedPolicyContinuation(neural,horizon=instance.cap_moves,insertion_candidates=4)
    completed=[];completed_simulations=0;incumbent=None;censored=False;stop_reason='simulation_ceiling'
    started=time.process_time();wall=time.monotonic()
    def timeout(*_):raise TimeoutError('equal30processCPU budget')
    previous=signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,30.)
    try:
        with torch.no_grad():
            for iteration in range(cap):
                pending=tree.select_leaf()
                if pending is None:
                    if tree.finished:stop_reason='tree_finished';break
                    continue
                tree.prepare_pending(pending)
                values=evaluator.evaluate([(instance,pending.state,pending.actions)]) if pending.state.key!=M.SOLVED and pending.actions else []
                rollout=None
                if not pending.widening and pending.state.key!=M.SOLVED:
                    rollout=continuation.run([(instance,pending.state)],seeds=[tree.seed*(cap+1)+iteration])[0]
                tree.apply(pending,values[0] if values else None,rollout)
                # Publish only a completely applied simulation.
                completed=list(tree.continuations)
                incumbent=tree.best_witness
                completed_simulations=tree.simulations
    except TimeoutError:
        censored=True;stop_reason='CPU_budget'
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGPROF,previous)
    cpu=time.process_time()-started;elapsed=time.monotonic()-wall
    episodes=[]
    config=dict(simulations=cap,seed=seed,horizon=instance.cap_moves,insertion_candidates=4,widening_batch=4,c_puct=1.5,temperature=1.)
    for simulation,prefix,c in completed:
        e=json.loads(json.dumps(dict(simulation=simulation,tree_prefix=list(prefix),continuation=asdict(c))))
        audit_episode(instance,e,config);episodes.append(e)
    if incumbent:incumbent.verify()
    return dict(solved=incumbent is not None,witness=incumbent.to_dict() if incumbent else None,
        CC=incumbent.crossing_changes if incumbent else None,moves=incumbent.moves if incumbent else None,
        L10=10*incumbent.crossing_changes+incumbent.moves if incumbent else None,
        L1000=1000*incumbent.crossing_changes+incumbent.moves if incumbent else None,
        cpu_seconds=cpu,wall_seconds=elapsed,stop_reason=stop_reason,cpu_censored=censored,
        actual_simulations=completed_simulations,episodes=episodes,
        neural_evaluations=neural.calls+heuristic.evaluations,
        incomplete_simulation_discarded=censored)

def run_evaluation(candidate):
    from rf_knots import mcts_loop as M
    from freeze_learning_comparison_v1 import state_id,inspect
    from rf_knots.evidence import UnknotWitness
    b=read(STUDY/'baseline.json');panel=read(STUDY/'panel.json');torch.set_num_threads(1)
    # Mandatory future-bank check. Any leak invalidates the frozen study; no substitution.
    ref=read(Path('artifacts/continuous-r2-current-training.json'));spec=read(Path(ref['spec']))
    groups=set();states=set()
    for k in ['bank','positive_bank','random_positive_bank']:
        bank=Path(spec[k]);m=read(bank/'manifest.json')
        for e in m.get('paths',m.get('roots',[])):inspect(read(bank/e['file']),groups,states)
    for row in panel['rows']:
        assert tuple(row['query_group']) not in groups
        w=UnknotWitness.from_dict(row['witness'])
        assert all(state_id(x.word,x.strands) not in states for x in [w.start,*[s.after for s in w.steps]] if len(x.word)>=20)
    actors={}
    for name,entry in [('baseline',b),('candidate',candidate)]:
        actors[name]=policy(dict(file=entry['checkpoint'],sha256=entry['checkpoint_sha256'],step=entry['step'],contract_sha256=entry['contract_sha256']))
        assert all(p.device.type=='cpu' and p.dtype==torch.float32 for obj in actors[name] for p in obj.parameters())
    out=STUDY/'evaluation';out.mkdir(exist_ok=True)
    for root,row in enumerate(panel['rows']):
        for wi,weight in enumerate([10,1000]):
            for seed in [20291019,20291020]:
                key=f'{root}:{weight}:{seed}';order=['baseline','candidate']
                if hashlib.sha256(key.encode()).digest()[0]%2:order.reverse()
                d=row['instances'][wi];instance=M.Instance(**{**d,'word':tuple(d['word'])})
                for name in order:
                    path=out/f'{root:03d}-{weight}-{seed}-{name}.json'
                    actor_sha=b['checkpoint_sha256'] if name=='baseline' else candidate['checkpoint_sha256']
                    if path.exists():
                        old=read(path)
                        assert old['panel_sha256']==sha(STUDY/'panel.json') and old['actor_sha256']==actor_sha
                        assert (old['root'],old['weight'],old['seed'],old['arm'])==(root,weight,seed,name)
                        continue
                    assert shutil.disk_usage(out).free>=12*1024**3
                    result=timed_search(instance,*actors[name],seed)
                    atomic(path,dict(root=root,weight=weight,seed=seed,arm=name,panel_sha256=sha(STUDY/'panel.json'),actor_sha256=actor_sha,result=result))
    atomic(out/'COMPLETED.json',dict(status='completed_search_needs_paired_analysis',cells=256))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-if-ready',action='store_true');a=p.parse_args()
    g=gate();print(g,flush=True)
    if a.run_if_ready and g['ready']:
        with (STUDY/'evaluation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert shutil.disk_usage(STUDY).free>=12*1024**3,'storage guard before evaluation'
            candidate=freeze_candidate(g);run_evaluation(candidate)
            from analyze_learning_comparison_v1 import main as analyze
            if not (STUDY/'evaluation/analysis.json').exists():analyze()

if __name__=='__main__':main()
