"""Separate bounded random-representation collection; no optimizer writes."""
import argparse
import hashlib
import random
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path

import torch
from build_mcts_training_supervision import atomic, sha
from build_continuous_train_roster_v1 import exclusions
from evaluate_embedding_cost_search import query_instance
from run_continuous_train_mcts_v1 import read, policy
from run_continuous_train_mcts_v2 import snapshot
from resource_cost_search import run_batch
from rf_knots import mcts_loop as M
from rf_knots.actions import ActionSpec, REDUCE
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

PROTOCOL=Path('research/416-random-representation-mcts-relative-success-pilot-v1.md')
ROSTER=Path('artifacts/continuous-train-roster-v1/roster.json')
OUT=Path('artifacts/random-representation-mcts-pilot-v1')


def timeout(*_):
    raise TimeoutError('20CPU-second condition ceiling')


def prepare():
    if OUT.exists():raise ValueError('immutable output exists')
    rows=read(ROSTER)['rows']; _,held=exclusions(); roots=[]
    for stage in range(4):
        pool=[r for r in rows if r['stage']==stage]
        row=min(pool,key=lambda r:hashlib.sha256(f'20261016416:{r["query_group"]}'.encode()).hexdigest())
        w=UnknotWitness.from_dict(row['witness']);w.verify()
        rng=random.Random(20261016416+stage);word=list(w.start.word);undos=[]
        for _ in range(2):
            p=rng.randrange(len(word)+1);g=rng.choice((-1,1))*rng.randrange(1,w.start.strands)
            word[p:p]=[g,-g];undos.append(p)
        spec=ActionSpec(max(len(word)+2,max(len(s.after.word) for s in w.steps)+2),
                        max(w.start.strands+1,max(s.after.strands for s in w.steps)+1))
        actions=[spec.encode(REDUCE,p) for p in reversed(undos)]
        actions += [s.action.to_flat(spec) for s in w.steps]
        aug=UnknotWitness.from_actions(tuple(word),w.start.strands,spec,actions);aug.verify()
        fp=invariants(tuple(word),w.start.strands)
        if owner(fp)!='train' or (fp[0],abs(fp[1]))!=tuple(row['query_group']) or (tuple(word),w.start.strands) in held:
            raise ValueError('augmented root owner/exclusion differs; no substitution')
        inst,fits=query_instance(stage,aug,witness_caps=True)
        assert fits
        roots.append(dict(stage=stage,query_group=row['query_group'],witness=aug.to_dict(),
                          instances=[replace(inst,cc_weight=k).to_dict() for k in (10,1000)]))
    assert len({(tuple(r['instances'][0]['word']),r['instances'][0]['strands']) for r in roots})==4
    actor=snapshot()
    pins=[PROTOCOL,ROSTER,Path(__file__),Path('scripts/resource_cost_search.py'),
          Path('scripts/run_continuous_train_mcts_v1.py'),Path('scripts/run_continuous_train_mcts_v2.py'),
          Path('scripts/audit_random_representation_mcts_v1.py'),Path('scripts/audit_f_short_positive_r2_yield_v1.py'),
          *Path('src/rf_knots').glob('*.py'),
          *Path('/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench').glob('*.py')]
    OUT.mkdir()
    atomic(OUT/'roster.json',dict(rows=roots,role='random exact representations, NOT new knot groups or F labels'))
    atomic(OUT/'manifest.json',dict(status='prepared',next_task=0,shards=[],cpu_seconds=0,
        actor=actor,roster_sha256=sha(OUT/'roster.json'),inputs={str(p):sha(p) for p in pins},
        simulations=8,task_cpu_limit=20,total_tasks=32,seeds=list(range(20281016,20281020))))


def run(count):
    import fcntl
    if shutil.disk_usage(OUT).free<12*1024**3:raise ValueError('disk guard')
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read(OUT/'manifest.json')
        if m['status']=='completed':raise ValueError('completed pilot; never restart')
        for p,h in m['inputs'].items():assert sha(Path(p))==h,p
        assert sha(OUT/'roster.json')==m['roster_sha256']
        for e in m['shards']:assert sha(OUT/e['file'])==e['sha256']
        roots=read(OUT/'roster.json')['rows'];torch.set_num_threads(1);net,obj=policy(m['actor'])
        start=m['next_task']
        while m['next_task']<32 and m['next_task']-start<count:
            if shutil.disk_usage(OUT).free<12*1024**3:raise ValueError('disk guard')
            p=m['next_task'];root=p//8;wi=(p//4)%2;seed=20281016+p%4
            data=roots[root]['instances'][wi];started=time.process_time()
            signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,20)
            try:
                rows,counts=run_batch([dict(condition=root,panel_index=root,instance=data)],net,obj,0,
                    dict(simulations=8,insertion_candidates=4,widening_batch=4,c_puct=1.5,
                         temperature=1.0,seed=seed,horizon=data['cap_moves']),prior='distance',cached=True)
                row=rows[0];censored=False
            except TimeoutError:row=None;counts={};censored=True
            finally:signal.setitimer(signal.ITIMER_PROF,0)
            cpu=time.process_time()-started;file=OUT/f'attempt-{p:04d}.json'
            atomic(file,dict(position=p,root=root,weight=data['cc_weight'],seed=seed,row=row,
                             counts=dict(counts),cpu_censored=censored,cpu_seconds=cpu))
            m['shards'].append(dict(file=file.name,sha256=sha(file)));m['next_task']+=1;m['cpu_seconds']+=cpu
            m['status']='completed' if m['next_task']==32 else 'collecting';atomic(OUT/'manifest.json',m)
            print(dict(next_task=m['next_task'],cpu=cpu,censored=censored),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.stop_after and a.stop_after>0:run(a.stop_after)
    else:p.error('prepare or positive bounded stop-after required')
