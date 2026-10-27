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

STUDY=Path('artifacts/paired-learning-48h100k-v3')
PROTOCOL=Path('artifacts/paired-learning-48h100k-v3-protocol.json')

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
    count=0;conditions=0;seen_batches=set();seen_shards=set()
    for a in m['audits']:
        if a['batch']*32<first:continue
        assert a['batch'] not in seen_batches,'duplicate audited batch is not new experience'
        seen_batches.add(a['batch'])
        p=Path(a['report']);assert sha(p)==a['sha256'];r=read(p)
        assert not r['partial'] and r['conditions']==32
        bp=data/f"batch-{a['batch']:06d}"/'manifest.json';assert sha(bp)==r['batch_manifest_sha256']
        batch=read(bp);actual=0
        for e in batch['shards']:
            assert e['file'] not in seen_shards,'duplicate shard is not new experience'
            seen_shards.add(e['file'])
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
    # SAME verified mechanical reductions for both networks: isolate learning.
    from verified_reduction_search_v28 import search
    r=search(instance,net,obj,seed,cpu_seconds=30.,simulations=4096)
    r.update(L10=10*r['CC']+r['moves'] if r['solved'] else None,
        L1000=1000*r['CC']+r['moves'] if r['solved'] else None,
        incomplete_simulation_discarded=r['cpu_censored'])
    return r

def run_evaluation(candidate):
    from rf_knots import mcts_loop as M
    from freeze_learning_comparison_v3 import state_id,inspect,substantial
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
        assert state_id(w.start.word,w.start.strands) not in states
        assert all(state_id(x.word,x.strands) not in states for x in [w.start,*[s.after for s in w.steps]] if substantial(x.word,x.strands))
    actors={}
    for name,entry in [('baseline',b),('candidate',candidate)]:
        actors[name]=policy(dict(file=entry['checkpoint'],sha256=entry['checkpoint_sha256'],step=entry['step'],contract_sha256=entry['contract_sha256']))
        assert all(p.device.type=='cpu' and p.dtype==torch.float32 for obj in actors[name] for p in obj.parameters())
    out=STUDY/'evaluation';out.mkdir(exist_ok=True)
    for root,row in enumerate(panel['rows']):
        for wi,weight in enumerate([10,1000]):
            for seed in [20291023,20291024]:
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
            from analyze_learning_comparison_v3 import main as analyze
            if not (STUDY/'evaluation/analysis.json').exists():analyze()

if __name__=='__main__':main()
