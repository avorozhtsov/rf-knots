"""Versioned generation/search/audit/additive-data loop. Never modifies a trainer."""
import argparse,fcntl,hashlib,os,random,shutil,signal,subprocess,sys,time
from pathlib import Path
from dataclasses import replace
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from build_short_horizon_random_roster_v5 import BASE,exclusions,suffix
from build_normalized_random_roster_v4 import move_choices
from export_mcts_root_positives_v1 import external_groups
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from evaluate_embedding_cost_search import query_instance

OUT=Path('artifacts/continuous-random-TRAIN-stream-v2')
PROTOCOL=Path('research/439-random-stream-v2-horizon24-and-base-novelty-review.md')
SEED=20281017436
HORIZON=24

def input_files():
    return [PROTOCOL,Path(__file__),Path('scripts/audit_continuous_random_stream_v2.py'),
        Path('scripts/continuous_random_positive_bank_v1.py'),Path('tests/test_continuous_random_stream_v2.py'),
        Path('scripts/build_short_horizon_random_roster_v5.py'),Path('scripts/build_normalized_random_roster_v4.py'),
        Path('scripts/random_representation_mcts_pilot_v1.py'),Path('scripts/audit_random_representation_mcts_v1.py'),
        Path('scripts/export_mcts_root_positives_v1.py'),Path('scripts/audit_mcts_root_positives_v1.py'),
        Path('scripts/resource_cost_search.py'),Path('scripts/run_continuous_train_mcts_v1.py'),
        Path('scripts/run_continuous_train_mcts_v2.py'),BASE,
        Path('artifacts/continuous-additive-f-bank-v59/manifest.json'),
        Path('artifacts/continuous-additive-f-bank-audit-v59/report.json'),
        Path('artifacts/normalized-random-representation-roster-v4/roster.json'),
        Path('artifacts/short-horizon-random-roster-v5/roster.json'),
        *Path('src/rf_knots').glob('*.py'),*Path('/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench').glob('*.py')]

def checkpins(pins):
    for f,h in pins.items():
        if sha(Path(f))!=h:raise ValueError('pin changed: '+f)

def seed_for(batch,opportunity):return SEED+batch*1000+opportunity

def known_from(priors):
    known=exclusions()
    for f in [Path('artifacts/short-horizon-random-roster-v5/roster.json'),*map(Path,priors)]:
        for row in read(f)['rows']:
            w=UnknotWitness.from_dict(row['witness']);known.add(norm(w.start.word,w.start.strands))
    return known

def generate(batchdir):
    c=read(batchdir/'supply-contract.json');checkpins(c['inputs']);known=known_from(c['prior_rosters']);excluded=external_groups()
    start=time.process_time();rows=[];bases=[];groups=set();op=0
    def timeout(*_):raise TimeoutError('supply30CPU ceiling')
    signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,30)
    try:
        source=read(BASE)['rows'];order=sorted(range(len(source)),key=lambda i:hashlib.sha256(f'{SEED}:{c["batch"]}:{i}'.encode()).hexdigest())
        for i in order:
            w=suffix(UnknotWitness.from_dict(source[i]['witness']),remaining=HORIZON)
            if w is None or not w.start.word:continue
            fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
            if owner(fp)!='train' or g in excluded or g in groups:continue
            if op>=32:break
            rng=random.Random(seed_for(c['batch'],op));op+=1;spec=ActionSpec(w.max_len,w.max_strands,w.cyclic_band_generators)
            state=(w.start.word,w.start.strands);visited={norm(*state)};walk=[]
            for _ in range(2):
                choices=move_choices(*state,spec,visited)
                if not choices:break
                action,state=rng.choice(choices);walk.append(action);visited.add(norm(*state))
            if len(walk)!=2 or norm(*state) in known:continue
            fp2=invariants(*state);assert owner(fp2)=='train' and (fp2[0],abs(fp2[1]))==g
            proof=UnknotWitness.from_actions(*state,spec,list(reversed(walk))+[s.action.to_flat(spec) for s in w.steps])
            inst,fits=query_instance(len(rows),proof,witness_caps=True)
            if not fits or proof.moves>192 or len(state[0])>192 or state[1]>20:continue
            bases.append(dict(source_index=i,witness=w.to_dict(),query_group=list(g)))
            k=norm(*state);rows.append(dict(base_ordinal=len(bases)-1,opportunity=op-1,query_group=list(g),walk=walk,witness=proof.to_dict(),
                normalized_word=list(k[0]),normalized_strands=k[1],instances=[replace(inst,cc_weight=v).to_dict() for v in (10,1000)]))
            known.add(k);groups.add(g)
            if len(rows)==4:break
        status='completed' if len(rows)==4 else 'exhausted'
    except TimeoutError:status='cpu_ceiling'
    finally:signal.setitimer(signal.ITIMER_PROF,0)
    atomic(batchdir/'supply.json',dict(status=status,rows=rows,bases=bases,inputs=c['inputs'],prior_rosters=c['prior_rosters'],
        batch=c['batch'],opportunities=op,cpu_seconds=time.process_time()-start,normalized_views=len(rows),query_groups=len(groups)))

def prepare_search(batchdir):
    from run_continuous_train_mcts_v2 import snapshot
    a=read(batchdir/'supply-audit/report.json');assert a['status']=='passed' and a['roster_sha256']==sha(batchdir/'supply.json')
    d=batchdir/'search';d.mkdir();atomic(d/'roster.json',dict(rows=read(batchdir/'supply.json')['rows']))
    pins={str(f):sha(f) for f in input_files()+[batchdir/'supply.json',batchdir/'supply-audit/report.json']}
    atomic(d/'manifest.json',dict(status='prepared',next_task=0,shards=[],cpu_seconds=0,actor=snapshot(),roster_sha256=sha(d/'roster.json'),
        inputs=pins,simulations=8,task_cpu_limit=20,total_tasks=32,seeds=list(range(20281016,20281020))))

def invoke(*args):subprocess.run([sys.executable,*args],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

def manager(resume):
    if not resume:
        if OUT.exists():raise ValueError('immutable stream exists')
        OUT.mkdir();atomic(OUT/'manifest.json',dict(status='active',batch=0,completed=[],inputs={str(f):sha(f) for f in input_files()},
            current_bank='artifacts/continuous-random-TRAIN-stream-v1/batch-000002/bank',current_audit='artifacts/continuous-random-TRAIN-stream-v1/batch-000002/bank-audit/report.json'))
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=read(OUT/'manifest.json')
        if m['status'] not in ('active','interrupted'):raise ValueError('stopped stream requires review, not blind resume')
        checkpins(m['inputs']);m['pid']=os.getpid();atomic(OUT/'manifest.json',m)
        while True:
            checkpins(m['inputs'])
            if (OUT/'STOP_REQUEST').exists() or shutil.disk_usage(OUT).free<12*1024**3:
                m.update(status='resource_or_user_stop');atomic(OUT/'manifest.json',m);return
            d=OUT/f'batch-{m["batch"]:06d}'
            if not d.exists():
                d.mkdir();priors=[str(Path('artifacts/continuous-random-TRAIN-stream-v1')/f'batch-{i:06d}'/'supply.json') for i in range(4)]+[str(OUT/f'batch-{i:06d}'/'supply.json') for i in range(m['batch'])]
                pins={**m['inputs'],**{f:sha(Path(f)) for f in priors}}
                atomic(d/'supply-contract.json',dict(batch=m['batch'],inputs=pins,prior_rosters=priors,seed=SEED,supplyCPU=30,opportunities=32))
            if not (d/'supply.json').exists():invoke(__file__,'--generate',str(d))
            if read(d/'supply.json')['status']!='completed':
                m.update(status='supply_exhausted');atomic(OUT/'manifest.json',m);return
            if not (d/'supply-audit/report.json').exists():invoke('scripts/audit_continuous_random_stream_v2.py','--supply',str(d))
            if not (d/'search').exists():prepare_search(d)
            sm=read(d/'search/manifest.json')
            if sm['next_task']==0:invoke(__file__,'--search',str(d),'--count','1')
            if not (d/'first-audit/report.json').exists():invoke('scripts/audit_continuous_random_stream_v2.py','--search',str(d),'--partial')
            sm=read(d/'search/manifest.json')
            if sm['next_task']<32:invoke(__file__,'--search',str(d),'--count',str(32-sm['next_task']))
            if not (d/'full-audit/report.json').exists():invoke('scripts/audit_continuous_random_stream_v2.py','--search',str(d))
            if not (d/'bank/manifest.json').exists():invoke('scripts/continuous_random_positive_bank_v1.py','--build',str(d),'--old-bank',m['current_bank'],'--old-audit',m['current_audit'])
            if not (d/'bank-audit/report.json').exists():invoke('scripts/continuous_random_positive_bank_v1.py','--audit',str(d))
            report=read(d/'bank-audit/report.json')
            if not report['yield_gate']:
                m.update(status='yield_stop');atomic(OUT/'manifest.json',m);return
            m['current_bank']=str(d/'bank');m['current_audit']=str(d/'bank-audit/report.json')
            m['completed'].append(dict(batch=m['batch'],bank_sha256=sha(d/'bank/manifest.json'),audit_sha256=sha(d/'bank-audit/report.json'),report=report))
            m['batch']+=1;atomic(OUT/'manifest.json',m)
            atomic(OUT/'READY.json',dict(bank=m['current_bank'],audit=m['current_audit'],manifest_sha256=sha(d/'bank/manifest.json'),audit_sha256=sha(d/'bank-audit/report.json'),optimizer_connected=False))
            print(dict(batch=m['batch'],ready=m['current_bank'],report=report),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--generate',type=Path);p.add_argument('--search',type=Path);p.add_argument('--count',type=int);a=p.parse_args()
    if a.generate:generate(a.generate)
    elif a.search:
        import random_representation_mcts_pilot_v1 as e
        assert a.count and 0<a.count<=32;e.OUT=a.search/'search';e.run(a.count)
    else:manager(a.resume)
if __name__=='__main__':main()
