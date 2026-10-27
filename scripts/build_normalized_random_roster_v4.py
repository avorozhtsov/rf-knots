"""Bounded random legal braid walks, novelty after actual normalization."""
import argparse
import hashlib
import random
import shutil
import signal
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from build_mcts_training_supervision import atomic,sha
from build_continuous_train_roster_v1 import exclusions
from run_continuous_train_mcts_v1 import read
from evaluate_embedding_cost_search import query_instance
from rf_knots import reference
from rf_knots.actions import COMMUTE,BRAID,ActionSpec
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.evidence import UnknotWitness

BASE=Path('artifacts/continuous-train-roster-v1/roster.json')
BANK=Path('artifacts/continuous-additive-f-bank-v55')
AUDIT=Path('artifacts/continuous-additive-f-bank-audit-v55/report.json')
PROTOCOL=Path('research/422-normalized-random-witness-format-correction-v4.md')


def move_choices(word,n,spec,visited):
    out=[]
    for kind in (COMMUTE,BRAID):
        for pos in range(len(word)):
            action=spec.encode(kind,pos)
            if reference.is_legal(spec,word,n,action,False):
                state=reference.apply(spec,word,n,action)
                if norm(*state) not in visited:out.append((action,state))
    return out


def known_states():
    _,held=exclusions();known={norm(*s) for s in held}
    for row in read(BASE)['rows']:
        w=UnknotWitness.from_dict(row['witness']);known.add(norm(w.start.word,w.start.strands))
    for e in read(BANK/'manifest.json')['roots']:
        p=BANK/e['file']
        if sha(p)!=e['sha256']:raise ValueError('old bank row pin differs')
        for g in read(p)['groups']:
            d=g['instance'];known.add(norm(tuple(d['word']),d['strands']))
    return known,held


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or shutil.disk_usage(a.out.parent).free<12*1024**3:raise ValueError('output/storage gate')
    assert sha(BASE)=='cea36964728381466158315252e6db2ea4f64eab17f3ba39e87f11d0a70586a0'
    assert sha(BANK/'manifest.json')=='9ab8d1643a60a5b5588a0d68a5a9677a8b9d540dd608a8b18666c8795d63b247'
    assert read(AUDIT)['manifest_sha256']==sha(BANK/'manifest.json') and read(AUDIT)['status']=='passed'
    pins=[BASE,BANK/'manifest.json',AUDIT,PROTOCOL,Path(__file__),Path('scripts/audit_normalized_random_roster_v4.py'),
          Path('tests/test_normalized_random_roster_v4.py'),Path('scripts/build_continuous_train_roster_v1.py'),
          *Path('src/rf_knots').glob('*.py')]
    report=dict(status='building',rows=[],bases=[],rejections={},inputs={str(f):sha(f) for f in pins},
                opportunities=32,cpu_ceiling=60,new_query_groups=0,F_labels=0,optimizer_steps=0)
    a.out.mkdir();atomic(a.out/'roster.json',report);start=time.process_time();rejected=Counter()
    def timeout(*_):raise TimeoutError('60CPU-second supply ceiling')
    signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,60)
    try:
        known,held=known_states();rows=read(BASE)['rows'];selected=[]
        for stage in range(4):
            pool=[r for r in rows if r['stage']==stage]
            pool.sort(key=lambda r:hashlib.sha256(f'20261017421:{r["query_group"]}:{r["source_index"]}'.encode()).hexdigest())
            selected.extend(pool[:4])
        assert len(selected)==16
        for bi,base in enumerate(selected):
            w=UnknotWitness.from_dict(base['witness']);w.verify();spec=ActionSpec(w.max_len,w.max_strands,w.cyclic_band_generators)
            report['bases'].append(dict(base_ordinal=bi,query_group=base['query_group'],stage=base['stage'],
                                       source_index=base['source_index'],word=list(w.start.word),strands=w.start.strands))
            for rep in range(2):
                seed=2028101700+2*bi+rep;rng=random.Random(seed)
                for attempt in range(32):
                    state=(w.start.word,w.start.strands);visited={norm(*state)};walk=[];ok=True
                    for depth in range(8):
                        choices=move_choices(*state,spec,visited)
                        if not choices:ok=False;rejected['walk_deadend']+=1;break
                        action,after=rng.choice(choices)
                        assert reference.is_legal(spec,*after,action,False) and reference.apply(spec,*after,action)==state
                        walk.append(action);state=after;visited.add(norm(*state))
                    if not ok:continue
                    key=norm(*state)
                    if key in known or state in held:rejected['normalized_known_or_heldout']+=1;continue
                    fp=invariants(*state)
                    if owner(fp)!='train' or (fp[0],abs(fp[1]))!=tuple(base['query_group']):raise ValueError('walk ownership changed')
                    inverse=list(reversed(walk));actions=inverse+[s.action.to_flat(spec) for s in w.steps]
                    proof=UnknotWitness.from_actions(*state,spec,actions);proof.verify()
                    if proof.moves>192 or len(state[0])>192 or state[1]>20:rejected['caps']+=1;continue
                    instance,fits=query_instance(bi*2+rep,proof,witness_caps=True)
                    if not fits:rejected['reference_caps']+=1;continue
                    report['rows'].append(dict(base_ordinal=bi,rep=rep,seed=seed,attempt=attempt,walk=walk,
                        stage=base['stage'],query_group=base['query_group'],witness=proof.to_dict(),
                        normalized_word=list(key[0]),normalized_strands=key[1],
                        instances=[replace(instance,cc_weight=v).to_dict() for v in (10,1000)]))
                    known.add(key);break
                else:rejected['opportunity_exhausted']+=1
            report.update(rejections=dict(rejected),cpu_seconds=time.process_time()-start);atomic(a.out/'roster.json',report)
        groups={tuple(r['query_group']) for r in report['rows']};stages={r['stage'] for r in report['rows']}
        report.update(status='completed',distinct_normalized_representations=len(report['rows']),base_groups_with_views=len(groups),
            decision='separate_small_MCTS_contract_only' if len(report['rows'])>=16 and len(groups)>=8 and len(stages)==4 else 'stop_generator_review')
    except TimeoutError:report.update(status='cpu_ceiling',decision='partial_no_scale')
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);report.update(cpu_seconds=time.process_time()-start,rejections=dict(rejected));atomic(a.out/'roster.json',report)
    print({k:v for k,v in report.items() if k not in ['rows','bases','inputs']},flush=True)


if __name__=='__main__':main()
