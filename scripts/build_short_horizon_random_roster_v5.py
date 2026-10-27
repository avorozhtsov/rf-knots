"""Prospective430: score-blind short-horizon curriculum, supply only."""
import argparse, hashlib, random, shutil, signal, time
from pathlib import Path
from dataclasses import replace
from collections import Counter
from build_mcts_training_supervision import atomic, sha
from build_normalized_random_roster_v4 import move_choices, known_states
from run_continuous_train_mcts_v1 import read
from evaluate_embedding_cost_search import query_instance
from rf_knots.actions import ActionSpec
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm

BASE=Path('artifacts/continuous-train-roster-v1/roster.json')
BANK=Path('artifacts/continuous-additive-f-bank-v59')
PRIOR=Path('artifacts/normalized-random-representation-roster-v4/roster.json')
PROTOCOL=Path('research/430-short-horizon-random-train-supply-proposal.md')
SEED=20281017430

def suffix(w,remaining=16):
    if len(w.steps)<remaining:return None
    i=len(w.steps)-remaining
    start=w.start if i==0 else w.steps[i-1].after
    return UnknotWitness(start,w.max_len,w.max_strands,w.steps[i:],w.cyclic_band_generators)

def exclusions():
    known,held=known_states();known.update(norm(*s) for s in held)
    for e in read(BANK/'manifest.json')['roots']:
        p=BANK/e['file'];assert sha(p)==e['sha256']
        for g in read(p)['groups']:
            d=g['instance'];known.add(norm(tuple(d['word']),d['strands']))
    for r in read(PRIOR)['rows']:
        w=UnknotWitness.from_dict(r['witness']);known.add(norm(w.start.word,w.start.strands))
    return known

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists() or shutil.disk_usage(a.out.parent).free<12*1024**3:raise ValueError('output/storage gate')
    pins=[BASE,BANK/'manifest.json',Path('artifacts/continuous-additive-f-bank-audit-v59/report.json'),PRIOR,
          Path('artifacts/normalized-random-representation-roster-audit-v4/report.json'),PROTOCOL,Path(__file__),
          Path('scripts/audit_short_horizon_random_roster_v5.py'),Path('tests/test_short_horizon_random_roster_v5.py'),
          Path('scripts/build_normalized_random_roster_v4.py'),*Path('src/rf_knots').glob('*.py')]
    r=dict(status='building',inputs={str(f):sha(f) for f in pins},seed=SEED,rows=[],bases=[],cpu_ceiling=30,
           F_labels=0,optimizer_steps=0,source='constructive_reference_NOT_MCTS',decision='pending')
    a.out.mkdir();atomic(a.out/'contract.json',r);atomic(a.out/'roster.json',r)
    start=time.process_time();reject=Counter()
    def timeout(*_):raise TimeoutError('30CPU seconds')
    signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,30)
    try:
        known=exclusions();candidates=[]
        for i,row in enumerate(read(BASE)['rows']):
            w=suffix(UnknotWitness.from_dict(row['witness']))
            if w is None or not w.start.word:continue
            fp=invariants(w.start.word,w.start.strands)
            if owner(fp)!='train' or norm(w.start.word,w.start.strands) in known:continue
            key=hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest()
            candidates.append((key,i,w,(fp[0],abs(fp[1]))))
        groups=set()
        for _,i,w,group in sorted(candidates):
            if group in groups:continue
            w.verify();groups.add(group);r['bases'].append(dict(source_index=i,query_group=list(group),witness=w.to_dict()))
            if len(groups)==16:break
        for bi,b in enumerate(r['bases']):
            w=UnknotWitness.from_dict(b['witness']);spec=ActionSpec(w.max_len,w.max_strands,w.cyclic_band_generators)
            for rep in range(2):
                rng=random.Random(SEED+bi*2+rep);state=(w.start.word,w.start.strands);visited={norm(*state)};walk=[]
                for _ in range(2):
                    choices=move_choices(*state,spec,visited)
                    if not choices:break
                    action,state=rng.choice(choices);walk.append(action);visited.add(norm(*state))
                if len(walk)!=2:reject['deadend']+=1;continue
                k=norm(*state)
                if k in known:reject['normalized_known']+=1;continue
                fp=invariants(*state)
                if owner(fp)!='train' or (fp[0],abs(fp[1]))!=tuple(b['query_group']):raise ValueError('owner/group changed')
                proof=UnknotWitness.from_actions(*state,spec,list(reversed(walk))+[s.action.to_flat(spec) for s in w.steps])
                inst,fits=query_instance(bi*2+rep,proof,witness_caps=True)
                if not fits or len(state[0])>192 or state[1]>20 or proof.moves>192:reject['caps']+=1;continue
                r['rows'].append(dict(base_ordinal=bi,rep=rep,walk=walk,query_group=b['query_group'],witness=proof.to_dict(),
                    normalized_word=list(k[0]),normalized_strands=k[1],instances=[replace(inst,cc_weight=v).to_dict() for v in (10,1000)]));known.add(k)
        n=len(r['rows']);g=len({tuple(x['query_group']) for x in r['rows']})
        r.update(status='completed',distinct_normalized_views=n,query_groups=g,
                 decision='separate_MCTS_contract_only' if n>=8 and g>=4 else 'stop_configuration')
    except TimeoutError:r.update(status='cpu_ceiling',decision='partial_no_scale')
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);r.update(cpu_seconds=time.process_time()-start,rejections=dict(reject));atomic(a.out/'roster.json',r)
    print({k:v for k,v in r.items() if k not in ['inputs','rows','bases']},flush=True)

if __name__=='__main__':main()
