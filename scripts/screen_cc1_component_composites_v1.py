"""Bounded structural TRAIN supply screen; NEVER emits F training labels."""
import argparse
import hashlib
import itertools
import json
import signal
import time
from collections import Counter
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from build_continuous_train_roster_v1 import exclusions
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness

ROSTER = Path('artifacts/continuous-train-roster-v1/roster.json')
ROSTER_SHA = 'cea36964728381466158315252e6db2ea4f64eab17f3ba39e87f11d0a70586a0'
AUDIT = Path('artifacts/continuous-train-roster-audit-v1/report.json')
AUDIT_SHA = 'ea8270f30074d3a365deb539b997302b0211dabcc5941a6ba4b6473932849cbf'
PROTOCOL = Path('research/413-fixed210-prepared-intake-and-cc1-composite-supply-screen.md')
LOCAL = {'REDUCE', 'COMMUTE', 'BRAID', 'CROSSING_CHANGE', 'INSERT',
         'STABILIZE_POS', 'STABILIZE_NEG', 'DESTABILIZE'}


def read(p):
    return json.loads(p.read_text())


def rank(value):
    return hashlib.sha256(('20261016413:'+str(value)).encode()).hexdigest()


def compose(a, b):
    word = tuple(a[0]) + tuple((1 if g > 0 else -1)*(abs(g)+a[1]-1) for g in b[0])
    return word, a[1]+b[1]-1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--bank-audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or sha(ROSTER) != ROSTER_SHA or sha(AUDIT) != AUDIT_SHA:
        raise ValueError('immutable output/input identity differs')
    bank_audit=read(args.bank_audit)
    if bank_audit['status']!='passed' or bank_audit['manifest_sha256']!=sha(args.bank/'manifest.json'):
        raise ValueError('prepared exclusion bank audit differs')
    pins = {str(x):sha(x) for x in [ROSTER,AUDIT,PROTOCOL,Path(__file__),args.bank/'manifest.json',
        args.bank_audit,Path('scripts/build_continuous_train_roster_v1.py'),
        Path('src/rf_knots/embedding_replay_corpus.py'),Path('src/rf_knots/evidence.py')]}
    rows = read(ROSTER)['rows']
    raw = [r for r in rows if r['witness']['crossing_changes'] == 1]
    raw.sort(key=lambda r:rank((r['query_group'],r['origin'],r['source_index'])))
    chosen = []
    groups = set()
    for r in raw:
        key = tuple(r['query_group'])
        if key not in groups:
            groups.add(key);chosen.append(r)
        if len(chosen)==32:break
    if len(chosen)!=32:raise ValueError('insufficient fixed CC1 basegroups')
    pairs = sorted(itertools.combinations(range(32),2),key=lambda x:rank(('pair',x)))[:64]
    report = dict(status='building', role='STRUCTURAL_CANDIDATES_ONLY_NO_F_LABELS',
        inputs=pins, base_pool_CC1=len(raw), fixed_basegroups=32, fixed_pairs=64,
        cpu_ceiling_seconds=45, bases=[], pairs=[], rejection_counts={}, admitted_F_roots=0)
    args.out.mkdir(parents=True)
    started = time.process_time()
    def ceiling(signum, frame):raise TimeoutError('fixed45CPU-second ceiling')
    signal.signal(signal.SIGPROF,ceiling)
    signal.setitimer(signal.ITIMER_PROF,45)
    rejections = Counter()
    try:
        excluded, held_states = exclusions()
        for e in read(args.bank/'manifest.json')['roots']:
            if sha(args.bank/e['file'])!=e['sha256']:raise ValueError('prepared bank row SHA differs')
            excluded.add(tuple(read(args.bank/e['file'])['query_group']))
        report['excluded_group_count']=len(excluded)
        report['excluded_group_digest']=hashlib.sha256(json.dumps(sorted(excluded)).encode()).hexdigest()
        report['held_exact_state_count']=len(held_states)
        report['held_exact_state_digest']=hashlib.sha256(json.dumps(sorted(held_states)).encode()).hexdigest()
        for row in chosen:
            w = UnknotWitness.from_dict(row['witness']);w.verify()
            if w.crossing_changes!=1:raise ValueError('base reference CC differs')
            prefix=[(w.start,0)]
            for i,s in enumerate(w.steps):
                if s.action.kind=='CROSSING_CHANGE':break
                prefix.append((s.after,i+1))
            state,cut=min(prefix,key=lambda x:(len(x[0].word),x[0].strands,x[1]))
            fp=invariants(tuple(state.word),state.strands)
            if owner(fp)!='train' or (fp[0],abs(fp[1]))!=tuple(row['query_group']):
                raise ValueError('base prefix invariant/ownership mismatch')
            suffix=w.steps[cut:]
            report['bases'].append(dict(source_index=row['source_index'],origin=row['origin'],
                query_group=row['query_group'],word=list(state.word),strands=state.strands,
                fingerprint=list(fp),prefix_moves=cut,suffix_moves=len(suffix),
                semantic_kinds=sorted({s.action.kind for s in suffix})))
        seen=set()
        for i,j in pairs:
            a,b=report['bases'][i],report['bases'][j]
            word,strands=compose((a['word'],a['strands']),(b['word'],b['strands']))
            e=dict(base_indices=[i,j],word_length=len(word),strands=strands,
                   proposed_reference_CC=2,proposed_suffix_moves=a['suffix_moves']+b['suffix_moves'])
            if len(word)>192 or strands>20 or e['proposed_suffix_moves']>160:
                e['status']='structural_caps'
            elif set(a['semantic_kinds']+b['semantic_kinds'])-LOCAL:
                e['status']='unmapped_reference_action'
            else:
                fp=invariants(word,strands)
                expected=(a['fingerprint'][0]*b['fingerprint'][0],a['fingerprint'][1]+b['fingerprint'][1])
                if fp!=expected:raise ValueError('actual composite invariants differ from proposed composition')
                key=(fp[0],abs(fp[1]));e.update(fingerprint=list(fp),query_group=list(key),owner=owner(fp))
                if owner(fp)!='train':e['status']='not_train'
                elif key in excluded:e['status']='old_or_heldout_query_group'
                elif (word,strands) in held_states:e['status']='heldout_exact_state'
                elif key in seen:e['status']='duplicate_novel_group'
                else:
                    seen.add(key);e.update(status='novel_structural_TRAIN_candidate',word=list(word))
            rejections[e['status']]+=1;report['pairs'].append(e)
        report.update(status='completed', novel_distinct_TRAIN_query_groups=len(seen),
            decision='allow_separate_reference_lifting_feasibility_test' if len(seen)>=16 else 'stop_sampled_recipe')
    except TimeoutError:
        report.update(status='cpu_ceiling',decision='no_scale_partial_censored')
    finally:
        signal.setitimer(signal.ITIMER_PROF,0)
        report.update(cpu_seconds=time.process_time()-started,rejection_counts=dict(rejections))
        atomic(args.out/'report.json',report)
    print(json.dumps(dict(status=report['status'],decision=report.get('decision'),
        processed_pairs=len(report['pairs']),novel_groups=report.get('novel_distinct_TRAIN_query_groups'),
        admitted_F_roots=0,report_sha256=sha(args.out/'report.json'))),flush=True)


if __name__=='__main__':main()
