"""Independent STRUCTURAL-only check; composed solution paths remain unproved."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from build_mcts_training_supervision import atomic, sha
from build_continuous_train_roster_v1 import exclusions
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.evidence import UnknotWitness


def read(p):return json.loads(p.read_text())
def rank(x):return hashlib.sha256(('20261016413:'+str(x)).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('immutable audit exists')
    r=read(args.data/'report.json')
    if r['status']!='completed' or len(r['bases'])!=32 or len(r['pairs'])!=64 or r['admitted_F_roots']!=0:
        raise ValueError('full structural screen missing')
    for name,h in r['inputs'].items():
        if sha(Path(name))!=h:raise ValueError('input SHA differs '+name)
    rows=read(Path('artifacts/continuous-train-roster-v1/roster.json'))['rows']
    rows=sorted((x for x in rows if x['witness']['crossing_changes']==1),
        key=lambda x:rank((x['query_group'],x['origin'],x['source_index'])))
    chosen=[];seen=set()
    for row in rows:
        key=tuple(row['query_group'])
        if key not in seen:seen.add(key);chosen.append(row)
        if len(chosen)==32:break
    for row,b in zip(chosen,r['bases']):
        w=UnknotWitness.from_dict(row['witness']);w.verify()
        states=[(w.start,0)]
        for i,s in enumerate(w.steps):
            if s.action.kind=='CROSSING_CHANGE':break
            states.append((s.after,i+1))
        state,cut=min(states,key=lambda x:(len(x[0].word),x[0].strands,x[1]))
        if (b['source_index']!=row['source_index'] or b['origin']!=row['origin']
                or b['word']!=list(state.word) or b['strands']!=state.strands
                or b['prefix_moves']!=cut or b['suffix_moves']!=len(w.steps)-cut
                or b['semantic_kinds']!=sorted({s.action.kind for s in w.steps[cut:]})):
            raise ValueError('base deterministic replay/state differs')
        if list(invariants(tuple(state.word),state.strands))!=b['fingerprint']:raise ValueError('base invariant differs')
    excluded,held=exclusions()
    manifests=[Path(x) for x in r['inputs'] if x.endswith('/manifest.json') and 'bank-v' in x]
    if len(manifests)!=1:raise ValueError('exclusion bank ambiguous')
    bank=manifests[0].parent
    for e in read(manifests[0])['roots']:
        if sha(bank/e['file'])!=e['sha256']:raise ValueError('bank row SHA differs')
        excluded.add(tuple(read(bank/e['file'])['query_group']))
    # Additionally prove disjointness against every existing prior candidate roster,
    # not only roots already admitted as F labels. No reselection/substitution.
    prior=set();roster_pins={}
    def collect(x):
        if isinstance(x,dict):
            q=x.get('query_group')
            if isinstance(q,list) and len(q)==2 and all(isinstance(v,int) for v in q):prior.add(tuple(q))
            for v in x.values():collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    files=set(Path('artifacts').glob('*roster*/roster.json'))|set(Path('artifacts').glob('*roster*.json'))
    for f in sorted(files):
        collect(read(f));roster_pins[str(f)]=sha(f)
    pairs=sorted(itertools.combinations(range(32),2),key=lambda x:rank(('pair',x)))[:64]
    local={'REDUCE','COMMUTE','BRAID','CROSSING_CHANGE','INSERT','STABILIZE_POS','STABILIZE_NEG','DESTABILIZE'}
    novel=set()
    for (i,j),e in zip(pairs,r['pairs']):
        if e['base_indices']!=[i,j]:raise ValueError('fixed pair order differs')
        a,b=r['bases'][i],r['bases'][j]
        word=tuple(a['word'])+tuple((1 if g>0 else -1)*(abs(g)+a['strands']-1) for g in b['word'])
        strands=a['strands']+b['strands']-1
        if len(word)>192 or strands>20 or a['suffix_moves']+b['suffix_moves']>160:status='structural_caps'
        elif set(a['semantic_kinds']+b['semantic_kinds'])-local:status='unmapped_reference_action'
        else:
            fp=invariants(word,strands);key=(fp[0],abs(fp[1]))
            if fp!=(a['fingerprint'][0]*b['fingerprint'][0],a['fingerprint'][1]+b['fingerprint'][1]):raise ValueError('composite invariants differ')
            if owner(fp)!='train':status='not_train'
            elif key in excluded:status='old_or_heldout_query_group'
            elif (word,strands) in held:status='heldout_exact_state'
            elif key in novel:status='duplicate_novel_group'
            else:
                if key in prior:raise ValueError('new candidate overlaps prior roster group')
                status='novel_structural_TRAIN_candidate';novel.add(key)
                if e.get('word')!=list(word) or e['query_group']!=list(key):raise ValueError('candidate root differs')
        if e['status']!=status:raise ValueError('independent disposition differs')
    if len(novel)!=r['novel_distinct_TRAIN_query_groups']:raise ValueError('distinct accounting differs')
    report=dict(status='passed_structural_only',screen_sha256=sha(args.data/'report.json'),
        bases_full_witness_replays=32,actual_composite_pairs_checked=64,
        novel_distinct_TRAIN_candidate_groups=len(novel),prior_roster_groups=len(prior),
        prior_roster_pins=roster_pins,admitted_F_roots=0,composed_paths_replayed=0,
        decision='separate_frozen_reference_lifting_test_only' if len(novel)>=16 else 'stop_sampled_recipe',
        auditor_sha256=sha(Path(__file__)))
    args.out.mkdir(parents=True);atomic(args.out/'report.json',report)
    print(json.dumps(dict(report_sha256=sha(args.out/'report.json'),summary=report)),flush=True)


if __name__=='__main__':main()
