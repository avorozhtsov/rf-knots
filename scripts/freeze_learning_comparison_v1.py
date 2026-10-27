"""Freeze baseline, search cursor, exclusions and unseen score-blind panel."""
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
from build_f_curriculum_roster_v1 import excluded_groups
from screen_legacy_unselected_root_f_v1 import ROSTERS
from evaluate_embedding_cost_search import query_instance
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import invariants, owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from rf_knots.evidence import UnknotWitness

OUT=Path('artifacts/paired-learning-48h100k-v1')
PROTOCOL=Path('artifacts/paired-learning-48h100k-v1-protocol.json')

def state_id(word,strands):
    w,n=norm(tuple(word),strands)
    return hashlib.sha256(json.dumps([list(w),n],separators=(',',':')).encode()).hexdigest()

def inspect(data,groups,states):
    if isinstance(data,dict):
        g=data.get('query_group')
        if isinstance(g,list) and len(g)==2 and all(isinstance(x,int) for x in g):groups.add(tuple(g))
        if 'start' in data and 'steps' in data and 'max_len' in data:
            w=UnknotWitness.from_dict(data)
            for s in [w.start,*[x.after for x in w.steps]]:
                if len(s.word)>=20:states.add(state_id(s.word,s.strands))
            fp=invariants(w.start.word,w.start.strands);groups.add((fp[0],abs(fp[1])))
            return
        if isinstance(data.get('word'),list) and isinstance(data.get('strands'),int):
            if len(data['word'])>=20:states.add(state_id(data['word'],data['strands']))
        for v in data.values():inspect(v,groups,states)
    elif isinstance(data,list):
        for v in data:inspect(v,groups,states)

def main():
    assert not OUT.exists(),'immutable study already exists'
    assert shutil.disk_usage('.').free>=12*1024**3
    started=time.time(); protocol=read(PROTOCOL)
    ref=read(Path('artifacts/continuous-r2-current-training.json'));specpath=Path(ref['spec']);spec=read(specpath)
    run=Path(spec['out']);manifest=read(run/'manifest.json');entry=manifest['checkpoints'][-1]
    cp=run/entry['file'];assert sha(cp)==entry['sha256']
    # Only complete immutable actor weights; never claim the live unsaved state.
    OUT.mkdir();shutil.copy2(cp,OUT/'baseline.pt');assert sha(OUT/'baseline.pt')==entry['sha256']
    atomic(OUT/'baseline-training-manifest.json',manifest)
    atomic(OUT/'baseline-training-spec.json',spec)
    mainstream=Path('artifacts/continuous-train-mcts-v2');mm=read(mainstream/'manifest.json')
    atomic(OUT/'baseline-search-manifest.json',mm)
    atomic(OUT/'baseline.json',dict(created_unix=started,checkpoint=str(OUT/'baseline.pt'),
        checkpoint_sha256=entry['sha256'],step=entry['step'],source_checkpoint=str(cp),
        run=str(run),spec=str(specpath),spec_sha256=sha(specpath),contract_sha256=manifest['contract_sha256'],
        checkpointed_active_origin_seconds=manifest['new_active_seconds'],main_search=str(mainstream),
        first_new_task=mm['next_task'],note='Baseline is latest full checkpoint, not live unsaved weights.'))
    panels=sorted(Path('artifacts').glob('*/panel.json'))
    files=set(panels)|set(ROSTERS)|{Path('artifacts/continuous-train-roster-v1/roster.json')}
    for k in ['bank','positive_bank','random_positive_bank']:
        b=Path(spec[k]);bm=read(b/'manifest.json');files.add(b/'manifest.json')
        files.update(b/e['file'] for e in bm.get('paths',bm.get('roots',[])))
    groups=excluded_groups();states=set();pins={}
    for f in sorted(files):
        pins[str(f)]=sha(f);inspect(read(f),groups,states)
    registry=dict(inputs=pins,excluded_query_groups=[list(g) for g in sorted(groups)],
        excluded_normalized_state_sha256=sorted(states),previous_panel_paths=list(map(str,panels)))
    atomic(OUT/'exclusions.json',registry)
    seed=protocol['holdout']['generator_seed'];maximum=protocol['holdout']['maximum_generator_indices']
    quotas=[8]*4;bands=[[] for _ in quotas];seen=set(groups);seenstates=set();rejects={}
    atomic(OUT/'status.json',dict(status='building_score_blind_panel',baseline_sha256=entry['sha256']))
    for i in sorted(range(maximum),key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()):
        w=conjugate_witness(seed,i,max_strands=20,min_depth=8,max_depth=60)
        n=len(w.start.word);band=0 if 20<=n<=64 else 1 if 65<=n<=96 else 2 if 97<=n<=128 else 3 if 129<=n<=192 else None
        reason=None
        if band is None or len(bands[band])==8:reason='length_or_filled'
        else:
            fp=invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
            if owner(fp)!='validation':reason='not_validation'
            elif g in seen:reason='group_exclusion'
            else:
                ss={state_id(s.word,s.strands) for s in [w.start,*[x.after for x in w.steps]] if len(s.word)>=20}
                if ss & states or state_id(w.start.word,w.start.strands) in seenstates:reason='normalized_overlap'
                else:
                    w.verify();base,fits=query_instance(i,w,witness_caps=True)
                    assert fits
                    bands[band].append(dict(generator_index=i,band=band,query_group=list(g),witness=w.to_dict(),
                        instances=[replace(base,cc_weight=v,uid=f'paired48h100k-{i}-{v}').to_dict() for v in (10,1000)]))
                    seen.add(g);seenstates.add(state_id(w.start.word,w.start.strands))
        if reason:rejects[reason]=rejects.get(reason,0)+1
        if [len(b) for b in bands]==quotas:break
    assert [len(b) for b in bands]==quotas,('supply_failure',[len(b) for b in bands],rejects)
    atomic(OUT/'panel.json',dict(rows=sum(bands,[]),protocol_sha256=sha(PROTOCOL),
        exclusions_sha256=sha(OUT/'exclusions.json'),source_sha256=sha(Path(__file__)),rejects=rejects,
        selection='score-blind SHA order; no network evaluation performed'))
    atomic(OUT/'status.json',dict(status='FROZEN_AWAITING_INDEPENDENT_AUDIT',roots=32,
        panel_sha256=sha(OUT/'panel.json'),baseline_sha256=entry['sha256'],elapsed_seconds=time.time()-started))
    print(read(OUT/'status.json'))

if __name__=='__main__':main()
