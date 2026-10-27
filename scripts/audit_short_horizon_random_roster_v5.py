"""Independent proof/start/suffix/inverse/novelty audit; no training writes."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from build_short_horizon_random_roster_v5 import BASE,exclusions
from rf_knots import reference,mcts_loop as M
from rf_knots.actions import ActionSpec,COMMUTE,BRAID
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm

def main():
    p=argparse.ArgumentParser();p.add_argument('--roster',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('immutable output')
    r=read(a.roster);assert r['status']=='completed'
    for f,h in r['inputs'].items():assert sha(Path(f))==h,f
    known=exclusions();seen=set();groups=set();sources=read(BASE)['rows']
    for b in r['bases']:
        w=UnknotWitness.from_dict(b['witness']);w.verify();original=UnknotWitness.from_dict(sources[b['source_index']]['witness'])
        assert len(w.steps)==16 and w.steps==original.steps[-16:]
        assert w.start==(original.start if len(original.steps)==16 else original.steps[-17].after)
    for row in r['rows']:
        w=UnknotWitness.from_dict(row['witness']);w.verify();b=UnknotWitness.from_dict(r['bases'][row['base_ordinal']]['witness'])
        spec=ActionSpec(w.max_len,w.max_strands,w.cyclic_band_generators);state=(b.start.word,b.start.strands)
        assert len(row['walk'])==2 and w.steps[2:]==b.steps
        for action in row['walk']:
            assert spec.decode(action)[0] in (COMMUTE,BRAID) and reference.is_legal(spec,*state,action,False)
            after=reference.apply(spec,*state,action);assert reference.apply(spec,*after,action)==state;state=after
        assert state==(w.start.word,w.start.strands)
        assert [s.action.to_flat(spec) for s in w.steps[:2]]==list(reversed(row['walk']))
        k=norm(*state);assert k not in known and k not in seen
        assert k==(tuple(row['normalized_word']),row['normalized_strands']);seen.add(k)
        fp=invariants(*state);assert owner(fp)=='train' and (fp[0],abs(fp[1]))==tuple(row['query_group']);groups.add(tuple(row['query_group']))
        for weight,d in zip((10,1000),row['instances'],strict=True):
            inst=M.Instance(**{**d,'word':tuple(d['word'])});assert inst.cc_weight==weight and (inst.word,inst.strands)==state
            assert w.moves<=inst.cap_moves and w.crossing_changes<=inst.cap_crossing_changes
    assert len(seen)==r['distinct_normalized_views'] and len(groups)==r['query_groups']
    assert r['decision']==('separate_MCTS_contract_only' if len(seen)>=8 and len(groups)>=4 else 'stop_configuration')
    a.out.mkdir();atomic(a.out/'report.json',dict(status='passed',roster_sha256=sha(a.roster),auditor_sha256=sha(Path(__file__)),
        full_paths_replayed=len(seen),query_groups=len(groups),decision=r['decision'],MCTS_labels=0,optimizer_draws=0))
    print('AUDIT_PASSED',len(seen),len(groups),r['decision'])
if __name__=='__main__':main()
