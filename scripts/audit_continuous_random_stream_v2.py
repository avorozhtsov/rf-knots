"""Independent supply and canonical episode replay audit subprocess."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
from continuous_random_stream_v2 import BASE,known_from,checkpins,HORIZON
from rf_knots import reference,mcts_loop as M
from rf_knots.actions import ActionSpec,COMMUTE,BRAID
from rf_knots.evidence import UnknotWitness
from rf_knots.embedding_replay_corpus import invariants,owner
from rf_knots.embedding_normalization import fast_normalize_global_view as norm
from export_mcts_root_positives_v1 import external_groups

def supply(d):
    r=read(d/'supply.json');assert r['status']=='completed';checkpins(r['inputs']);known=known_from(r['prior_rosters']);seen=set();groups=set()
    originals=read(BASE)['rows'];excluded=external_groups();assert len(r['rows'])==4 and r['opportunities']<=32
    for row in r['rows']:
        b=r['bases'][row['base_ordinal']];base=UnknotWitness.from_dict(b['witness']);original=UnknotWitness.from_dict(originals[b['source_index']]['witness'])
        assert len(base.steps)==HORIZON and base.steps==original.steps[-HORIZON:]
        assert base.start==(original.start if len(original.steps)==HORIZON else original.steps[-HORIZON-1].after);base.verify()
        w=UnknotWitness.from_dict(row['witness']);w.verify();spec=ActionSpec(w.max_len,w.max_strands,w.cyclic_band_generators)
        state=(base.start.word,base.start.strands);assert len(row['walk'])==2 and w.steps[2:]==base.steps
        for action in row['walk']:
            assert spec.decode(action)[0] in (COMMUTE,BRAID) and reference.is_legal(spec,*state,action,False)
            after=reference.apply(spec,*state,action);assert reference.apply(spec,*after,action)==state;state=after
        assert state==(w.start.word,w.start.strands) and [s.action.to_flat(spec) for s in w.steps[:2]]==list(reversed(row['walk']))
        k=norm(*state);assert k not in known and k not in seen and k==(tuple(row['normalized_word']),row['normalized_strands']);seen.add(k)
        fp=invariants(*state);g=(fp[0],abs(fp[1]));assert owner(fp)=='train' and g==tuple(row['query_group']) and g not in excluded;groups.add(g)
        for weight,data in zip((10,1000),row['instances'],strict=True):
            i=M.Instance(**{**data,'word':tuple(data['word'])});assert (i.word,i.strands)==state and i.cc_weight==weight
            assert w.moves<=i.cap_moves and w.crossing_changes<=i.cap_crossing_changes
    assert len(seen)==len(groups)==4
    (d/'supply-audit').mkdir();atomic(d/'supply-audit/report.json',dict(status='passed',roster_sha256=sha(d/'supply.json'),full_proofs=4,normalized_views=4,TRAIN_groups=4,auditor_sha256=sha(Path(__file__))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--supply',type=Path);p.add_argument('--search',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args()
    if a.supply:supply(a.supply)
    else:
        import audit_random_representation_mcts_v1 as audit
        import sys
        audit.DATA=a.search/'search';sys.argv=[__file__,'--out',str(a.search/('first-audit' if a.partial else 'full-audit'))]+(['--partial'] if a.partial else []);audit.main()
if __name__=='__main__':main()
