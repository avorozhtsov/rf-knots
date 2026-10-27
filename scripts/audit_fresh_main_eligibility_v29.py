"""Independent freshness review: uid and L-weight are not braid-state identity."""
from pathlib import Path
from learning_pipeline_v28 import read,sha,pins_ok,UnknotWitness,semantic_id,admit,instance,protected_registry


def semantic_instance(d):
    # UID is a display/provenance label and includes the L-weight suffix. Neither
    # UID nor cc_weight is part of the braid/caps identity used for a path target.
    return {k:v for k,v in d.items() if k not in ['uid','cc_weight']}


def validate_fresh(bank):
    m=read(bank/'manifest.json');pins_ok(m['inputs'])
    ids=set(m['fresh_main_path_ids']);rows={}
    for e in m['paths']:
        assert sha(bank/e['file'])==e['sha256'];r=read(bank/e['file']);rows[r['path_id']]=r
    groups,states,_,_=protected_registry();seen=set()
    for e in m['fresh_main_evidence']:
        ident=e['path_id'];o=e['observation'];sp=Path(o['shard'])
        assert int(sp.stem.split('-')[-1])>=39168 and sha(sp)==o['shard_sha256']
        saved=read(sp);w=UnknotWitness.from_dict(saved['row']['witness']);w.verify()
        assert semantic_id(w)==ident and ident in rows
        report=Path(o['batch_report']);assert sha(report)==o['batch_report_sha256']
        r=read(report);assert not r['partial'] and r['conditions']==32 and r['actor']==o['actor']
        assert sha(Path(o['actor']['file']))==o['actor']['sha256']
        roster=Path(o['roster']);assert sha(roster)==o['roster_sha256']
        assert saved['seed']==o['seed'] and saved['weight']==o['weight'] and saved['root_index']==o['root_index']
        observed=read(roster)['rows'][o['root_index']]['instances'][0 if o['weight']==10 else 1]
        assert semantic_instance(rows[ident]['instance'])==semantic_instance(observed),'semantic braid/caps differ'
        _,ok=admit(w,instance(observed),groups,states);assert ok
        assert rows[ident]['path_id']==semantic_id(UnknotWitness.from_dict(rows[ident]['witness']))
        seen.add(ident)
    assert ids==seen and ids,'fresh MAIN evidence absent'
    return ids
