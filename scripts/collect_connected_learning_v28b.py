"""Prospective shallow curriculum; closed v28a yield failure is never retried."""
import argparse
import collections
import signal
import time
from dataclasses import replace
from pathlib import Path
import learning_pipeline_v28 as pipeline
import eligible_positive_bank_v28a as eligible

PROTOCOL=Path('artifacts/connected-learning-live-v28b-protocol.json')
FIRST_BATCH=2
MIN_DEPTH,MAX_DEPTH,MAX_STRANDS=4,8,6
ORIGINAL_PINS=pipeline.code_pins
ORIGINAL_WRITE=pipeline.write_bank


def prior_rosters():
    return sorted({p for name in ['random-TRAIN-stream-v28','random-TRAIN-stream-v28a',
                                  'random-TRAIN-stream-v28b']
                   for p in Path('artifacts',name).glob('batch-*/supply.json')})


def code_pins():
    return {**ORIGINAL_PINS(),**{str(p):pipeline.sha(p) for p in [Path(__file__),PROTOCOL,
        Path(eligible.__file__),Path('tests/test_connected_learning_v28b.py')]}}


def write_bank(out,parent,audit,candidates,inputs,**metadata):
    metadata['quarantined_path_ids']=pipeline.read(parent/'manifest.json')['quarantined_path_ids']
    ORIGINAL_WRITE(out,parent,audit,candidates,inputs,**metadata)


def generate(out,batch,prior_paths):
    from rf_knots.conjugate_witnesses import conjugate_witness
    from evaluate_embedding_cost_search import query_instance
    assert batch>=FIRST_BATCH,'closed v28/v28a indices cannot be retried'
    groups,protected,known,pins=pipeline.protected_registry()
    for p in sorted(set(prior_rosters())|set(prior_paths)):
        pins[str(p)]=pipeline.sha(p)
        for row in pipeline.read(p)['rows']:
            w=pipeline.UnknotWitness.from_dict(row['witness'])
            known.add(pipeline.norm(w.start.word,w.start.strands))
    pins.update(code_pins())
    if out.exists():raise ValueError('immutable supply attempt; no retry')
    pipeline.disk_guard();out.mkdir(parents=True)
    pipeline.atomic(out/'contract.json',dict(seed=pipeline.SEED,batch=batch,opportunities=256,
        supply_CPU=30,requested_views=4,minimum_groups=4,inputs=pins,
        min_depth=MIN_DEPTH,max_depth=MAX_DEPTH,max_strands=MAX_STRANDS,
        first_generator_index=batch*256,reference_is_training_label=False))
    rows=[];seen_groups=set();rejected=collections.Counter();op=-1
    start=time.process_time();status='supply_exhausted'
    def timeout(*_):raise TimeoutError('supply CPU ceiling')
    previous=signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,30)
    try:
        for op in range(256):
            index=batch*256+op
            w=conjugate_witness(pipeline.SEED,index,max_strands=MAX_STRANDS,
                                min_depth=MIN_DEPTH,max_depth=MAX_DEPTH)
            k=pipeline.norm(w.start.word,w.start.strands)
            if len(w.start.word)>192 or w.moves>192 or len(k[0])<8:
                rejected['size_or_trivial']+=1;continue
            fp=pipeline.invariants(w.start.word,w.start.strands);g=(fp[0],abs(fp[1]))
            if pipeline.owner(fp)!='train' or g in groups:
                rejected['not_admissible_TRAIN_group']+=1;continue
            if k in known or g in seen_groups:
                rejected['normalized_known_or_batch_group_duplicate']+=1;continue
            inst,fits=query_instance(op,w,witness_caps=True)
            if not fits:rejected['caps']+=1;continue
            w.verify();known.add(k);seen_groups.add(g)
            rows.append(dict(query_group=list(g),generator_index=index,witness=w.to_dict(),
                instances=[replace(inst,cc_weight=v).to_dict() for v in [10,1000]]))
            if len(rows)==4:status='completed';break
    except TimeoutError:status='cpu_ceiling'
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);signal.signal(signal.SIGPROF,previous)
    pipeline.atomic(out/'supply.json',dict(status=status,rows=rows,opportunities=op+1,
        rejections=dict(rejected),cpu_seconds=time.process_time()-start,
        normalized_views=len(rows),query_groups=len(seen_groups),training_labels=0))
    return pipeline.read(out/'supply.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent-ready',type=Path,required=True);p.add_argument('--batches',type=int,default=1)
    a=p.parse_args();ready=pipeline.read(a.parent_ready)
    bank,audit=Path(ready['bank']),Path(ready['audit'])
    assert pipeline.sha(bank/'manifest.json')==ready['manifest_sha256']
    assert pipeline.sha(audit)==ready['audit_sha256']
    pipeline.disk_guard();a.out.mkdir(exist_ok=True)
    mp=a.out/'manifest.json'
    if not mp.exists():pipeline.atomic(mp,dict(status='ready_for_next_batch',next_batch=FIRST_BATCH,
        current_bank=str(bank),current_audit=str(audit),completed=[],
        reviewed_failure='artifacts/random-TRAIN-stream-v28a/batch-000001/collection.json'))
    pipeline.code_pins=code_pins;pipeline.write_bank=write_bank
    pipeline.generate=generate;pipeline.audit_bank=eligible.audit
    pipeline.collect(a.out,bank,audit,a.batches)


if __name__=='__main__':main()
