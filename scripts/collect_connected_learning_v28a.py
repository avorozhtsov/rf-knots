"""Bounded continuation after reviewed v28 legacy-root quarantine.

Starts at batch1; freezes ALL v28 rosters, including its unsolved fourth view.
Never retries batch0 or changes the fixed seed. READY is not optimizer intake.
"""
import argparse
from pathlib import Path
import learning_pipeline_v28 as pipeline
import eligible_positive_bank_v28a as eligible

ORIGINAL_WRITE=pipeline.write_bank
ORIGINAL_PINS=pipeline.code_pins
ORIGINAL_GENERATE=pipeline.generate


def code_pins():
    extra=[Path(__file__),Path(eligible.__file__),
        Path('scripts/positive_sampler_v28a.py'),
        Path('scripts/train_continuous_random_view_r2_v28a.py'),
        Path('scripts/smoke_connected_learning_cpu_v28a.py'),Path('tests/test_connected_learning_v28a.py')]
    return {**ORIGINAL_PINS(),**{str(p):pipeline.sha(p) for p in extra}}


def write_bank(out,parent,audit,candidates,inputs,**metadata):
    metadata['quarantined_path_ids']=pipeline.read(parent/'manifest.json')['quarantined_path_ids']
    ORIGINAL_WRITE(out,parent,audit,candidates,inputs,**metadata)


def generate(out,batch,prior_paths):
    assert batch>=1,'v28 batch0 evidence is closed, never rerun'
    return ORIGINAL_GENERATE(out,batch,[*Path('artifacts/random-TRAIN-stream-v28').glob('batch-*/supply.json'),*prior_paths])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent-ready',type=Path,required=True);p.add_argument('--batches',type=int,default=1)
    a=p.parse_args();ready=pipeline.read(a.parent_ready)
    bank,audit=Path(ready['bank']),Path(ready['audit'])
    assert pipeline.sha(bank/'manifest.json')==ready['manifest_sha256']
    assert pipeline.sha(audit)==ready['audit_sha256']
    pipeline.disk_guard();a.out.mkdir(exist_ok=True)
    mp=a.out/'manifest.json'
    if not mp.exists():
        pipeline.atomic(mp,dict(status='ready_for_next_batch',next_batch=1,
            current_bank=str(bank),current_audit=str(audit),completed=[],
            prior_supply='artifacts/random-TRAIN-stream-v28/batch-000000/supply.json'))
    pipeline.code_pins=code_pins;pipeline.write_bank=write_bank
    pipeline.generate=generate;pipeline.audit_bank=eligible.audit
    pipeline.collect(a.out,bank,audit,a.batches)
