"""Immutable audited horizon48 positive intake, retaining the full v26 anchor."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
import random_view_positive_sampler_v2 as mixture
from random_positive_sampler_v1 import source_ordinal

def commissioning(step, bank, oldids):
    mixture.initialize(bank)
    for i in range(64):
        if source_ordinal(step+i)[0] == 'random':
            row, _ = mixture.select(step+i)
            if row['path_id'] not in oldids:
                return i+1
    raise ValueError('no newly admitted draw within64updates')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--cpu-report',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    out=Path('artifacts/continuous-r2-start-spec-v27.json')
    assert not out.exists() and not Path('artifacts/continuous-r2-v27').exists()
    ref=read(Path('artifacts/continuous-r2-current-training.json')); s=read(Path(ref['spec']))
    old=Path(s['random_positive_bank']); oldrows=read(old/'manifest.json')['paths']
    oldids={read(old/e['file'])['path_id'] for e in oldrows}
    newrows=read(a.bank/'manifest.json')['paths']
    assert {e['sha256'] for e in oldrows} <= {e['sha256'] for e in newrows}
    for e in newrows: assert sha(a.bank/e['file'])==e['sha256']
    audit=read(a.audit)
    assert audit['status']=='passed_data_only' and audit['yield_gate']
    assert audit['manifest_sha256']==sha(a.bank/'manifest.json')
    cpu=read(a.cpu_report)
    assert cpu['status']=='passed_isolated_CPU' and cpu['updates']==2 and cpu['AdamW_states']==71
    assert cpu['random_bank_sha256']==sha(a.bank/'manifest.json') and cpu['audit_sha256']==sha(a.audit)
    parent=Path(s['out']); m=read(parent/'manifest.json')
    if a.preflight:
        commissioning(m['optimizer_steps'],a.bank,oldids)
        print('preflight passed: retained rows, full audit, CPU gate, metadata draw')
        return
    assert m['status']=='checkpointed' and m['stop_reason']=='user_stop_request'
    c=m['checkpoints'][-1]; cp=parent/c['file']; assert sha(cp)==c['sha256']
    count=commissioning(c['step'],a.bank,oldids)
    s.update(parent=str(cp),parent_step=c['step'],parent_sha256=c['sha256'],
        parent_manifest_sha256=sha(parent/'manifest.json'),out='artifacts/continuous-r2-v27',
        random_positive_bank=str(a.bank),random_positive_bank_sha256=sha(a.bank/'manifest.json'),
        random_positive_audit=str(a.audit),random_positive_audit_sha256=sha(a.audit),commissioning_updates=count)
    atomic(out,s)
    print(dict(parent_step=c['step'],native_smoke_updates=count,spec_sha256=sha(out)))

if __name__=='__main__':main()
