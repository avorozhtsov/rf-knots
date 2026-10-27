"""Freeze an already audited READY bank after validated graceful handoff."""
import argparse
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
import random_view_positive_sampler_v2 as mixture
from random_positive_sampler_v1 import source_ordinal
def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--cpu-report',type=Path,required=True);a=p.parse_args()
    out=Path('artifacts/continuous-r2-start-spec-v24.json')
    if out.exists():raise ValueError('immutable successor spec')
    ref=read(Path('artifacts/continuous-r2-current-training.json'));s=read(Path(ref['spec']));parent=Path(s['out']);m=read(parent/'manifest.json')
    assert m['status']=='checkpointed' and m['stop_reason']=='user_stop_request';c=m['checkpoints'][-1];cp=parent/c['file'];assert sha(cp)==c['sha256']
    audit=read(a.audit);assert audit['status']=='passed_data_only' and audit['manifest_sha256']==sha(a.bank/'manifest.json')
    cpu=read(a.cpu_report);assert cpu['status']=='passed_isolated_CPU' and cpu['random_bank_sha256']==sha(a.bank/'manifest.json') and cpu['audit_sha256']==sha(a.audit)
    assert cpu['updates']==2 and cpu['AdamW_states']==71
    s.update(parent=str(cp),parent_step=c['step'],parent_sha256=c['sha256'],parent_manifest_sha256=sha(parent/'manifest.json'),out='artifacts/continuous-r2-v24',
        random_positive_bank=str(a.bank),random_positive_bank_sha256=sha(a.bank/'manifest.json'),random_positive_audit=str(a.audit),random_positive_audit_sha256=sha(a.audit))
    mixture.initialize(a.bank);count=None
    for i in range(64):
        if source_ordinal(c['step']+i)[0]!='random':continue
        row,_=mixture.select(c['step']+i)
        if 'continuous-random-TRAIN-stream-v1/' in row['observations'][0]['shard']:count=i+1;break
    if count is None:raise ValueError('no refreshed stream draw within64updates')
    s['commissioning_updates']=count
    atomic(out,s);print(dict(parent_step=c['step'],native_smoke_updates=count,spec_sha256=sha(out),bank=str(a.bank)))
if __name__=='__main__':main()
