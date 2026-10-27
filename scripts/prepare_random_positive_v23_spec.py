"""Freeze successor only from naturally checkpointed parent and audited data."""
from pathlib import Path
from build_mcts_training_supervision import atomic,sha
from run_continuous_train_mcts_v1 import read
def main():
    out=Path('artifacts/continuous-r2-start-spec-v23.json')
    if out.exists():raise ValueError('immutable spec exists')
    s=read(Path('artifacts/continuous-r2-start-spec-v22.json'));d=Path(s['out']);m=read(d/'manifest.json')
    assert m['status']=='checkpointed' and m['stop_reason']=='user_stop_request';c=m['checkpoints'][-1];cp=d/c['file'];assert sha(cp)==c['sha256']
    s.update(parent=str(cp),parent_sha256=c['sha256'],parent_step=c['step'],parent_manifest_sha256=sha(d/'manifest.json'),out='artifacts/continuous-r2-v23',
        random_positive_bank='artifacts/random-root-positive-bank-v1',random_positive_audit='artifacts/random-root-positive-bank-audit-v1/report.json')
    s['random_positive_bank_sha256']=sha(Path(s['random_positive_bank'])/'manifest.json');s['random_positive_audit_sha256']=sha(Path(s['random_positive_audit']))
    a=read(Path(s['random_positive_audit']));assert a['status']=='passed_data_only' and a['manifest_sha256']==s['random_positive_bank_sha256']
    cpu=read(Path('artifacts/random-positive-joint-cpu-smoke-v1/report.json'));assert cpu['status']=='passed_isolated_CPU' and cpu['updates']==2
    atomic(out,s);print(dict(parent_step=c['step'],smoke_updates=1 if c['step']%2 else 2,spec_sha256=sha(out)))
if __name__=='__main__':main()
