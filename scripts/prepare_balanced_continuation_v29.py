"""One explicit version migration; no retry of failed or partially written work."""
import subprocess
import sys
from pathlib import Path
import shutil
import connected_learning_supervisor_v29 as supervisor


def prepare():
    assert shutil.disk_usage('.').free>=16*1024**3,'need stable16GiB before intake/optimizer diagnostics/native launch'
    assert not supervisor.has_process('scripts/train_connected_learning_live_'),'existing learner must not be duplicated'
    assert not supervisor.has_process('scripts/connected_learning_supervisor_v28b.py'),'old supervisor still owns state'
    assert not (supervisor.ROOT/'failure.json').exists(),'failed preparation needs review, not retry'
    supervisor.validate_parent()
    state=supervisor.read(Path('artifacts/connected-learning-live-v28b/state.json'))
    assert state['status']=='STOPPED_LEARNER' and state['exit_code']==0 and state['stop_reason']=='storage_gate'
    main=supervisor.checked_ready(Path(state['latest_main_ready']))
    random_ready=Path('artifacts/random-TRAIN-stream-v28b/READY.json');supervisor.checked_ready(random_ready)
    dest=Path('artifacts/v29-fresh-main-initial');ready=dest/'READY.json'
    if ready.exists():supervisor.checked_ready(ready)
    else:
        assert not dest.exists(),'partial fresh intake must be reviewed, not retried'
        snapshot=supervisor.read(Path('artifacts/continuous-train-mcts-v2/manifest.json'))
        full={a['batch'] for a in snapshot['audits']};last=1224
        while last in full and last<1240:last+=1
        assert last>1224,'no fully audited post-A MAIN data'
        subprocess.run([sys.executable,'scripts/intake_fresh_main_v29a.py','--out',str(dest),
            '--parent-bank',main['bank'],'--parent-audit',main['audit'],
            '--first-batch','1224','--batches',str(last-1224)],check=True)
    report=Path('artifacts/v29-launch-CPU/report.json')
    if not report.exists():
        assert not report.parent.exists(),'partial CPU diagnostic must be reviewed, not retried'
        subprocess.run([sys.executable,'scripts/smoke_balanced_new_mcts_f_v29.py',
            '--main-ready',str(ready),'--random-ready',str(random_ready),'--out',str(report.parent)],check=True)
    prepared=supervisor.ROOT/'prepared.json'
    assert not prepared.exists(),'already prepared; never pick another initial parent'
    supervisor.ROOT.mkdir(exist_ok=True)
    spec=supervisor.prepare(0,ready,random_ready,report)
    supervisor.atomic(prepared,dict(spec=str(spec),main_ready=str(ready),random_ready=str(random_ready)))
    print(dict(status='PREPARED_NOT_NATIVE_LAUNCHED',spec=str(spec)))


if __name__=='__main__':prepare()
