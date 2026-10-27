"""Reviewed full-state migration from v29; original A and balance phase unchanged."""
import shutil
from pathlib import Path
import connected_learning_supervisor_v30 as supervisor

def prepare():
    assert shutil.disk_usage('.').free>=16*1024**3
    assert not supervisor.has_process('scripts/train_connected_learning_live_')
    assert not supervisor.has_process('scripts/connected_learning_supervisor_v29.py')
    state=supervisor.read(Path('artifacts/connected-learning-live-v29/state.json'))
    assert state['status']=='STOPPED_LEARNER' and state['exit_code']==0
    assert state['stop_reason']=='user_stop_request'
    supervisor.validate_parent()
    assert not (supervisor.ROOT/'prepared.json').exists()
    supervisor.ROOT.mkdir(exist_ok=True)
    main=supervisor.checked_ready(Path(state['latest_main_ready']))
    supervisor.atomic(supervisor.INITIAL_MAIN,main)
    supervisor.checked_ready(supervisor.INITIAL_RANDOM)
    report=supervisor.cpu_check(supervisor.INITIAL_MAIN,supervisor.INITIAL_RANDOM,
        supervisor.ROOT/'initial-CPU')
    spec=supervisor.prepare(0,supervisor.INITIAL_MAIN,supervisor.INITIAL_RANDOM,report)
    supervisor.atomic(supervisor.ROOT/'prepared.json',dict(spec=str(spec),
        main_ready=str(supervisor.INITIAL_MAIN),random_ready=str(supervisor.INITIAL_RANDOM)))
    print(dict(status='PREPARED',spec=str(spec)))

if __name__=='__main__':prepare()
