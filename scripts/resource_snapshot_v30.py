"""Read-only disk/swap/own-worker observations; never purge or change thresholds."""
import json
import shutil
import subprocess
import time

def snapshot():
    listing=subprocess.check_output(['ps','-axo','pid=,ppid=,rss=,pcpu=,command='],text=True)
    names=('scripts/connected_learning_supervisor_v30.py','scripts/train_connected_learning_live_v30.py',
        'scripts/run_continuous_train_mcts_v2.py','scripts/collect_connected_learning_v30.py')
    workers=[]
    for line in listing.splitlines():
        fields=line.split(None,4)
        if len(fields)==5 and fields[4].startswith('/') and '/Python.app/' in fields[4] and any(n in fields[4] for n in names):
            workers.append(dict(pid=int(fields[0]),ppid=int(fields[1]),rss_KiB=int(fields[2]),
                CPU_percent=float(fields[3]),command=fields[4]))
    return dict(timestamp=time.time(),free_GiB=shutil.disk_usage('.').free/1024**3,
        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip(),
        workers=workers,cause_inference='observations only; no causal attribution from one snapshot')

if __name__=='__main__':print(json.dumps(snapshot()))
