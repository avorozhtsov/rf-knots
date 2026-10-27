"""Wait for the already-running builder, audit ONCE, never build or evaluate."""
import fcntl
import subprocess
import sys
import time
from pathlib import Path
from build_mcts_training_supervision import atomic

ROOT=Path('artifacts/paired-learning-48h100k-v3')


def main():
    with (ROOT/'preparation-finisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'study.json').exists():
            if (ROOT/'panel.json').exists():
                result=subprocess.run([sys.executable,'scripts/audit_learning_comparison_v3.py'])
                if result.returncode:
                    atomic(ROOT/'preparation-failure.json',dict(stage='independent_audit',exit_code=result.returncode,no_retry=True))
                    raise RuntimeError('independent study audit failed; preserve, no retry')
                print('Fresh v3 panel independently audited; no network evaluation performed',flush=True)
                return
            listing=subprocess.check_output(['ps','-axo','command='],text=True)
            if not any('scripts/freeze_learning_comparison_v3.py' in line and not line.startswith(('rg ','grep ')) for line in listing.splitlines()):
                atomic(ROOT/'preparation-failure.json',dict(stage='builder',error='builder exited without frozen panel',no_retry=True))
                raise RuntimeError('builder exited: preserve supply evidence, never retry seed')
            time.sleep(30)


if __name__=='__main__':main()
