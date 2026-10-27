"""Single bounded horizon48 MCTS pilot; frozen actor and no auto-retry."""
import argparse
import shutil
from pathlib import Path
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
import random_representation_mcts_pilot_v1 as engine

ROOT = Path('artifacts/random-horizon48-feasibility-v3')

def prepare():
    out = ROOT/'search'
    assert not out.exists(), 'immutable search already prepared'
    assert shutil.disk_usage(ROOT).free >= 12*1024**3
    c = read(ROOT/'supply-contract.json'); r = read(ROOT/'supply.json')
    audit = read(ROOT/'supply-audit/report.json')
    assert c['horizon'] == 48 and r['status'] == 'completed'
    assert audit['status'] == 'passed' and audit['roster_sha256'] == sha(ROOT/'supply.json')
    for p,h in c['inputs'].items():
        assert sha(Path(p)) == h, p
    pins = {**c['inputs'], str(Path(__file__)):sha(Path(__file__))}
    for p in [ROOT/'supply-contract.json', ROOT/'supply.json', ROOT/'supply-audit/report.json']:
        pins[str(p)] = sha(p)
    out.mkdir()
    atomic(out/'roster.json', dict(rows=r['rows']))
    atomic(out/'manifest.json', dict(status='prepared', next_task=0, shards=[], cpu_seconds=0,
        actor=c['actor'], roster_sha256=sha(out/'roster.json'), inputs=pins,
        simulations=8, task_cpu_limit=20, total_tasks=32, seeds=list(range(20281016,20281020))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--count', type=int)
    a=p.parse_args()
    if a.prepare:
        prepare()
    else:
        m=read(ROOT/'search/manifest.json')
        assert a.count and 0 < a.count <= 32-m['next_task']
        engine.OUT=ROOT/'search'
        engine.run(a.count)

if __name__ == '__main__':
    main()
