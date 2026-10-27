"""Separate bounded comparison on independently replayed normalized-new views."""
import argparse
import hashlib
from pathlib import Path
import random_representation_mcts_pilot_v1 as engine
from build_mcts_training_supervision import atomic,sha

engine.OUT=Path('artifacts/normalized-random-mcts-pilot-v3')
PROTOCOL=Path('research/423-normalized-random-mcts-relative-pilot-v3.md')
SOURCE=Path('artifacts/normalized-random-representation-roster-v4/roster.json')
AUDIT=Path('artifacts/normalized-random-representation-roster-audit-v4/report.json')


def prepare():
    import shutil
    if engine.OUT.exists() or shutil.disk_usage(engine.OUT.parent).free<12*1024**3:raise ValueError('output/disk gate')
    assert sha(SOURCE)=='b0ef8bb1e267e4e7e701344b4da2b0c885c80966fb8b686b14894200942fc30d'
    assert sha(AUDIT)=='cb5176367eed3657fe75fc898951eb9f947b02fbff2f283179a832509652297b'
    a=engine.read(AUDIT);assert a['status']=='passed' and a['decision']=='separate_small_MCTS_contract_only'
    src=engine.read(SOURCE);selected=[]
    for stage in range(4):
        pool=[r for r in src['rows'] if r['stage']==stage]
        selected.append(min(pool,key=lambda r:hashlib.sha256(f'20261017423:{r["base_ordinal"]}:{r["rep"]}'.encode()).hexdigest()))
    actor=engine.snapshot()
    files=[SOURCE,AUDIT,PROTOCOL,Path(__file__),Path('scripts/random_representation_mcts_pilot_v1.py'),
        Path('scripts/audit_normalized_random_mcts_pilot_v3.py'),Path('scripts/audit_random_representation_mcts_v1.py'),
        Path('tests/test_random_representation_mcts_v2.py'),Path('tests/test_normalized_random_roster_v4.py'),
        Path('scripts/resource_cost_search.py'),Path('scripts/run_continuous_train_mcts_v1.py'),
        Path('scripts/run_continuous_train_mcts_v2.py'),*Path('src/rf_knots').glob('*.py'),
        *Path('/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench').glob('*.py')]
    engine.OUT.mkdir();atomic(engine.OUT/'roster.json',dict(rows=selected,role='four normalized-new views of oldTRAINgroups; noFlabels'))
    atomic(engine.OUT/'manifest.json',dict(status='prepared',next_task=0,shards=[],cpu_seconds=0,actor=actor,
        roster_sha256=sha(engine.OUT/'roster.json'),inputs={str(f):sha(f) for f in files},simulations=8,
        task_cpu_limit=20,total_tasks=32,seeds=list(range(20281016,20281020))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.stop_after and a.stop_after>0:engine.run(a.stop_after)
    else:p.error('prepare or positive bounded stop-after required')
