"""Separate bounded short-horizon curriculum, unchanged canonical engine."""
import argparse,hashlib,shutil
from pathlib import Path
import random_representation_mcts_pilot_v1 as engine
from build_mcts_training_supervision import atomic,sha

engine.OUT=Path('artifacts/short-horizon-random-mcts-pilot-v4')
SOURCE=Path('artifacts/short-horizon-random-roster-v5/roster.json')
AUDIT=Path('artifacts/short-horizon-random-roster-audit-v5/report.json')
def prepare():
    if engine.OUT.exists() or shutil.disk_usage(engine.OUT.parent).free<12*1024**3:raise ValueError('output/storage gate')
    a=engine.read(AUDIT);assert a['status']=='passed' and a['decision']=='separate_MCTS_contract_only' and a['roster_sha256']==sha(SOURCE)
    source=engine.read(SOURCE)
    for f,h in source['inputs'].items():assert sha(Path(f))==h,f
    selected=[];groups=set()
    for r in sorted(source['rows'],key=lambda r:hashlib.sha256(f'20281017431:{r["base_ordinal"]}:{r["rep"]}'.encode()).hexdigest()):
        g=tuple(r['query_group'])
        if g in groups:continue
        selected.append(r);groups.add(g)
        if len(selected)==4:break
    assert len(selected)==4
    actor=engine.snapshot()
    files=[SOURCE,AUDIT,Path(__file__),Path('research/431-short-horizon-random-mcts-pilot-v4.md'),
        Path('scripts/random_representation_mcts_pilot_v1.py'),Path('scripts/audit_short_horizon_random_mcts_pilot_v4.py'),
        Path('scripts/audit_random_representation_mcts_v1.py'),Path('tests/test_random_representation_mcts_v2.py'),
        Path('tests/test_short_horizon_random_roster_v5.py'),Path('scripts/resource_cost_search.py'),
        Path('scripts/run_continuous_train_mcts_v1.py'),Path('scripts/run_continuous_train_mcts_v2.py'),
        *Path('src/rf_knots').glob('*.py'),*Path('/Users/artemvorozhtsov/projects/pgx-mcts-bench/src/pgx_mcts_bench').glob('*.py')]
    engine.OUT.mkdir();atomic(engine.OUT/'roster.json',dict(rows=selected,role='short-horizon random normalized views; noMCTSlabels yet'))
    atomic(engine.OUT/'manifest.json',dict(status='prepared',next_task=0,shards=[],cpu_seconds=0,actor=actor,
        roster_sha256=sha(engine.OUT/'roster.json'),inputs={str(f):sha(f) for f in files},simulations=8,
        task_cpu_limit=20,total_tasks=32,seeds=list(range(20281016,20281020))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.stop_after and a.stop_after>0:engine.run(a.stop_after)
    else:p.error('prepare or positive bounded stop-after required')
