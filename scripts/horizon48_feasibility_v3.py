"""One frozen horizon48 supply feasibility test; no loop or optimizer writes."""
import shutil
from pathlib import Path
import continuous_random_stream_v2 as core
from build_mcts_training_supervision import atomic, sha
from run_continuous_train_mcts_v1 import read
from run_continuous_train_mcts_v2 import snapshot

OUT = Path('artifacts/random-horizon48-feasibility-v3')
PROTOCOL = Path('artifacts/random-horizon48-feasibility-v3-prospective.json')
HORIZON = 48

def prior_rosters():
    return sorted([*Path('artifacts/continuous-random-TRAIN-stream-v1').glob('batch-*/supply.json'),
                   *Path('artifacts/continuous-random-TRAIN-stream-v2').glob('batch-*/supply.json')])

def main():
    assert not OUT.exists(), 'immutable single attempt: never rerun'
    assert shutil.disk_usage('.').free >= 12*1024**3
    protocol = read(PROTOCOL)
    assert protocol['search']['total_CPU_upper_bound'] == 640
    assert protocol['supply']['max_opportunities'] == 32
    priors = prior_rosters()
    assert len(priors) == 57 and any('batch-000052' in str(p) for p in priors)
    files = [*core.input_files(), PROTOCOL, Path(__file__),
             Path('scripts/audit_horizon48_feasibility_v3.py'),
             Path('tests/test_horizon48_feasibility_v3.py'), *priors]
    pins = {str(p): sha(p) for p in files}
    actor = snapshot()
    OUT.mkdir()
    atomic(OUT/'supply-contract.json', dict(batch=0, inputs=pins,
        prior_rosters=list(map(str,priors)), horizon=HORIZON, seed=core.SEED,
        actor=actor, search_permitted_only_after_independent_supply_audit=True))
    core.HORIZON = HORIZON
    core.generate(OUT)
    print({k:v for k,v in read(OUT/'supply.json').items()
           if k in ('status','opportunities','cpu_seconds','normalized_views','query_groups')})

if __name__ == '__main__':
    main()
