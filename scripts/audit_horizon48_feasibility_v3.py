"""Separate-process full proof audit at the explicitly frozen horizon48."""
from pathlib import Path
import audit_continuous_random_stream_v2 as independent
from build_mcts_training_supervision import sha
from run_continuous_train_mcts_v1 import read

def main():
    d = Path('artifacts/random-horizon48-feasibility-v3')
    c = read(d/'supply-contract.json')
    assert c['horizon'] == 48
    assert sha(Path(__file__)) == c['inputs'][__file__ if __file__ in c['inputs'] else 'scripts/audit_horizon48_feasibility_v3.py']
    independent.HORIZON = 48
    independent.supply(d)
    print(read(d/'supply-audit/report.json'))

if __name__ == '__main__':
    main()
