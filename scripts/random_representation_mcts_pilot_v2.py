"""Separate output/pinned corrected tests, unchanged v1 collection algorithm."""
import argparse
from pathlib import Path
import random_representation_mcts_pilot_v1 as engine
from build_mcts_training_supervision import atomic,sha

engine.OUT=Path('artifacts/random-representation-mcts-pilot-v2')
engine.PROTOCOL=Path('research/417-random-representation-pilot-test-expectation-v2.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    if a.prepare:
        engine.prepare()
        m=engine.read(engine.OUT/'manifest.json')
        for f in [Path(__file__),Path('scripts/audit_random_representation_mcts_v2.py'),
                  Path('tests/test_random_representation_mcts_v2.py'),
                  Path('research/416-random-representation-mcts-relative-success-pilot-v1.md')]:
            m['inputs'][str(f)]=sha(f)
        atomic(engine.OUT/'manifest.json',m)
    elif a.stop_after and a.stop_after>0:engine.run(a.stop_after)
    else:p.error('prepare or positive stop-after required')
