"""Fixed four-batch yield windows; never replace or retry a low-yield batch."""
from pathlib import Path
import collect_connected_learning_v28b as parent
import collect_connected_learning_v29 as prior
import learning_pipeline_v30 as pipeline

OLD_ROSTERS=prior.rosters

def rosters():
    return sorted(set(OLD_ROSTERS())|set(Path('artifacts/random-TRAIN-stream-v30').glob('batch-*/supply.json')))

def pins():
    return {**prior.pins(),**{str(p):pipeline.sha(p) for p in [Path(__file__),
        Path(pipeline.__file__),Path('artifacts/connected-learning-live-v30-protocol.json'),
        Path('tests/test_random_yield_window_v30.py')]}}

if __name__=='__main__':
    parent.pipeline=pipeline;parent.FIRST_BATCH=61
    parent.prior_rosters=rosters;parent.code_pins=pins
    parent.main()
