from pathlib import Path
from continuous_random_stream_v2 import HORIZON,OUT,seed_for
from build_mcts_training_supervision import sha
def test_new_namespace_and_horizon():
    assert HORIZON==24 and OUT.name=='continuous-random-TRAIN-stream-v2'
def test_original_source_preserved():
    assert sha(Path('scripts/continuous_random_stream_v1.py'))=='afe51255000b0b849fe934a5f60a175a36c3be5011f62c5ea335738db42f27a4'
def test_seed_not_outcome_retry():
    assert seed_for(0,0)==20281017436 and seed_for(1,0)==20281018436
