"""Separate reviewed continuation; preserve the storage-interrupted batch15."""
from pathlib import Path
import collect_connected_learning_v28b as parent

OLD_ROSTERS=parent.prior_rosters
OLD_PINS=parent.code_pins


def rosters():
    return sorted(set(OLD_ROSTERS())|set(Path('artifacts/random-TRAIN-stream-v29').glob('batch-*/supply.json')))


def pins():
    return {**OLD_PINS(),**{str(p):parent.pipeline.sha(p) for p in [Path(__file__),
        Path('artifacts/connected-learning-live-v29-protocol.json')]}}


if __name__=='__main__':
    parent.FIRST_BATCH=16;parent.prior_rosters=rosters;parent.code_pins=pins
    parent.main()
