"""Prepared v28 successor; legacy protected-root collisions never sampled."""
from pathlib import Path
import train_continuous_random_view_r2_v28 as parent
import positive_sampler_v28a as eligible


def setup(path):
    settings,order,frozen=parent.setup(path)
    eligible.initialize(settings['positive_bank'],settings['random_positive_bank'])
    for f in [Path(__file__),Path(eligible.__file__)]:frozen[str(f)]=parent.old.guard.core.base.sha(f)
    return settings,order,frozen


if __name__=='__main__':
    parent.old.guard.core.PROTOCOL=Path('research/460-v28-connected-learning-protocol.md')
    parent.old.guard.core.setup=setup
    parent.old.guard.core.numerical_gate=parent.old.guard.gate
    parent.old.guard.core.main()
