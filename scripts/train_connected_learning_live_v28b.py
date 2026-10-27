"""Versioned supervisor repair; unchanged native optimizer and source mixture."""
from pathlib import Path
import train_connected_learning_live_v28a as parent


def setup(path):
    s,order,frozen=parent.setup(path)
    for f in [Path(__file__),Path('scripts/connected_learning_supervisor_v28b.py'),
              Path('scripts/collect_connected_learning_v28b.py'),
              Path('artifacts/connected-learning-live-v28b-protocol.json')]:
        frozen[str(f)]=parent.connected.old.guard.core.base.sha(f)
    return s,order,frozen


if __name__=='__main__':
    core=parent.connected.old.guard.core
    core.PROTOCOL=Path('artifacts/connected-learning-live-v28b-protocol.json')
    core.setup=setup;core.numerical_gate=parent.connected.old.guard.gate
    core.main()
