"""Read-only real child/grandchild probe using the exact pinned spawn override."""
import json
import os
from pathlib import Path
import subprocess
import sys
from build_mcts_training_supervision import sha


def main():
    path = Path('artifacts/connected-learning-live-v30/main-intake-001563-001566/bank/path-00000.json')
    expected = sha(path)
    code = "from pathlib import Path;from build_mcts_training_supervision import sha;import compact_storage_v31;assert compact_storage_v31.INSTALLED is not None;assert sha(Path(%r))==%r;print('verified_child_archive_read')" % (str(path), expected)
    reset = 'src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src'
    env = {**os.environ, 'PYTHONPATH':reset}
    child = subprocess.check_output([sys.executable, '-c', code], env=env, text=True)
    grandchild_code = "import os,subprocess,sys;env={**os.environ,'PYTHONPATH':%r};print(subprocess.check_output([sys.executable,'-c',%r],env=env,text=True))" % (reset, code)
    grandchild = subprocess.check_output([sys.executable, '-c', grandchild_code], env=env, text=True)
    assert 'verified_child_archive_read' in child and 'verified_child_archive_read' in grandchild
    print(json.dumps(dict(status='PASSED_RESET_PYTHONPATH_CHILD_AND_GRANDCHILD', path=str(path), sha256=expected)))


if __name__ == '__main__': main()
