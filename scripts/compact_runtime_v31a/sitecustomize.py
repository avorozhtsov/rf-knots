"""Separate pinned child-inheritance repair; original v31 reader remains unchanged."""
import os
if os.environ.get('RF_KNOTS_COMPACT_V31') == '1':
    try:
        import hashlib
        import json
        import runpy
        import sys
        from pathlib import Path
        root = Path(os.environ['RF_KNOTS_COMPACT_ROOT'])
        pins = json.loads((root/'artifacts/compact-storage-v31/runtime-v31a-pins.json').read_text())
        for name, expected in pins['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('compact v31a transport pin changed: '+name)
        runpy.run_path(str(root/'scripts/compact_runtime_v31/sitecustomize.py'))
        from compact_child_env_v31a import install
        install(root)
    except BaseException as error:
        print('COMPACT_V31A_FAIL_CLOSED: '+repr(error), file=sys.stderr, flush=True)
        os._exit(78)
