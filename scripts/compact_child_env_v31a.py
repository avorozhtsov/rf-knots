"""Opt-in transport fix: pinned v30 spawn resets PYTHONPATH in Python children."""
import os
from pathlib import Path
import subprocess
import sys


def child_env(command, env, root):
    if not isinstance(command, (list, tuple)) or not command:
        return env
    if os.path.realpath(str(command[0])) != os.path.realpath(sys.executable):
        return env
    result = dict(os.environ if env is None else env)
    source = os.environ
    for name in ('RF_KNOTS_COMPACT_V31', 'RF_KNOTS_COMPACT_ROOT', 'RF_KNOTS_COMPACT_WRITE'):
        result[name] = source[name]
    bridge = str(Path(root)/'scripts/compact_runtime_v31a')
    result['PYTHONPATH'] = os.pathsep.join([bridge, *[part for part in result.get('PYTHONPATH','').split(os.pathsep)
        if part and part != bridge]])
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    return result


def install(root):
    original = subprocess.Popen
    class CompactPythonChild(original):
        def __init__(self, args, *positional, **kwargs):
            # Repository workers always pass env by keyword; reject ambiguous positional env.
            if len(positional) > 9:
                raise ValueError('positional Popen env unsupported by explicit compact bridge')
            kwargs['env'] = child_env(args, kwargs.get('env'), root)
            super().__init__(args, *positional, **kwargs)
    subprocess.Popen = CompactPythonChild
