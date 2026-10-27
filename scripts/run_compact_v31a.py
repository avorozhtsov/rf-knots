"""Explicit compact v31 reader plus versioned Python-child transport repair."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    parser.add_argument('script'); parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(); root = Path(__file__).resolve().parent.parent
    script = (root/args.script).resolve()
    if not script.is_relative_to(root/'scripts') or not script.is_file():
        raise ValueError('only explicit repository scripts')
    env = dict(os.environ, RF_KNOTS_COMPACT_V31='1', RF_KNOTS_COMPACT_ROOT=str(root),
        RF_KNOTS_COMPACT_WRITE='1' if args.write else '0', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(root/'scripts/compact_runtime_v31a')+os.pathsep+env.get('PYTHONPATH','')
    os.chdir(root)
    os.execve(sys.executable, [sys.executable, '-u', str(script), *args.args], env)


if __name__ == '__main__': main()
