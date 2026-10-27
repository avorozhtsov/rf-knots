"""Explicit versioned launcher; storage adapter only, no optimizer/spec/clock reset."""
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='allow new CAS bank rows; never needed for checks')
    parser.add_argument('script')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    script = (root/args.script).resolve()
    if not script.is_relative_to(root/'scripts') or not script.is_file():
        raise ValueError('only explicit repository scripts')
    env = dict(os.environ, RF_KNOTS_COMPACT_V31='1', RF_KNOTS_COMPACT_ROOT=str(root),
               RF_KNOTS_COMPACT_WRITE='1' if args.write else '0', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(root/'scripts/compact_runtime_v31')+os.pathsep+env.get('PYTHONPATH','')
    import sys
    os.chdir(root)
    os.execve(sys.executable, [sys.executable, '-u', str(script), *args.args], env)

if __name__ == '__main__':
    main()
