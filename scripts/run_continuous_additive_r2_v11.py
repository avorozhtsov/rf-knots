"""Continuing single-writer optimizer scheduler for an audited bank spec."""

import argparse
import os
import sys
from pathlib import Path

import run_continuous_r2_f_v10 as core
from train_joint_f_r2_r4_v1 import read


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    spec = read(args.spec)
    core.OUT = Path(spec["out"])
    core.TRAINER = Path("scripts/train_continuous_additive_r2_v11.py")
    os.environ["R2_CONTINUATION_SPEC"] = str(args.spec)
    sys.argv = [sys.argv[0], *remaining]
    core.main()


if __name__ == "__main__":
    main()
