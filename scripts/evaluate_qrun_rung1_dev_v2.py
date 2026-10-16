"""JSON-normalized wrapper for the frozen rung-1 matched DEV search."""

from pathlib import Path

import evaluate_qrun_rung1_dev_v1 as old
from build_mcts_training_supervision import sha


def main():
    old.SEEDS = list(old.SEEDS)
    old.CONFIG["seeds"] = old.SEEDS
    old.CONFIG["source_v2_sha256"] = sha(Path(__file__))
    old.CONFIG["protocol_v2_sha256"] = sha(Path("research/287-qrun-rung1-dev-evaluator-json-fix-v2.md"))
    old.main()


if __name__ == "__main__":
    main()
