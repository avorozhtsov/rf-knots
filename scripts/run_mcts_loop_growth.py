"""The same three arms in the growth-heavy regime, with no further training.

§2.3 and §3.6: controlled inflation is a first-class requirement, and a
length-ordered beam is disqualified for anything past the second change *because*
it will never grow a diagram.  The main run gives the search one letter of slack
(`growth_len = 2`), which is enough for `INSERT` to be legal at all.  Widening the
slack multiplies the branching factor without adding a single new instance, and a
length-ordered beam drowns in it.  That is the regime where a learned prior is
supposed to be the thing that narrows the search, so it is worth asking the
question directly -- with the checkpoints the main run already produced and no
retraining, so the answer is about the slack and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rf_knots import mcts_loop as M  # noqa: E402
from scripts.run_mcts_loop import build_rungs, evaluate_arm, log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/mcts-loop")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--doses", default="800,3200")
    parser.add_argument("--growth-lens", default="4,6")
    parser.add_argument("--cap-moves", type=int, default=32)
    parser.add_argument("--rungs", default="A")
    args = parser.parse_args()

    out = Path(args.out)
    seeds = tuple(range(args.seeds))
    doses = [int(x) for x in args.doses.split(",") if x]
    wanted = [x for x in args.rungs.split(",") if x]
    started = time.monotonic()

    net = M.SitePolicyValueNet()
    checkpoints = {
        "untrained": torch.load(out / "untrained.pt"),
        "warmup": torch.load(out / "warmup.pt"),
        "trained": torch.load(out / "trained.pt"),
    }

    rows: list[dict] = []
    payload = {"schema": "mcts-loop-growth-v1", "config": vars(args), "arms": rows}

    def flush() -> None:
        payload["elapsed_seconds"] = time.monotonic() - started
        (out / "growth.json").write_text(json.dumps(payload, indent=2, default=str))

    for growth in [int(x) for x in args.growth_lens.split(",") if x]:
        rungs = {
            name: instances
            for name, instances in build_rungs(growth, args.cap_moves).items()
            if name in wanted
        }
        for rung, instances in rungs.items():
            spec_widths = sorted({i.spec.num_actions for i in instances})
            log(f"growth_len={growth} rung {rung}: action-space widths {spec_widths}")
            for dose in doses:
                rows.append(
                    {"growth_len": growth, "rung": rung, **evaluate_arm(
                        "classical-beam", instances, None, simulations=dose, seeds=seeds)}
                )
                flush()
                rows.append(
                    {"growth_len": growth, "rung": rung, **evaluate_arm(
                        "uniform-mcts", instances, M.UniformEvaluator(),
                        simulations=dose, seeds=seeds)}
                )
                flush()
                for tag, state in checkpoints.items():
                    net.load_state_dict(state)
                    rows.append(
                        {"growth_len": growth, "rung": rung, **evaluate_arm(
                            f"{tag}-puct", instances, M.NetworkEvaluator(net),
                            simulations=dose, seeds=seeds)}
                    )
                    flush()
    log(f"done in {time.monotonic() - started:.0f}s")


if __name__ == "__main__":
    main()
