"""Engineering-only replay diagnostic, not an independent model evaluation."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots.anytime_cost_mcts import run_anytime_cost_searches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("research/48-anytime-cost-mcts-contract.md")]
    sources += sorted(Path("src/rf_knots").glob("*.py"))
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {"status": "running", "pid": os.getpid(), "sources": pins, "rows": [],
              "scope": "engineering-only; uniform priors; no embedding or RL benefit claim"}

    def save():
        tmp = args.out / "report.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.out / "report.json")

    save()
    for weight in (10, 1000):
        instances = [M.Instance("trefoil", (1, 1, 1), 2, 1, cap_moves=8, cc_weight=weight),
                     M.Instance("reducible", (1, -1, 1), 2, 0, cap_moves=8, cc_weight=weight)]
        started = time.monotonic()
        trees = run_anytime_cost_searches(instances, M.UniformEvaluator(), simulations=256)
        for t in trees:
            assert t.best_witness is not None
            t.best_witness.verify()
            assert t.simulations > t.incumbents[0]["simulation"]
            report["rows"].append({"name": t.instance.name, "weight": weight,
                                   "simulations": t.simulations, "expansions": t.expansions,
                                   "incumbents": t.incumbents,
                                   "witness": t.best_witness.to_dict(),
                                   "exhausted": t.exhausted})
        print(json.dumps({"weight": weight, "rows": len(report["rows"]),
                          "seconds": time.monotonic() - started}), flush=True)
        save()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in pins.items())
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
