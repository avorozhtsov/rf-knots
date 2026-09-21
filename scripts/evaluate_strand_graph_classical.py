"""Classical best-first control on an already frozen MCTS query selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots.embedding_replay_corpus import verify_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an evaluation")
    selection = json.loads(args.selection_report.read_text())
    panel_sha = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    if panel_sha != selection["panel_sha256"]:
        raise ValueError("selection panel SHA mismatch")
    panel = json.loads(args.panel.read_text())
    config = selection["config"]
    rows = []
    started = time.time()
    for selected in selection["selected"]:
        i = selected["panel_index"]
        w = verify_record(panel["validation"][i])
        instance = M.Instance(
            f"short-u-{i}",
            w.start.word,
            w.start.strands,
            w.crossing_changes,
            cap_moves=config["cap_moves"],
            uid=f"short-u-{i}",
        )
        for seed in range(config["seeds"]):
            result = M.classical_beam(instance, simulations=config["simulations"], seed=seed)
            witness = result.witness
            if witness:
                witness.verify()
            rows.append(
                {
                    "instance": instance.name,
                    "seed": seed,
                    "solved": result.solved,
                    "expansions": result.expansions,
                    "L10": 10 * witness.crossing_changes + witness.moves if witness else None,
                    "L1000": 1000 * witness.crossing_changes + witness.moves if witness else None,
                    "witness": witness.to_dict() if witness else None,
                }
            )
        print(
            json.dumps({"finished": len(rows), "solved": sum(r["solved"] for r in rows)}),
            flush=True,
        )
    report = {
        "selection_sha256": hashlib.sha256(args.selection_report.read_bytes()).hexdigest(),
        "panel_sha256": panel_sha,
        "config": config,
        "selected": selection["selected"],
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/reference.py"),
            ]
        },
        "rows": rows,
        "solved": sum(r["solved"] for r in rows),
        "trials": len(rows),
        "seconds": time.time() - started,
        "caveat": (
            "Classical first-found best-first witnesses; equal expansions, "
            "not equal wall time or optimal costs."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
