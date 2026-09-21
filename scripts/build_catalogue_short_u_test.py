"""Model-blind bounded certificate search for a small independent u1/u2 panel.

All cached catalogue knots with <=8 crossings in the frozen test partition.
Try one original-diagram CC if signature allows it, otherwise two for sigma=4.
Then zero-CC length-guided search. Failed budgets are NOT lower bounds.
"""

import hashlib
import itertools
import json
import time
from pathlib import Path

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE
from rf_knots.embedding_cost_search import greedy_cost_search
from rf_knots.embedding_replay_corpus import invariants, lift, owner, verify_record
from rf_knots.evidence import UnknotWitness


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("src/rf_knots/data/knot_table.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())
    selected = []
    for name, row in sorted(source["knots"].items(), key=lambda kv: (kv[1]["crossings"], kv[0])):
        if row["crossings"] > 8:
            continue
        fp = invariants(tuple(row["braid"]), row["strands"])
        if owner(fp) == "test" and abs(fp[1]) <= 4:
            selected.append((name, row, fp))
    report = {
        "status": "running",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "selection": [{"name": n, "fingerprint": fp} for n, _, fp in selected],
        "protocol": "All catalogue <=8-crossing owner=test; one CC if |sigma|<=2, "
        "else two CC. All original-position combinations, 2048 zero-CC expansions each, "
        "64 total moves, +4 letters/+1 strand. Stop at first replayed witness per root.",
        "attempts": [],
        "seconds": 0.0,
    }
    started = time.time()

    def save():
        report["seconds"] = time.time() - started
        temp = args.out / "search.partial"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.out / "search.json")

    save()
    records, large = [], []
    for name, row, fp in selected:
        word, n = tuple(row["braid"]), row["strands"]
        cc = max(1, abs(fp[1]) // 2)
        source_instance = M.Instance(name, word, n, cc, cap_moves=64, growth_len=4)
        spec = source_instance.spec
        for positions in itertools.combinations(range(len(word)), cc):
            changed, ns = word, n
            prefix = tuple(spec.encode(CROSSING_CHANGE, p) for p in positions)
            for action in prefix:
                changed, ns = reference.apply(spec, changed, ns, action)
            problem = M.Instance(name, changed, ns, 0, cap_moves=64 - cc, growth_len=4)
            result = greedy_cost_search(
                problem, lambda states: [len(s.word) for s in states], expansions=2048, cc_weight=10
            )
            report["attempts"].append(
                {
                    "name": name,
                    "positions": positions,
                    "expansions": result.expansions,
                    "solved": result.witness is not None,
                }
            )
            save()
            if result.witness is None:
                continue
            actions = prefix + tuple(s.action.to_flat(spec) for s in result.witness.steps)
            witness = UnknotWitness.from_actions(word, n, spec, actions)
            witness.verify()
            record = {
                "split": "test",
                "catalogue_name": name,
                "left_fingerprint": list(fp),
                "right_fingerprint": [1, 0],
                "pair_prefix_length": witness.moves,
                "witness": witness.to_dict(),
                "interval": {
                    "metric": "cc",
                    "left": witness.start.to_dict(),
                    "right": witness.steps[-1].after.to_dict(),
                    "lower": abs(fp[1]) // 2,
                    "upper": witness.crossing_changes,
                    "witness_semantic_length": witness.moves,
                },
            }
            verify_record(record)
            records.append(record)
            bigger, _ = lift(witness, max(9, n + 2), max(60, len(word) + 16))
            large_record = {
                **record,
                "evaluation_view": "size_shift",
                "witness": bigger.to_dict(),
                "pair_prefix_length": bigger.moves,
                "interval": {
                    **record["interval"],
                    "left": bigger.start.to_dict(),
                    "witness_semantic_length": bigger.moves,
                },
            }
            verify_record(large_record)
            large.append(large_record)
            print(
                json.dumps({"name": name, "cc": witness.crossing_changes, "moves": witness.moves}),
                flush=True,
            )
            break
    panel = {
        "schema": "rf-knots-short-u-panel-v1",
        "partition": "test",
        "complete": True,
        "source_sha256": report["source_sha256"],
        "validation": records,
        "size_shift": large,
        "caveat": "Model-blind but successful-certificate-search-selected, tiny catalogue cohort. "
        "Unsolved source roots excluded, NOT certified harder. Strengthened determinant lower "
        "bounds must be reported explicitly. No independent-trial claim for repeated groups.",
    }
    (args.out / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    report.update(status="completed", certified_roots=len(records))
    save()


if __name__ == "__main__":
    main()
