"""Engineering smoke of real sparse MCTS on large connected braid words.

Eight simulations and two semantic moves deliberately do not claim solving these
knots. The question is whether search expands actual successors without L*N work.
"""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch
from run_strand_graph_pilot import PolicyEvaluator

from rf_knots import mcts_loop as M
from rf_knots.reference import num_components
from rf_knots.site_proposals import SiteProposalProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-proposals", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite benchmark")
    sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["status"] != "completed" or sha != manifest["checkpoint_sha256"]:
        raise ValueError("require completed hash-verified checkpoint")
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    net = M.SitePolicyValueNet(**checkpoint["architecture"]).eval()
    net.load_state_dict(checkpoint["model"], strict=True)
    report = {
        "status": "running",
        "pid": os.getpid(),
        "checkpoint_sha256": sha,
        "config": {
            "simulations": 8,
            "cap_moves": 2,
            "widening_batch": 4,
            "seed": 0,
            "threads": 1,
            "device": "cpu",
            "site_proposals": args.site_proposals,
        },
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/lazy_actions.py"),
                Path("src/rf_knots/actions.py"),
                Path("src/rf_knots/strand_graph_encoder.py"),
                Path("src/rf_knots/embedding_encoder.py"),
                Path("src/rf_knots/embedding_normalization.py"),
                Path("src/rf_knots/site_head.py"),
                Path("src/rf_knots/site_proposals.py"),
                Path("scripts/run_strand_graph_pilot.py"),
            )
        },
        "rows": [],
        "caveat": "Real successor expansion and policy inference, NOT solving evidence. "
        "Only eight simulations/two moves. No full-enumeration baseline at huge sizes. "
        "No failure labels generated; RSS is cumulative macOS process high-water bytes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix(".partial")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    save()
    for strands in (33, 2049, 8193):
        word = tuple(range(1, strands)) * 2
        assert num_components(word, strands) == 1
        instance = M.Instance(f"large-{strands}", word, strands, 2, cap_moves=2)
        for mode in ("uniform", "trained"):
            evaluator = M.UniformEvaluator() if mode == "uniform" else PolicyEvaluator(net, "cpu")
            provider = SiteProposalProvider(net) if mode == "trained" and args.site_proposals else None
            started = time.perf_counter()
            result = M.run_searches(
                [instance], evaluator, simulations=8, seeds=(0,), widening_batch=4,
                proposal_provider=provider,
            )[0]
            elapsed = time.perf_counter() - started
            nodes, stack = [], [result.root]
            while stack:
                node = stack.pop()
                nodes.append(node)
                stack.extend(c for c in node.children if c is not None)
            row = {
                "strands": strands,
                "length": len(word),
                "mode": mode,
                "site_proposals": provider is not None,
                "proposal_compute": provider.statistics() if provider is not None else None,
                "total_neural_state_evaluations": (
                    evaluator.calls + (provider.statistics()["state_evaluations"] if provider else 0)
                    if mode == "trained" else 0
                ),
                "seconds": elapsed,
                "simulations": result.simulations,
                "expansions_including_widening": result.expansions,
                "retained_nodes": len(nodes),
                "retained_child_states": sum(len(n.children_states) for n in nodes),
                "root_catalogue_size": len(result.root.catalogue),
                "root_catalogue_examined": result.root.catalogue_cursor,
                "root_actions_open": len(result.root.actions),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            assert row["root_catalogue_examined"] <= 4 * result.simulations
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in report["code_sha256"].items()):
        raise ValueError("benchmark source changed during execution")
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
