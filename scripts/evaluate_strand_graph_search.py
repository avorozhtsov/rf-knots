"""Matched MCTS on replayed short-u holdout queries, with pinned checkpoints.

Query selection uses only pre-existing witnesses and invariant groups, not model
scores. First-found costs are reported honestly, not called optimal witnesses.
"""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import torch
from evaluate_embedding_cost_search import query_instance
from run_strand_graph_pilot import search_report

from rf_knots import mcts_loop as M
from rf_knots.embedding_checkpoint_io import load_initial_checkpoint
from rf_knots.site_proposals import SiteProposalProvider
from rf_knots.embedding_replay_corpus import verify_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--panel-view",
        choices=("validation", "size_shift"),
        default="validation",
        help="which frozen panel view to evaluate; size_shift keeps the same roots but uses larger words/strand counts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--widening-batch", type=int, default=0)
    parser.add_argument("--witness-caps", action="store_true")
    parser.add_argument("--learned-value", action="store_true")
    parser.add_argument("--factorized-proposals", action="store_true")
    parser.add_argument("--site-proposals", action="store_true")
    parser.add_argument("--cc-weight", type=float, default=1000.0)
    parser.add_argument("--instances", type=int, default=9)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=1024)
    args = parser.parse_args()
    if args.site_proposals and (args.factorized_proposals or args.uniform or args.widening_batch < 1):
        raise ValueError("site proposals require neural progressive search and exclude factorized/uniform modes")
    if args.output.exists():
        raise SystemExit("refusing to overwrite a search evaluation")
    manifest = json.loads((args.run / "manifest.json").read_text())
    checkpoint_sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    if manifest["status"] != "completed" or checkpoint_sha != manifest["checkpoint_sha256"]:
        raise SystemExit("require a completed hash-verified checkpoint")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(manifest["config"]["seed"])
    net = None
    if not args.uniform:
        net = M.SitePolicyValueNet(**checkpoint["architecture"])
        if not args.initial:
            net.load_state_dict(checkpoint["model"], strict=True)
        else:
            initial = load_initial_checkpoint(args.run, manifest, checkpoint["architecture"])
            if initial is not None:
                net.load_state_dict(initial["model"], strict=True)
    if args.factorized_proposals and (net is None or net.factorized_head is None):
        raise ValueError("--factorized-proposals requires a factorized checkpoint")
    proposal_provider = (
        M.FactorizedProposalProvider(net, "cpu") if args.factorized_proposals else None
    )
    if args.site_proposals:
        proposal_provider = SiteProposalProvider(net, "cpu")
    panel = json.loads(args.panel.read_text())
    if panel.get("complete") is False:
        raise ValueError("refusing to evaluate a partial panel")
    panel_rows = panel.get(args.panel_view)
    if not isinstance(panel_rows, list) or not panel_rows:
        raise ValueError(f"panel view {args.panel_view!r} is missing or empty")
    choices = [(i, r, verify_record(r)) for i, r in enumerate(panel_rows)]
    choices.sort(key=lambda x: (x[2].crossing_changes, x[2].moves, len(x[2].start.word), x[0]))
    instances, selected, seen = [], [], set()
    for i, row, w in choices:
        det, sigma = row["left_fingerprint"]
        fp = det, abs(sigma)
        if fp in seen:
            continue
        seen.add(fp)
        instance, fits = query_instance(i, w, witness_caps=args.witness_caps)
        instance = replace(instance, cc_weight=args.cc_weight)
        instances.append(instance)
        selected.append(
            {
                "panel_index": i,
                "fingerprint": list(fp),
                "witness_moves": w.moves,
                "witness_cc": w.crossing_changes,
                "reference_witness_fits": fits,
                "resource_caps": instance.to_dict(),
            }
        )
        if len(instances) == args.instances:
            break
    report = {
        "status": "running",
        "pid": os.getpid(),
        "run": str(args.run),
        "uniform": args.uniform,
        "initial": args.initial,
        "partition": f"{panel.get('partition', 'validation')}:{args.panel_view}",
        "checkpoint_sha256": checkpoint_sha,
        "initial_checkpoint_sha256": manifest.get("initial_checkpoint_sha256") if args.initial and not args.uniform else None,
        "initial_state_source": (
            "saved_tensors" if manifest.get("initial_checkpoint_sha256") else "legacy_seed_reconstruction"
        ) if args.initial and not args.uniform else None,
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "config": {
            "instances": len(instances),
            "seeds": args.seeds,
            "simulations": args.simulations,
            "cap_moves": 64,
            "root_noise": 0.25,
            "widening_batch": args.widening_batch,
            "witness_caps": args.witness_caps,
            "learned_value": args.learned_value,
            "factorized_proposals": args.factorized_proposals,
            "site_proposals": args.site_proposals,
            "cc_weight": args.cc_weight,
        },
        "selected": selected,
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/rf_knots/embedding_checkpoint_io.py"),
                Path("scripts/run_strand_graph_pilot.py"),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/reference.py"),
                Path("src/rf_knots/strand_graph_encoder.py"),
                Path("src/rf_knots/embedding_encoder.py"),
                Path("src/rf_knots/site_head.py"),
                Path("src/rf_knots/embedding_normalization.py"),
                Path("src/rf_knots/lazy_actions.py"),
                Path("src/rf_knots/site_proposals.py"),
                Path("src/rf_knots/actions.py"),
                Path("scripts/evaluate_embedding_cost_search.py"),
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / "source_snapshot"
    for name, sha in report["code_sha256"].items():
        source = Path(name).resolve()
        relative = source.relative_to(Path.cwd().resolve())
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise ValueError("evaluation directory contains a different source revision")
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    report["search"] = search_report(
        net,
        instances,
        args.simulations,
        args.seeds,
        "cpu",
        0.25,
        args.widening_batch,
        args.learned_value,
        proposal_provider,
    )
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha
        for p, sha in report["code_sha256"].items()
    ):
        raise ValueError("evaluation source changed while running")
    report["status"] = "completed"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in report["search"].items() if k != "rows"}))


if __name__ == "__main__":
    main()
