"""Direct embedding-guided, anytime cost search with length/random-net controls.

Independent of policy logits and the untrained value head. The immutable query
selection comes from the prior MCTS panel, not from model success/failure.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from rf_knots import mcts_loop as M
from rf_knots.embedding_checkpoint_io import load_initial_checkpoint
from rf_knots.embedding_cost_search import greedy_cost_search
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.embedding_normalization import fast_normalize_global_view
from rf_knots.embedding_replay_corpus import verify_record


class KnotDistanceHeuristic:
    def __init__(self, net, objective):
        if not net.encoder.normalized_knot:
            raise ValueError("normalized-view cache requires normalized_knot architecture")
        self.net, self.objective = net.eval(), objective.eval()
        self.embeddings = {}
        self.evaluations = 0
        self.scores = {}
        self._embed([((), 1)])
        self.anchor = self.embeddings[((), 1)]

    @torch.no_grad()
    def _embed(self, keys):
        for start in range(0, len(keys), 64):
            chunk = keys[start : start + 64]
            padded = max(1, max(len(w) for w, _ in chunk))
            words = torch.tensor(
                [list(w) + [0] * (padded - len(w)) for w, _ in chunk], dtype=torch.long
            )
            lengths = torch.tensor([len(w) for w, _ in chunk])
            strands = torch.tensor([n for _, n in chunk])
            encoded = self.net.encoder(words, lengths, strands)["knot"]
            for key, vector in zip(chunk, encoded, strict=True):
                self.embeddings[key] = vector.clone()
            self.evaluations += len(chunk)

    @torch.no_grad()
    def __call__(self, states):
        keys = [fast_normalize_global_view(s.word, s.strands) for s in states]
        missing = list(dict.fromkeys(k for k in keys if k not in self.embeddings))
        self._embed(missing)
        for key in dict.fromkeys(keys):
            if key not in self.scores:
                # Roundoff in normalized dot products can exceed 1 by one ULP.
                # Clamp the mathematical cosine domain, not the search result.
                cosine = (self.embeddings[key] * self.anchor).sum().clamp(-1.0, 1.0)
                self.scores[key] = float(
                    self.objective.scale("cc") * self.objective.cc_distance(cosine)
                )
        return [self.scores[k] for k in keys]


def query_instance(index, witness, *, witness_caps=False):
    states = [witness.start] + [s.after for s in witness.steps]
    peak_length = max(len(s.word) for s in states)
    peak_strands = max(s.strands for s in states)
    instance = M.Instance(
        f"short-u-{index}",
        witness.start.word,
        witness.start.strands,
        witness.crossing_changes,
        cap_moves=max(64, witness.moves) if witness_caps else 64,
        growth_len=max(2, peak_length - len(witness.start.word)) if witness_caps else 2,
        growth_strands=max(1, peak_strands - witness.start.strands) if witness_caps else 1,
        uid=f"short-u-{index}",
    )
    fits = (
        peak_length <= instance.spec.max_len
        and peak_strands <= instance.spec.max_strands
        and witness.moves <= instance.cap_moves
    )
    if witness_caps and not fits:
        raise AssertionError("reference witness must fit the expanded evaluation contract")
    return instance, fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("length", "initial", "learned"), required=True)
    parser.add_argument("--expansions", type=int, default=128)
    parser.add_argument("--instances", type=int, default=9)
    parser.add_argument("--layout-tiebreak", action="store_true")
    parser.add_argument(
        "--witness-caps",
        action="store_true",
        help="ensure the known witness fits the search resource caps",
    )
    parser.add_argument(
        "--all-queries", action="store_true", help="every frozen validation representation"
    )
    args = parser.parse_args()
    if not args.selection_report and not args.all_queries:
        raise SystemExit("provide a selection report or explicitly evaluate all queries")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
    checkpoint_sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    if manifest["status"] != "completed" or checkpoint_sha != manifest["checkpoint_sha256"]:
        raise SystemExit("require a completed hash-verified checkpoint")
    panel_sha = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    selection = json.loads(args.selection_report.read_text()) if args.selection_report else None
    if selection and selection["panel_sha256"] != panel_sha:
        raise ValueError("selection panel SHA mismatch")
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    torch.manual_seed(manifest["config"]["seed"])
    net = M.SitePolicyValueNet(**checkpoint["architecture"])
    objective = MetricEmbeddingObjective(cc_rank_weight=0.1, witness_priority=False)
    objective.calibrate_cc_scale(reference_cosine=0.4)
    if args.mode == "learned":
        net.load_state_dict(checkpoint["model"], strict=True)
        objective.load_state_dict(checkpoint["objective"], strict=True)
    elif args.mode == "initial":
        initial = load_initial_checkpoint(args.run, manifest, checkpoint["architecture"])
        if initial is not None:
            net.load_state_dict(initial["model"], strict=True)
            objective.load_state_dict(initial["objective"], strict=True)
    panel = json.loads(args.panel.read_text())
    if panel.get("complete") is False:
        raise ValueError("refusing to evaluate a partial panel")
    selected_queries = selection["selected"][: args.instances] if selection else []
    if args.all_queries:
        selected_queries = [
            {
                "panel_index": i,
                "fingerprint": [r["left_fingerprint"][0], abs(r["left_fingerprint"][1])],
            }
            for i, r in enumerate(panel["validation"])
        ]
    report = {
        "status": "running",
        "pid": os.getpid(),
        "run": str(args.run),
        "mode": args.mode,
        "checkpoint_sha256": checkpoint_sha,
        "initial_checkpoint_sha256": manifest.get("initial_checkpoint_sha256") if args.mode == "initial" else None,
        "initial_state_source": (
            "saved_tensors" if manifest.get("initial_checkpoint_sha256") else "legacy_seed_reconstruction"
        ) if args.mode == "initial" else None,
        "panel_sha256": panel_sha,
        "partition": panel.get("partition", "validation"),
        "selection_sha256": hashlib.sha256(args.selection_report.read_bytes()).hexdigest()
        if args.selection_report
        else None,
        "config": {
            "expansions": args.expansions,
            "cc_weights": [10, 1000],
            "cap_moves": 64,
            "layout_tiebreak": args.layout_tiebreak,
            "all_queries": args.all_queries,
            "witness_caps": args.witness_caps,
        },
        "selected": selected_queries,
        "rows": [],
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/rf_knots/embedding_checkpoint_io.py"),
                Path("src/rf_knots/embedding_cost_search.py"),
                Path("src/rf_knots/embedding_normalization.py"),
                Path("src/rf_knots/mcts_loop.py"),
                Path("src/rf_knots/strand_graph_encoder.py"),
            ]
        },
        "caveat": (
            "Anytime greedy, NOT MCTS. Equal expansions; "
            "not equal neural evaluations or wall time. "
            "Costs are best found within budget, NOT proven optima. "
            "Cache uses only zero-CC rewrites."
        ),
    }

    def save():
        temp = args.out / "report.partial"
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.out / "report.json")

    save()
    for selected in report["selected"]:
        i = selected["panel_index"]
        witness = verify_record(panel["validation"][i])
        instance, fits = query_instance(i, witness, witness_caps=args.witness_caps)
        for weight in (10, 1000):
            # No cross-root or cross-budget warm cache advantage.
            heuristic = (
                (lambda states: [len(s.word) for s in states])
                if args.mode == "length"
                else KnotDistanceHeuristic(net, objective)
            )
            started = time.time()
            result = greedy_cost_search(
                instance,
                heuristic,
                expansions=args.expansions,
                cc_weight=weight,
                layout_tiebreak=args.layout_tiebreak,
            )
            w = result.witness
            row = {
                "instance": instance.name,
                "resource_caps": {
                    "moves": instance.cap_moves,
                    "max_len": instance.spec.max_len,
                    "max_strands": instance.spec.max_strands,
                },
                "reference_witness_fits": fits,
                "cc_weight": weight,
                "solved": w is not None,
                "expansions": result.expansions,
                "generated": result.generated,
                "neural_state_evaluations": getattr(heuristic, "evaluations", 0),
                "seconds": time.time() - started,
                "incumbents": result.incumbents,
                "frontier_exhausted": result.exhausted,
                "witness": w.to_dict() if w else None,
                "L10": 10 * w.crossing_changes + w.moves if w else None,
                "L1000": 1000 * w.crossing_changes + w.moves if w else None,
            }
            report["rows"].append(row)
            save()
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("witness", "incumbents")}),
                flush=True,
            )
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
