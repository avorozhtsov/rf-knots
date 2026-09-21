"""Known-good root-action coverage, not a classification of unknown actions as bad.

Changes only the budget inputs, holding root and legal candidates fixed. A replay
completion within M moves also fits any larger cap: the positive set stays valid.
This diagnostic does not train or count itself as a search success.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an audit")
    manifest = json.loads((args.run / "manifest.json").read_text())
    checkpoint_sha = hashlib.sha256((args.run / "checkpoint.pt").read_bytes()).hexdigest()
    if manifest["status"] != "completed" or checkpoint_sha != manifest["checkpoint_sha256"]:
        raise SystemExit("require a completed, hash-verified checkpoint")
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    net = M.SitePolicyValueNet(**checkpoint["architecture"])
    net.load_state_dict(checkpoint["model"], strict=True)
    evaluator = M.NetworkEvaluator(net)
    corpus_path = args.run / "corpus.json"
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_sha != manifest["corpus_sha256"]:
        raise SystemExit("corpus SHA mismatch")
    corpus = json.loads(corpus_path.read_text())
    report = {
        "checkpoint_sha256": checkpoint_sha,
        "corpus_sha256": corpus_sha,
        "caveat": "Known-good coverage is a lower bound; misses are NOT bad actions.",
        "splits": {},
    }
    for split in ("train", "validation"):
        samples = []
        for i, record in enumerate(corpus[split][: args.count]):
            witness = UnknotWitness.from_dict(record["witness"])
            witness.verify()
            candidates = M.samples_from_witness(
                witness,
                name=f"{split}-{i}",
                cap_moves=witness.moves,
                growth_len=witness.max_len,
                growth_strands=witness.max_strands,
            )
            if candidates and candidates[0].state.word == witness.start.word:
                samples.append(candidates[0])
        regimes = {}
        for cap in (None, 64, 128):
            requests = []
            for s in samples:
                instance = (
                    s.instance
                    if cap is None
                    else replace(s.instance, cap_moves=max(cap, s.instance.cap_moves))
                )
                # Enlarging a positive move budget cannot remove a legal first action.
                assert {a for a, _ in M.successors_of(s.state, instance, instance.spec)} == set(
                    s.actions
                )
                requests.append((instance, s.state, s.actions))
            rows = []
            for s, request, evaluation in zip(
                samples, requests, evaluator.evaluate(requests), strict=True
            ):
                positive = {s.actions.index(a) for a in s.positives}
                ranking = np.argsort(-evaluation.priors, kind="stable")
                rows.append(
                    {
                        "name": s.instance.name,
                        "moves": request[0].cap_moves,
                        "candidates": len(s.actions),
                        "known_good": len(positive),
                        "good_mass": sum(float(evaluation.priors[i]) for i in positive),
                        "uniform_mass": len(positive) / len(s.actions),
                        "hit1": int(ranking[0] in positive),
                        "hit10": int(bool(set(ranking[:10]) & positive)),
                    }
                )
            regimes["witness_cap" if cap is None else f"at_least_{cap}"] = {
                "count": len(rows),
                "rows": rows,
                **{
                    k: float(np.mean([r[k] for r in rows])) if rows else None
                    for k in ("good_mass", "uniform_mass", "hit1", "hit10")
                },
            }
        report["splits"][split] = regimes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                s: {c: {k: v for k, v in r.items() if k != "rows"} for c, r in regimes.items()}
                for s, regimes in report["splits"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
