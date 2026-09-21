"""Measure factorized proposal recall without changing the MCTS default path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rf_knots import reference
from rf_knots.actions import INSERT, NUM_KINDS, ActionSpec
from rf_knots.lazy_actions import factorized_action_candidates


def action_score(action: int, spec: ActionSpec, kind, positions, generators, signs) -> float:
    action_kind, position, generator, sign = spec.decode(action)
    score = float(kind[action_kind])
    if action_kind != INSERT and action_kind <= INSERT:
        score += float(positions[action_kind, position])
    elif action_kind == INSERT:
        score += float(positions[INSERT, position])
        score += float(generators[generator - 1])
        score += float(signs[0 if sign > 0 else 1])
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite benchmark")
    rng = np.random.default_rng(args.seed)
    rows = []
    for max_len, max_strands, word in (
        (32, 8, (1, -2, 3, 2, -1)),
        (32, 8, (1, 2, -3, 1, -2, 3, 2, -1, 1)),
        (64, 12, (1, 2, 3, -2, 1, -3, 2, 1, -2, 3, 1, -1, 2)),
    ):
        spec = ActionSpec(max_len, max_strands)
        kind = rng.normal(size=NUM_KINDS)
        positions = rng.normal(size=(NUM_KINDS, len(word)))
        generators = rng.normal(size=max_strands - 1)
        signs = rng.normal(size=2)
        legal = reference.successors(spec, word, max_strands - 1, True)
        scored = [
            (
                action,
                action_score(action, spec, kind, positions, generators, signs),
            )
            for action, _, _ in legal
        ]
        scored.sort(key=lambda row: (-row[1], row[0]))
        target = {action for action, _ in scored[: min(16, len(scored))]}
        row = {
            "max_len": max_len,
            "strands": max_strands - 1,
            "length": len(word),
            "legal_actions": len(legal),
            "full_insertions": 2 * len(word) * (max_strands - 2),
            "recall": {},
        }
        for limit in (1, 2, 4, 8):
            proposals = factorized_action_candidates(
                spec,
                word,
                max_strands - 1,
                kind_scores=kind,
                position_scores=positions,
                generator_scores=generators,
                sign_scores=signs,
                top_positions=limit,
                top_generators=limit,
            )
            proposed = {action for action, _ in proposals}
            row["recall"][str(limit)] = {
                "top16": len(proposed & target) / max(len(target), 1),
                "proposed": len(proposals),
            }
        rows.append(row)

    large_word = tuple((index % 99999) + 1 for index in range(64))
    large_spec = ActionSpec(100_000, 100_000)
    large_kind = np.zeros(NUM_KINDS)
    large_positions = np.zeros((NUM_KINDS, len(large_word)))
    large_generators = np.zeros(99_999)
    large_signs = np.zeros(2)
    proposals = factorized_action_candidates(
        large_spec,
        large_word,
        100_000,
        kind_scores=large_kind,
        position_scores=large_positions,
        generator_scores=large_generators,
        top_positions=8,
        top_generators=8,
    )
    rows.append(
        {
            "large_case": True,
            "length": len(large_word),
            "strands": 100_000,
            "full_insertions": 2 * len(large_word) * 99_999,
            "proposed": len(proposals),
            "theoretical_factorized_insertions": 2 * 8 * 8 * 2,
        }
    )
    report = {
        "status": "completed",
        "seed": args.seed,
        "rows": rows,
        "contract": (
            "proposal recall is measured against additive synthetic scores; "
            "legality is exact, but this is not a search-quality claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
