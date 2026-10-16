"""Protocol80 score-blind common panel and native witness accessibility gate."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from rf_knots.actions import ActionSpec
from rf_knots.conjugate_witnesses import conjugate_witness
from rf_knots.embedding_replay_corpus import canonical, invariants, owner
from rf_knots.evidence import UnknotWitness
from rf_knots.mcts_loop import Instance, SearchState
from rf_knots.site_proposals import proposal_candidates

CONFIG = {
    "seed": 20261040, "attempts": 4096, "per_stratum": 4,
    "max_strands": 8, "min_depth": 2, "max_depth": 8,
    "length_split": 20, "capacity_length": 48, "capacity_strands": 12,
    "native_horizon": 128, "cc_weights": [10, 1000],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_panel(config=CONFIG):
    rows, counts, seen, decisions = [], Counter(), set(), Counter()
    for index in range(config["attempts"]):
        witness = conjugate_witness(
            config["seed"], index, max_strands=config["max_strands"],
            min_depth=config["min_depth"], max_depth=config["max_depth"],
        )
        start = witness.start
        stratum = f"cc{witness.crossing_changes}-" + (
            "short" if len(start.word) <= config["length_split"] else "long"
        )
        key = canonical(start.word, start.strands)
        if counts[stratum] >= config["per_stratum"]:
            decisions["stratum_full"] += 1
            continue
        if key in seen:
            decisions["duplicate_cyclic_word"] += 1
            continue
        states = [start, *(step.after for step in witness.steps)]
        fingerprints = [invariants(s.word, s.strands) for s in states]
        if fingerprints[0][0] <= 1:
            decisions["root_not_certified_nontrivial_by_determinant"] += 1
            continue
        if any(owner(f) != "train" for f in fingerprints):
            decisions["non_train_path"] += 1
            continue
        spec = ActionSpec(config["capacity_length"], config["capacity_strands"])
        if any(len(s.word) > spec.max_len or s.strands > spec.max_strands for s in states):
            raise ValueError("constructive root/path exceeds declared common domain")
        witness = UnknotWitness.from_actions(
            start.word, start.strands, spec,
            [step.action.to_flat(spec) for step in witness.steps],
        )
        rows.append({
            "index": index, "stratum": stratum, "fingerprints": fingerprints,
            "witness": witness.to_dict(),
        })
        seen.add(key)
        counts[stratum] += 1
        decisions["accepted"] += 1
        if len(rows) == 4 * config["per_stratum"]:
            break
    return {
        "config": config, "rows": rows, "strata": dict(counts),
        "selection_counts": dict(decisions),
        "complete": all(counts[s] == config["per_stratum"] for s in (
            "cc1-short", "cc1-long", "cc2-short", "cc2-long",
        )),
        "scope": "TRAIN diagnostic representations; Q training exposure unknown",
    }


def rf_coverage(witness):
    instance = Instance(
        "common-witness", witness.start.word, witness.start.strands,
        cap_crossing_changes=128, cap_moves=128,
        growth_len=48 - len(witness.start.word),
        growth_strands=12 - witness.start.strands,
    )
    state = SearchState(witness.start.word, witness.start.strands, 0, 0)
    from rf_knots.cost_mcts_rollout import advance

    for step in witness.steps:
        action = step.action.to_flat(instance.spec)
        if action not in proposal_candidates(instance, state, 4):
            return False
        state = advance(instance, state, action)
        if (state.word, state.strands) != (step.after.word, step.after.strands):
            raise ValueError("RF proposal replay differs from reference")
    return (state.word, state.strands) == ((), 1)


def check_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError(f"input changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--baseline-check", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    baseline = json.loads(args.baseline_check.read_text())
    if baseline["status"] != "passed":
        raise ValueError("native checkpoint check must pass first")
    pins = dict(baseline["inputs"])
    for path in (args.baseline_check, Path(__file__), Path(
        "research/80-qrun-common-domain-diagnostic.md"
    )):
        pins[str(path.resolve())] = sha(path)
    if not all(str(p.resolve()) in pins for p in (args.pgx_root / "src").rglob("*.py")):
        raise ValueError("unverified native source files")
    check_pins(pins)
    panel = build_panel()
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "panel.json", panel)
    if not panel["complete"]:
        raise ValueError("fixed diagnostic supply insufficient; panel retained")
    pins[str((args.out / "panel.json").resolve())] = sha(args.out / "panel.json")
    write_new(args.out / "inputs.json", pins)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    from pgx_mcts_bench.adaptive_scientists import KnotItem, load_scientist
    from pgx_mcts_bench.collaborative_scientists import (
        translate_semantic_record,
        verified_record_cost,
    )
    from pgx_mcts_bench.game import make_game

    torch.set_num_threads(1)
    results = []
    for label, name in (("q254", "raster-axial-12"), ("q304", "strand-graph-12")):
        expected = baseline["results"][label]["checkpoint_sha256"]
        candidates = [Path(p) for p, h in pins.items() if h == expected]
        if len(candidates) != 1:
            raise ValueError("ambiguous frozen checkpoint")
        scientist = load_scientist(
            name, candidates[0], seed=CONFIG["seed"], device="cpu", simulations=0,
            require_factorized=True, objective_budget_channel=True,
        )
        payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
        weights = payload.get("network", payload)
        loaded = scientist.network.state_dict()
        if set(loaded) != set(weights) or any(
            not torch.equal(loaded[k], value) for k, value in weights.items()
        ):
            raise ValueError("native load changed checkpoint tensors")
        scientist.config = replace(scientist.config, game=replace(
            scientist.config.game, simplify_budget=CONFIG["native_horizon"],
        ))
        scientist.game = make_game(scientist.config.game)
        scientist.network.eval()
        for row in panel["rows"]:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            knot = KnotItem(str(row["index"]), len(witness.start.word),
                            witness.start.word, witness.start.strands)
            actions = [s.action.to_flat(scientist.game.spec) for s in witness.steps]
            for weight in CONFIG["cc_weights"]:
                record = translate_semantic_record(
                    scientist, knot, weight, actions, seed=CONFIG["seed"],
                )
                verified = (
                    verified_record_cost(scientist.game, knot, weight, record)
                    if record is not None else None
                )
                result = {
                    "model": label, "index": row["index"], "cc_weight": weight,
                    "rf_proposal_coverage": rf_coverage(witness),
                    "native_route_found": verified is not None,
                    "native_actions": [int(p.action) for p in record] if record else [],
                    "native_plies": len(record) if record is not None else None,
                }
                if verified is not None:
                    cc, moves, semantic = verified
                    replay = UnknotWitness.from_actions(
                        knot.word, knot.strands, scientist.game.spec, semantic,
                    )
                    replay.verify()
                    if (cc, moves) != (replay.crossing_changes, replay.moves):
                        raise ValueError("native and portable objective disagree")
                    result.update(
                        crossing_changes=cc, semantic_moves=moves,
                        internal_plies=len(record) - moves,
                        objective=weight * cc + moves, witness=replay.to_dict(),
                    )
                results.append(result)
                write_new(args.out / f"route-{len(results):03d}.json", result)
                print(json.dumps({k: v for k, v in result.items()
                                  if k not in ("witness", "native_actions")}), flush=True)
    check_pins(pins)
    passed = all(r["native_route_found"] and r["rf_proposal_coverage"] for r in results)
    plies = [r["native_plies"] for r in results if r["native_route_found"]]
    report = {
        "status": "passed" if passed else "not_ready", "inputs": pins,
        "conditions": len(results), "routes_found": sum(r["native_route_found"] for r in results),
        "native_plies_range": [min(plies), max(plies)] if plies else None,
        "policy_scored": False, "training_performed": False,
        "scope": "Oracle witness accessibility, NOT network solve rate or generalization",
        "route_hashes": {p.name: sha(p) for p in sorted(args.out.glob("route-*.json"))},
    }
    write_new(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "route_hashes")}))


if __name__ == "__main__":
    main()
