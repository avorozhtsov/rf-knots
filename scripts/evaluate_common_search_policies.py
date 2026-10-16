"""Protocol84 native Q MCTS versus RF continuation-cost search, diagnostic only."""

import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from check_qrun_common_witnesses import check_pins, sha
from evaluate_common_greedy_policies import (
    GATE,
    RF,
    ForwardCounter,
    atomic,
    load_model,
    raw_state,
    replay_step,
    snapshot,
)
from evaluate_joint_cost_mcts import run_batch

from rf_knots import mcts_loop as M
from rf_knots.actions import CROSSING_CHANGE, PASS
from rf_knots.evidence import UnknotWitness

GREEDY = Path("artifacts/common-greedy-policy-diagnostic-v1")
CONFIG = {
    "seed": 20261042,
    "models": ["q254", "q304", "rf-joint57"],
    "weights": [10, 1000],
    "decisions": 128,
    "cc_cap": 2,
    "max_len": 48,
    "max_strands": 12,
    "q_simulations_per_decision": 8,
    "rf_total_simulations": 8,
    "rf_insertion_candidates": 1056,
}


class SearchForwardCounter(ForwardCounter):
    def __call__(self, module, inputs, output):
        super().__call__(module, inputs, output)
        if not torch.isfinite(output[1]).all():
            raise ValueError("nonfinite native MCTS value")


class NativeWitnessSearch:
    """Mask all search transitions and retain proofs from unselected branches.

    Native states remain unchanged. An epoch retains the actual state objects
    while mapping their IDs to complete action prefixes; this prevents ID reuse
    and keeps memory bounded to one decision's small MCTS tree.
    """

    def __init__(self, fixed_game, start, weight):
        self.base = fixed_game
        self.start = start
        self.weight = weight
        self.spec = fixed_game.base.spec
        self.paths = {}
        self.best = None
        self.incumbents = []
        self.transitions = 0

    def __getattr__(self, key):
        return getattr(self.base, key)

    def mask(self, transition):
        legal = np.asarray(transition.legal_actions).copy()
        raw = self.base.unwrap(transition.state)
        cc = int(np.asarray(raw._crossing_changes))
        head = int(transition.state.base_state.head)
        length = len(raw_state(self.base, transition.state)[0])
        for native in np.flatnonzero(legal):
            action = self.base.base.underlying_action(int(native), head, length)
            if action is not None:
                kind = self.spec.decode(action)[0]
                if kind == PASS or (kind == CROSSING_CHANGE and cc >= CONFIG["cc_cap"]):
                    legal[native] = False
        if not transition.terminated and not legal.any():
            raise ValueError("empty nonterminal native mask under frozen search contract")
        return replace(transition, legal_actions=legal)

    def begin(self, transition, native_prefix=(), semantic_prefix=()):
        self.paths = {id(transition.state): (transition.state, native_prefix, semantic_prefix)}

    def reset(self, seed):
        transition = self.mask(self.base.reset(seed))
        if raw_state(self.base, transition.state) != self.start:
            raise ValueError("native reset changed root")
        self.begin(transition)
        return transition

    def step(self, state, native):
        stored, path, semantic = self.paths[id(state)]
        if stored is not state:
            raise ValueError("native state identity was reused")
        before = raw_state(self.base, state)
        head = int(state.base_state.head)
        action = self.base.base.underlying_action(native, head, len(before[0]))
        raw = self.base.unwrap(state)
        cc = int(np.asarray(raw._crossing_changes))
        if action is not None and (
            self.spec.decode(action)[0] == PASS
            or (self.spec.decode(action)[0] == CROSSING_CHANGE and cc >= CONFIG["cc_cap"])
        ):
            raise ValueError("masked action reached a virtual or actual transition")
        transition = self.mask(self.base.step(state, native))
        after = raw_state(self.base, transition.state)
        replay_step(self.spec, before, action, after, cc)
        path = (*path, int(native))
        semantic = semantic if action is None else (*semantic, int(action))
        if len(path) > CONFIG["decisions"]:
            raise ValueError("native search escaped path horizon")
        self.paths[id(transition.state)] = (transition.state, path, semantic)
        self.transitions += 1
        if after == ((), 1):
            witness = UnknotWitness.from_actions(*self.start, self.spec, semantic)
            witness.verify()
            cost = self.weight * witness.crossing_changes + witness.moves
            if self.best is None or cost < self.best["cost"]:
                self.best = {
                    "cost": cost,
                    "witness": witness.to_dict(),
                    "native_actions": list(path),
                    "semantic_actions": list(semantic),
                    "discovery_transition": self.transitions,
                }
                self.incumbents.append(self.best)
        return transition


def native_episode(model, row, weight, seed):
    from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem
    from pgx_mcts_bench.search import NeuralMCTS

    witness = UnknotWitness.from_dict(row["witness"])
    start = witness.start.word, witness.start.strands
    fixed = FixedWordGame(model.game, KnotItem(str(row["index"]), len(start[0]), *start), weight)
    game = NativeWitnessSearch(fixed, start, weight)
    search_config = replace(model.config.search, simulations=CONFIG["q_simulations_per_decision"])
    search = NeuralMCTS(game, model.network, search_config, "cpu")
    if getattr(model.network, "forward_with_states", None) is not None:
        raise ValueError("counter contract requires ordinary native forward")
    counter = SearchForwardCounter()
    hook = model.network.register_forward_hook(counter)
    transition = game.reset(seed)
    native_path, semantic_path, transcript = (), (), []
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    try:
        for _ in range(CONFIG["decisions"]):
            if transition.terminated:
                break
            game.begin(transition, native_path, semantic_path)
            result = search.run(
                transition.state,
                transition.observation,
                transition.legal_actions,
                rng,
                temperature=0.0,
                add_root_noise=False,
            )
            before = raw_state(game, transition.state)
            head = int(transition.state.base_state.head)
            action = model.game.underlying_action(result.action, head, len(before[0]))
            transition = game.step(transition.state, result.action)
            _, native_path, semantic_path = game.paths[id(transition.state)]
            transcript.append(
                {
                    "native_action": result.action,
                    "semantic_action": action,
                    "head_before": head,
                    "before": snapshot(before),
                    "after": snapshot(raw_state(game, transition.state)),
                }
            )
    finally:
        hook.remove()
    if counter.samples != search.inference_cache_stats["misses"]:
        raise ValueError("native actual forward count differs from cache misses")
    return {
        "best": game.best,
        "solved": game.best is not None,
        "executed_solved": raw_state(game, transition.state) == ((), 1),
        "stop_reason": str(transition.termination_reason)
        if transition.terminated
        else "decision_cap",
        "executed_transcript": transcript,
        "incumbents": game.incumbents,
        "actual_neural_samples": counter.samples,
        "actual_forward_batches": counter.batches,
        "inference_cache": search.inference_cache_stats,
        "search_config": asdict(search_config),
        "virtual_and_executed_transitions": game.transitions,
        "seconds": time.perf_counter() - started,
    }


def rf_episode(checkpoint, row, weight, condition):
    witness = UnknotWitness.from_dict(row["witness"])
    start = witness.start
    instance = M.Instance(
        "common-search",
        start.word,
        start.strands,
        cap_crossing_changes=2,
        cap_moves=128,
        growth_len=48 - len(start.word),
        growth_strands=12 - start.strands,
        cc_weight=weight,
        uid=f"common-{row['index']}-L{weight}",
    )
    roots = [{"condition": condition, "panel_index": row["index"], "instance": instance.to_dict()}]
    config = {
        "seed": CONFIG["seed"],
        "simulations": CONFIG["rf_total_simulations"],
        "horizon": 128,
        "insertion_candidates": CONFIG["rf_insertion_candidates"],
        "widening_batch": 4,
        "c_puct": 1.5,
        "temperature": 1.0,
    }
    started = time.perf_counter()
    rows, counts = run_batch(roots, checkpoint, condition, config)
    return {
        "search": rows[0],
        "counts": counts,
        "roots": roots,
        "rf_config": config,
        "solved": rows[0]["solved"],
        "seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit matching resume")
    audit_path = GREEDY / "replay-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["conditions"] != 96:
        raise ValueError("require completed audited greedy diagnostic")
    pins = dict(audit["inputs"])
    sources = [
        audit_path,
        Path(__file__),
        Path("research/84-common-domain-search-diagnostic-protocol.md"),
        Path("scripts/evaluate_joint_cost_mcts.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("scripts/check_cost_mcts_continuations.py"),
        Path("scripts/evaluate_embedding_cost_search.py"),
        Path("scripts/audit_joint_policy_metric_panel.py"),
        Path("scripts/build_mcts_training_supervision.py"),
    ]
    for path in sources:
        key, value = str(path.resolve()), sha(path)
        if key in pins and pins[key] != value:
            raise ValueError("inherited source pin changed")
        pins[key] = value
    check_pins(pins)
    if not all(str(p.resolve()) in pins for p in (args.pgx_root / "src").rglob("*.py")):
        raise ValueError("unexpected native source scope")
    panel = json.loads((GATE / "panel.json").read_text())
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mp = args.out / "manifest.json"
    manifest = (
        json.loads(mp.read_text())
        if args.resume
        else {
            "inputs": pins,
            "config": CONFIG,
            "conditions": {},
            "status": "running",
        }
    )
    if manifest["inputs"] != pins or manifest["config"] != CONFIG:
        raise ValueError("resume contract differs")
    for name, expected in manifest["conditions"].items():
        if sha(args.out / name) != expected:
            raise ValueError("committed condition changed")
    if manifest["status"] == "completed":
        print("completed diagnostic hashes verified")
        return
    manifest.update(pid=os.getpid(), status="running")
    atomic(mp, manifest)
    torch.set_num_threads(1)
    condition = 0
    for label in CONFIG["models"]:
        torch.manual_seed(CONFIG["seed"])
        model = (
            torch.load(RF / "checkpoint.pt", map_location="cpu", weights_only=False)
            if label == "rf-joint57"
            else load_model(label, baseline, pins)
        )
        for row in panel["rows"]:
            for weight in CONFIG["weights"]:
                name = f"{label}-{row['index']:04d}-L{weight}.json"
                condition += 1
                if name in manifest["conditions"]:
                    continue
                if (args.out / name).exists():
                    raise ValueError("uncommitted condition needs explicit recovery inspection")
                result = (
                    rf_episode(model, row, weight, condition)
                    if label == "rf-joint57"
                    else native_episode(model, row, weight, CONFIG["seed"] + condition)
                )
                result.update(
                    model=label, index=row["index"], cc_weight=weight, condition=condition
                )
                atomic(args.out / name, result)
                manifest["conditions"][name] = sha(args.out / name)
                atomic(mp, manifest)
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "model": label,
                            "index": row["index"],
                            "cc_weight": weight,
                            "solved": result["solved"],
                            "seconds": result["seconds"],
                        }
                    ),
                    flush=True,
                )
        del model
    check_pins(pins)
    if len(manifest["conditions"]) != 96:
        raise ValueError("incomplete search grid")
    manifest.update(status="completed")
    atomic(mp, manifest)
    print("completed all 96 search conditions; independent audit still required", flush=True)


if __name__ == "__main__":
    main()
