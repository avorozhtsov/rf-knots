"""Protocol82: actual frozen Q/RF greedy policies, not witness routing or MCTS."""

import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from check_qrun_common_witnesses import check_pins, sha

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, PASS, ActionSpec
from rf_knots.cost_mcts_rollout import advance
from rf_knots.evidence import UnknotWitness
from rf_knots.lazy_actions import LazyActionCatalogue

GATE = Path("artifacts/qrun-common-witness-access-v1")
RF = Path("artifacts/mcts-policy-metric-joint-v1")
RF_SHA = "e58270a34569bfad11127ef4f69a8e67bcdcf4825213f88bce03a73a2233ef84"
CONFIG = {"seed": 20261041, "decisions": 128, "cc_cap": 2, "max_len": 48,
          "max_strands": 12, "weights": [10, 1000], "models": ["q254", "q304", "rf-joint57"]}


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def choose(scores, actions):
    scores = np.asarray(scores)
    if not len(actions) or scores.shape != (len(actions),) or not np.isfinite(scores).all():
        raise ValueError("empty/nonfinite/malformed legal policy scores")
    return int(actions[min(range(len(actions)), key=lambda i: (-float(scores[i]), actions[i]))])


def raw_state(game, state):
    raw = game.unwrap(state)
    return tuple(int(g) for g in np.asarray(raw._word) if int(g)), int(np.asarray(raw._n))


def snapshot(state):
    return {"word": list(state[0]), "strands": state[1]}


def replay_step(spec, before, action, after, used_cc):
    if action is None:
        if before != after:
            raise ValueError("controller action changed the braid")
        return used_cc
    if not reference.is_legal(spec, *before, action, used_cc < CONFIG["cc_cap"]):
        raise ValueError("policy selected an illegal semantic move")
    if reference.apply(spec, *before, action) != after:
        raise ValueError("native/reference transition mismatch")
    return used_cc + int(spec.decode(action)[0] == CROSSING_CHANGE)


class ForwardCounter:
    def __init__(self, rf=False):
        self.samples = self.batches = 0
        self.rf = rf

    def __call__(self, module, inputs, output):
        self.samples += int(inputs[0].shape[0])
        self.batches += 1
        if self.rf:
            logits, _, effective = output
            if not effective.any(dim=1).all() or not torch.isfinite(logits[effective]).all():
                raise ValueError("invalid RF logits; uniform fallback forbidden")
        elif not torch.isfinite(output[0]).all():
            raise ValueError("invalid Q logits")


def episode(label, model, row, weight):
    original = UnknotWitness.from_dict(row["witness"])
    original.verify()
    start = original.start.word, original.start.strands
    spec = ActionSpec(CONFIG["max_len"], CONFIG["max_strands"])
    is_rf = label == "rf-joint57"
    net = model if is_rf else model.network
    counter = ForwardCounter(rf=is_rf)
    hook = net.register_forward_hook(counter)
    transcript, semantic, used_cc, scored = [], [], 0, 0
    before = start
    seen, repeated = {start}, 0
    reason = "decision_cap"
    if is_rf:
        instance = M.Instance(label, *start, cap_crossing_changes=CONFIG["cc_cap"],
                              cap_moves=CONFIG["decisions"], growth_len=48 - len(start[0]),
                              growth_strands=12 - start[1], cc_weight=weight)
        state = M.SearchState(*start, 0, 0)
        evaluator = M.NetworkEvaluator(net)
    else:
        from pgx_mcts_bench.adaptive_scientists import FixedWordGame, KnotItem

        base = model.game
        game = FixedWordGame(base, KnotItem(str(row["index"]), len(start[0]), *start), weight)
        transition = game.reset(CONFIG["seed"])
        if raw_state(game, transition.state) != start:
            raise ValueError("native reset changed common root")
    started = time.perf_counter()
    try:
        for _ in range(CONFIG["decisions"]):
            if before == ((), 1):
                reason = "solved"
                break
            if not is_rf and transition.terminated:
                reason = str(transition.termination_reason)
                break
            if is_rf:
                catalogue = LazyActionCatalogue(spec, *before, used_cc < CONFIG["cc_cap"])
                actions = sorted(catalogue.action_at(i) for i in range(len(catalogue)))
                if not actions:
                    reason = "no_legal_actions"
                    break
                result = evaluator.evaluate([(instance, state, actions)])[0]
                action = choose(result.priors, actions)
                state = advance(instance, state, action)
                after = state.word, state.strands
                native, head = None, None
            else:
                controller = transition.state.base_state
                head = int(controller.head)
                mapping = {}
                for native in np.flatnonzero(transition.legal_actions):
                    action = base.underlying_action(int(native), head, len(before[0]))
                    if action is not None:
                        kind = spec.decode(action)[0]
                        capped_cc = kind == CROSSING_CHANGE and used_cc >= CONFIG["cc_cap"]
                        if kind == PASS or capped_cc:
                            continue
                    mapping[int(native)] = action
                actions = sorted(mapping)
                if not actions:
                    reason = "no_legal_actions"
                    break
                tensor = torch.from_numpy(np.asarray(transition.observation).copy())
                tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2).float()
                with torch.inference_mode():
                    logits, _ = net(tensor)
                native = choose(logits[0, actions].cpu().numpy(), actions)
                action = mapping[native]
                transition = game.step(transition.state, native)
                after = raw_state(game, transition.state)
            scored += len(actions)
            used_cc = replay_step(spec, before, action, after, used_cc)
            if action is not None:
                semantic.append(action)
                repeated += int(after in seen)
                seen.add(after)
            transcript.append({"before": snapshot(before), "after": snapshot(after),
                               "native_action": native, "head_before": head,
                               "semantic_action": action, "legal_scores": len(actions)})
            before = after
        if before == ((), 1):
            reason = "solved"
        elif not is_rf and transition.terminated:
            reason = str(transition.termination_reason)
    finally:
        hook.remove()
    if counter.samples != len(transcript) or counter.batches != len(transcript):
        raise ValueError("actual neural counter differs from decision count")
    witness = None
    if reason == "solved":
        witness = UnknotWitness.from_actions(*start, spec, semantic)
        witness.verify()
        if witness.crossing_changes != used_cc:
            raise ValueError("CC count mismatch")
    return {
        "model": label, "index": row["index"], "cc_weight": weight,
        "solved": witness is not None, "stop_reason": reason,
        "neural_samples": counter.samples, "forward_batches": counter.batches,
        "scored_actions": scored, "semantic_moves": len(semantic), "used_cc": used_cc,
        "controller_only_decisions": len(transcript) - len(semantic),
        "repeated_semantic_states": repeated, "seconds": time.perf_counter() - started,
        "cost": weight * used_cc + len(semantic) if witness else None,
        "witness": witness.to_dict() if witness else None,
        "transcript": transcript, "final": snapshot(before),
    }


def load_model(label, baseline, pins):
    if label == "rf-joint57":
        checkpoint = torch.load(RF / "checkpoint.pt", map_location="cpu", weights_only=False)
        net = M.SitePolicyValueNet(**checkpoint["architecture"])
        net.load_state_dict(checkpoint["model"], strict=True)
        net.eval()
        return net
    from pgx_mcts_bench.adaptive_scientists import load_scientist
    from pgx_mcts_bench.game import make_game

    expected = baseline["results"][label]["checkpoint_sha256"]
    paths = [Path(p) for p, h in pins.items() if h == expected]
    if len(paths) != 1:
        raise ValueError("ambiguous native checkpoint")
    model = load_scientist(
        "raster-axial-12" if label == "q254" else "strand-graph-12", paths[0],
        seed=CONFIG["seed"], device="cpu", simulations=0,
        require_factorized=True, objective_budget_channel=True,
    )
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    weights, loaded = payload.get("network", payload), model.network.state_dict()
    if set(weights) != set(loaded) or any(
        not torch.equal(v, loaded[k]) for k, v in weights.items()
    ):
        raise ValueError("native weights changed on load")
    model.config = replace(model.config, game=replace(model.config.game, simplify_budget=128))
    model.game = make_game(model.config.game)
    model.network.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    gate = json.loads((GATE / "report.json").read_text())
    if gate["status"] != "passed" or gate["conditions"] != 64:
        raise ValueError("reference accessibility gate incomplete")
    pins = dict(gate["inputs"])
    for path in [GATE / "report.json", Path(__file__), RF / "checkpoint.pt", RF / "manifest.json",
                 Path("research/82-common-domain-greedy-policy-protocol.md")]:
        pins[str(path.resolve())] = sha(path)
    for name, expected in gate["route_hashes"].items():
        pins[str((GATE / name).resolve())] = expected
    if sha(RF / "checkpoint.pt") != RF_SHA:
        raise ValueError("RF initial checkpoint changed")
    rf_manifest = json.loads((RF / "manifest.json").read_text())
    if rf_manifest["status"] != "completed" or rf_manifest["checkpoint_sha256"] != RF_SHA:
        raise ValueError("RF checkpoint is not a committed final")
    if not all(str(p.resolve()) in pins for p in (args.pgx_root / "src").rglob("*.py")):
        raise ValueError("native source scope differs from verified baseline")
    check_pins(pins)
    panel = json.loads((GATE / "panel.json").read_text())
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if args.resume else {
        "inputs": pins, "config": CONFIG, "episodes": {}, "status": "running",
    }
    if manifest["inputs"] != pins or manifest["config"] != CONFIG:
        raise ValueError("resume contract differs")
    for name, expected in manifest["episodes"].items():
        if sha(args.out / name) != expected:
            raise ValueError("committed episode changed")
    manifest.update(pid=os.getpid(), status="running")
    atomic(manifest_path, manifest)
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    for label in CONFIG["models"]:
        model = load_model(label, baseline, pins)
        for row in panel["rows"]:
            for weight in CONFIG["weights"]:
                name = f"{label}-{row['index']:04d}-L{weight}.json"
                if name in manifest["episodes"]:
                    continue
                if (args.out / name).exists():
                    raise ValueError("uncommitted episode exists; inspect before recovery")
                result = episode(label, model, row, weight)
                atomic(args.out / name, result)
                manifest["episodes"][name] = sha(args.out / name)
                atomic(manifest_path, manifest)
                print(json.dumps({k: v for k, v in result.items()
                                  if k not in ("transcript", "witness", "final")}), flush=True)
        del model
    check_pins(pins)
    rows = [json.loads((args.out / p).read_text()) for p in manifest["episodes"]]
    if len(rows) != 96:
        raise ValueError("incomplete condition grid")
    summary = {}
    for label in CONFIG["models"]:
        for weight in CONFIG["weights"]:
            subset = [r for r in rows if r["model"] == label and r["cc_weight"] == weight]
            summary[f"{label}-L{weight}"] = {
                "attempts": len(subset), "solved": sum(r["solved"] for r in subset),
                "neural_samples": sum(r["neural_samples"] for r in subset),
                "costs_of_successes_only": [r["cost"] for r in subset if r["solved"]],
            }
    manifest.update(status="completed", summary=summary)
    atomic(manifest_path, manifest)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
