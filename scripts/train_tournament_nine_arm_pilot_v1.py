"""Resumable, matched nine-arm TRAIN pilot using audited S10 rollout paths.

D uses a declared 16-slot stochastic suffix signal, not an exact full-suffix
gradient. This is exploratory rollout-tournament screening, not AlphaZero MCTS.
"""

import argparse
import copy
import fcntl
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, load_actor
from build_mcts_training_supervision import atomic, sha
from graph_metric_sampling import load_graph_sampler
from long_general_training_data import CONFIG as BASE, load_long_source, paired_draw
from mcts_joint_training import load_metric_samplers, load_policy_targets
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from pretrained_raster_policy import policy_backward, reference_preferences
from raster_log_training import CORPUS, GRAPH, SUPERVISION, digest, relative_due
from run_strand_graph_pilot import metric_loss
from tournament_s10_dataset_v1 import DATA, MANIFEST_SHA, load_groups
from tournament_term_sampler_v1 import sample_coefficients
from trajectory_log_preference import relative_backward

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import witness_priority_weight
from rf_knots.site_proposals import proposal_candidates


SCHEMA = "rf-knots-tournament-nine-arm-pilot-v1"
CONFIG = {
    "arms": list("ABCDEFGHI"),
    "steps_per_arm": 144,
    "checkpoint_interval": 16,
    "seed": 20261003,
    "signal_slots_per_step": 16,
    "k": 4,
    "near_margin": 2,
    "gap_cap": 4,
    "gap_scale": "lambda",
    "lr": BASE["lr"],
    "weight_decay": BASE["weight_decay"],
    "grad_clip": BASE["grad_clip"],
    "positive_weight": BASE["positive_weight"],
    "old_relative_weight": BASE["relative_weight"] * BASE["relative_every"],
    "tournament_weight": 0.002,
    "h_late_multiplier": 2.0,
    "g_cc_extra_weight": 0.5,
    "device": "mps",
    "optimizer_reset_for_all_arms": True,
    "d_signal": "stochastic 16-slot coefficient estimator, not exact full-group loss",
}
LONG_DATA = Path("artifacts/long-general-cc-supply-v1")
DEV = Path("artifacts/mcts-tournament-dev-panel-v1/panel.json")


def read(path):
    return json.loads(path.read_text())


def term_requests(instance, coefficients, slots):
    pools = {}
    for state, action in coefficients:
        pools.setdefault(state, set()).add(action)
    rows = []
    for (state, action), coefficient in sorted(
        coefficients.items(),
        key=lambda item: (item[0][0].used_moves, item[0][0].used_cc, item[0][1]),
    ):
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | pools[state]))
        rows.append(((instance, state, actions), actions.index(action), coefficient))
    if not rows:
        raise ValueError("no tournament terms")
    if len(rows) > slots:
        raise ValueError("signal exceeded frozen term slots")
    dummy = (rows[0][0], rows[0][1], 0.0)
    rows.extend([dummy] * (slots - len(rows)))
    return rows


def signal_bank(roots, groups):
    bank = {}
    for (base, weight), witnesses in sorted(groups.items()):
        root = roots[base * 2]
        instance = M.Instance(
            **{**root["instance"], "word": tuple(root["instance"]["word"]), "cc_weight": weight}
        )
        traces = verified_success_traces(instance, witnesses)
        if len({t["cost"] for t in traces}) < 2:
            continue
        bank[(base, weight)] = (instance, traces)
    if len(bank) != 132:
        raise ValueError("audited cost-different group count changed")
    return bank


def signal_rows(bank, key, arm, step):
    instance, traces = bank[key]
    if arm == "A":
        # Control performs equal extra neural work with a zero-gradient row.
        arm_for_rows = "B"
    else:
        arm_for_rows = arm
    signals = _groups(
        traces, arm_for_rows, seed=CONFIG["seed"] + step + key[0] * 2 + (key[1] == 1000),
        k=CONFIG["k"], near_margin=CONFIG["near_margin"],
        gap_scale=key[1], gap_cap=CONFIG["gap_cap"],
    )
    if not signals:
        # F can have no strictly worse path after admitting near-best positives.
        signals = _groups(traces, "B", seed=0, k=4, near_margin=0,
                          gap_scale=key[1], gap_cap=4)
        if not signals:
            raise ValueError("audited cost group has no fallback signal")
        signals[0]["coefficients"] = {k: 0.0 for k in signals[0]["coefficients"]}
    if arm == "G" and len(signals) == 2:
        allocations = (8, 8)
    else:
        allocations = (16,)
        signals = signals[:1]
    rows = []
    for index, (signal, slots) in enumerate(zip(signals, allocations, strict=True)):
        coefficients = signal["coefficients"]
        if arm == "A":
            coefficients = {k: 0.0 for k in coefficients}
        elif len(coefficients) > slots:
            coefficients = sample_coefficients(
                coefficients, terms=slots, seed=CONFIG["seed"] * 100000 + step * 13 + key[0] * 2 + index
            )
        rows.append(term_requests(instance, coefficients, slots))
    if sum(map(len, rows)) != CONFIG["signal_slots_per_step"]:
        raise ValueError("training signal neural-work slots differ")
    return rows


def optimize(net, objective, optimizer, draw, old_relative, tournament):
    net.train()
    optimizer.zero_grad(set_to_none=True)
    weights = [
        witness_priority_weight("cc", r.spec.distance_upper, r.witness_semantic_length)
        for r in draw.metrics
    ]
    metric_value = 0.0
    for row, weight in zip(draw.metrics, weights, strict=True):
        loss, _ = metric_loss(net, objective, [row], "mps")
        scaled = loss * (weight / sum(weights))
        if not torch.isfinite(scaled):
            raise ValueError("nonfinite metric loss")
        scaled.backward()
        metric_value += float(scaled.detach())
    policy = policy_backward(net, draw.positives, None, microbatch=1,
                             positive_weight=CONFIG["positive_weight"])
    if old_relative is not None:
        relative_backward(net, old_relative, microbatch=1,
                          weight=CONFIG["old_relative_weight"])
    tourney_terms = 0
    for relative, weight in tournament:
        result = relative_backward(net, relative, microbatch=1, weight=weight)
        tourney_terms += result["relative_states"]
    norm = torch.nn.utils.clip_grad_norm_(
        [*net.parameters(), *objective.parameters()], CONFIG["grad_clip"],
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {"metric_loss": metric_value, "positive_loss": policy["positive_loss"],
            "gradient_norm_before_clip": float(norm), "tournament_terms": tourney_terms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx-mcts-bench is missing")
    sys.path.insert(0, str(pgx_src))
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("require native MPS without CPU fallback")
    torch.set_num_threads(1)
    pins = {str(p): sha(p) for p in (
        ACTOR, DATA / "manifest.json", DEV, LONG_DATA / "corpus.json",
        LONG_DATA / "consumer-audit.json", CORPUS, GRAPH / "manifest.json",
        Path(__file__), Path("scripts/tournament_s10_dataset_v1.py"),
        Path("scripts/tournament_term_sampler_v1.py"),
        Path("scripts/mcts_tournament_signals_v1.py"),
    )}
    if pins[str(ACTOR)] != ACTOR_SHA or pins[str(DATA / "manifest.json")] != MANIFEST_SHA:
        raise ValueError("initial actor or teacher manifest changed")
    contract = {"schema": SCHEMA, "config": CONFIG, "inputs": pins}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = read(manifest_path) if args.resume else {
            "schema": SCHEMA, "contract": contract, "contract_sha256": contract_sha,
            "arm_index": 0, "completed_steps": 0, "status": "preparing", "checkpoints": {},
            "seconds_committed": 0.0,
        }
        if manifest["contract"] != contract or manifest["contract_sha256"] != contract_sha:
            raise ValueError("pilot resume contract differs")
        if manifest["arm_index"] == len(CONFIG["arms"]):
            print("nine arms already completed", flush=True)
            return
        atomic(manifest_path, manifest)
        roots, groups, _ = load_groups()
        bank = signal_bank(roots, groups)
        keys = sorted(bank)
        schedule = keys[:]
        random.Random(CONFIG["seed"]).shuffle(schedule)
        schedule += schedule[: CONFIG["steps_per_arm"] - len(schedule)]
        if len(schedule) != CONFIG["steps_per_arm"]:
            raise ValueError("balanced schedule length differs")
        long_source, _, _ = load_long_source(LONG_DATA)
        original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
        graph = load_graph_sampler(GRAPH)
        positives, raw_relative = load_policy_targets(SUPERVISION)
        net0, objective0 = load_actor()
        net0.to("mps")
        objective0.to("mps")
        old_references = reference_preferences(net0, raw_relative, microbatch=1)
        for arm_index in range(manifest["arm_index"], len(CONFIG["arms"])):
            arm = CONFIG["arms"][arm_index]
            net, objective = copy.deepcopy(net0), copy.deepcopy(objective0)
            optimizer = torch.optim.AdamW(
                [*net.parameters(), *objective.parameters()],
                lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
            )
            rng = random.Random(BASE["seed"])
            long_rng = random.Random(BASE["long_seed"])
            torch.manual_seed(BASE["seed"])
            torch.mps.manual_seed(BASE["seed"])
            start = manifest["completed_steps"] if arm_index == manifest["arm_index"] else 0
            if start:
                entry = manifest["checkpoints"][arm]
                path = args.out / entry["file"]
                if sha(path) != entry["sha256"] or entry["step"] != start:
                    raise ValueError("resumed checkpoint differs")
                state = torch.load(path, map_location="cpu", weights_only=False)
                if state["contract_sha256"] != contract_sha or state["arm"] != arm:
                    raise ValueError("resumed checkpoint contract differs")
                net.load_state_dict(state["model"])
                objective.load_state_dict(state["objective"])
                optimizer.load_state_dict(state["optimizer"])
                rng.setstate(state["rng"])
                long_rng.setstate(state["long_rng"])
                torch.set_rng_state(state["torch_rng"])
                torch.mps.set_rng_state(state["mps_rng"])
            manifest.update(status="training", phase="optimizer", arm_index=arm_index,
                            current_arm=arm, completed_steps=start, pid=os.getpid())
            atomic(manifest_path, manifest)
            arm_started = time.monotonic()
            for step in range(start, CONFIG["steps_per_arm"]):
                draw = paired_draw("long", rng, long_rng, original, suffix, graph,
                                   long_source, positives, old_references)
                old_relative = draw.relative if relative_due(step) else None
                rows = signal_rows(bank, schedule[step], arm, step)
                references = [reference_preferences(net0, [group], microbatch=1)[0]
                              for group in rows]
                if arm == "G" and len(references) == 2:
                    weights = [CONFIG["tournament_weight"],
                               CONFIG["tournament_weight"] * CONFIG["g_cc_extra_weight"]]
                else:
                    multiplier = CONFIG["h_late_multiplier"] if arm == "H" and step >= 72 else 1.0
                    weights = [CONFIG["tournament_weight"] * multiplier]
                if arm == "A":
                    weights = [0.0]
                losses = optimize(net, objective, optimizer, draw, old_relative,
                                  list(zip(references, weights, strict=True)))
                if (step + 1) % CONFIG["checkpoint_interval"] == 0:
                    torch.mps.synchronize()
                    payload = {
                        "contract_sha256": contract_sha, "arm": arm, "step": step + 1,
                        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        "objective": {k: v.detach().cpu() for k, v in objective.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "rng": rng.getstate(),
                        "long_rng": long_rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "mps_rng": torch.mps.get_rng_state(),
                    }
                    path = args.out / f"{arm}-step-{step + 1:04d}.pt"
                    temporary = path.with_suffix(".partial")
                    torch.save(payload, temporary)
                    temporary.replace(path)
                    manifest["checkpoints"][arm] = {"file": path.name, "sha256": sha(path), "step": step + 1}
                    manifest.update(completed_steps=step + 1, seconds_committed=manifest["seconds_committed"] + time.monotonic() - arm_started, **losses)
                    arm_started = time.monotonic()
                    atomic(manifest_path, manifest)
                    print(json.dumps({"arm": arm, "step": step + 1, **losses}, sort_keys=True), flush=True)
                    if args.stop_after and CONFIG["steps_per_arm"] * arm_index + step + 1 >= args.stop_after:
                        return
            manifest.update(arm_index=arm_index + 1, completed_steps=0,
                            status="arm_completed", current_arm=None)
            atomic(manifest_path, manifest)
        manifest.update(status="completed", phase="completed", pid=None)
        atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
