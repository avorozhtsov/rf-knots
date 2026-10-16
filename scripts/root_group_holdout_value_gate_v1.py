"""Two preregistered root-group-disjoint value-head training/evaluation arms."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, ROOTS, load_actor
from budgeted_cost_value_head_v1 import BudgetedCostValueHead, value_loss
from build_mcts_training_supervision import atomic, sha
from raster_log_training import digest
from train_root_budget16_value_head_v1 import encode_roots


LABELS = Path("artifacts/root-budget16-multioutput-labels-v1/labels.json")
PROTOCOL = Path("research/268-root-group-holdout-value-gate-v1.md")
TESTS = Path("tests/test_root_group_holdout_value_gate_v1.py")
CONFIG = {
    "split_salts": [20261005, 20261006],
    "holdout_fraction": 0.25,
    "seed": 20261004,
    "steps": 512,
    "checkpoint_interval": 64,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "device": "mps",
    "local_simulations": 16,
    "reference_simulations": 1024,
}


def read(path):
    return json.loads(path.read_text())


def stratum(instance):
    cap = instance["cap_crossing_changes"]
    return ("1" if cap == 1 else "2" if cap == 2 else "3+",
            "<100" if len(instance["word"]) < 100 else ">=100")


def partition(roots, salt):
    if len(roots) != 256 or salt not in CONFIG["split_salts"]:
        raise ValueError("unexpected roots or split salt")
    groups = defaultdict(list)
    for index in range(0, 256, 2):
        left, right = roots[index:index + 2]
        a, b = left["instance"], right["instance"]
        if (left["base_root_index"] != right["base_root_index"]
                or (a["word"], a["strands"], a["cap_crossing_changes"], a["cap_moves"])
                != (b["word"], b["strands"], b["cap_crossing_changes"], b["cap_moves"])
                or {a["cc_weight"], b["cc_weight"]} != {10, 1000}):
            raise ValueError("paired root identity differs")
        groups[stratum(a)].append((left["base_root_index"], index))
    if sorted(len(v) for v in groups.values()) != [10, 11, 17, 20, 31, 39]:
        raise ValueError("fixed stratum roster changed")
    holdout = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda pair: (
            hashlib.sha256(f"{salt}:{pair[0]}".encode()).hexdigest(), pair[0]))
        holdout.extend(index for _, index in ordered[:round(len(ordered) * 0.25)])
    if len(holdout) != 32 or len(set(holdout)) != 32:
        raise ValueError("holdout size differs")
    return sorted(holdout)


def records_for(roots, labels, heldout):
    seen = Counter((row["root_index"], row["seed"]) for row in labels)
    if (len(labels) != 2560 or any(seen[(index, seed)] != 1
                                    for index in range(256)
                                    for seed in range(20270000, 20270010))):
        raise ValueError("label root/seed coverage differs")
    holdout_roots = {index + offset for index in heldout for offset in (0, 1)}
    fit = [row for row in labels if row["root_index"] not in holdout_roots]
    test = [row for row in labels if row["root_index"] in holdout_roots]
    if len(fit) != 1920 or len(test) != 640:
        raise ValueError("fit/holdout attempt count differs")
    for row in labels:
        instance = roots[row["root_index"]]["instance"]
        if (row["lambda"] != instance["cc_weight"]
                or row["cc_left"] != instance["cap_crossing_changes"]
                or row["moves_left"] != instance["cap_moves"]
                or row["local_simulations"] != 16):
            raise ValueError("label-root context differs")
    return fit, test


def metrics(rows, prediction, baseline):
    def scores(field):
        p = min(1 - 1e-8, max(1e-8, field))
        y = float(rows["solved"])
        return (p - y) ** 2, -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return scores(prediction), scores(baseline)


def evaluate(net, head, roots, fit, test):
    """No labels from test are used in the baselines or forward computation."""
    by_key = defaultdict(list)
    by_lambda = defaultdict(list)
    for row in fit:
        instance = roots[row["root_index"]]["instance"]
        key = (row["lambda"], *stratum(instance))
        by_key[key].append(row)
        by_lambda[row["lambda"]].append(row)
    constants = {}
    for key, group in by_key.items():
        successful = [row for row in group if row["solved"]]
        fallback = [row for row in by_lambda[key[0]] if row["solved"]]
        cost = successful or fallback
        constants[key] = {
            "p": (sum(row["solved"] for row in group) + 1) / (len(group) + 2),
            "cc": sum(row["cc_remaining"] for row in cost) / len(cost),
            "moves": sum(row["moves_remaining"] for row in cost) / len(cost),
            "fit_attempts": len(group),
        }
    net.eval()
    head = head.cpu().eval()
    cache = {}
    cold_cpu = warm_cpu = 0.0
    with torch.no_grad():
        for pair_index in sorted({row["root_index"] // 2 for row in test}):
            instance = roots[2 * pair_index]["instance"]
            word = instance["word"]
            begin = time.process_time()
            encoded = net.encoder(torch.tensor([word], dtype=torch.long),
                                  torch.tensor([len(word)]),
                                  torch.tensor([instance["strands"]]))
            for offset in (0, 1):
                root_index = 2 * pair_index + offset
                item = roots[root_index]["instance"]
                output = head(encoded,
                              cc_left=[item["cap_crossing_changes"]],
                              moves_left=[item["cap_moves"]],
                              cc_weight=[item["cc_weight"]],
                              local_simulations=[16])
                cache[root_index] = {name: float(output[name][0]) for name in
                                     ("p_solve", "cc_remaining", "moves_remaining")}
            cold_cpu += time.process_time() - begin
        for pair_index in sorted({row["root_index"] // 2 for row in test}):
            instance = roots[2 * pair_index]["instance"]
            word = instance["word"]
            begin = time.process_time()
            encoded = net.encoder(torch.tensor([word], dtype=torch.long),
                                  torch.tensor([len(word)]),
                                  torch.tensor([instance["strands"]]))
            for offset in (0, 1):
                item = roots[2 * pair_index + offset]["instance"]
                head(encoded, cc_left=[item["cap_crossing_changes"]],
                     moves_left=[item["cap_moves"]], cc_weight=[item["cc_weight"]],
                     local_simulations=[16])
            warm_cpu += time.process_time() - begin
    grouped = defaultdict(list)
    for row in test:
        instance = roots[row["root_index"]]["instance"]
        key = (row["lambda"], *stratum(instance))
        p = cache[row["root_index"]]
        fixed = constants[key]
        score, control = metrics(row, p["p_solve"], fixed["p"])
        item = {"root_index": row["root_index"], "solved": row["solved"],
                "p": p["p_solve"], "baseline_p": fixed["p"],
                "brier": score[0], "logloss": score[1],
                "baseline_brier": control[0], "baseline_logloss": control[1]}
        if row["solved"]:
            item.update(cc_mae=abs(p["cc_remaining"] - row["cc_remaining"]),
                        cc_cap_mae=abs(row["cc_left"] - row["cc_remaining"]),
                        cc_constant_mae=abs(fixed["cc"] - row["cc_remaining"]),
                        moves_mae=abs(p["moves_remaining"] - row["moves_remaining"]),
                        moves_constant_mae=abs(fixed["moves"] - row["moves_remaining"]))
        grouped[(row["lambda"], "all")].append(item)
        grouped[(row["lambda"], *stratum(instance))].append(item)
    report = {}
    for key, group in grouped.items():
        metric_names = ("brier", "logloss", "baseline_brier", "baseline_logloss")
        result = {"attempts": len(group), "roots": len({x["root_index"] for x in group}),
                  "solved": sum(x["solved"] for x in group),
                  "mean_prediction": sum(x["p"] for x in group) / len(group),
                  "observed_fraction": sum(x["solved"] for x in group) / len(group)}
        result.update({name: sum(x[name] for x in group) / len(group)
                       for name in metric_names})
        roots_group = defaultdict(list)
        for x in group:
            roots_group[x["root_index"]].append(x)
        for name in metric_names:
            result[f"root_weighted_{name}"] = sum(
                sum(x[name] for x in rr) / len(rr) for rr in roots_group.values()
            ) / len(roots_group)
        success = [x for x in group if x["solved"]]
        if success:
            result.update({name: sum(x[name] for x in success) / len(success)
                           for name in ("cc_mae", "cc_cap_mae", "cc_constant_mae",
                                        "moves_mae", "moves_constant_mae")})
        report["-".join(map(str, key))] = result
    return {"slices": report,
            "fit_only_conditional_constants": {"-".join(map(str, key)): value
                                               for key, value in constants.items()},
            "cpu_inference": {"heldout_paired_roots": len(cache) // 2,
                              "head_queries_per_pass": len(cache),
                              "cold_cpu_seconds": cold_cpu,
                              "warm_cpu_seconds": warm_cpu}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--split-salt", type=int, choices=CONFIG["split_salts"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output requires no --resume; existing output requires --resume")
    pgx_src = args.pgx_root.resolve() / "src"
    if not (pgx_src / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical dependency missing")
    sys.path.insert(0, str(pgx_src))
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("native MPS required without fallback")
    if sha(ACTOR) != ACTOR_SHA:
        raise ValueError("actor SHA changed")
    labels = read(LABELS)
    if labels["schema"] != "rf-knots-root-budget16-multioutput-labels-v1" or sha(LABELS) != "57e7fa1d021ddc54aaeb220d68e415e6ab91d6765749b8087bd1eda8ebd77201":
        raise ValueError("label source changed")
    if any(sha(Path(path)) != digest for path, digest in labels["inputs"].items()):
        raise ValueError("label input pin changed")
    roots = read(ROOTS)
    heldout = partition(roots, args.split_salt)
    fit, test = records_for(roots, labels["labels"], heldout)
    inputs = {str(path): sha(path) for path in (
        ACTOR, LABELS, ROOTS, PROTOCOL, TESTS, Path(__file__),
        Path("scripts/budgeted_cost_value_head_v1.py"),
        Path("scripts/train_root_budget16_value_head_v1.py"),
        Path("scripts/bootstrap_cost_value_teacher_v1.py"),
    )}
    contract = {"schema": "rf-knots-root-group-holdout-value-gate-v1",
                "config": {**CONFIG, "split_salt": args.split_salt}, "inputs": inputs,
                "actor_sha256": ACTOR_SHA, "holdout_base_indices": heldout}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "schema": contract["schema"], "contract": contract,
            "contract_sha256": contract_sha, "status": "preparing",
            "optimizer_steps": 0, "checkpoints": []}
        if (state["contract"] != contract or state["contract_sha256"] != contract_sha
                or state["optimizer_steps"] != len(state["checkpoints"]) * CONFIG["checkpoint_interval"]):
            raise ValueError("resume contract or checkpoint coverage differs")
        for index, entry in enumerate(state["checkpoints"]):
            if (entry["step"] != (index + 1) * CONFIG["checkpoint_interval"]
                    or sha(args.out / entry["file"]) != entry["sha256"]):
                raise ValueError("prior checkpoint SHA differs")
        if state["status"] == "completed":
            print("already completed", flush=True)
            return
        atomic(manifest_path, state)
        torch.set_num_threads(1)
        net, _ = load_actor()
        net.to("mps").eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        feature = encode_roots(net, roots)
        examples = fit
        encoded = {name: torch.stack([feature[(row["root_index"], name)] for row in examples])
                   for name in ("knot", "layout", "isotopy")}
        cc_left = torch.tensor([r["cc_left"] for r in examples], dtype=torch.float32, device="mps")
        moves_left = torch.tensor([r["moves_left"] for r in examples], dtype=torch.float32, device="mps")
        lam = torch.tensor([r["lambda"] for r in examples], dtype=torch.float32, device="mps")
        solved = torch.tensor([r["solved"] for r in examples], dtype=torch.float32, device="mps")
        cc = torch.tensor([float("nan") if r["cc_remaining"] is None else r["cc_remaining"]
                           for r in examples], dtype=torch.float32, device="mps")
        moves = torch.tensor([float("nan") if r["moves_remaining"] is None else r["moves_remaining"]
                              for r in examples], dtype=torch.float32, device="mps")
        head = BudgetedCostValueHead(reference_simulations=CONFIG["reference_simulations"]).to("mps")
        optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        rng = torch.Generator(device="cpu")
        rng.manual_seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        start = state["optimizer_steps"]
        if start:
            entry = state["checkpoints"][-1]
            payload = torch.load(args.out / entry["file"], map_location="cpu", weights_only=False)
            if payload["contract_sha256"] != contract_sha or payload["step"] != start:
                raise ValueError("resume checkpoint identity differs")
            head.load_state_dict(payload["head"])
            optimizer.load_state_dict(payload["optimizer"])
            rng.set_state(payload["draw_rng"])
            torch.set_rng_state(payload["torch_rng"])
            torch.mps.set_rng_state(payload["mps_rng"])
        state.update(status="training", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        for step in range(start, end):
            indices = torch.randint(len(examples), (CONFIG["batch_size"],), generator=rng).to("mps")
            output = head({name: vector[indices] for name, vector in encoded.items()},
                          cc_left=cc_left[indices], moves_left=moves_left[indices],
                          cc_weight=lam[indices],
                          local_simulations=torch.full((len(indices),), 16.0, device="mps"))
            losses = value_loss(output, solved=solved[indices], cc=cc[indices],
                                moves=moves[indices], cc_left=cc_left[indices],
                                moves_left=moves_left[indices])
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite value loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["grad_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % CONFIG["checkpoint_interval"] == 0:
                torch.mps.synchronize()
                payload = {"contract_sha256": contract_sha, "step": step + 1,
                           "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                           "optimizer": optimizer.state_dict(), "draw_rng": rng.get_state(),
                           "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state()}
                checkpoint_path = args.out / f"step-{step + 1:04d}.pt"
                temporary = checkpoint_path.with_suffix(".partial")
                torch.save(payload, temporary)
                temporary.replace(checkpoint_path)
                state["checkpoints"].append({"file": checkpoint_path.name,
                                             "sha256": sha(checkpoint_path), "step": step + 1})
                state.update(optimizer_steps=step + 1, value_loss=float(losses["total"].detach()),
                             solve_loss=float(losses["solve"].detach()),
                             cost_loss=float(losses["cost"].detach()),
                             gradient_norm_before_clip=float(norm))
                atomic(manifest_path, state)
                print(json.dumps({"salt": args.split_salt, "step": step + 1,
                                  "loss": state["value_loss"]}), flush=True)
        if state["optimizer_steps"] == CONFIG["steps"]:
            net.cpu().eval()
            report = evaluate(net, head, roots, fit, test)
            report.update(schema="rf-knots-root-group-holdout-value-gate-report-v1",
                          split_salt=args.split_salt, contract_sha256=contract_sha,
                          checkpoint_sha256=state["checkpoints"][-1]["sha256"],
                          heldout_base_indices=heldout)
            atomic(args.out / "report.json", report)
            state.update(status="completed", pid=None,
                         report_sha256=sha(args.out / "report.json"))
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
