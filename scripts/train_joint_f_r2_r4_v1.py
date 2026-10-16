"""Matched, resumable native-MPS F-dominant R2/R4 trainer (protocol303)."""

import argparse
import fcntl
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import atomic, digest, sha
from mcts_tournament_signals_v1 import _groups
from mcts_training_supervision import verified_success_traces
from pretrained_raster_policy import policy_log_probabilities
from tournament_term_sampler_v1 import sample_coefficients
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.site_proposals import proposal_candidates

BANK = Path("artifacts/joint-f-bank-v2")
BANK_SHA = "6a61a0ec0bf47c59cdb601b633c962dc4b2d4df3b46ba9a2ca429fab49f19a12"
BANK_AUDIT = Path("artifacts/joint-f-bank-audit-v1/report.json")
BANK_AUDIT_SHA = "2a4a7a19f6d55da6d64533fba41a9594ce85ad89c808205b0984382ba297dc73"
PARENT = Path("artifacts/qrun-like-policy-curriculum-rung1-v1-envfix/step-1024.pt")
PARENT_SHA = "03506e0c572864dc207c0b4a330648dc86bef18b18af2420955a69697b9fe9e8"
PROTOCOL = Path("research/303-matched-f-r2-r4-smoke-and-training-v1.md")
CONFIG = {"seed": 20261007, "lr": 5e-5, "weight_decay": 0.01,
          "grad_clip": 1.0, "checkpoint_interval": 16,
          "sampled_F_terms": 16, "max_aux_encoder_ratio": 0.1,
          "max_rehearsal_encoder_ratio": 0.2, "device": "mps"}


class AuxiliaryCostHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(68, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, embedding, features):
        return self.layers(torch.cat((embedding, features), dim=-1))


def read(path):
    return json.loads(path.read_text())


def bank_order(manifest):
    order = []
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / entry["file"]
        if entry["file"] != f"root-{index:04d}.json" or sha(path) != entry["sha256"]:
            raise ValueError(f"bank root SHA differs: {index}")
        if read(path)["groups"]:
            order.append(index)
    if len(order) != 392:
        raise ValueError("F-usable bank coverage differs")
    return order


def selected_group(order, step):
    root_index = order[step % len(order)]
    row = read(BANK / f"root-{root_index:04d}.json")
    groups = row["groups"]
    group_index = (step // len(order)) % len(groups)
    return row, groups[group_index], group_index


def group_data(group):
    instance_dict = group["instance"]
    instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
    witnesses = [UnknotWitness.from_actions(instance.word, instance.strands,
                                             instance.spec, path["actions"])
                 for path in group["paths"]]
    traces = verified_success_traces(instance, witnesses)
    signals = _groups(traces, "F", seed=20261006, k=4,
                      near_margin=2, gap_scale=group["weight"], gap_cap=4)
    if len(signals) != 1 or len(signals[0]["coefficients"]) != group["signal_terms"]:
        raise ValueError("bank F signal changed")
    return instance, traces, signals[0]["coefficients"]


def f_rows(instance, coefficients, step):
    sampled = sample_coefficients(coefficients, terms=CONFIG["sampled_F_terms"],
                                  seed=CONFIG["seed"] + step)
    pools = {}
    for state, action in sampled:
        pools.setdefault(state, set()).add(action)
    rows = []
    for (state, action), coefficient in sorted(
        sampled.items(), key=lambda item: (item[0][0].used_moves,
                                           item[0][0].used_cc, item[0][1])
    ):
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | pools[state]))
        rows.append(((instance, state, actions), actions.index(action), coefficient))
    if len(rows) > CONFIG["sampled_F_terms"] or not rows:
        raise ValueError("invalid sampled F term count")
    return rows


def gradient_snapshot(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


def encoder_norm(grads, parameters, encoder):
    ids = {id(p) for p in encoder.parameters()}
    terms = [g.float().square().sum() for p, g in zip(parameters, grads, strict=True)
             if id(p) in ids and g is not None]
    return math.sqrt(float(torch.stack(terms).sum().cpu())) if terms else 0.0


def encoder_cosine(first, second, parameters, encoder):
    ids = {id(p) for p in encoder.parameters()}
    terms = [(a.float() * b.float()).sum()
             for p, a, b in zip(parameters, first, second, strict=True)
             if id(p) in ids and a is not None and b is not None]
    dot = float(torch.stack(terms).sum().cpu()) if terms else 0.0
    denominator = encoder_norm(first, parameters, encoder) * encoder_norm(second, parameters, encoder)
    return dot / denominator if denominator else 0.0


def root_embedding(net, instance):
    device = next(net.parameters()).device
    word = torch.tensor([instance.word], dtype=torch.long, device=device)
    length = torch.tensor([len(instance.word)], dtype=torch.long, device=device)
    strands = torch.tensor([instance.strands], dtype=torch.long, device=device)
    return net.encoder(word, length, strands)["knot"]


def auxiliary_loss(net, head, instance, trace):
    embedding = root_embedding(net, instance)
    features = embedding.new_tensor([[math.log1p(instance.cc_weight) / math.log1p(1000),
                                      instance.cap_crossing_changes / 4,
                                      instance.cap_moves / 224,
                                      len(instance.word) / 192]])
    target = embedding.new_tensor([[trace["cc"] / 2, trace["moves"] / 224]])
    return F.smooth_l1_loss(head(embedding, features), target)


def rehearsal_loss(net, objective, instance, reference, step):
    state, action = reference["rows"][step % len(reference["rows"])]
    actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
    logp = policy_log_probabilities(net, [(instance, state, actions)])
    imitation = -logp[0, actions.index(action)]
    embedding = root_embedding(net, instance)
    device = embedding.device
    anchor = net.encoder(torch.zeros((1, 1), dtype=torch.long, device=device),
                         torch.zeros((1,), dtype=torch.long, device=device),
                         torch.ones((1,), dtype=torch.long, device=device))["knot"]
    cosine = (embedding * anchor).sum(dim=-1).clamp(-1, 1)
    predicted_cc = objective.scale("cc") * objective.cc_distance(cosine)
    target_cc = predicted_cc.new_tensor([reference["cc"]])
    distance = F.smooth_l1_loss(predicted_cc, target_cc)
    return imitation + distance, float(imitation.detach()), float(distance.detach())


def train_step(net, objective, head, optimizer, group, step):
    instance, traces, coefficients = group_data(group)
    rows = f_rows(instance, coefficients, step)
    parameters = [*net.parameters(), *objective.parameters(), *head.parameters()]
    optimizer.zero_grad(set_to_none=True)
    f_loss = 0.0
    for request, action_index, coefficient in rows:
        logp = policy_log_probabilities(net, [request])
        loss = -coefficient * logp[0, action_index]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite F loss")
        loss.backward()
        f_loss += float(loss.detach())
    f_grads = gradient_snapshot(parameters)
    f_norm = encoder_norm(f_grads, parameters, net.encoder)
    if not math.isfinite(f_norm) or f_norm <= 1e-12:
        raise ValueError("zero/nonfinite F gradient: skip, do not auxiliary-train")
    best = min(traces, key=lambda t: (t["cost"], tuple(a for _, a in t["rows"])))
    optimizer.zero_grad(set_to_none=True)
    aux = auxiliary_loss(net, head, instance, best)
    aux.backward()
    aux_grads = gradient_snapshot(parameters)
    aux_norm = encoder_norm(aux_grads, parameters, net.encoder)
    aux_cos = encoder_cosine(f_grads, aux_grads, parameters, net.encoder)
    reference = traces[0]
    optimizer.zero_grad(set_to_none=True)
    rehearsal, imitation, distance = rehearsal_loss(net, objective, instance, reference, step)
    rehearsal.backward()
    rehearsal_grads = gradient_snapshot(parameters)
    rehearsal_norm = encoder_norm(rehearsal_grads, parameters, net.encoder)
    rehearsal_cos = encoder_cosine(f_grads, rehearsal_grads, parameters, net.encoder)
    aux_scale = min(1.0, CONFIG["max_aux_encoder_ratio"] * f_norm / aux_norm) if aux_norm else 0.0
    rehearsal_scale = (min(1.0, CONFIG["max_rehearsal_encoder_ratio"] * f_norm / rehearsal_norm)
                       if rehearsal_norm else 0.0)
    optimizer.zero_grad(set_to_none=True)
    for p, f_grad, aux_grad, rehearsal_grad in zip(
        parameters, f_grads, aux_grads, rehearsal_grads, strict=True
    ):
        pieces = [g * scale for g, scale in ((f_grad, 1.0),
                                             (aux_grad, aux_scale),
                                             (rehearsal_grad, rehearsal_scale))
                  if g is not None and scale]
        if pieces:
            p.grad = sum(pieces[1:], pieces[0])
    combined_norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["grad_clip"],
                                                   error_if_nonfinite=True)
    optimizer.step()
    return {"F_loss": f_loss, "aux_loss": float(aux.detach()),
            "imitation_loss": imitation, "distance_loss": distance,
            "F_encoder_grad_norm": f_norm, "aux_encoder_grad_norm": aux_norm,
            "rehearsal_encoder_grad_norm": rehearsal_norm,
            "aux_F_cosine": aux_cos, "rehearsal_F_cosine": rehearsal_cos,
            "aux_scale": aux_scale, "rehearsal_scale": rehearsal_scale,
            "combined_grad_norm": float(combined_norm),
            "sampled_F_terms": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("R2", "R4"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() != args.resume or args.stop_after < 0:
        raise ValueError("new output without --resume; existing output with --resume")
    if (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("require native MPS without CPU fallback")
    torch.set_num_threads(1)
    inputs = {str(path): sha(path) for path in
              (BANK / "manifest.json", BANK_AUDIT, PARENT, PROTOCOL, Path(__file__),
               Path("scripts/mcts_tournament_signals_v1.py"),
               Path("scripts/tournament_term_sampler_v1.py"),
               Path("scripts/pretrained_raster_policy.py"),
               Path("scripts/mcts_training_supervision.py"))}
    if (inputs[str(BANK / "manifest.json")] != BANK_SHA
            or inputs[str(BANK_AUDIT)] != BANK_AUDIT_SHA
            or inputs[str(PARENT)] != PARENT_SHA):
        raise ValueError("frozen bank/audit/parent differs")
    bank_manifest = read(BANK / "manifest.json")
    order = bank_order(bank_manifest)
    contract = {"schema": "rf-knots-matched-F-R2-R4-v1", "arm": args.arm,
                "graph_rounds": 2 if args.arm == "R2" else 4,
                "config": CONFIG, "inputs": inputs, "eligible_roots": order,
                "role": "native-MPS F-dominant optimizer training"}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        manifest = read(manifest_path) if args.resume else {
            "contract": contract, "contract_sha256": contract_sha,
            "status": "preparing", "optimizer_steps": 0, "checkpoints": [],
            "active_seconds": 0.0, "cpu_seconds": 0.0,
            "stage_draws": [0, 0, 0, 0]}
        if (manifest["contract"] != json.loads(json.dumps(contract))
                or manifest["contract_sha256"] != contract_sha
                or manifest["optimizer_steps"] != (manifest["checkpoints"][-1]["step"]
                                                if manifest["checkpoints"] else 0)):
            raise ValueError("resume contract/checkpoint state differs")
        for entry in manifest["checkpoints"]:
            if sha(args.out / entry["file"]) != entry["sha256"]:
                raise ValueError("committed checkpoint SHA differs")
        net, objective = load_actor()
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)
        if parent["step"] != 1024:
            raise ValueError("parent checkpoint step differs")
        net.load_state_dict(parent["model"], strict=True)
        objective.load_state_dict(parent["objective"], strict=True)
        net.encoder.graph_rounds = contract["graph_rounds"]
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        rng = random.Random(CONFIG["seed"])
        head = AuxiliaryCostHead()
        net.to("mps")
        objective.to("mps")
        head.to("mps")
        parameters = [*net.parameters(), *objective.parameters(), *head.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        if manifest["optimizer_steps"]:
            entry = manifest["checkpoints"][-1]
            saved = torch.load(args.out / entry["file"], map_location="cpu",
                               weights_only=False)
            if saved["contract_sha256"] != contract_sha or saved["step"] != entry["step"]:
                raise ValueError("resume checkpoint identity differs")
            net.load_state_dict(saved["model"], strict=True)
            objective.load_state_dict(saved["objective"], strict=True)
            head.load_state_dict(saved["auxiliary_head"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            rng.setstate(saved["rng"])
            torch.set_rng_state(saved["torch_rng"])
            torch.mps.set_rng_state(saved["mps_rng"])
            if saved["stage_draws"] != manifest["stage_draws"]:
                raise ValueError("resume stage draws differ")
        start = manifest["optimizer_steps"]
        end = start + (args.stop_after or CONFIG["checkpoint_interval"])
        manifest.update(status="training", pid=os.getpid())
        atomic(manifest_path, manifest)
        pending_wall = pending_cpu = 0.0
        for step in range(start, end):
            if any(sha(Path(path)) != expected for path, expected in inputs.items()):
                raise ValueError("pinned trainer input changed")
            started_wall, started_cpu = time.monotonic(), time.process_time()
            net.train()
            objective.train()
            head.train()
            root, group, group_index = selected_group(order, step)
            metrics = train_step(net, objective, head, optimizer, group, step)
            torch.mps.synchronize()
            pending_wall += time.monotonic() - started_wall
            pending_cpu += time.process_time() - started_cpu
            manifest["stage_draws"][root["stage"]] += 1
            if ((step + 1) % CONFIG["checkpoint_interval"] == 0 or step + 1 == end):
                saved = {"contract_sha256": contract_sha, "step": step + 1,
                         "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                         "objective": {k: v.detach().cpu() for k, v in objective.state_dict().items()},
                         "auxiliary_head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                         "optimizer": optimizer.state_dict(), "rng": rng.getstate(),
                         "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
                         "stage_draws": manifest["stage_draws"].copy(),
                         "metrics": metrics, "root_index": root["root_index"],
                         "group_index": group_index}
                path = args.out / f"step-{step + 1:06d}.pt"
                partial = path.with_suffix(".partial")
                torch.save(saved, partial)
                partial.replace(path)
                manifest["checkpoints"].append({"file": path.name,
                                                "sha256": sha(path), "step": step + 1})
                manifest["optimizer_steps"] = step + 1
                manifest["active_seconds"] += pending_wall
                manifest["cpu_seconds"] += pending_cpu
                pending_wall = pending_cpu = 0.0
                manifest.update(metrics=metrics, pid=os.getpid())
                atomic(manifest_path, manifest)
                print(json.dumps({"arm": args.arm, "optimizer_steps": step + 1,
                                  "root_index": root["root_index"],
                                  "F_encoder_grad_norm": metrics["F_encoder_grad_norm"],
                                  "aux_scale": metrics["aux_scale"],
                                  "rehearsal_scale": metrics["rehearsal_scale"]}), flush=True)
        manifest.update(status="checkpointed", pid=None)
        atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
