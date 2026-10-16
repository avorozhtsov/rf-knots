"""Frozen-encoder, three-output b=16 root-value bootstrap for protocol264."""

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import ACTOR_SHA, ROOTS, load_actor
from budgeted_cost_value_head_v1 import BudgetedCostValueHead, value_loss
from build_mcts_training_supervision import atomic, sha
from raster_log_training import digest


LABELS = Path("artifacts/root-budget16-multioutput-labels-v1/labels.json")
LABELS_SHA = "57e7fa1d021ddc54aaeb220d68e415e6ab91d6765749b8087bd1eda8ebd77201"
PROTOCOL = Path("research/264-root-budget16-multioutput-value-bootstrap-v1.md")
CONFIG = {
    "seed": 20261004,
    "steps": 512,
    "checkpoint_interval": 64,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "device": "mps",
    "frozen_encoder": True,
    "local_simulations": 16,
    "reference_simulations": 1024,
}


def read(path):
    return json.loads(path.read_text())


def encode_roots(net, roots):
    """One frozen raster encoding per paired representation, reused by lambda."""
    features = {}
    net.eval()
    with torch.no_grad():
        for index in range(0, len(roots), 2):
            left, right = roots[index : index + 2]
            if (left["instance"]["word"], left["instance"]["strands"]) != (
                right["instance"]["word"], right["instance"]["strands"]
            ):
                raise ValueError("lambda pair has different encoder input")
            word = left["instance"]["word"]
            encoded = net.encoder(
                torch.tensor([word], dtype=torch.long, device="mps"),
                torch.tensor([len(word)], device="mps"),
                torch.tensor([left["instance"]["strands"]], device="mps"),
            )
            if any(not torch.isfinite(v).all() for v in encoded.values()):
                raise ValueError("nonfinite frozen encoder output")
            for name in ("knot", "layout", "isotopy"):
                features[(index, name)] = encoded[name][0].detach()
                features[(index + 1, name)] = encoded[name][0].detach()
    return features


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
        raise ValueError("canonical dependency missing")
    sys.path.insert(0, str(pgx_src))
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("native MPS required without CPU fallback")
    if sha(LABELS) != LABELS_SHA:
        raise ValueError("value labels changed")
    labels = read(LABELS)
    if labels["schema"] != "rf-knots-root-budget16-multioutput-labels-v1" or len(labels["labels"]) != 2560:
        raise ValueError("fixed teacher labels missing")
    if any(sha(Path(p)) != h for p, h in labels["inputs"].items()):
        raise ValueError("label provenance changed")
    roots = read(ROOTS)
    if len(roots) != 256:
        raise ValueError("TRAIN root roster changed")
    inputs = {str(p): sha(p) for p in (
        LABELS, ROOTS, PROTOCOL, Path(__file__),
        Path("scripts/budgeted_cost_value_head_v1.py"),
        Path("scripts/bootstrap_cost_value_teacher_v1.py"),
    )}
    contract = {"schema": "rf-knots-root-budget16-value-head-training-v1",
                "config": CONFIG, "inputs": inputs, "actor_sha256": ACTOR_SHA}
    contract_sha = digest(contract)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.out / "manifest.json"
        state = read(manifest_path) if args.resume else {
            "schema": contract["schema"], "contract": contract,
            "contract_sha256": contract_sha, "status": "preparing",
            "optimizer_steps": 0, "checkpoints": [],
        }
        if state["contract"] != contract or state["contract_sha256"] != contract_sha:
            raise ValueError("training contract changed")
        if state["optimizer_steps"] == CONFIG["steps"]:
            print("already completed", flush=True)
            return
        atomic(manifest_path, state)
        torch.set_num_threads(1)
        net, _ = load_actor()
        net.to("mps")
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        feature = encode_roots(net, roots)
        examples = labels["labels"]
        encoded = {
            name: torch.stack([feature[(r["root_index"], name)] for r in examples])
            for name in ("knot", "layout", "isotopy")
        }
        root_cc = torch.tensor([r["cc_left"] for r in examples], dtype=torch.float32, device="mps")
        root_moves = torch.tensor([r["moves_left"] for r in examples], dtype=torch.float32, device="mps")
        lam = torch.tensor([r["lambda"] for r in examples], dtype=torch.float32, device="mps")
        solved = torch.tensor([r["solved"] for r in examples], dtype=torch.float32, device="mps")
        cc = torch.tensor([float("nan") if r["cc_remaining"] is None else r["cc_remaining"]
                           for r in examples], dtype=torch.float32, device="mps")
        moves = torch.tensor([float("nan") if r["moves_remaining"] is None else r["moves_remaining"]
                              for r in examples], dtype=torch.float32, device="mps")
        head = BudgetedCostValueHead(reference_simulations=CONFIG["reference_simulations"]).to("mps")
        optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["lr"],
                                      weight_decay=CONFIG["weight_decay"])
        draw_rng = torch.Generator(device="cpu")
        draw_rng.manual_seed(CONFIG["seed"])
        torch.manual_seed(CONFIG["seed"])
        torch.mps.manual_seed(CONFIG["seed"])
        start = state["optimizer_steps"]
        if start:
            entry = state["checkpoints"][-1]
            checkpoint_path = args.out / entry["file"]
            if entry["step"] != start or sha(checkpoint_path) != entry["sha256"]:
                raise ValueError("resume checkpoint SHA differs")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["contract_sha256"] != contract_sha or checkpoint["step"] != start:
                raise ValueError("resume checkpoint contract differs")
            head.load_state_dict(checkpoint["head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            draw_rng.set_state(checkpoint["draw_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            torch.mps.set_rng_state(checkpoint["mps_rng"])
        state.update(status="training", pid=os.getpid())
        atomic(manifest_path, state)
        end = min(CONFIG["steps"], start + (args.stop_after or CONFIG["steps"]))
        for step in range(start, end):
            indices = torch.randint(len(examples), (CONFIG["batch_size"],), generator=draw_rng)
            indices = indices.to("mps")
            output = head(
                {name: vector[indices] for name, vector in encoded.items()},
                cc_left=root_cc[indices], moves_left=root_moves[indices],
                cc_weight=lam[indices],
                local_simulations=torch.full((len(indices),), 16.0, device="mps"),
            )
            losses = value_loss(output, solved=solved[indices], cc=cc[indices],
                                moves=moves[indices], cc_left=root_cc[indices],
                                moves_left=root_moves[indices])
            if not torch.isfinite(losses["total"]):
                raise ValueError("nonfinite value loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["grad_clip"],
                                                  error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % CONFIG["checkpoint_interval"] == 0:
                torch.mps.synchronize()
                payload = {
                    "contract_sha256": contract_sha, "step": step + 1,
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "draw_rng": draw_rng.get_state(),
                    "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state(),
                }
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
                print(json.dumps({"step": step + 1, "loss": state["value_loss"],
                                  "solve": state["solve_loss"], "cost": state["cost_loss"]},
                                 sort_keys=True), flush=True)
        if state["optimizer_steps"] == CONFIG["steps"]:
            state.update(status="completed", pid=None)
            atomic(manifest_path, state)


if __name__ == "__main__":
    main()
