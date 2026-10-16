"""TRAIN-only first-stage value-head fit to one-continuation teacher outcomes.

The actor encoder is frozen. This does not fit a multi-simulation value curve,
change the actor policy, or train on MCTS tournament comparisons.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from audit_cost_mcts_continuations import audit_episode
from bootstrap_cost_value_teacher_v1 import ACTOR, ACTOR_SHA, ROOTS, load_actor, read
from budgeted_cost_value_head_v1 import BudgetedCostValueHead, value_loss
from build_mcts_training_supervision import atomic, sha

from rf_knots import mcts_loop as M

SOURCE = Path("artifacts/raster-budgeted-value-teacher-diverse-v1")
TRAIN_GROUPS = (35, 122)
VALIDATION_GROUP = 80  # a disjoint TRAIN representation; not the true DEV split
SEED = 20261204
STEPS = 512
LEARNING_RATE = 0.001


def load_rows():
    manifest = read(SOURCE / "manifest.json")
    if manifest["status"] != "completed" or len(manifest["shards"]) != 60:
        raise ValueError("teacher collection incomplete")
    if manifest["contract"]["actor_sha256"] != ACTOR_SHA:
        raise ValueError("teacher actor differs")
    rows, pins = [], {str(SOURCE / "manifest.json"): sha(SOURCE / "manifest.json")}
    for position, entry in enumerate(manifest["shards"]):
        path = SOURCE / entry["file"]
        if entry["file"] != f"attempt-{position:03d}.json" or sha(path) != entry["sha256"]:
            raise ValueError("teacher shard identity differs")
        pins[str(path)] = entry["sha256"]
        rows.append(read(path)["row"])
    return rows, pins


def extract_examples(rows, roots):
    examples = []
    for row in rows:
        root = roots[row["condition"]]
        group = root["base_root_index"]
        if group not in (*TRAIN_GROUPS, VALIDATION_GROUP):
            raise ValueError("teacher row outside pinned TRAIN roster")
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        config = {
            **read(SOURCE / "manifest.json")["contract"]["config"],
            "horizon": instance.cap_moves,
        }
        if len(row["episodes"]) != len(row["episodes_train_owned"]):
            raise ValueError("episode ownership coverage differs")
        for episode, eligible in zip(row["episodes"], row["episodes_train_owned"], strict=True):
            audit_episode(instance, episode, config)
            if not eligible:
                continue
            start, end = episode["continuation"]["start"], episode["continuation"]["end"]
            success = episode["continuation"]["stop_reason"] == "solved"
            cc_left = instance.cap_crossing_changes - start["used_cc"]
            moves_left = instance.cap_moves - start["used_moves"]
            cc = end["used_cc"] - start["used_cc"] if success else math.nan
            moves = end["used_moves"] - start["used_moves"] if success else math.nan
            if cc_left < 0 or moves_left < 0 or (success and (cc > cc_left or moves > moves_left)):
                raise ValueError("leaf-relative resource label invalid")
            examples.append(
                dict(
                    group=group,
                    word=tuple(start["word"]),
                    strands=start["strands"],
                    cc_left=cc_left,
                    moves_left=moves_left,
                    cc_weight=row["cc_weight"],
                    solved=int(success),
                    cc=cc,
                    moves=moves,
                )
            )
    train_states = {(e["word"], e["strands"]) for e in examples if e["group"] in TRAIN_GROUPS}
    validation = [e for e in examples if e["group"] == VALIDATION_GROUP]
    overlap = sum((e["word"], e["strands"]) in train_states for e in validation)
    if overlap:
        raise ValueError(f"exact representation leaked across TRAIN groups: {overlap}")
    return examples


def encode_examples(net, examples):
    cache = {}
    net.eval()
    with torch.no_grad():
        for example in examples:
            key = (example["word"], example["strands"])
            if key in cache:
                continue
            word = torch.tensor([list(example["word"])], dtype=torch.long)
            length = torch.tensor([len(example["word"])], dtype=torch.long)
            strands = torch.tensor([example["strands"]], dtype=torch.long)
            encoded = net.encoder.encode_sites(word, length, strands)
            cache[key] = {
                name: encoded[name][0].detach().clone() for name in ("knot", "layout", "isotopy")
            }
    return {
        name: torch.stack([cache[(e["word"], e["strands"])][name] for e in examples])
        for name in ("knot", "layout", "isotopy")
    }, len(cache)


def subset(values, indices):
    return {name: tensor[indices] for name, tensor in values.items()}


def evaluate(head, features, examples, indices):
    rows = [examples[i] for i in indices]
    kwargs = {
        "cc_left": [e["cc_left"] for e in rows],
        "moves_left": [e["moves_left"] for e in rows],
        "cc_weight": [e["cc_weight"] for e in rows],
        "local_simulations": [1] * len(rows),
    }
    predicted = head(subset(features, indices), **kwargs)
    loss = value_loss(
        predicted,
        solved=[e["solved"] for e in rows],
        cc=[e["cc"] for e in rows],
        moves=[e["moves"] for e in rows],
        cc_left=kwargs["cc_left"],
        moves_left=kwargs["moves_left"],
    )
    labels = torch.tensor([e["solved"] for e in rows], dtype=torch.float32)
    positive = labels.bool()
    return loss, {
        "count": len(rows),
        "solved": int(labels.sum()),
        "solve_brier": float(((predicted["p_solve"] - labels) ** 2).mean()),
        "solve_mean_prediction": float(predicted["p_solve"].mean()),
        "cc_mae_on_solved": float(
            (
                predicted["cc_remaining"][positive]
                - torch.tensor([e["cc"] for e in rows if e["solved"]])
            )
            .abs()
            .mean()
        ),
        "moves_mae_on_solved": float(
            (
                predicted["moves_remaining"][positive]
                - torch.tensor([e["moves"] for e in rows if e["solved"]])
            )
            .abs()
            .mean()
        ),
        "loss": float(loss["total"].detach()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new output required; preserve any earlier attempt")
    import sys

    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rows, pins = load_rows()
    pins.update(
        {
            str(p): sha(p)
            for p in (ACTOR, ROOTS, Path(__file__), Path("scripts/budgeted_cost_value_head_v1.py"))
        }
    )
    if pins[str(ACTOR)] != ACTOR_SHA:
        raise ValueError("actor checkpoint hash differs")
    roots = read(ROOTS)
    examples = extract_examples(rows, roots)
    groups = Counter(e["group"] for e in examples)
    if set(groups) != {*TRAIN_GROUPS, VALIDATION_GROUP}:
        raise ValueError("insufficient group coverage")
    net, _ = load_actor()
    features, unique_states = encode_examples(net, examples)
    train = [i for i, e in enumerate(examples) if e["group"] in TRAIN_GROUPS]
    validation = [i for i, e in enumerate(examples) if e["group"] == VALIDATION_GROUP]
    head = BudgetedCostValueHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE)
    args.out.mkdir(parents=True)
    contract = dict(
        schema="rf-knots-budgeted-value-head-bootstrap-v1",
        teacher_role="single policy continuation, local_simulations=1",
        train_groups=TRAIN_GROUPS,
        heldout_train_group=VALIDATION_GROUP,
        steps=STEPS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        inputs=pins,
    )
    atomic(
        args.out / "manifest.json", dict(status="training", contract=contract, counts=dict(groups))
    )
    head.eval()
    with torch.no_grad():
        _, baseline = evaluate(head, features, examples, validation)
    for _ in range(STEPS):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = evaluate(head, features, examples, train)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    head.eval()
    with torch.no_grad():
        _, train_metrics = evaluate(head, features, examples, train)
        _, validation_metrics = evaluate(head, features, examples, validation)
    checkpoint = args.out / "head.pt"
    torch.save(
        dict(model=head.state_dict(), optimizer=optimizer.state_dict(), contract=contract),
        checkpoint,
    )
    report = dict(
        status="completed",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        unique_encoded_states=unique_states,
        admitted_episodes=len(examples),
        train=train_metrics,
        heldout_train_initial=baseline,
        heldout_train_final=validation_metrics,
        note="Not a true DEV result; no multi-simulation training or search utility claim.",
    )
    atomic(args.out / "report.json", report)
    atomic(
        args.out / "manifest.json",
        dict(
            status="completed",
            contract=contract,
            counts=dict(groups),
            report_sha256=sha(args.out / "report.json"),
        ),
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
