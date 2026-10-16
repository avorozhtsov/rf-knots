"""Native-MPS R2 continuation with retained rehearsal and new root-only F."""

import math
from pathlib import Path

import torch

import train_joint_f_r2_continuation_v1 as previous
import train_joint_f_r2_r4_v1 as base
import train_joint_f_r2_r4_bounded_v2 as bounded
from rf_knots import mcts_loop as M, reference

BANK = Path("artifacts/additive-root-only-f-bank-v3")
BANK_SHA = "e1129f31cacb538a9b3a8824f8f552dabdd80803d7fd95c5feb0fb1aaa449c33"
BANK_AUDIT = Path("artifacts/additive-root-only-f-bank-audit-v3/report.json")
BANK_AUDIT_SHA = "cd23cae6c0b93d1f0dcace39f343a94cefa86e3e110f62974a180966ecff376f"
PARENT = Path("artifacts/additive-f-r2-stage-v2/step-152964.pt")
PARENT_SHA = "12f4a3675dd6880203a2968431d3c7bdd44a45b586e92937a25373facc32803e"
START_STEP = 152964
PROTOCOL = Path("research/345-additive-root-only-r2-training-v3.md")
PATTERN = (0,) * 5 + (1,) * 7 + (2,) * 7 + (3,)
OLD_BY_STAGE = None
NEW_ORDER = None
PREVIOUS_SETUP = previous.setup_training


def bank_order(manifest):
    order = []
    for index, entry in enumerate(manifest["roots"]):
        path = BANK / f"root-{index:04d}.json"
        if entry["file"] != path.name or base.sha(path) != entry["sha256"]:
            raise ValueError(f"root SHA/order differs: {index}")
        if base.read(path)["groups"]:
            order.append(index)
    if (len(order) != 402 or sum(i < 1007 for i in order) != 399
            or sum(i >= 1007 for i in order) != 3):
        raise ValueError("audited F bank coverage differs")
    return order


def initialize_order():
    global OLD_BY_STAGE, NEW_ORDER
    order = bank_order(base.read(BANK / "manifest.json"))
    old, new = [[] for _ in range(4)], []
    for index in order:
        row = base.read(BANK / f"root-{index:04d}.json")
        (new if index >= 1007 else old[row["stage"]]).append(index)
    if [len(rows) for rows in old] != [115, 195, 81, 8] or len(new) != 3:
        raise ValueError("retained/new stage coverage differs")
    OLD_BY_STAGE, NEW_ORDER = old, new
    return order


def selected_group(order, step):
    if OLD_BY_STAGE is None or NEW_ORDER is None:
        raise ValueError("stage-aware order not initialized")
    local = step - START_STEP
    if local < 0:
        raise ValueError("step precedes continuation")
    if local % 10 == 0:
        root_index = NEW_ORDER[(local // 10) % len(NEW_ORDER)]
        group_cycle = local // 30
    else:
        old_slot = local - local // 10 - 1
        position = old_slot % len(PATTERN)
        stage = PATTERN[position]
        within_cycle = sum(value == stage for value in PATTERN[:position])
        per_cycle = PATTERN.count(stage)
        ordinal = (old_slot // len(PATTERN)) * per_cycle + within_cycle
        rows = OLD_BY_STAGE[stage]
        root_index = rows[ordinal % len(rows)]
        group_cycle = step // len(order)
    row = base.read(BANK / f"root-{root_index:04d}.json")
    groups = row["groups"]
    group_index = group_cycle % len(groups)
    return row, groups[group_index], group_index


def root_only_step(net, objective, head, optimizer, group, step):
    if (group.get("gradient_scope") != "exact_zero_spent_root_only"
            or group["root_index"] < 1007):
        raise ValueError("unexpected new-root gradient scope")
    data = group["instance"]
    instance = M.Instance(**{**data, "word": tuple(data["word"])})
    state = M.SearchState(instance.word, instance.strands, 0, 0)
    coefficients = {}
    for item in group["root_coefficients"]:
        action, value = item["action"], float(item["value"])
        if (not reference.is_legal(instance.spec, instance.word,
                                   instance.strands, action, True)
                or not math.isfinite(value)):
            raise ValueError("illegal/nonfinite new root F term")
        coefficients[(state, action)] = value
    if len(coefficients) != group["signal_terms"]:
        raise ValueError("root F term count differs")
    positive = [action for (_, action), value in coefficients.items() if value > 0]
    negative = [action for (_, action), value in coefficients.items() if value < 0]
    successors = {action: reference.apply(instance.spec, instance.word,
                                          instance.strands, action)
                  for _, action in coefficients}
    if not any(successors[a] != successors[b]
               for a in positive for b in negative):
        raise ValueError("root F term lost strict successor divergence")
    parameters = [*net.parameters(), *objective.parameters(), *head.parameters()]
    optimizer.zero_grad(set_to_none=True)
    f_loss, pair_count = bounded.bounded_f_loss(net, instance, coefficients)
    f_loss.backward()
    f_grads = base.gradient_snapshot(parameters)
    f_norm = base.encoder_norm(f_grads, parameters, net.encoder)
    if not math.isfinite(f_norm) or f_norm <= 1e-12:
        raise ValueError("new root produced zero/nonfinite F encoder gradient")
    combined_norm = torch.nn.utils.clip_grad_norm_(
        parameters, base.CONFIG["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"F_loss": float(f_loss.detach()), "aux_loss": 0.0,
            "imitation_loss": 0.0, "distance_loss": 0.0,
            "F_encoder_grad_norm": f_norm,
            "aux_encoder_grad_norm": 0.0,
            "rehearsal_encoder_grad_norm": 0.0,
            "aux_F_cosine": 0.0, "rehearsal_F_cosine": 0.0,
            "aux_scale": 0.0, "rehearsal_scale": 0.0,
            "combined_grad_norm": float(combined_norm),
            "sampled_F_terms": len(coefficients), "F_pairs": pair_count,
            "new_root_only": True}


def train_step(net, objective, head, optimizer, group, step):
    if group.get("gradient_scope") == "exact_zero_spent_root_only":
        return root_only_step(net, objective, head, optimizer, group, step)
    return bounded.train_step(net, objective, head, optimizer, group, step)


def setup_training():
    PREVIOUS_SETUP()
    base.train_step = train_step


def inputs():
    paths = (BANK / "manifest.json", BANK_AUDIT, PARENT,
             PARENT.parent / "manifest.json", PROTOCOL, Path(__file__),
             Path("scripts/train_joint_f_r2_continuation_v1.py"),
             Path("scripts/train_joint_f_r2_r4_v1.py"),
             Path("scripts/train_joint_f_r2_r4_bounded_v2.py"),
             Path("scripts/train_joint_f_r2_r4_smoothed_v3.py"),
             Path("scripts/mcts_tournament_signals_v1.py"),
             Path("scripts/tournament_term_sampler_v1.py"),
             Path("scripts/pretrained_raster_policy.py"),
             Path("scripts/mcts_training_supervision.py"))
    hashes = {str(path): base.sha(path) for path in paths}
    if (hashes[str(BANK / "manifest.json")] != BANK_SHA
            or hashes[str(BANK_AUDIT)] != BANK_AUDIT_SHA
            or hashes[str(PARENT)] != PARENT_SHA):
        raise ValueError("bank/audit/parent SHA differs")
    parent = base.read(PARENT.parent / "manifest.json")
    if (parent["optimizer_steps"] != START_STEP
            or parent["checkpoints"][-1]["sha256"] != PARENT_SHA
            or parent["contract"]["arm"] != "R2"):
        raise ValueError("parent checkpoint/manifest differs")
    return hashes


def main():
    initialize_order()
    previous.PARENT = PARENT
    previous.PARENT_SHA = PARENT_SHA
    previous.PARENT_MANIFEST = PARENT.parent / "manifest.json"
    previous.START_STEP = START_STEP
    previous.PROTOCOL = PROTOCOL
    previous.inputs = inputs
    previous.setup_training = setup_training
    base.BANK = BANK
    base.bank_order = bank_order
    base.selected_group = selected_group
    base.CONFIG["data_bank"] = "additive-root-only-F-v3"
    base.CONFIG["sampler"] = "retained-stage-90/new-root-only-10"
    base.CONFIG["retained_root_draw_fraction"] = 0.9
    base.CONFIG["new_root_only_F_no_aux_rehearsal"] = True
    previous.main()


if __name__ == "__main__":
    main()
