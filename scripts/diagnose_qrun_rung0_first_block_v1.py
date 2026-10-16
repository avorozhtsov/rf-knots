"""Compare fixed unseen-reference action probabilities before/after rung-0 step16."""

import json
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from build_mcts_training_supervision import sha
from mcts_training_supervision import verified_success_traces
from pretrained_raster_policy import policy_log_probabilities
from rf_knots import mcts_loop as M
from rf_knots.evidence import UnknotWitness
from rf_knots.site_proposals import proposal_candidates

from train_qrun_like_policy_bootstrap_rung0_v1 import PARENT, PARENT_SHA, ROSTER, build_bank

CHECKPOINT = Path("artifacts/qrun-like-policy-bootstrap-rung0-v1-envfix/step-0016.pt")


def main():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("parent changed")
    bank = build_bank()
    roster = json.loads(ROSTER.read_text())
    requests = []
    for index in range(64, 84):
        instance, trace = bank[index][0]
        state, action = trace[len(trace) // 2]
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
        requests.append((f"stage0-{index}", (instance, state, actions), actions.index(action)))
    for index, root in enumerate(roster["stages"][3][:8]):
        witness = UnknotWitness.from_dict(root["witness"])
        instance_dict = root["instances"][0]
        instance = M.Instance(**{**instance_dict, "word": tuple(instance_dict["word"])})
        trace = verified_success_traces(instance, [witness])[0]["rows"]
        state, action = trace[len(trace) // 2]
        actions = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
        requests.append((f"stage3-{index}", (instance, state, actions), actions.index(action)))
    results = {}
    for label, checkpoint in (("parent_F144", PARENT), ("rung0_step16", CHECKPOINT)):
        net, _ = load_actor()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(payload["model"], strict=True)
        net.to("mps").eval()
        stats = {"stage0": {"n": 0, "nll": 0.0, "top1": 0},
                 "stage3": {"n": 0, "nll": 0.0, "top1": 0}}
        with torch.no_grad():
            for name, request, target in requests:
                logp = policy_log_probabilities(net, [request])[0, :len(request[2])]
                key = name.split("-")[0]
                stats[key]["n"] += 1
                stats[key]["nll"] -= float(logp[target])
                stats[key]["top1"] += int(int(logp.argmax()) == target)
        for item in stats.values():
            item["mean_nll"] = item.pop("nll") / item["n"]
        results[label] = stats
    print(json.dumps({"checkpoint_sha256": sha(CHECKPOINT), "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
