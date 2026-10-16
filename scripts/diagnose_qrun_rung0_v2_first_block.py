"""Fixed unseen-root policy diagnostic for parent, v1-16 and mixed v2-16."""

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
from train_qrun_like_policy_bootstrap_rung0_v1 import PARENT, ROSTER


CHECKPOINTS = {
    "parent_F144": PARENT,
    "short_only_v1_step16": Path("artifacts/qrun-like-policy-bootstrap-rung0-v1-envfix/step-0016.pt"),
    "mixed_v2_step16": Path("artifacts/qrun-like-policy-bootstrap-rung0-v2/step-0016.pt"),
}


def cases():
    roster = json.loads(ROSTER.read_text())
    for stage, indices in ((0, range(64, 84)), (3, range(20, 28))):
        for index in indices:
            root = roster["stages"][stage][index]
            witness = UnknotWitness.from_dict(root["witness"])
            data = root["instances"][0]
            instance = M.Instance(**{**data, "word": tuple(data["word"])})
            trace = verified_success_traces(instance, [witness])[0]["rows"]
            state, action = trace[len(trace) // 2]
            pool = tuple(sorted(set(proposal_candidates(instance, state, 4)) | {action}))
            yield f"stage{stage}", (instance, state, pool), pool.index(action)


def main():
    rows = list(cases())
    results = {}
    for label, path in CHECKPOINTS.items():
        net, _ = load_actor()
        saved = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(saved["model"], strict=True)
        net.to("mps").eval()
        stats = {"stage0": {"n": 0, "nll": 0., "top1": 0},
                 "stage3": {"n": 0, "nll": 0., "top1": 0}}
        with torch.no_grad():
            for key, request, target in rows:
                logp = policy_log_probabilities(net, [request])[0, :len(request[2])]
                stats[key]["n"] += 1
                stats[key]["nll"] -= float(logp[target])
                stats[key]["top1"] += int(int(logp.argmax()) == target)
        for value in stats.values():
            value["mean_nll"] = value.pop("nll") / value["n"]
        results[label] = stats
    print(json.dumps({"checkpoints": {k: sha(v) for k, v in CHECKPOINTS.items()},
                      "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
