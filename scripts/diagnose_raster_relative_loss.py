"""Read-only numerical diagnosis of the intentionally interrupted protocol94 arm."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from mcts_joint_training import load_policy_targets
from pretrained_raster_policy import policy_log_probabilities
from raster_pair_training import CONFIG, SUPERVISION, matched_initializations

from rf_knots.rollout_value import clipped_relative_loss


@torch.no_grad()
def selected_logs(net, group):
    result = []
    net.eval()
    for request, action, _ in group:
        result.append(float(policy_log_probabilities(net, [request])[0, action]))
    return torch.tensor(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    parent = json.loads((args.run / "manifest.json").read_text())
    run = args.run / "q254"
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        parent["status"] != "failed"
        or parent["returncode"] != -2
        or manifest["completed_steps"] != 16
    ):
        raise ValueError("require the declared intentionally interrupted16-step run")
    if sha(run / "progress.pt") != manifest["progress_sha256"]:
        raise ValueError("saved progress changed")
    pins = {**manifest["inputs"], **manifest["sources"]}
    for path in (
        args.run / "manifest.json",
        run / "manifest.json",
        run / "progress.pt",
        run / "initial.pt",
        Path(__file__),
    ):
        pins[str(path)] = sha(path)
    check_pins(pins)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, states, _, _ = matched_initializations(manifest["inputs"])
    progress = torch.load(run / "progress.pt", map_location="cpu", weights_only=False)
    positives, preferences = load_policy_targets(SUPERVISION)
    rng = random.Random(CONFIG["seed"])
    # Only replay the declared RNG consumption, not expensive metric witnesses.
    for _ in range(progress["steps"]):
        for _ in range(
            3 * (CONFIG["old_pairs"] + CONFIG["suffix_pairs"]) + 4 * CONFIG["graph_pairs"]
        ):
            rng.random()
        for _ in range(CONFIG["positive_states"]):
            rng.choice(rng.choice(positives))
        group_index = rng.randrange(len(preferences))
        group = preferences[group_index]
        indices = [rng.randrange(len(group)) for _ in range(CONFIG["relative_rows"])]
    if rng.getstate() != progress["rng"]:
        raise ValueError("diagnostic did not reconstruct the exact saved draw state")
    net.load_state_dict(states["q254"], strict=True)
    initial = selected_logs(net, group)
    net.load_state_dict(progress["model"], strict=True)
    current = selected_logs(net, group)
    coefficients = torch.tensor([row[2] for row in group])
    sampled = current[indices].clone().requires_grad_()
    scaled_coefficients = coefficients[indices] * (len(group) / len(indices))
    old_loss = clipped_relative_loss(sampled, initial[indices], scaled_coefficients, CONFIG["clip"])
    old_gradient = torch.autograd.grad(old_loss, sampled)[0]
    full = current.clone().requires_grad_()
    score = (coefficients * (full - initial)).sum()
    logistic = torch.nn.functional.softplus(-score)
    logistic_gradient = torch.autograd.grad(logistic, full)[0]
    if not torch.isfinite(logistic) or not torch.isfinite(logistic_gradient).all():
        raise ValueError("log-space comparison is not finite")
    if (logistic_gradient.abs() > coefficients.abs() + 1e-6).any():
        raise ValueError("logistic log-probability gradient bound failed")
    report = {
        "status": "completed",
        "inputs": pins,
        "checkpoint_steps": progress["steps"],
        "last_training_manifest": {
            k: manifest[k]
            for k in (
                "metric_loss",
                "positive_loss",
                "relative_loss",
                "gradient_norm_before_clip",
                "seconds_this_run",
            )
        },
        "reference_group": group_index,
        "group_states": len(group),
        "sample_indices": indices,
        "rng_reconstruction_matches": True,
        "sample_initial_logp": initial[indices].tolist(),
        "sample_current_logp": current[indices].tolist(),
        "sample_coefficients": scaled_coefficients.tolist(),
        "sample_log_ratios": (sampled.detach() - initial[indices]).tolist(),
        "sample_log_ratios_at_or_above20": int(((sampled.detach() - initial[indices]) >= 20).sum()),
        "sample_ratio_loss_at_saved_checkpoint": float(old_loss.detach()),
        "sample_ratio_logp_gradients": old_gradient.tolist(),
        "full_group_logistic_score": float(score.detach()),
        "full_group_logistic_loss": float(logistic.detach()),
        "full_group_logistic_max_abs_logp_gradient": float(logistic_gradient.abs().max()),
        "full_group_max_abs_coefficient": float(coefficients.abs().max()),
        "scope": "Post-update16 TRAIN-only diagnosis, not exact replay of pre-update16 loss. "
        "Log-space comparison is a candidate objective, not an implemented training fix. "
        "No optimizer updates or checkpoint mutation. Do not resume94 unchanged.",
    }
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(args.out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "inputs"}), flush=True)


if __name__ == "__main__":
    main()
