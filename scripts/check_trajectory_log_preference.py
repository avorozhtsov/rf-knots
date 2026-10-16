"""Protocol96: actual raster-network gradients for every audited relative group."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from mcts_joint_training import load_policy_targets
from pretrained_raster_policy import policy_log_probabilities, reference_preferences
from raster_pair_training import SUPERVISION, matched_initializations
from trajectory_log_preference import relative_backward, trajectory_preference

RUN = Path("artifacts/pretrained-raster-training-pair-v1")
DIAGNOSTIC = Path("artifacts/pretrained-raster-relative-loss-diagnostic-v1/report.json")


def gradient_summary(net):
    gradients = [p.grad for p in net.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise ValueError("missing or nonfinite parameter gradients")
    return {
        "gradient_l2": float(torch.sqrt(sum(g.square().sum() for g in gradients))),
        "gradient_max_abs": max(float(g.abs().max()) for g in gradients),
        "gradient_tensor_count": len(gradients),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    diagnostic = json.loads(DIAGNOSTIC.read_text())
    if diagnostic["status"] != "completed" or not diagnostic["rng_reconstruction_matches"]:
        raise ValueError("require completed exact-draw diagnosis")
    pins = dict(diagnostic["inputs"])
    for path in (
        DIAGNOSTIC,
        Path(__file__),
        Path("scripts/trajectory_log_preference.py"),
        Path("tests/test_trajectory_log_preference.py"),
        Path("research/96-log-preference-real-network-gate.md"),
    ):
        pins[str(path.resolve())] = sha(path)
    check_pins(pins)
    parent = json.loads((RUN / "manifest.json").read_text())
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    net, initial_states, _, _ = matched_initializations(parent["inputs"])
    saved = torch.load(RUN / "q254/progress.pt", map_location="cpu", weights_only=False)
    _, groups = load_policy_targets(SUPERVISION)
    if len(groups) != 4:
        raise ValueError("complete relative group coverage differs")
    net.load_state_dict(initial_states["q254"], strict=True)
    references = reference_preferences(net, groups, microbatch=1)
    results = []
    for state_name, state in (("initial", initial_states["q254"]), ("update16", saved["model"])):
        net.load_state_dict(state, strict=True)
        net.train()
        for index, relative in enumerate(references):
            net.zero_grad(set_to_none=True)
            started = time.perf_counter()
            row = relative_backward(net, relative, microbatch=1, weight=0.01)
            row.update(
                state=state_name,
                group=index,
                seconds=time.perf_counter() - started,
                total_word_length=sum(len(r[0][1].word) for r in relative[0]),
                **gradient_summary(net),
            )
            if state_name == "initial" and (
                abs(row["relative_log_score"]) > 1e-3 or row["gradient_l2"] <= 0
            ):
                raise ValueError("initial reference or nonzero-gradient identity failed")
            results.append(row)
            print(json.dumps(row), flush=True)
        if any(not torch.equal(value, net.state_dict()[key]) for key, value in state.items()):
            raise ValueError("backward gate changed model tensors")
    smallest = min(range(len(groups)), key=lambda i: sum(len(r[0][1].word) for r in groups[i]))
    relative = references[smallest]
    net.load_state_dict(initial_states["q254"], strict=True)
    net.zero_grad(set_to_none=True)
    bounded = relative_backward(net, relative, microbatch=1, weight=0.01)
    bounded_gradients = {
        k: None if p.grad is None else p.grad.clone() for k, p in net.named_parameters()
    }
    net.zero_grad(set_to_none=True)
    group, old = relative
    probabilities = policy_log_probabilities(net, [r[0] for r in group])
    selected = probabilities[torch.arange(len(group)), [r[1] for r in group]]
    loss, score, mass = trajectory_preference(
        selected, old, selected.new_tensor([r[2] for r in group])
    )
    (0.01 * loss).backward()
    torch.testing.assert_close(
        loss.detach(), loss.new_tensor(bounded["relative_loss"]), rtol=1e-4, atol=1e-3
    )
    max_error = 0.0
    for name, parameter in net.named_parameters():
        expected = bounded_gradients[name]
        if expected is None:
            if parameter.grad is not None:
                raise ValueError("gradient coverage differs")
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-3, atol=5e-5)
            max_error = max(max_error, float((parameter.grad - expected).abs().max()))
    if any(not torch.equal(v, net.state_dict()[k]) for k, v in initial_states["q254"].items()):
        raise ValueError("full-autograd check changed model tensors")
    check_pins(pins)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "passed",
        "inputs": pins,
        "cases": results,
        "optimizer_steps": 0,
        "full_autograd_group": smallest,
        "full_autograd_loss": float(loss.detach()),
        "full_autograd_score": float(score.detach()),
        "coefficient_mass": float(mass),
        "gradient_max_abs_error": max_error,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "scope": "All-group real-network gradient/cost gate, not trained quality",
    }
    write_new(args.out / "report.json", report)
    print(json.dumps({"status": "passed", "gradient_max_abs_error": max_error}), flush=True)


if __name__ == "__main__":
    main()
