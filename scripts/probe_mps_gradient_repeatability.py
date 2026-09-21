"""Attribute MPS bit differences with repeated original and bulk TRAIN backward passes.

No optimizer step, saved model, TEST data, or modification of the failed probe.
The numeric tolerance is the CPU/MPS tolerance declared before the first probe.
"""

import argparse
import random
from pathlib import Path

import benchmark_policy_training_device as probe
import torch

PREVIOUS = Path("artifacts/mcts-policy-training-device-probe-v1/report.json")


def compare(before, after):
    gate = probe.difference(before, after, exact=False)
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    differences = sorted(
        ((float((before[name] - after[name]).abs().max()), name) for name in changed), reverse=True
    )
    return {**gate, "non_bit_equal_tensors": len(changed), "largest_differences": differences[:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.backends.mps.is_available():
        raise ValueError("require MPS and a new output")
    previous = probe.read(PREVIOUS)
    pins = {
        **previous["inputs"],
        str(PREVIOUS): probe.sha(PREVIOUS),
        str(Path(__file__).resolve()): probe.sha(Path(__file__).resolve()),
    }
    probe.check_pins(pins)
    torch.set_num_threads(1)
    torch.manual_seed(probe.CONFIG["seed"])
    groups, _ = probe.load_policy_targets(probe.DATA)
    rng = random.Random(probe.CONFIG["seed"])
    selection = [
        rng.choice(groups[c])
        for c in rng.sample(range(len(groups)), max(probe.CONFIG["batch_sizes"]))
    ][:16]
    checkpoint = torch.load(probe.RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    net = probe.M.SitePolicyValueNet(**checkpoint["architecture"]).to("mps").train()
    net.load_state_dict(checkpoint["model"], strict=True)
    batch = probe.M.build_batch([r[0] for r in selection], "mps", budget_dim=net.budget_dim)
    old_helper = probe.graph.crossing_adjacency
    adjacency_equal = torch.equal(
        old_helper(batch["word"], batch["lengths"]),
        probe.bulk_crossing_adjacency(batch["word"], batch["lengths"]),
    )
    if not adjacency_equal:
        raise ValueError("MPS integer adjacency differs")
    series = {}
    for name, helper in (("original", old_helper), ("bulk", probe.bulk_crossing_adjacency)):
        with probe.use_adjacency(helper):
            probe.step(net, selection, "mps")  # Warmup, no parameter update.
            series[name] = [
                probe.capture(net, *probe.step(net, selection, "mps")) for _ in range(4)
            ]
    for name, value in net.state_dict().items():
        if not torch.equal(value.detach().cpu(), checkpoint["model"][name]):
            raise ValueError("model weights/buffers changed")
    results = {
        "original_repeats": [compare(series["original"][0], r) for r in series["original"][1:]],
        "bulk_repeats": [compare(series["bulk"][0], r) for r in series["bulk"][1:]],
        "bulk_vs_original": [compare(series["original"][0], r) for r in series["bulk"]],
    }
    probe.check_pins(pins)
    report = {
        "status": "completed",
        "inputs": pins,
        "batch_size": 16,
        "adjacency_bit_equal_on_mps": adjacency_equal,
        "weights_unchanged": True,
        "results": results,
        "scope": (
            "Four repeated backward passes per helper, same TRAIN batch and weights; "
            "no claim of bit-reproducible MPS training or searches."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    probe.atomic(args.out / "report.json", report)
    print(results, flush=True)


if __name__ == "__main__":
    main()
