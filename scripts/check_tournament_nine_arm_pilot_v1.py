"""Non-publishing one-step MPS execution gate for every pilot arm."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

from bootstrap_cost_value_teacher_v1 import load_actor
from graph_metric_sampling import load_graph_sampler
from long_general_training_data import CONFIG as BASE, load_long_source, paired_draw
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import reference_preferences
from raster_log_training import CORPUS, GRAPH, SUPERVISION
from tournament_s10_dataset_v1 import load_groups
from train_tournament_nine_arm_pilot_v1 import (
    CONFIG, LONG_DATA, optimize, signal_bank, signal_rows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    args = parser.parse_args()
    pgx = args.pgx_root.resolve() / "src"
    if not (pgx / "pgx_mcts_bench").is_dir():
        raise ValueError("canonical pgx source absent")
    sys.path.insert(0, str(pgx))
    if not torch.backends.mps.is_available():
        raise RuntimeError("native MPS unavailable")
    torch.set_num_threads(1)
    roots, groups, _ = load_groups()
    bank = signal_bank(roots, groups)
    key = sorted(bank)[0]
    long_source, _, _ = load_long_source(LONG_DATA)
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    positives, raw_relative = load_policy_targets(SUPERVISION)
    net0, objective0 = load_actor()
    net0.to("mps")
    objective0.to("mps")
    old_references = reference_preferences(net0, raw_relative, microbatch=1)
    results = {}
    for arm in CONFIG["arms"]:
        net, objective = copy.deepcopy(net0), copy.deepcopy(objective0)
        optimizer = torch.optim.AdamW([*net.parameters(), *objective.parameters()],
                                      lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        draw = paired_draw("long", random.Random(BASE["seed"]),
                           random.Random(BASE["long_seed"]), original, suffix, graph,
                           long_source, positives, old_references)
        rows = signal_rows(bank, key, arm, 0)
        refs = [reference_preferences(net0, [group], microbatch=1)[0] for group in rows]
        weights = ([CONFIG["tournament_weight"],
                    CONFIG["tournament_weight"] * CONFIG["g_cc_extra_weight"]]
                   if arm == "G" and len(refs) == 2 else [CONFIG["tournament_weight"]])
        if arm == "A":
            weights = [0.0]
        losses = optimize(net, objective, optimizer, draw, None,
                          list(zip(refs, weights, strict=True)))
        if not all(torch.isfinite(t).all() for t in net.state_dict().values()):
            raise ValueError("nonfinite real-network state after optimizer step")
        results[arm] = {"tournament_terms": losses["tournament_terms"],
                        "gradient_norm_before_clip": losses["gradient_norm_before_clip"]}
        print(json.dumps({"arm": arm, **results[arm]}, sort_keys=True), flush=True)
    if set(results) != set(CONFIG["arms"]):
        raise ValueError("nine-arm execution coverage missing")


if __name__ == "__main__":
    main()
