"""Checkpoint-specific139 inference cache parity, without any TEST scoring."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from audit_joint_policy_metric_panel import check_pins, merge_pins
from check_qrun_common_witnesses import sha, write_new
from pretrained_raster_graph import GLOBALS
from probe_raster_cache_full_network import fixtures, full_output, state_digest
from probe_raster_cache_search import CONFIG, SEEDS, discrete_rows, roots
from probe_raster_tile_cache import TileCacheProbe
from resource_cost_search import run_batch
from resource_evaluation_inputs import load_model

from rf_knots import mcts_loop as M


def compare_paths(a_rows, a_counts, b_rows, b_counts):
    if discrete_rows(a_rows) != discrete_rows(b_rows):
        raise ValueError("cached/uncached discrete search differs")
    if a_counts != {k: v for k, v in b_counts.items() if not k.startswith("cache_")}:
        raise ValueError("cached/uncached logical work differs")
    error = 0.0
    for a, b in zip(a_rows, b_rows, strict=True):
        for left, right in zip(a["episodes"], b["episodes"], strict=True):
            p = torch.tensor(left["continuation"]["probabilities"], dtype=torch.float64)
            q = torch.tensor(right["continuation"]["probabilities"], dtype=torch.float64)
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-5)
            error = max(error, float((p - q).abs().max()) if p.numel() else 0.0)
    return error


def check_model(net, objective):
    cached = copy.deepcopy(net).eval()
    probe = TileCacheProbe(cached.encoder.tiles, capacity=512)
    cached.encoder.tiles.forward = probe
    before = state_digest(net, objective)
    forward_cases, max_error = 0, 0.0
    with torch.no_grad():
        for weight in (10, 1000):
            requests = fixtures(weight)
            for begin in range(0, len(requests), 2):
                chunk = requests[begin : begin + 2]
                batch = M.build_batch([r for _, r in chunk], budget_dim=5)
                for heads in (
                    None,
                    torch.zeros_like(batch["lengths"]),
                    (batch["lengths"] - 1).clamp_min(0),
                ):
                    a, b = full_output(net, batch, heads), full_output(cached, batch, heads)
                    if not torch.equal(a["effective"], b["effective"]):
                        raise ValueError("cache masks differ")
                    for name in ("tokens", *GLOBALS, "logits", "probabilities"):
                        x, y = a[name], b[name]
                        if name == "logits":
                            x, y = x[a["effective"]], y[b["effective"]]
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                            raise ValueError("cache forward nonfinite")
                        max_error = max(max_error, float((x - y).abs().max()))
                    if not torch.equal(a["logits"].argmax(-1), b["logits"].argmax(-1)):
                        raise ValueError("cache changes greedy action")
                    forward_cases += len(chunk)
    if state_digest(cached, objective) != before:
        raise ValueError("cached forward changed checkpoint")
    searches = []
    for prior in ("distance", "uniform"):
        for seed in SEEDS:
            config = {**CONFIG, "seed": seed}
            a, counts_a = run_batch(roots(), net, objective, 0, config, prior=prior)
            b, counts_b = run_batch(roots(), net, objective, 0, config, prior=prior, cached=True)
            error = compare_paths(a, counts_a, b, counts_b)
            searches.append(
                dict(
                    prior=prior,
                    seed=seed,
                    probability_max_error=error,
                    logical_counts=dict(counts_a),
                    cache_counts={k: v for k, v in counts_b.items() if k.startswith("cache_")},
                )
            )
    if state_digest(net, objective) != before:
        raise ValueError("parity gate changed checkpoint")
    return dict(
        status="passed",
        forward_cases=forward_cases,
        forward_max_error=max_error,
        searches=searches,
        cache_capacity=512,
        state_sha256=before,
        weights_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    pins, results = {}, {}
    for arm in ("units", "raw"):
        for stage in ("initial", "final"):
            net, objective, extra, checkpoint = load_model(args.run, args.pair_audit, arm, stage)
            merge_pins(pins, extra)
            result = check_model(net, objective)
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            results[f"{arm}-{stage}"] = result
            print(json.dumps(dict(arm=arm, stage=stage, status="passed")), flush=True)
    for path in (
        Path(__file__).resolve(),
        Path("scripts/resource_cost_search.py"),
        Path("scripts/probe_raster_tile_cache.py"),
        Path("scripts/probe_raster_cache_search.py"),
        Path("scripts/probe_raster_cache_full_network.py"),
        Path("tests/test_resource_search_cache.py"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    write_new(
        args.out,
        dict(
            status="passed",
            inputs=pins,
            models=results,
            cache_capacity=512,
            scope="Non-TEST fixtures only; no scientific quality or speed claim",
        ),
    )


if __name__ == "__main__":
    main()
