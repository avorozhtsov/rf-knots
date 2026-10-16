"""Protocol114 tiny paired searches; separate outputs, no active-run integration."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from audit_cost_mcts_continuations import audit_episode
from pretrained_raster_graph import RasterGraphPolicy
from pretrained_raster_tiles import PretrainedRasterTiles
from probe_raster_cache_full_network import ARCHITECTURE, CHECKPOINTS, state_digest
from probe_raster_tile_cache import TileCacheProbe, file_sha
from raster_cost_search import run_batch

from rf_knots import mcts_loop as M
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.evidence import UnknotWitness

CONFIG = dict(
    horizon=8, insertion_candidates=4, simulations=4, widening_batch=4, c_puct=1.5, temperature=1.0
)
SEEDS = (20261056, 20261057, 20261058)


def roots():
    result = []
    for panel_index, (word, cc, moves) in enumerate((((1,), 0, 4), ((1, 1, 1), 1, 8))):
        for weight in (10, 1000):
            instance = M.Instance(
                f"cache-engineering-{len(word)}-{weight}",
                word,
                2,
                cc,
                cap_moves=moves,
                growth_len=2,
                growth_strands=1,
                cc_weight=weight,
            )
            result.append(
                dict(condition=len(result), panel_index=panel_index, instance=instance.to_dict())
            )
    return result


def discrete_rows(rows):
    result = copy.deepcopy(rows)
    for row in result:
        for episode in row["episodes"]:
            episode["continuation"].pop("probabilities")
    return result


def compare_runs(reference, cached):
    equal = discrete_rows(reference["rows"]) == discrete_rows(cached["rows"])
    discrepancies, close = [], equal
    if equal:
        for left, right in zip(reference["rows"], cached["rows"], strict=True):
            for a, b in zip(left["episodes"], right["episodes"], strict=True):
                for p, q in zip(
                    a["continuation"]["probabilities"],
                    b["continuation"]["probabilities"],
                    strict=True,
                ):
                    discrepancies.append(abs(p - q))
                    close = (
                        close
                        and math.isfinite(p)
                        and math.isfinite(q)
                        and (abs(p - q) <= 1e-5 + 1e-5 * abs(q))
                    )
    counts_equal = reference["counts"] == cached["counts"]
    return dict(
        discrete_transcripts_equal=equal,
        logical_counts_equal=counts_equal,
        chosen_probabilities_close=close,
        maximum_probability_error=max(discrepancies, default=0.0) if equal else None,
        equal_work_timing=equal and counts_equal,
        uncached_seconds=reference["seconds"],
        cached_seconds=cached["seconds"],
        uncached_over_cached_seconds=reference["seconds"] / cached["seconds"],
        passed=equal and close and counts_equal,
    )


def load_network(checkpoint, arm):
    from pgx_mcts_bench.networks import MaskedAxialCylinderResidualBlock

    if file_sha(checkpoint) != CHECKPOINTS[arm]:
        raise ValueError("trained checkpoint identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != ARCHITECTURE:
        raise ValueError("architecture changed")
    teacher = SimpleNamespace(
        variant="axial",
        recurrent_steps=0,
        max_strands=12,
        input=torch.nn.Conv2d(4, 64, 1, bias=False),
        blocks=torch.nn.ModuleList([MaskedAxialCylinderResidualBlock(64) for _ in range(4)]),
    )
    net = RasterGraphPolicy(PretrainedRasterTiles(teacher)).eval()
    net.load_state_dict(payload["model"], strict=True)
    objective = MetricEmbeddingObjective().eval()
    objective.load_state_dict(payload["objective"], strict=True)
    return net, objective


def measure(net, objective, mode, seed, panel):
    before = state_digest(net, objective)
    cache = TileCacheProbe(net.encoder.tiles) if mode == "cached" else None
    if cache is not None:
        net.encoder.tiles.forward = cache
    cnn = dict(tiles=0, batches=0)

    def count_input(_module, args, _output):
        cnn["tiles"] += int(args[0].shape[0])
        cnn["batches"] += 1

    handle = net.encoder.tiles.input.register_forward_hook(count_input)
    config = {**CONFIG, "seed": seed}
    try:
        with torch.no_grad():
            started = time.perf_counter()
            rows, counts = run_batch(panel, net, objective, 0, config)
            seconds = time.perf_counter() - started
    finally:
        handle.remove()
    if state_digest(net, objective) != before:
        raise ValueError("search changed model/objective weights")
    if cache is not None and cnn["tiles"] != cache.computed_tiles:
        raise ValueError("CNN-input hook disagrees with cache misses")
    return dict(
        mode=mode,
        seed=seed,
        config=config,
        rows=rows,
        counts=dict(counts),
        seconds=seconds,
        actual_cnn=cnn,
        weights_unchanged=True,
        cache=None
        if cache is None
        else dict(
            logical_tiles=cache.logical_tiles,
            computed_tiles=cache.computed_tiles,
            hits=cache.hits,
            invalidations=cache.invalidations,
            retained=len(cache.entries),
        ),
    )


def replay_saved(panel, result):
    episodes = steps = scores = solved = 0
    for root, row in zip(panel, result["rows"], strict=True):
        instance = M.Instance(**{**root["instance"], "word": tuple(root["instance"]["word"])})
        if (row["condition"], row["cc_weight"]) != (root["condition"], instance.cc_weight):
            raise ValueError("saved root identity differs")
        for episode in row["episodes"]:
            count, scored, _ = audit_episode(instance, episode, result["config"])
            episodes, steps, scores = episodes + 1, steps + count, scores + scored
        if row["witness"] is not None:
            witness = UnknotWitness.from_dict(row["witness"])
            witness.verify()
            if (witness.start.word, witness.start.strands) != (instance.word, instance.strands):
                raise ValueError("saved witness root differs")
            if (
                witness.moves > instance.cap_moves
                or witness.crossing_changes > instance.cap_crossing_changes
            ):
                raise ValueError("saved witness exceeds resources")
            if (
                row["L10"] != 10 * witness.crossing_changes + witness.moves
                or row["L1000"] != 1000 * witness.crossing_changes + witness.moves
            ):
                raise ValueError("saved witness cost differs")
            solved += 1
        if row["solved"] != (row["witness"] is not None):
            raise ValueError("saved solution flag differs")
    counts = result["counts"]
    if (episodes, steps, scores, solved) != (
        counts["episodes"],
        counts["continuation_moves"],
        counts["policy_action_scores"],
        counts["solved_conditions"],
    ):
        raise ValueError("saved replay counts differ")
    return dict(episodes=episodes, steps=steps, action_scores=scores, solved_conditions=solved)


def write_new(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    gate_path = Path("artifacts/raster-cache-full-network-gate-v1/report.json")
    if file_sha(gate_path) != "4c2323439bd19d90da9d8d31935ed904fca9dc4e1606cfbc7950c9010838f3bd":
        raise ValueError("full-network gate identity differs")
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed":
        raise ValueError("full-network gate did not pass")
    pins = dict(gate["inputs"])
    for path in [
        gate_path,
        Path(__file__),
        Path("scripts/raster_cost_search.py"),
        Path("scripts/pretrained_raster_policy.py"),
        Path("scripts/audit_cost_mcts_continuations.py"),
        Path("research/114-raster-cache-paired-search-gate.md"),
        Path("tests/test_raster_cache_search_probe.py"),
    ]:
        pins[str(path.resolve())] = file_sha(path)
    if any(file_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("inherited source changed")
    args.out.mkdir(parents=True, exist_ok=False)
    panel, comparisons, saved = roots(), [], []
    write_new(args.out / "inputs.json", dict(inputs=pins, roots=panel, seeds=SEEDS, config=CONFIG))
    for arm_index, arm in enumerate(CHECKPOINTS):
        checkpoint = (
            Path("artifacts/pretrained-raster-log-training-pair-v1") / arm / "checkpoint.pt"
        )
        net, objective = load_network(checkpoint, arm)
        with torch.no_grad():
            for _ in range(2):
                net.encoder(torch.tensor([[1]]), torch.tensor([1]), torch.tensor([2]))
        for index, seed in enumerate(SEEDS):
            order = (
                ("uncached", "cached") if (arm_index + index) % 2 == 0 else ("cached", "uncached")
            )
            pair = {}
            for mode in order:
                result = measure(copy.deepcopy(net), copy.deepcopy(objective), mode, seed, panel)
                path = args.out / f"{arm}-{seed}-{mode}.json"
                write_new(path, result)
                reloaded = json.loads(path.read_text())
                replay = replay_saved(panel, reloaded)
                saved.append(dict(path=str(path.resolve()), sha256=file_sha(path), replay=replay))
                pair[mode] = reloaded
            comparison = dict(
                arm=arm,
                seed=seed,
                run_order=order,
                **compare_runs(pair["uncached"], pair["cached"]),
            )
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    if any(file_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("source/checkpoint changed during gate")
    if any(file_sha(record["path"]) != record["sha256"] for record in saved):
        raise ValueError("saved search artifact changed")
    report = dict(
        status="passed" if all(p["passed"] for p in comparisons) else "diverged",
        inputs=pins,
        roots=panel,
        comparisons=comparisons,
        runs=saved,
        torch_version=torch.__version__,
        optimizer_updates=0,
        scope="Tiny synthetic paired-search engineering gate; no scientific promotion",
    )
    write_new(args.out / "report.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
