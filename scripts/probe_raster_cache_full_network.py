"""Protocol112 synthetic full-network parity gate, isolated from active search."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from pretrained_raster_graph import GLOBALS, RasterGraphPolicy
from pretrained_raster_tiles import PretrainedRasterTiles
from probe_raster_tile_cache import TileCacheProbe, file_sha, synthetic_word

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    KIND_NAMES,
    PASS,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
)
from rf_knots.embedding_metric import MetricEmbeddingObjective
from rf_knots.site_head import SiteHead

CHECKPOINTS = {
    "q254": "c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f",
    "random": "17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6",
}
ARCHITECTURE = dict(
    width=64,
    embedding_dim=64,
    graph_rounds=2,
    tile_rows=12,
    tile_columns=7,
    tile_width=64,
    tile_batch=64,
)


def candidate_subset(instance, state):
    spec = instance.spec
    candidates = [spec.encode(k) for k in (DESTABILIZE, STABILIZE_POS, STABILIZE_NEG, PASS)]
    length = len(state.word)
    positions = range(length) if length <= 16 else sorted({0, length // 2, length - 1})
    for position in positions:
        candidates.extend(
            spec.encode(k, position) for k in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE)
        )
        for generator in sorted({1, max(1, state.strands - 1)}):
            candidates.extend(spec.encode(INSERT, position, generator, sign) for sign in (-1, 1))
    actions = tuple(
        a
        for a in dict.fromkeys(candidates)
        if reference.is_legal(
            spec, state.word, state.strands, a, state.used_cc < instance.cap_crossing_changes
        )
    )
    if not actions:
        raise ValueError("synthetic fixture has no legal candidates")
    return actions


def fixtures(weight):
    base = (1, -1, 1, 2, 1, 3, 1)
    words = [
        ("empty", (), 1),
        ("single", (1,), 2),
        ("negative", (-1, -1, -1), 2),
        ("all-kinds", base, 4),
        ("crop-boundary", (1, 11, -12, 6, 12), 13),
        ("long64", tuple(synthetic_word(64, 33)), 33),
        ("long256", tuple(synthetic_word(256, 65)), 65),
    ]
    result = []
    for label, word, strands in words:
        instance = M.Instance(
            label, word, strands, 2, cap_moves=128, growth_len=8, growth_strands=4, cc_weight=weight
        )
        state = M.SearchState(word, strands, 0, 0)
        result.append((label, (instance, state, candidate_subset(instance, state))))
    instance, state, actions = next(row for name, row in result if name == "all-kinds")
    chosen = {}
    for action in actions:
        chosen.setdefault(instance.spec.decode(action)[0], action)
    if set(chosen) != set(range(9)) - {PASS}:
        raise ValueError("fixture does not expose all eight non-PASS action kinds")
    for kind, action in sorted(chosen.items()):
        word, strands = reference.apply(instance.spec, state.word, state.strands, action)
        successor = M.SearchState(tuple(word), strands, int(kind == CROSSING_CHANGE), 1)
        result.append(
            (
                f"after-{KIND_NAMES[kind]}",
                (instance, successor, candidate_subset(instance, successor)),
            )
        )
    # PASS is legal only when no other move exists, not on the all-kinds word.
    constrained = M.Instance(
        "pass-only",
        (1, 1, 1),
        2,
        0,
        cap_moves=128,
        growth_len=0,
        growth_strands=0,
        cc_weight=weight,
    )
    start = M.SearchState(constrained.word, 2, 0, 0)
    actions = candidate_subset(constrained, start)
    if actions != (constrained.spec.encode(PASS),):
        raise ValueError("PASS fixture is not exclusively terminal-forfeit")
    word, strands = reference.apply(constrained.spec, start.word, start.strands, actions[0])
    result.append(("pass-only", (constrained, start, actions)))
    result.append(("after-PASS", (constrained, M.SearchState(tuple(word), strands, 0, 1), actions)))
    return result


def state_digest(net, objective):
    digest = hashlib.sha256()
    for prefix, module in (("net", net), ("objective", objective)):
        for name, tensor in module.state_dict().items():
            digest.update((prefix + name + str(tuple(tensor.shape)) + str(tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def full_output(net, batch, heads):
    encoded = net.encoder.encode_sites(batch["word"], batch["lengths"], batch["strands"])
    logits, effective = net.score_encoded(
        encoded, batch["sites"], batch["site_mask"], batch["budgets"], heads=heads
    )
    if not torch.equal(effective, batch["site_mask"]):
        raise ValueError("network masked a legal candidate")
    probabilities = SiteHead.masked_log_policy(logits, effective).exp()
    return {**encoded, "logits": logits, "effective": effective, "probabilities": probabilities}


def run_arm(path, arm):
    from pgx_mcts_bench.networks import MaskedAxialCylinderResidualBlock

    if file_sha(path) != CHECKPOINTS[arm]:
        raise ValueError("trained checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    cached = copy.deepcopy(net)
    probe = TileCacheProbe(cached.encoder.tiles)
    # Ephemeral override on this private in-memory copy only; state_dict stays identical.
    cached.encoder.tiles.forward = probe
    before = state_digest(net, objective)
    if state_digest(cached, objective) != before:
        raise ValueError("cached/reference weights differ")
    records, failures, argmax_differences = [], [], 0
    with torch.no_grad():
        zero, lengths, strands = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.tensor([0]),
            torch.tensor([1]),
        )
        anchors = [m.encoder(zero, lengths, strands)["knot"][0] for m in (net, cached)]
        for weight in (10, 1000):
            requests = fixtures(weight)
            for begin in range(0, len(requests), 2):
                chunk = requests[begin : begin + 2]
                batch = M.build_batch([r for _, r in chunk], budget_dim=5)
                for mode in ("absent", "zero", "last"):
                    heads = (
                        None
                        if mode == "absent"
                        else (
                            torch.zeros_like(batch["lengths"])
                            if mode == "zero"
                            else (batch["lengths"] - 1).clamp_min(0)
                        )
                    )
                    outputs = [full_output(m, batch, heads) for m in (net, cached)]
                    for output, anchor in zip(outputs, anchors, strict=True):
                        cosine = (output["knot"] * anchor).sum(-1).clamp(-1, 1)
                        output["distance"] = objective.scale("cc") * objective.cc_distance(cosine)
                    if not torch.equal(outputs[0]["effective"], outputs[1]["effective"]):
                        raise ValueError("effective action mask differs")
                    for i, (label, _) in enumerate(chunk):
                        errors = {}
                        for name in ("tokens", *GLOBALS, "logits", "probabilities", "distance"):
                            left, right = outputs[0][name][i], outputs[1][name][i]
                            if name == "logits":
                                left, right = (
                                    left[outputs[0]["effective"][i]],
                                    right[outputs[1]["effective"][i]],
                                )
                            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                                raise ValueError("nonfinite output")
                            errors[name] = float((left - right).abs().max())
                            if not torch.allclose(left, right, rtol=1e-5, atol=1e-5):
                                failures.append(
                                    dict(label=label, weight=weight, head=mode, field=name)
                                )
                        mismatch = int(
                            outputs[0]["logits"][i].argmax() != outputs[1]["logits"][i].argmax()
                        )
                        argmax_differences += mismatch
                        records.append(
                            dict(
                                label=label,
                                weight=weight,
                                head=mode,
                                errors=errors,
                                argmax_changed=bool(mismatch),
                            )
                        )
    if state_digest(net, objective) != before or state_digest(cached, objective) != before:
        raise ValueError("gate mutated model/objective weights")
    return dict(
        arm=arm,
        status="failed" if failures else "passed",
        cases=records,
        failures=failures,
        argmax_differences=argmax_differences,
        weights_unchanged=True,
        logical_tiles=probe.logical_tiles,
        computed_tiles=probe.computed_tiles,
        cache_hits=probe.hits,
        cache_invalidations=probe.invalidations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    run = Path("artifacts/pretrained-raster-log-training-pair-v1")
    paths = [
        Path(__file__),
        Path("scripts/probe_raster_tile_cache.py"),
        Path("scripts/pretrained_raster_graph.py"),
        Path("scripts/pretrained_raster_tiles.py"),
        Path("research/112-raster-cache-full-network-gate.md"),
        Path("tests/test_raster_cache_full_network_probe.py"),
        Path("tests/test_probe_raster_tile_cache.py"),
        *Path("src/rf_knots").glob("*.py"),
        *args.pgx_root.joinpath("src").rglob("*.py"),
        *(run / arm / "checkpoint.pt" for arm in CHECKPOINTS),
    ]
    pins = {str(p.resolve()): file_sha(p) for p in paths}
    results = []
    for arm in CHECKPOINTS:
        results.append(run_arm(run / arm / "checkpoint.pt", arm))
        print(
            json.dumps({k: results[-1][k] for k in ("arm", "status", "argmax_differences")}),
            flush=True,
        )
    if any(file_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("input changed while gate ran")
    report = dict(
        status="passed" if all(r["status"] == "passed" for r in results) else "failed",
        inputs=pins,
        arms=results,
        torch_version=torch.__version__,
        rtol=1e-5,
        atol=1e-5,
        optimizer_updates=0,
        mcts_runs=0,
        scope="Synthetic full-network numerical parity; not trajectory or speedup evidence",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
