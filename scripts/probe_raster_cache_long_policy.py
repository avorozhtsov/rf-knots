"""Protocol116 synthetic edit-chain full-policy timings; no MCTS integration."""

import argparse
import copy
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from pretrained_raster_graph import GLOBALS
from probe_raster_cache_full_network import CHECKPOINTS, full_output, state_digest
from probe_raster_cache_search import load_network, write_new
from probe_raster_tile_cache import TileCacheProbe, file_sha, synthetic_word

from rf_knots import mcts_loop as M
from rf_knots import reference
from rf_knots.actions import CROSSING_CHANGE, INSERT, STABILIZE_POS

SIZES = ((64, 5), (256, 17), (1024, 65))
FAMILIES = ("structured", "random-inserted-trefoil")


def initial_word(family, length, strands):
    if family == "structured":
        return tuple(synthetic_word(length, strands))
    if family != "random-inserted-trefoil":
        raise ValueError("unknown synthetic family")
    rng = random.Random(20261060 + length + strands)
    word = [1, 1, 1, *range(2, strands)]
    if length < len(word) or (length - len(word)) % 2:
        raise ValueError("invalid stabilized-trefoil length")
    while len(word) < length:
        position = rng.randrange(len(word) + 1)
        letter = rng.randint(1, strands - 1) * rng.choice((-1, 1))
        word[position:position] = [letter, -letter]
    return tuple(word)


def edit_chain(family, length, strands):
    word = initial_word(family, length, strands)
    instance = M.Instance(
        f"cost-probe-{family}-{length}",
        word,
        strands,
        4,
        cap_moves=128,
        growth_len=8,
        growth_strands=4,
        cc_weight=1000,
    )
    state = M.SearchState(word, strands, 0, 0)
    states, applied = [state], []
    edits = [
        (CROSSING_CHANGE, length // 3),
        (CROSSING_CHANGE, 2 * length // 3),
        (INSERT, length // 2),
        (STABILIZE_POS, 0),
    ]
    for kind, position in edits:
        action = instance.spec.encode(kind, position, 1, 1)
        if not reference.is_legal(instance.spec, state.word, state.strands, action, True):
            raise ValueError("illegal synthetic edit")
        new_word, new_n = reference.apply(instance.spec, state.word, state.strands, action)
        state = M.SearchState(
            tuple(new_word),
            new_n,
            state.used_cc + int(kind == CROSSING_CHANGE),
            state.used_moves + 1,
        )
        if reference.num_components(state.word, state.strands) != 1:
            raise ValueError("edit fixture is not a knot")
        states.append(state)
        applied.append(action)
    batches, descriptions = [], []
    for step, state in enumerate(states):
        size = len(state.word)
        actions = [instance.spec.encode(CROSSING_CHANGE, p) for p in (0, size // 2, size - 1)]
        actions.extend(
            [
                instance.spec.encode(INSERT, 0, 1, 1),
                instance.spec.encode(INSERT, size - 1, state.strands - 1, -1),
                instance.spec.encode(STABILIZE_POS),
            ]
        )
        if len(set(actions)) != 6 or not all(
            reference.is_legal(
                instance.spec,
                state.word,
                state.strands,
                a,
                state.used_cc < instance.cap_crossing_changes,
            )
            for a in actions
        ):
            raise ValueError("invalid legal-candidate subset")
        head = step % size
        batches.append(
            (M.build_batch([(instance, state, actions)], budget_dim=5), torch.tensor([head]))
        )
        descriptions.append(
            dict(
                word=list(state.word),
                strands=state.strands,
                used_cc=state.used_cc,
                used_moves=state.used_moves,
                actions=actions,
                head=head,
            )
        )
    return batches, dict(
        family=family,
        length=length,
        strands=strands,
        instance=instance.to_dict(),
        states=descriptions,
        applied_actions=applied,
    )


def measure(net, objective, batches, mode):
    before = state_digest(net, objective)
    cache = TileCacheProbe(net.encoder.tiles) if mode == "cached" else None
    if cache is not None:
        net.encoder.tiles.forward = cache
    actual_tiles = [0]

    def count(_module, args, _output):
        actual_tiles[0] += int(args[0].shape[0])

    handle = net.encoder.tiles.input.register_forward_hook(count)
    measurements, outputs = [], []
    try:
        with torch.no_grad():
            for step, (batch, head) in enumerate(batches):
                tiles_before = actual_tiles[0]
                started = time.perf_counter()
                output = full_output(net, batch, head)
                elapsed = time.perf_counter() - started
                outputs.append(output)
                measurements.append(
                    dict(
                        step=step,
                        seconds=elapsed,
                        actual_cnn_tiles=actual_tiles[0] - tiles_before,
                        retained_entries=0 if cache is None else len(cache.entries),
                        retained_tensor_bytes=0
                        if cache is None
                        else sum(v.numel() * v.element_size() for v in cache.entries.values()),
                        retained_key_bytes=0
                        if cache is None
                        else sum(len(k[0]) for k in cache.entries),
                    )
                )
    finally:
        handle.remove()
    if state_digest(net, objective) != before:
        raise ValueError("model/objective changed")
    if cache is not None and (actual_tiles[0] != cache.computed_tiles or len(cache.entries) > 512):
        raise ValueError("cache count/bound mismatch")
    return measurements, outputs


def compare_outputs(reference_outputs, cached_outputs):
    rows, passed = [], True
    for step, (left, right) in enumerate(zip(reference_outputs, cached_outputs, strict=True)):
        if not torch.equal(left["effective"], right["effective"]):
            raise ValueError("legal mask differs")
        errors, failures = {}, []
        for name in ("tokens", *GLOBALS, "logits", "probabilities"):
            a, b = left[name], right[name]
            if name == "logits":
                a, b = a[left["effective"]], b[right["effective"]]
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("nonfinite outputs")
            errors[name] = float((a - b).abs().max())
            if not torch.allclose(a, b, rtol=1e-5, atol=1e-5):
                failures.append(name)
        passed = passed and not failures
        rows.append(
            dict(
                step=step,
                errors=errors,
                failures=failures,
                argmax_changed=bool(left["logits"].argmax() != right["logits"].argmax()),
            )
        )
    return rows, bool(passed)


def summarize(pairs):
    result = []
    for arm in CHECKPOINTS:
        for family in FAMILIES:
            for length, strands in SIZES:
                selected = [
                    p
                    for p in pairs
                    if (p["arm"], p["family"], p["length"]) == (arm, family, length)
                ]
                if len(selected) != 3:
                    raise ValueError("incomplete timing repetitions")
                row = dict(arm=arm, family=family, length=length, strands=strands)
                for label, indices in (("cold", [0]), ("edits", [1, 2, 3, 4]), ("total", range(5))):
                    ratios = [
                        sum(p["measurements"]["uncached"][i]["seconds"] for i in indices)
                        / sum(p["measurements"]["cached"][i]["seconds"] for i in indices)
                        for p in selected
                    ]
                    row[label] = dict(
                        median_uncached_over_cached=statistics.median(ratios),
                        minimum=min(ratios),
                        maximum=max(ratios),
                    )
                result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    torch.set_num_threads(1)
    gate_path = Path("artifacts/raster-cache-paired-search-gate-v1/report.json")
    if file_sha(gate_path) != "31b0bc8530fe03b355283d5ae3167a49bb8c39137b7a181ff8e44027ced077c1":
        raise ValueError("paired-search gate changed")
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "passed":
        raise ValueError("paired-search gate failed")
    pins = dict(gate["inputs"])
    for path in (
        gate_path,
        Path(__file__),
        Path("research/116-raster-cache-long-policy-cost-gate.md"),
        Path("tests/test_raster_cache_long_policy_probe.py"),
    ):
        pins[str(path.resolve())] = file_sha(path)
    if any(file_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("source/checkpoint identity differs")
    workloads = {
        (family, length): edit_chain(family, length, strands)
        for family in FAMILIES
        for length, strands in SIZES
    }
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json", dict(inputs=pins, workloads=[v[1] for v in workloads.values()])
    )
    pairs, artifacts = [], []
    for ai, arm in enumerate(CHECKPOINTS):
        path = Path("artifacts/pretrained-raster-log-training-pair-v1") / arm / "checkpoint.pt"
        net, objective = load_network(path, arm)
        with torch.no_grad():
            for _ in range(2):
                net.encoder(torch.tensor([[1]]), torch.tensor([1]), torch.tensor([2]))
        for fi, family in enumerate(FAMILIES):
            for si, (length, strands) in enumerate(SIZES):
                batches, _ = workloads[family, length]
                for repetition in range(3):
                    order = (
                        ("uncached", "cached")
                        if (ai + fi + si + repetition) % 2 == 0
                        else ("cached", "uncached")
                    )
                    measurements, outputs = {}, {}
                    for mode in order:
                        measurements[mode], outputs[mode] = measure(
                            copy.deepcopy(net), copy.deepcopy(objective), batches, mode
                        )
                    comparisons, passed = compare_outputs(outputs["uncached"], outputs["cached"])
                    pair = dict(
                        arm=arm,
                        family=family,
                        length=length,
                        strands=strands,
                        repetition=repetition,
                        mode_order=order,
                        measurements=measurements,
                        comparisons=comparisons,
                        passed=passed,
                    )
                    path = args.out / f"pair-{len(pairs):03d}.json"
                    write_new(path, pair)
                    pairs.append(pair)
                    artifacts.append(dict(path=str(path.resolve()), sha256=file_sha(path)))
                print(
                    json.dumps(
                        dict(arm=arm, family=family, length=length, completed_pairs=len(pairs))
                    ),
                    flush=True,
                )
    if any(file_sha(path) != digest for path, digest in pins.items()):
        raise ValueError("source/checkpoint changed during benchmark")
    report = dict(
        status="passed" if all(p["passed"] for p in pairs) else "numerical_mismatch",
        inputs=pins,
        artifacts=artifacts,
        summary=summarize(pairs),
        torch_version=torch.__version__,
        rtol=1e-5,
        atol=1e-5,
        scope="Synthetic full-policy edit-chain cost, not MCTS or knot quality",
    )
    write_new(args.out / "report.json", report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
