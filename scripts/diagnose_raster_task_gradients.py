"""Protocol118: fixed TRAIN-only gradients, no optimizer or evaluation changes."""

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha, write_new
from graph_metric_sampling import load_graph_sampler
from mcts_joint_training import load_metric_samplers, load_policy_targets
from pretrained_raster_policy import policy_backward, reference_preferences
from probe_raster_cache_full_network import ARCHITECTURE, state_digest
from probe_raster_cache_search import load_network
from profile_pretrained_raster_training import backward_metric
from raster_log_training import CONFIG, CORPUS, GRAPH, SUPERVISION, digest, draw_batch
from trajectory_log_preference import relative_backward

RUN = Path("artifacts/pretrained-raster-log-training-pair-v1")
PROTOCOL = Path("research/118-raster-training-gradient-diagnostic-protocol.md")
GROUP_SIZES = [334, 52, 74, 54]
INITIAL_SHA = {
    "q254": "953228e82a12deba1beabdc4f47aec2f8520638a9efb2f4348f50ec15d25ad55",
    "random": "bc07c0408cbf3e6de8d9cde1c6146473a13369025cd840ba2b74441c2a98cd1a",
}


def block(name):
    if name.startswith("net.encoder.tiles."):
        return "local_raster"
    if name.startswith("net.encoder.heads."):
        return "global_" + name.split(".")[3]
    if name.startswith("net.encoder."):
        return "shared_graph_pool"
    if name.startswith("net.position."):
        return "position"
    if name.startswith("net.head."):
        return "policy_head"
    if name.startswith("objective."):
        return "metric_objective"
    raise ValueError(f"unclassified parameter {name}")


def named_parameters(net, objective):
    return [
        (prefix + name, parameter)
        for prefix, module in (("net.", net), ("objective.", objective))
        for name, parameter in module.named_parameters()
    ]


def gradient_snapshot(parameters):
    vectors, layout, offset = [], [], 0
    for name, parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise ValueError("nonfinite gradient")
        vector = torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
        vectors.append(vector.cpu().reshape(-1))
        layout.append(
            dict(
                name=name,
                shape=list(parameter.shape),
                block=block(name),
                start=offset,
                stop=offset + parameter.numel(),
                absent=gradient is None,
            )
        )
        offset += parameter.numel()
    return torch.cat(vectors), layout


def vector_stats(vectors):
    if not vectors or len({v.shape for v in vectors.values()}) != 1:
        raise ValueError("matching nonempty vectors required")
    vectors = {key: value.double() for key, value in vectors.items()}
    if any(v.ndim != 1 or not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError("finite one-dimensional vectors required")
    norms = {key: float(v.norm()) for key, v in vectors.items()}
    pairs = {}
    for a, b in itertools.combinations(vectors, 2):
        dot = float(torch.dot(vectors[a], vectors[b]))
        denom = norms[a] * norms[b]
        pairs[f"{a}__{b}"] = dict(dot=dot, cosine=dot / denom if denom else None)
    return dict(norms=norms, pairs=pairs)


def summarize(vectors, layout):
    count = next(iter(vectors.values())).numel()
    covered = [i for p in layout for i in range(p["start"], p["stop"])]
    if covered != list(range(count)):
        raise ValueError("parameter layout does not partition gradient vector")
    result = {"all": vector_stats(vectors), "blocks": {}, "sums": {}}
    for group in sorted({p["block"] for p in layout}):
        indices = [i for p in layout if p["block"] == group for i in range(p["start"], p["stop"])]
        result["blocks"][group] = vector_stats({key: v[indices] for key, v in vectors.items()})
    base = vectors["metric"].double() + vectors["positive"].double()
    sums = {"ordinary": base}
    sums.update(
        {key: base + value.double() for key, value in vectors.items() if key.startswith("relative")}
    )
    for name, value in sums.items():
        norm = float(value.norm())
        result["sums"][name] = dict(
            norm_before_clip=norm,
            hypothetical_clip_factor=min(1.0, CONFIG["grad_clip"] / (norm + 1e-6)),
            dot_with_metric=float(torch.dot(value, vectors["metric"].double())),
            dot_with_positive=float(torch.dot(value, vectors["positive"].double())),
        )
    return result


def request_record(request):
    instance, state, actions = request
    return dict(instance=instance.to_dict(), state=asdict(state), actions=list(actions))


def provenance(pgx_root):
    pins, manifests = {}, {}

    def add(name, expected):
        name = str(Path(name).resolve())
        if name in pins and pins[name] != expected:
            raise ValueError("conflicting provenance")
        pins[name] = expected

    for arm in INITIAL_SHA:
        path = RUN / arm / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest["status"] != "completed" or manifest["completed_steps"] != CONFIG["steps"]:
            raise ValueError("require completed protocol97 training")
        if manifest["config"] != {**CONFIG, "arm": arm}:
            raise ValueError("training config differs")
        for section in ("inputs", "sources"):
            for name, expected in manifest[section].items():
                add(name, expected)
        add(path, sha(path))
        add(RUN / arm / "initial.pt", INITIAL_SHA[arm])
        add(RUN / arm / "checkpoint.pt", manifest["checkpoint_sha256"])
        manifests[arm] = manifest
    # Freeze the diagnostic's transitive local Python sources as well as originals.
    for root in (Path("scripts"), Path("src/rf_knots"), pgx_root / "src"):
        for path in root.rglob("*.py"):
            add(path, sha(path))
    for path in (PROTOCOL, Path("tests/test_raster_task_gradients.py")):
        add(path, sha(path))
    check_pins(pins)
    return pins, manifests


def measure(net, objective, rows, positives, preferences, out):
    net.train()
    objective.train()
    parameters = named_parameters(net, objective)
    before = state_digest(net, objective)
    vectors, layouts, losses, durations = {}, {}, {}, {}
    tasks = ["metric", "positive", *[f"relative{i}" for i in range(4)]]
    for task in tasks:
        net.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)
        started = time.perf_counter()
        if task == "metric":
            losses[task] = dict(metric_loss=backward_metric(net, objective, rows, 1))
        elif task == "positive":
            losses[task] = policy_backward(
                net, positives, None, microbatch=1, positive_weight=CONFIG["positive_weight"]
            )
        else:
            losses[task] = relative_backward(
                net,
                preferences[int(task[-1])],
                microbatch=1,
                weight=CONFIG["relative_weight"] * CONFIG["relative_every"],
            )
        vectors[task], layouts[task] = gradient_snapshot(parameters)
        durations[task] = time.perf_counter() - started
        if not all(math.isfinite(v) for v in losses[task].values()):
            raise ValueError("nonfinite loss diagnostic")
        if state_digest(net, objective) != before:
            raise ValueError("gradient diagnostic changed model/objective state")
        print(
            json.dumps(
                dict(
                    case=out.name,
                    component=task,
                    seconds=durations[task],
                    norm=float(vectors[task].double().norm()),
                )
            ),
            flush=True,
        )
    for layout in layouts.values():
        if [{k: v for k, v in p.items() if k != "absent"} for p in layout] != [
            {k: v for k, v in p.items() if k != "absent"} for p in layouts["metric"]
        ]:
            raise ValueError("parameter layout changed")
    path = out / "gradients.pt"
    with path.open("xb") as stream:
        torch.save(vectors, stream)
    summary = summarize(vectors, layouts["metric"])
    result = dict(
        status="completed",
        vectors_sha256=sha(path),
        layout=layouts,
        unweighted_losses=losses,
        seconds=durations,
        statistics=summary,
        state_sha256=before,
        weights_unchanged=True,
        optimizer_steps=0,
    )
    write_new(out / "report.json", result)
    net.zero_grad(set_to_none=True)
    objective.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgx-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(CONFIG["seed"])
    sys.path.insert(0, str(args.pgx_root.resolve() / "src"))
    if args.out.exists():
        raise ValueError("refusing existing diagnostic output")
    pins, manifests = provenance(args.pgx_root)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "inputs.json",
        dict(
            protocol=118,
            pid=os.getpid(),
            inputs=pins,
            optimizer_steps=0,
            device="cpu",
            threads=1,
            torch_version=torch.__version__,
        ),
    )
    positives, groups = load_policy_targets(SUPERVISION)
    if [len(g) for g in groups] != GROUP_SIZES:
        raise ValueError("relative group coverage differs")
    original, suffix = load_metric_samplers(CORPUS, SUPERVISION)
    graph = load_graph_sampler(GRAPH)
    rng, traces = random.Random(CONFIG["seed"]), []
    dummy = [(group, torch.zeros(len(group))) for group in groups]
    for _ in range(16):
        rows, chosen, _, trace, group_index = draw_batch(
            rng, original, suffix, graph, positives, dummy
        )
        traces.append(trace)
    write_new(
        args.out / "selection.json",
        dict(
            draw_number=16,
            seed=CONFIG["seed"],
            draw_hashes=traces,
            selected_group=group_index,
            metrics=[asdict(r) for r in rows],
            positives=[dict(request=request_record(r), good=list(good)) for r, good in chosen],
            full_relative_groups=[
                [dict(request=request_record(r), index=i, coefficient=c) for r, i, c in g]
                for g in groups
            ],
        ),
    )
    del original, suffix, graph, positives, dummy
    reports = {}
    for arm in INITIAL_SHA:
        net, objective = load_network(RUN / arm / "checkpoint.pt", arm)
        initial = torch.load(RUN / arm / "initial.pt", map_location="cpu", weights_only=False)
        if initial["architecture"] != ARCHITECTURE or initial["arm"] != arm:
            raise ValueError("initial architecture/arm mismatch")
        net.load_state_dict(initial["model"], strict=True)
        objective.load_state_dict(initial["objective"], strict=True)
        before = state_digest(net, objective)
        preferences = reference_preferences(net, groups, microbatch=1)
        refs = [old.tolist() for _, old in preferences]
        if digest(refs) != manifests[arm]["reference_log_probabilities_sha256"]:
            raise ValueError("own initial reference hash changed")
        if state_digest(net, objective) != before:
            raise ValueError("reference computation changed weights")
        write_new(args.out / f"{arm}-references.json", dict(references=refs, sha256=digest(refs)))
        for stage in ("initial", "final"):
            if stage == "final":
                net, objective = load_network(RUN / arm / "checkpoint.pt", arm)
            directory = args.out / f"{arm}-{stage}"
            directory.mkdir()
            measure(net, objective, rows, chosen, preferences, directory)
            reports[directory.name] = sha(directory / "report.json")
    check_pins(pins)
    write_new(
        args.out / "report.json",
        dict(
            status="completed",
            protocol=118,
            cases=reports,
            inputs_sha256=sha(args.out / "inputs.json"),
            selection_sha256=sha(args.out / "selection.json"),
            optimizer_steps=0,
            references={arm: sha(args.out / f"{arm}-references.json") for arm in INITIAL_SHA},
            weights_unchanged=True,
            diagnostic_only=True,
            test_scoring=False,
        ),
    )
    print(json.dumps(dict(status="completed", cases=len(reports))), flush=True)


if __name__ == "__main__":
    main()
