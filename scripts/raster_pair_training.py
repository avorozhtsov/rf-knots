"""Frozen protocol94 data draws and matched raster-network initialization."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from check_qrun_common_witnesses import check_pins, sha
from evaluate_common_greedy_policies import load_model
from pretrained_raster_graph import RasterGraphPolicy
from pretrained_raster_tiles import PretrainedRasterTiles
from run_strand_graph_pilot import pair_row

INITIAL = Path("artifacts/pretrained-raster-graph-initialization-v1/initialization.pt")
CORPUS = Path("artifacts/embedding-diverse-short-u-corpus-v1/corpus.json")
SUPERVISION = Path("artifacts/diverse-mcts-supervision-diamonds-v1")
GRAPH = Path("artifacts/ordinary-graph-metric-corpus-v1")
PANELS = [
    Path("artifacts/graph-mixture-fresh-short-u-test-v1/panel.json"),
    Path("artifacts/pure-braid-short-distance-test-v1/panel.json"),
]
CONFIG = {
    "steps": 2048,
    "seed": 20261047,
    "random_local_seed": 20261048,
    "device": "cpu",
    "threads": 1,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "metric_microbatch": 1,
    "policy_microbatch": 1,
    "positive_weight": 0.1,
    "relative_weight": 0.01,
    "clip": 0.2,
    "old_pairs": 7,
    "suffix_pairs": 8,
    "graph_pairs": 16,
    "positive_states": 8,
    "relative_rows": 8,
    "checkpoint_interval": 16,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def inputs_and_sources():
    pins = {}
    reports = [
        Path("artifacts/pretrained-raster-policy-search-gate-v1/report.json"),
        Path("artifacts/pretrained-raster-training-cost-v1/report.json"),
    ]
    for path in reports:
        report = json.loads(path.read_text())
        if report["status"] != "passed":
            raise ValueError("require passed real-weight integration/cost gates")
        for name, expected in report["inputs"].items():
            if name in pins and pins[name] != expected:
                raise ValueError("conflicting input provenance")
            pins[name] = expected
        pins[str(path)] = sha(path)
    for path in [
        *PANELS,
        CORPUS,
        INITIAL,
        PANELS[0].parent / "independence-audit.json",
        PANELS[1].parent / "independence-audit.json",
    ]:
        if path.name == "independence-audit.json":
            audit = json.loads(path.read_text())
            if audit["status"] != "passed":
                raise ValueError("require passed held-out panel audit")
            check_pins(audit["inputs"])
        pins[str(path)] = sha(path)
    source_paths = [
        Path(__file__),
        Path("scripts/train_pretrained_raster_pair.py"),
        Path("scripts/run_pretrained_raster_pair.py"),
        Path("tests/test_raster_pair_training.py"),
        Path("research/94-pretrained-raster-paired-training-protocol.md"),
        Path("scripts/mcts_joint_training.py"),
        Path("scripts/balanced_metric_sampling.py"),
        Path("scripts/train_mcts_policy_metric.py"),
        Path("scripts/profile_pretrained_raster_training.py"),
        Path("scripts/pretrained_raster_policy.py"),
        Path("scripts/graph_metric_sampling.py"),
        *sorted(Path("src/rf_knots").glob("*.py")),
    ]
    sources = {str(p.resolve()): sha(p) for p in source_paths}
    check_pins(pins)
    check_pins(sources)
    return pins, sources


def matched_initializations(pins):
    baseline = json.loads(Path("artifacts/qrun-baseline-input-check-v2/report.json").read_text())
    q = load_model("q254", baseline, pins)
    net = RasterGraphPolicy(PretrainedRasterTiles(q.network.representation))
    payload = torch.load(INITIAL, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"], strict=True)
    pretrained = copy.deepcopy(net.state_dict())
    # The native constructor also resets custom MaskedGroupNorm parameters;
    # calling reset_parameters only on standard torch modules would miss them.
    with torch.random.fork_rng():
        torch.manual_seed(CONFIG["random_local_seed"])
        representation = type(q.network.representation)(q.config.game, q.config.model)
        random_tiles = PretrainedRasterTiles(representation)
    net.encoder.tiles.load_state_dict(random_tiles.state_dict(), strict=True)
    random = copy.deepcopy(net.state_dict())
    differences = [key for key in pretrained if not torch.equal(pretrained[key], random[key])]
    if not differences or any(not key.startswith("encoder.tiles.") for key in differences):
        raise ValueError("initialization contrast is not confined to local raster weights")
    return net, {"q254": pretrained, "random": random}, payload["architecture"], differences


def sample_relative(rng, reference, count):
    if count < 1:
        raise ValueError("positive relative sample count required")
    group, old = reference
    if not group or old.shape != (len(group),):
        raise ValueError("invalid complete relative reference group")
    indices = [rng.randrange(len(group)) for _ in range(count)]
    scale = len(group) / count
    sampled = [(group[i][0], group[i][1], group[i][2] * scale) for i in indices]
    return (sampled, old[indices].clone()), indices


def draw_batch(rng, original, suffix, graph, positives, preferences):
    metrics = (
        original.sample(rng, CONFIG["old_pairs"])
        + suffix.sample(rng, CONFIG["suffix_pairs"])
        + graph.sample(rng, CONFIG["graph_pairs"])
    )
    metrics.append(pair_row(-1, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6))
    chosen = [rng.choice(rng.choice(positives)) for _ in range(CONFIG["positive_states"])]
    group_index = rng.randrange(len(preferences))
    relative, indices = sample_relative(rng, preferences[group_index], CONFIG["relative_rows"])
    # Exclude model-dependent reference probabilities from the data-order digest.
    trace = {
        "metrics": [asdict(r) for r in metrics],
        "positives": [
            {
                "instance": r[0][0].to_dict(),
                "state": asdict(r[0][1]),
                "actions": r[0][2],
                "good": r[1],
            }
            for r in chosen
        ],
        "relative_group": group_index,
        "relative_indices": indices,
    }
    return metrics, chosen, relative, digest(trace)
