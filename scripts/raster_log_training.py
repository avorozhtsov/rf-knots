"""Protocol97 full-group sampling/lineage; original94 stays frozen and stopped."""

import json
from dataclasses import asdict
from pathlib import Path

from check_qrun_common_witnesses import check_pins, sha
from raster_pair_training import (
    CONFIG as OLD_CONFIG,
)
from raster_pair_training import (
    CORPUS,
    GRAPH,
    SUPERVISION,
    digest,
    matched_initializations,
)
from raster_pair_training import (
    inputs_and_sources as previous_inputs,
)
from run_strand_graph_pilot import pair_row

# Re-export the fixed data/initialization helpers for the dedicated97 runner.
__all__ = [
    "CONFIG",
    "CORPUS",
    "GRAPH",
    "SUPERVISION",
    "digest",
    "draw_batch",
    "inputs_and_sources",
    "matched_initializations",
    "relative_due",
]
CONFIG = {key: value for key, value in OLD_CONFIG.items() if key not in ("clip", "relative_rows")}
CONFIG.update(protocol=97, relative_every=16, relative_objective="full-group-scaled-logistic")
GATE = Path("artifacts/trajectory-log-preference-real-gate-v1/report.json")


def inputs_and_sources():
    pins, sources = previous_inputs()
    gate = json.loads(GATE.read_text())
    if gate["status"] != "passed" or len(gate["cases"]) != 8 or gate["optimizer_steps"] != 0:
        raise ValueError("require complete two-state/all-group real-network gradient gate")
    for name, expected in gate["inputs"].items():
        if name in pins and pins[name] != expected:
            raise ValueError("conflicting gradient-gate lineage")
        pins[name] = expected
    pins[str(GATE)] = sha(GATE)
    for path in (
        Path(__file__),
        Path("scripts/train_raster_log_pair.py"),
        Path("scripts/run_raster_log_pair.py"),
        Path("scripts/trajectory_log_preference.py"),
        Path("tests/test_raster_log_training.py"),
        Path("research/97-raster-log-preference-paired-training.md"),
    ):
        sources[str(path.resolve())] = sha(path)
    check_pins(pins)
    check_pins(sources)
    return pins, sources


def relative_due(step):
    if step < 0:
        raise ValueError("negative optimizer step")
    return (step + 1) % CONFIG["relative_every"] == 0


def validate_relative_counts(counts, groups, completed):
    if (
        not isinstance(counts, list)
        or len(counts) != groups
        or any(type(n) is not int or n < 0 for n in counts)
        or sum(counts) != completed // CONFIG["relative_every"]
    ):
        raise ValueError("resumed relative-group counters differ")


def draw_batch(rng, original, suffix, graph, positives, preferences):
    metrics = (
        original.sample(rng, CONFIG["old_pairs"])
        + suffix.sample(rng, CONFIG["suffix_pairs"])
        + graph.sample(rng, CONFIG["graph_pairs"])
    )
    metrics.append(pair_row(-1, "mirror", ((1, 1, 1), 2), ((-1, -1, -1), 2), 2, 2, 6))
    chosen = [rng.choice(rng.choice(positives)) for _ in range(CONFIG["positive_states"])]
    group_index = rng.randrange(len(preferences))
    relative = preferences[group_index]
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
    }
    return metrics, chosen, relative, digest(trace), group_index
