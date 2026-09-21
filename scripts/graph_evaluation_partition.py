"""Frozen protocol69 optimizer groups and protocol70 historical query exclusions.

Reuse hash-verified replay certificates; do not call invariant groups knot IDs.
This module does not import labels from mirror-quotient graph paths.
"""

import json
from pathlib import Path

from audit_joint_policy_metric_panel import AUDITS, check_pins, historical_groups, merge_pins
from audit_ordinary_graph_pair_supply import heldout_queries
from build_mcts_training_supervision import sha
from train_graph_metric_mixture import GRAPH, SUPERVISION, inputs_and_sources

from rf_knots.embedding_replay_corpus import owner


def partition_inputs():
    pins, sources = inputs_and_sources()
    merge_pins(pins, sources)
    merge_pins(pins, {str(Path(__file__).resolve()): sha(Path(__file__))})
    historical = {}
    for name, path in AUDITS.items():
        audit = json.loads(path.read_text())
        if audit["status"] != "passed":
            raise ValueError("historical replay audit incomplete")
        merge_pins(pins, {str(path): sha(path), **audit["inputs"]})
        historical[name] = audit
    old_metric, old_rollout, mcts = historical_groups(historical)
    graph = json.loads((GRAPH / "partition-replay-audit.json").read_text())
    graph_groups = set(map(tuple, graph["summary"]["support"]["eligible_invariant_groups"]))
    index_path = SUPERVISION / "short-u-views.json"
    # These suffix rows were checked against every witness by the pinned
    # certificate audit and the actual protocol69 optimizer loader.
    if historical["certificates"]["inputs"].get(str(index_path)) != sha(index_path):
        raise ValueError("metric suffix registry not covered by certificate audit")
    index = json.loads(index_path.read_text())
    suffix = {(r["fingerprint"][0], abs(r["fingerprint"][1])) for r in index["views"]}
    groups = {
        "old_metric": old_metric,
        "old_rollout": old_rollout,
        "mcts": mcts,
        "graph_paths": graph_groups,
        "metric_suffixes_and_sentinel": suffix | {(1, 0), (3, 2)},
    }
    training = set.union(*groups.values())
    if any(owner(g) != "train" for g in training):
        raise ValueError("non-TRAIN group in audited optimizer lineage")
    # Freeze exactly the 15 panels already declared in the full corpus audit,
    # never discover newly generated panels through a moving filesystem glob.
    panels = sorted(Path(p) for p in graph["inputs"] if Path(p).name == "panel.json")
    if len(panels) != 15:
        raise ValueError("historical panel coverage differs from protocol70")
    excluded = heldout_queries([json.loads(p.read_text()) for p in panels])
    check_pins(pins)
    return pins, groups, training, excluded
