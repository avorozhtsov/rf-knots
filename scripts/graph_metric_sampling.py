"""Audited ordinary graph TRAIN pairs, balanced by class and endpoint groups."""

import json
import sqlite3
from collections import defaultdict
from dataclasses import replace

from audit_joint_policy_metric_panel import check_pins
from build_mcts_training_supervision import sha
from run_strand_graph_pilot import pair_row


class GraphMetricSampler:
    def __init__(self, entries):
        bins = defaultdict(lambda: defaultdict(list))
        for row, left, right in entries:
            lo, hi = row.spec.distance_lower, row.spec.distance_upper
            if lo not in (0, 1, 2) or hi not in (0, 1, 2):
                raise ValueError("graph bounds must be integer CC counts")
            if lo == hi == 0:
                continue
            label = f"exact-{int(lo)}" if lo == hi else "interval"
            if label not in {"exact-1", "exact-2", "interval"} or not 0 <= lo <= hi <= 2:
                raise ValueError("unsupported graph TRAIN interval")
            group = tuple(sorted(((left[0], abs(left[1])), (right[0], abs(right[1])))))
            bins[label][group].append(row)
        if set(bins) != {"exact-1", "exact-2", "interval"}:
            raise ValueError("graph sampler requires all three fixed classes")
        self.bins = [tuple(tuple(g[k]) for k in sorted(g)) for _, g in sorted(bins.items())]
        self.support = {
            label: {"groups": len(g), "records": sum(map(len, g.values()))}
            for label, g in sorted(bins.items())
        }

    def sample(self, rng, count):
        if count < 0:
            raise ValueError("negative count")
        result = []
        for _ in range(count):
            category = self.bins[int(rng.random() * len(self.bins))]
            group = category[int(rng.random() * len(category))]
            row = group[int(rng.random() * len(group))]
            swap = rng.random() < 0.5
            if swap:
                row = replace(
                    row,
                    left_word=row.right_word,
                    right_word=row.left_word,
                    left_strands=row.right_strands,
                    right_strands=row.left_strands,
                )
            result.append(row)
        return result


def load_graph_sampler(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    audit = json.loads((directory / "partition-replay-audit.json").read_text())
    if manifest["status"] != "completed" or audit["status"] != "passed" or not audit["train_ready"]:
        raise ValueError("require completed corpus and passed full replay/partition audit")
    for path in (directory / "manifest.json", directory / "corpus.sqlite"):
        if audit["inputs"].get(str(path)) != sha(path):
            raise ValueError("audit belongs to a different graph corpus")
    check_pins(audit["inputs"])
    db = sqlite3.connect((directory / "corpus.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    states = {
        rep: (tuple(json.loads(word)), n, (int(det), sigma))
        for rep, word, n, det, sigma in db.execute("SELECT rep,word,strands,det,sigma FROM states")
    }
    entries = []
    for index, lo, hi, payload, witness_cc in db.execute(
        "SELECT t.path_index,t.lower_bound,t.upper_bound,p.source,p.witness_cc "
        "FROM training_pairs t JOIN paths p ON p.path_index=t.path_index ORDER BY t.pair_key"
    ):
        reps = json.loads(payload)["representations"]
        a, b = states[reps[0]], states[reps[-1]]
        row = pair_row(index, "ordinary-graph-general", a[:2], b[:2], lo, hi, witness_cc)
        entries.append((row, a[2], b[2]))
    db.close()
    if len(entries) != audit["summary"]["deduplicated_pairs"]:
        raise ValueError("graph pair coverage differs")
    return GraphMetricSampler(entries)


def paired_draws(rng, original, suffix, graph, positives, preferences, mode):
    if mode not in {"graph", "control"}:
        raise ValueError("unknown graph mixture mode")
    old = original.sample(rng, 95)
    short_u = suffix.sample(rng, 32)
    new = graph.sample(rng, 64)
    chosen = [rng.choice(rng.choice(positives)) for _ in range(32)]
    relative = rng.choice(preferences) if preferences else None
    metrics = (old[:31] + short_u + new) if mode == "graph" else old + short_u
    return metrics, chosen, relative
