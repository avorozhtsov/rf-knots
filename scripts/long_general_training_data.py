"""Fail-closed147 TRAIN loader and matched149 data intervention; no optimizer."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_long_general_cc_supply import CONFIG as DATA_CONFIG
from build_long_general_cc_supply import SCHEMA, group, pair_key, supply_summary
from build_mcts_training_supervision import sha
from raster_log_training import CONFIG as BASE_CONFIG
from raster_log_training import digest, draw_batch
from run_strand_graph_pilot import pair_row

from rf_knots.embedding_metric import witness_priority_weight
from rf_knots.embedding_replay_corpus import owner

CONFIG = {
    **BASE_CONFIG,
    "protocol": 149,
    "seed": 20261076,
    "long_seed": 20261077,
    "device": "mps",
    "initialization": "q254",
    "arms": ["control", "long"],
    "resource_features": "raw",
    "long_pairs": 8,
    "automatic_mps_fallback": False,
    "dev_steps": [0, 2048],
}


def materialize(record):
    pair = record["interval"]
    lo, hi = pair["lower"], pair["upper"]
    if (
        pair["metric"] != "cc"
        or any(type(v) is not int for v in (lo, hi))
        or not 0 <= lo <= hi <= 2
        or (record["lower"], record["upper"]) != (lo, hi)
        or pair["left"] != record["states"][0]
        or pair["right"] != record["states"][-1]
        or pair["witness_semantic_length"] != len(record["cc_positions"])
        or record["left_fingerprint"] != record["fingerprints"][0]
        or record["right_fingerprint"] != record["fingerprints"][-1]
        or record["pair_key"] != pair_key(record)
    ):
        raise ValueError("147 materialized source identity differs")
    row = pair_row(
        record["index"],
        f"long-general-{record['split']}",
        (tuple(pair["left"]["word"]), pair["left"]["strands"]),
        (tuple(pair["right"]["word"]), pair["right"]["strands"]),
        lo,
        hi,
        pair["witness_semantic_length"],
    )
    if witness_priority_weight(row.metric, hi, row.witness_semantic_length) != 1:
        raise ValueError("replacement rows must preserve unit graph weight")
    return row


class LongGeneralSampler:
    """Band/class/endpoint-group/record/swap, five random() calls per row."""

    def __init__(self, records, excluded):
        bins = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        seen = set()
        for record in records:
            if record["split"] != "train" or any(
                owner(fp) != "train" or group(fp) in excluded for fp in record["fingerprints"]
            ):
                raise ValueError("non-TRAIN or historically excluded path")
            row = materialize(record)
            if record["pair_key"] in seen:
                raise ValueError("duplicate147 canonical pair")
            seen.add(record["pair_key"])
            band = record["source"]["length_band"]
            if type(band) is not int or not 0 <= band < len(DATA_CONFIG["length_bands"]):
                raise ValueError("invalid length band")
            lo_size, hi_size = DATA_CONFIG["length_bands"][band]
            if any(
                not lo_size <= len(s["word"]) <= hi_size
                or not DATA_CONFIG["min_strands"] <= s["strands"] <= DATA_CONFIG["max_strands"]
                for s in record["states"]
            ):
                raise ValueError("147 source size differs")
            lo, hi = row.spec.distance_lower, row.spec.distance_upper
            label = f"exact-{lo}" if lo == hi else "interval"
            g = tuple(
                sorted((group(record["left_fingerprint"]), group(record["right_fingerprint"])))
            )
            bins[band][label][g].append(row)
        if set(bins) != set(range(len(DATA_CONFIG["length_bands"]))) or any(
            not {"exact-0", "exact-1", "exact-2"} <= set(classes) for classes in bins.values()
        ):
            raise ValueError("long sampler requires each band's three exact classes")
        self.bins = [
            [
                tuple(tuple(groups[g]) for g in sorted(groups))
                for _, groups in sorted(classes.items())
            ]
            for _, classes in sorted(bins.items())
        ]
        self.support = {
            str(band): {
                label: dict(
                    groups=len(groups),
                    records=sum(map(len, groups.values())),
                    probability=1 / len(bins) / len(classes),
                )
                for label, groups in sorted(classes.items())
            }
            for band, classes in sorted(bins.items())
        }

    def sample(self, rng, count):
        if type(count) is not int or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        result = []
        for _ in range(count):
            band = self.bins[int(rng.random() * len(self.bins))]
            category = band[int(rng.random() * len(band))]
            bucket = category[int(rng.random() * len(category))]
            row = bucket[int(rng.random() * len(bucket))]
            if rng.random() < 0.5:
                row = replace(
                    row,
                    left_word=row.right_word,
                    right_word=row.left_word,
                    left_strands=row.right_strands,
                    right_strands=row.left_strands,
                )
            result.append(row)
        return result


def load_long_source(directory):
    corpus_path, audit_path = directory / "corpus.json", directory / "consumer-audit.json"
    corpus = json.loads(corpus_path.read_text())
    audit = json.loads(audit_path.read_text())
    pins = dict(audit["inputs"])
    if (
        audit["status"] != "passed"
        or audit["training_ready"] is not True
        or audit["minimum_support_ready"] is not True
        or audit["corpus_sha256"] != sha(corpus_path)
        or Path(audit["corpus"]).resolve() != corpus_path.resolve()
        or pins.get(audit["corpus"]) != sha(corpus_path)
        or corpus["schema"] != SCHEMA
        or corpus["contract"]["config"] != DATA_CONFIG
        or corpus["construction_complete"] is not True
        or corpus["minimum_support_ready"] is not True
        or corpus["optimizer_ready"] is not False
        or audit["attempts"] != DATA_CONFIG["attempts"]
        or audit["selected_certificates_checked"] != sum(len(r) for r in corpus["records"].values())
    ):
        raise ValueError("long corpus requires a completed ready, hash-bound consumer")
    for path in (
        Path("scripts/build_long_general_cc_supply.py").resolve(),
        Path("scripts/audit_long_general_cc_supply.py").resolve(),
        directory / "progress.json",
    ):
        if pins.get(str(path)) != sha(path):
            raise ValueError("long consumer lacks required source/completion binding")
    merge_pins(pins, corpus["contract"]["inputs"])
    merge_pins(pins, {str(corpus_path): sha(corpus_path), str(audit_path): sha(audit_path)})
    check_pins(pins)
    summary, gaps = supply_summary(corpus["records"])
    if (
        gaps
        or corpus["support_gaps"] != []
        or audit["support_gaps"] != []
        or (summary != corpus["summary"] or summary != audit["summary"])
    ):
        raise ValueError("long corpus minimum support differs")
    excluded = set(map(tuple, corpus["excluded_query_groups"]))
    training = set.union(*(set(map(tuple, g)) for g in corpus["training_groups"].values()))
    sampler = LongGeneralSampler(corpus["records"]["train"], excluded)
    optimized = training | {
        group(fp) for r in corpus["records"]["train"] for fp in r["fingerprints"]
    }
    dev = corpus["records"]["validation"]
    if any(
        r["split"] != "validation"
        or owner(r["left_fingerprint"]) != "validation"
        or group(r["left_fingerprint"]) in optimized | excluded
        for r in dev
    ):
        raise ValueError("fresh DEV query exposure or owner differs")
    # TEST records are used only by the consumer's supply-count consistency,
    # never materialized into sampler/DEV rows or returned to the optimizer.
    return sampler, ([materialize(r) for r in dev], dev), pins


@dataclass(frozen=True)
class PairedDraw:
    metrics: list
    positives: list
    relative: tuple
    group_index: int
    shared_digest: str
    selected_digest: str


def paired_draw(arm, rng, long_rng, original, suffix, graph, long_source, positives, preferences):
    if arm not in CONFIG["arms"]:
        raise ValueError("unknown149 arm")
    old, chosen, relative, base_digest, index = draw_batch(
        rng, original, suffix, graph, positives, preferences
    )
    replacement = long_source.sample(long_rng, CONFIG["long_pairs"])
    stop = CONFIG["old_pairs"] + CONFIG["suffix_pairs"] + CONFIG["graph_pairs"]
    begin = stop - CONFIG["long_pairs"]
    if len(old) != stop + 1 or len(replacement) != CONFIG["long_pairs"]:
        raise ValueError("149 metric draw coverage differs")
    # Reject any accidental loss reweighting from mislabeled semantic lengths.
    for row in old[begin:stop] + replacement:
        if (
            witness_priority_weight(
                row.metric, row.spec.distance_upper, row.witness_semantic_length
            )
            != 1
        ):
            raise ValueError("149 replacement changes metric weight normalization")
    metrics = old if arm == "control" else old[:begin] + replacement + old[stop:]
    shared = digest([base_digest, [asdict(r) for r in replacement]])
    selected = digest([shared, [asdict(r) for r in metrics]])
    return PairedDraw(metrics, chosen, relative, index, shared, selected)
