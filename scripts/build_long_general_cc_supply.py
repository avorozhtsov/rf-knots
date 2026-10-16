"""Protocol147: replay-checked long general CC pairs with fixed TRAIN/DEV/TEST."""

import argparse
import fcntl
import json
import os
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_test_panel import trace_bounds
from build_mcts_training_supervision import atomic, digest, sha
from build_ordinary_graph_test_panel import cyclic_rotation
from build_resource_general_test import provenance as historical_provenance
from burau_integer_check import determinant

from rf_knots import reference
from rf_knots.embedding_replay_corpus import canonical, invariants, owner

SCHEMA = "rf-knots-long-general-cc-supply-v1"
CONFIG = dict(
    seed=20261075,
    attempts=6144,
    length_bands=[[49, 96], [97, 192], [193, 384]],
    min_strands=3,
    max_strands=33,
    shard_rows=64,
    group_caps=dict(train=8, validation=2, test=2),
    min_pairs=dict(train=32, validation=8, test=8),
    min_groups=dict(train=16, validation=4, test=4),
)
PANELS = [
    Path("artifacts/resource-units-fresh-short-u-test-v1/generator/panel.json"),
    Path("artifacts/resource-units-fresh-general-test-v1/panel.json"),
]
AUDITS = [
    Path("artifacts/resource-units-fresh-short-u-test-v1/consumer-audit.json"),
    Path("artifacts/resource-units-fresh-general-test-v1/consumer-audit.json"),
]


def construct(index):
    if type(index) is not int or not 0 <= index < CONFIG["attempts"]:
        raise ValueError("attempt index outside fixed contract")
    rng = random.Random(CONFIG["seed"] + index)
    k, band = index % 3, (index // 3) % 3
    strands = CONFIG["min_strands"] + (index // 9) % 31
    lo, hi = CONFIG["length_bands"][band]
    target = rng.randrange(lo, hi - 6)
    positive = (index // 279) % 2 == 0
    word, blocks, centers = list(range(1, strands)), [], []
    while len(word) < target:
        conjugator = [
            rng.randrange(1, strands) * rng.choice((-1, 1)) for _ in range(rng.randrange(4))
        ]
        generator = rng.randrange(1, strands)
        sign = 1 if positive else rng.choice((-1, 1))
        centers.append(len(word) + len(conjugator))
        blocks.append(dict(conjugator=conjugator, generator=generator, sign=sign))
        word.extend(conjugator + [sign * generator] * 2 + [-g for g in reversed(conjugator)])
    if not lo <= len(word) <= hi or reference.num_components(tuple(word), strands) != 1:
        raise ValueError("construction length or closure differs")
    source = dict(
        length_band=band,
        target_length=target,
        positive_centers=positive,
        reference_strands=strands,
        blocks=blocks,
        crossing_changes=k,
    )
    if not k:
        offset = rng.randrange(1, len(word))
        source.update(rotation_from_reference=offset, selected_blocks=[])
        return (
            source,
            [
                dict(word=word[offset:] + word[:offset], strands=strands),
                dict(word=word, strands=strands),
            ],
            [],
        )
    chosen = rng.sample(range(len(blocks)), k)
    positions = [centers[i] for i in chosen]
    states = [dict(word=word, strands=strands)]
    for position in positions:
        after = states[-1]["word"].copy()
        after[position] *= -1
        states.append(dict(word=after, strands=strands))
    source.update(rotation_from_reference=None, selected_blocks=chosen)
    return source, list(reversed(states)), list(reversed(positions))


@lru_cache(maxsize=1024)
def fingerprint(word, strands):
    return invariants(word, strands)


@lru_cache(maxsize=1024)
def integer_determinant(word, strands):
    return determinant(word, strands)


def make_row(index):
    source, states, positions = construct(index)
    fps = [list(fingerprint(tuple(s["word"]), s["strands"])) for s in states]
    rotation = cyclic_rotation(states[0], states[-1])
    return dict(
        index=index,
        source=source,
        states=states,
        cc_positions=positions,
        fingerprints=fps,
        cyclic_rotation=rotation,
        lower=max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0])),
        upper=0 if rotation is not None else len(positions),
    )


def audit_row(row, index):
    if type(row["index"]) is not int or row["index"] != index:
        raise ValueError("row index differs")
    if (row["source"], row["states"], row["cc_positions"]) != construct(index):
        raise ValueError("deterministic construction differs")
    if any(type(row[k]) is not int for k in ("lower", "upper")):
        raise ValueError("bounds must be integers")
    for state in row["states"]:
        if reference.num_components(tuple(state["word"]), state["strands"]) != 1:
            raise ValueError("closure is not a knot")
    if row["cc_positions"]:
        trace_bounds(row, fingerprint, integer_determinant)
    else:
        fps = []
        for state in row["states"]:
            word, n = tuple(state["word"]), state["strands"]
            fp = fingerprint(word, n)
            if fp[0] != integer_determinant(word, n):
                raise ValueError("independent determinant differs")
            fps.append(list(fp))
        if row["fingerprints"] != fps or row["lower"] != 0 or row["upper"] != 0:
            raise ValueError("zero-CC fingerprints or bounds differ")
    a, b = row["states"][0], row["states"][-1]
    rotations = [
        k
        for k in range(len(a["word"]))
        if a["strands"] == b["strands"] and a["word"][k:] + a["word"][:k] == b["word"]
    ]
    if row["cyclic_rotation"] != (min(rotations) if rotations else None):
        raise ValueError("cyclic certificate differs")
    if (row["upper"] == 0) != bool(rotations) or not 0 <= row["lower"] <= row["upper"] <= 2:
        raise ValueError("certified bounds differ")


def group(fp):
    return fp[0], abs(fp[1])


def category(row):
    return str(row["lower"]) if row["lower"] == row["upper"] else "interval"


def pair_key(row):
    return digest(
        sorted(
            canonical(tuple(s["word"]), s["strands"]) for s in (row["states"][0], row["states"][-1])
        )
    )


def select(rows, training, excluded):
    records = {split: [] for split in CONFIG["group_caps"]}
    decisions, counts, seen = [], Counter(), set()
    for row in rows:
        fps = row["fingerprints"]
        split, g = owner(fps[0]), group(fps[0])
        path_groups = set(map(group, fps))
        key = pair_key(row)
        cap_key = split, g, category(row), row["source"]["length_band"]
        reason = (
            "quarantine"
            if any(owner(fp) == "quarantine" for fp in fps)
            else "non_train_path"
            if split == "train" and any(owner(fp) != "train" for fp in fps)
            else "excluded_train_path"
            if split == "train" and path_groups & excluded
            else "known_evaluation_query"
            if split != "train" and g in training | excluded
            else "uncertified_reference"
            if group(fps[-1]) == (1, 0)
            else "identical_raw_endpoints"
            if row["states"][0] == row["states"][-1]
            else "duplicate_pair"
            if key in seen
            else "group_cap"
            if counts[cap_key] >= CONFIG["group_caps"][split]
            else "accepted"
        )
        decisions.append(reason)
        if reason != "accepted":
            continue
        records[split].append(
            dict(
                **row,
                split=split,
                pair_key=key,
                left_fingerprint=fps[0],
                right_fingerprint=fps[-1],
                interval=dict(
                    metric="cc",
                    left=row["states"][0],
                    right=row["states"][-1],
                    lower=row["lower"],
                    upper=row["upper"],
                    witness_semantic_length=len(row["cc_positions"]),
                ),
            )
        )
        seen.add(key)
        counts[cap_key] += 1
    return records, decisions


def supply_summary(records):
    summary, gaps = {}, []
    for split, rows in records.items():
        cells, classes = defaultdict(list), Counter()
        for row in rows:
            c = category(row)
            cells[(row["source"]["length_band"], c)].append(row)
            classes[c] += 1
        by_band = []
        for band in range(len(CONFIG["length_bands"])):
            entries = {}
            for c in ("0", "1", "2", "interval"):
                selected = cells[(band, c)]
                groups = {group(r["fingerprints"][0]) for r in selected}
                entries[c] = dict(pairs=len(selected), query_groups=len(groups))
                if c != "interval" and (
                    len(selected) < CONFIG["min_pairs"][split]
                    or len(groups) < CONFIG["min_groups"][split]
                ):
                    gaps.append(dict(split=split, band=band, category=c, **entries[c]))
            by_band.append(entries)
        endpoints = [s for r in rows for s in (r["states"][0], r["states"][-1])]
        summary[split] = dict(
            pairs=len(rows),
            classes=dict(classes),
            by_length_band=by_band,
            unique_canonical_pairs=len({pair_key(r) for r in rows}),
            unique_raw_pairs=len(
                {
                    digest(
                        sorted((s["strands"], s["word"]) for s in (r["states"][0], r["states"][-1]))
                    )
                    for r in rows
                }
            ),
            query_groups=len({group(r["fingerprints"][0]) for r in rows}),
            lengths=[min(len(s["word"]) for s in endpoints), max(len(s["word"]) for s in endpoints)]
            if endpoints
            else None,
            strands=[min(s["strands"] for s in endpoints), max(s["strands"] for s in endpoints)]
            if endpoints
            else None,
        )
    return summary, gaps


def provenance():
    pins, groups, training, excluded = historical_provenance()
    excluded = set(excluded)
    for path, audit_path in zip(PANELS, AUDITS, strict=True):
        audit = json.loads(audit_path.read_text())
        if audit["status"] != "passed" or audit["panel_sha256"] != sha(path):
            raise ValueError("fresh historical panel lacks a bound consumer audit")
        merge_pins(pins, audit["inputs"])
        merge_pins(pins, {str(path): sha(path), str(audit_path): sha(audit_path)})
        panel = json.loads(path.read_text())
        for row in panel["validation"] + panel["size_shift"]:
            excluded.add(group(row["left_fingerprint"]))
    for path in (
        Path(__file__).resolve(),
        Path("tests/test_long_general_cc_supply.py"),
        Path("research/147-long-general-cc-supply-protocol.md"),
    ):
        merge_pins(pins, {str(path): sha(path)})
    check_pins(pins)
    return pins, groups, training, excluded


def load_shards(out, progress, *, replay):
    rows = []
    for entry in progress["shards"]:
        start = len(rows)
        expected = f"attempts-{start:06d}.json"
        path = out / expected
        if entry["file"] != expected or sha(path) != entry["sha256"]:
            raise ValueError("shard identity or hash differs")
        part = json.loads(path.read_text())
        end = min(start + CONFIG["shard_rows"], CONFIG["attempts"])
        if len(part) != end - start or [r["index"] for r in part] != list(range(start, end)):
            raise ValueError("shard attempt coverage differs")
        if replay:
            for row in part:
                audit_row(row, row["index"])
        rows.extend(part)
    if len(rows) != progress["next_index"] or len(rows) > CONFIG["attempts"]:
        raise ValueError("progress attempt coverage differs")
    expected_files = {s["file"] for s in progress["shards"]}
    if {p.name for p in out.glob("attempts-*.json")} != expected_files:
        raise ValueError("uncommitted shard preserved; manual recovery required")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pins, groups, training, excluded = provenance()
    contract = dict(config=CONFIG, inputs=pins)
    progress_path = args.out / "progress.json"
    old = json.loads(progress_path.read_text()) if args.resume else None
    if old is not None and old["contract"] != contract:
        raise ValueError("resume contract differs")
    if old and old["status"] not in ("building", "completed"):
        raise ValueError("unknown progress status")
    rows = load_shards(args.out, old, replay=old["status"] != "completed") if old else []
    shards = old["shards"] if old else []
    if old and old["status"] == "completed":
        if len(rows) != CONFIG["attempts"] or sha(args.out / "corpus.json") != old["corpus_sha256"]:
            raise ValueError("completed corpus differs")
        print("already completed; source, shard and corpus hashes verified", flush=True)
        return

    def save(status, **extra):
        records, decisions = select(rows, training, excluded)
        summary, gaps = supply_summary(records)
        state = dict(
            schema=SCHEMA,
            status=status,
            pid=os.getpid(),
            contract=contract,
            shards=shards,
            next_index=len(rows),
            summary=summary,
            selection_counts=dict(Counter(decisions)),
            support_gaps=gaps,
            minimum_support_ready=not gaps,
            **extra,
        )
        atomic(progress_path, state)
        print(
            json.dumps(
                dict(
                    status=status,
                    next_index=len(rows),
                    selected={s: v["classes"] for s, v in summary.items()},
                    support_gaps=len(gaps),
                )
            ),
            flush=True,
        )

    save("building")
    for start in range(len(rows), CONFIG["attempts"], CONFIG["shard_rows"]):
        path = args.out / f"attempts-{start:06d}.json"
        if path.exists() or path.with_suffix(".partial").exists():
            raise ValueError("uncommitted artifact preserved; explicit recovery required")
        part = []
        for index in range(start, min(start + CONFIG["shard_rows"], CONFIG["attempts"])):
            row = make_row(index)
            audit_row(row, index)
            part.append(row)
        atomic(path, part)
        rows.extend(part)
        shards.append(dict(file=path.name, sha256=sha(path)))
        save("building")
    records, decisions = select(rows, training, excluded)
    summary, gaps = supply_summary(records)
    check_pins(pins)
    corpus = dict(
        schema=SCHEMA,
        contract=contract,
        construction_complete=True,
        minimum_support_ready=not gaps,
        optimizer_ready=False,
        audit_required="post-exit selected-certificate and partition consumer",
        records=records,
        summary=summary,
        support_gaps=gaps,
        training_groups={k: sorted(v) for k, v in groups.items()},
        excluded_query_groups=sorted(excluded),
        selection_decisions=decisions,
        selection_counts=dict(Counter(decisions)),
        model_forwards=0,
        optimizer_steps=0,
        scope="Synthetic query-disjoint RF-lineage pairs; references may be familiar; "
        "not named knots, full Q254 independence, or unknotting policy trajectories",
    )
    if (args.out / "corpus.json").exists():
        raise ValueError("uncommitted corpus preserved; explicit recovery required")
    atomic(args.out / "corpus.json", corpus)
    save("completed", corpus_sha256=sha(args.out / "corpus.json"))


if __name__ == "__main__":
    main()
