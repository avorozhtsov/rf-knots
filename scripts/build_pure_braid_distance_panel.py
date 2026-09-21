"""Protocol74 score-blind general-distance panel; no full unknot witness is assumed."""

import argparse
import fcntl
import json
import os
import random
from collections import Counter
from functools import cache
from pathlib import Path

from audit_joint_policy_metric_panel import check_pins, merge_pins
from build_mcts_training_supervision import atomic, digest, sha
from build_ordinary_graph_test_panel import cyclic_rotation
from burau_integer_check import determinant
from graph_evaluation_partition import partition_inputs
from probe_ordinary_graph_pairs import replay_cc

from rf_knots import reference
from rf_knots.embedding_replay_corpus import canonical, invariants, owner

CONFIG = {"seed": 20261036, "attempts": 3072, "per_category": 32, "per_query_group": 4}
PROBE = Path("artifacts/ordinary-neighborhood-fresh-query-probe-v1/independence-audit.json")
SCHEMA = "rf-knots-pure-braid-distance-panel-v1"


def construct(index, seed=CONFIG["seed"]):
    rng = random.Random(seed + index)
    n = 3 + (index // 3) % 31
    word = list(range(1, n))
    block_count, blocks, centers = 8 + rng.randrange(25), [], []
    positive = (index // 93) % 2 == 0
    for _ in range(block_count):
        conjugator = [rng.randrange(1, n) * rng.choice((-1, 1)) for _ in range(rng.randrange(4))]
        generator, drawn_sign = rng.randrange(1, n), rng.choice((-1, 1))
        sign = 1 if positive else drawn_sign
        centers.append(len(word) + len(conjugator))
        blocks.append({"conjugator": conjugator, "generator": generator, "sign": sign})
        word.extend(conjugator + [sign * generator] * 2 + [-g for g in reversed(conjugator)])
    if reference.num_components(tuple(word), n) != 1:
        raise ValueError("pure-block reference closure is not a knot")
    metadata = {"positive_centers": positive, "blocks": blocks, "reference_strands": n}
    k = index % 3
    if not k:
        offset = rng.randrange(len(word))
        query = word[offset:] + word[:offset]
        states = [{"word": query, "strands": n}, {"word": word, "strands": n}]
        metadata.update(crossing_changes=0, rotation_from_reference=offset, selected_blocks=[])
        return metadata, states, []
    chosen = rng.sample(range(block_count), k)
    positions = [centers[i] for i in chosen]
    states = [{"word": word, "strands": n}]
    for position in positions:
        after = states[-1]["word"].copy()
        after[position] *= -1
        states.append({"word": after, "strands": n})
    metadata.update(crossing_changes=k, rotation_from_reference=None, selected_blocks=chosen)
    return metadata, list(reversed(states)), list(reversed(positions))


def select(rows, training, excluded, config=CONFIG):
    records, decisions, classes, groups, seen = [], [], Counter(), Counter(), set()
    for r in rows:
        a, b = r["fingerprints"][0], r["fingerprints"][-1]
        g = a[0], abs(a[1])
        left, right = r["states"][0], r["states"][-1]
        key = digest(sorted(canonical(tuple(s["word"]), s["strands"]) for s in (left, right)))
        c = str(r["lower"]) if r["lower"] == r["upper"] else "interval"
        reason = (
            "other_owner"
            if owner(a) != "test"
            else "optimizer_collision"
            if g in training
            else "excluded_query"
            if g in excluded
            else "quarantine_reference"
            if owner(b) == "quarantine"
            else "uncertified_nontrivial_reference"
            if b == [1, 0]
            else "duplicate_pair"
            if key in seen
            else "group_cap"
            if groups[g] >= config["per_query_group"]
            else "class_cap"
            if classes[c] >= config["per_category"]
            else "accepted"
        )
        decisions.append(reason)
        if reason != "accepted":
            continue
        records.append(
            {
                **r,
                "split": "test",
                "left_fingerprint": a,
                "right_fingerprint": b,
                "interval": {
                    "metric": "cc",
                    "left": left,
                    "right": right,
                    "lower": r["lower"],
                    "upper": r["upper"],
                    "witness_semantic_length": len(r["cc_positions"]),
                },
            }
        )
        seen.add(key)
        classes[c] += 1
        groups[g] += 1
    adequate = (
        len(records) >= 32 and len(groups) >= 8 and all(classes[c] >= 8 for c in ("0", "1", "2"))
    )
    return records, decisions, dict(classes), len(groups), adequate


def inputs():
    pins, groups, training, historical = partition_inputs()
    probe = json.loads(PROBE.read_text())
    if probe["status"] != "passed" or probe["evaluation_ready"] is not False:
        raise ValueError("require audited diagnostic query exclusion")
    merge_pins(pins, probe["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (
                PROBE,
                Path(__file__).resolve(),
                Path("research/74-pure-braid-short-distance-test-protocol.md"),
            )
        },
    )
    excluded = historical | set(map(tuple, probe["summary"]["eligible_query_groups"]))
    check_pins(pins)
    return pins, groups, training, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() != args.resume:
        raise ValueError("new output required, or explicit resume")
    pins, groups, training, excluded = inputs()
    contract = {"config": CONFIG, "inputs": pins}
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = args.out / "progress.json"
    old = json.loads(progress.read_text()) if args.resume else None
    if old and old["contract"] != contract:
        raise ValueError("resume protocol differs")
    if old and old["status"] == "completed":
        if sha(args.out / "panel.json") != old["panel_sha256"]:
            raise ValueError("completed panel changed")
        print("already completed, frozen input and panel SHA verified")
        return
    rows = old["rows"] if old else []
    if len(rows) > CONFIG["attempts"]:
        raise ValueError("saved attempt coverage differs")
    for i, r in enumerate(rows):
        source, states, positions = construct(i)
        if (r["index"], r["source"], r["states"], r["cc_positions"]) != (
            i,
            source,
            states,
            positions,
        ):
            raise ValueError("saved deterministic construction differs")

    @cache
    def fingerprint(word, strands):
        fp = invariants(word, strands)
        if fp[0] != determinant(word, strands):
            raise ValueError("independent determinant differs")
        return list(fp)

    def save(status, **extra):
        records, decisions, classes, count, adequate = select(rows, training, excluded)
        atomic(
            progress,
            {
                "status": status,
                "pid": os.getpid(),
                "contract": contract,
                "rows": rows,
                "next_index": len(rows),
                "classes": classes,
                "groups": count,
                "selected": len(records),
                "adequate": adequate,
                "selection_counts": dict(Counter(decisions)),
                **extra,
            },
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "attempts": len(rows),
                    "classes": classes,
                    "groups": count,
                    "adequate": adequate,
                }
            ),
            flush=True,
        )

    save("building")
    for i in range(len(rows), CONFIG["attempts"]):
        source, states, positions = construct(i)
        if positions:
            actual = [
                replay_cc((tuple(a["word"]), a["strands"]), (tuple(b["word"]), b["strands"]))
                for a, b in zip(states[:-1], states[1:], strict=True)
            ]
            if actual != positions:
                raise ValueError("ordinary CC path differs")
        fps = [fingerprint(tuple(s["word"]), s["strands"]) for s in states]
        rotation = cyclic_rotation(states[0], states[-1])
        lo = max(abs(fps[0][1] - fps[-1][1]) // 2, int(fps[0][0] != fps[-1][0]))
        hi = 0 if rotation is not None else len(positions)
        if lo > hi or (not positions and rotation is None):
            raise ValueError("ordinary bounds/cyclic certificate contradict")
        rows.append(
            {
                "index": i,
                "source": source,
                "states": states,
                "cc_positions": positions,
                "fingerprints": fps,
                "cyclic_rotation": rotation,
                "lower": lo,
                "upper": hi,
            }
        )
        if len(rows) % 32 == 0:
            save("building")
    records, decisions, classes, count, adequate = select(rows, training, excluded)
    panel = {
        "schema": SCHEMA,
        "partition": "test",
        "complete": adequate,
        "construction_complete": True,
        "contract": contract,
        "validation": records,
        "size_shift": [],
        "classes": classes,
        "groups": count,
        "training_groups": {k: sorted(v) for k, v in groups.items()},
        "excluded_query_groups": sorted(excluded),
        "selection_decisions": decisions,
        "selection_counts": dict(Counter(decisions)),
        "scope": "Synthetic pure-block ordinary distance around certified nontrivial references, "
        "not short-u or catalogue coverage. No model scores, no lifted views.",
    }
    check_pins(pins)
    atomic(args.out / "panel.json", panel)
    save("completed", panel_sha256=sha(args.out / "panel.json"))


if __name__ == "__main__":
    main()
