"""Independent deterministic construction, ordinary replay and selection audit for74."""

import argparse
import json
import random
from collections import Counter
from functools import cache
from pathlib import Path

from audit_fresh_ordinary_neighborhood import endpoint_key
from audit_graph_mixture_short_u_panel import stratum
from audit_joint_policy_metric_panel import check_pins, merge_pins
from audit_ordinary_graph_test_panel import trace_bounds
from build_mcts_training_supervision import atomic, sha
from build_pure_braid_distance_panel import CONFIG, SCHEMA, inputs
from burau_integer_check import determinant

from rf_knots.embedding_replay_corpus import invariants, owner


def reconstruct(index):
    """Separate construction, including the declared RNG draws and actual permutation."""
    n = 3 + (index // 3) % 31
    rng = random.Random(CONFIG["seed"] + index)
    count = 8 + rng.randrange(25)
    pure_blocks, word, center = [], list(range(1, n)), []
    positive = (index // 93) % 2 == 0
    for _ in range(count):
        conjugator = []
        for _ in range(rng.randrange(4)):
            generator = rng.randrange(1, n)
            sign = rng.choice((-1, 1))
            conjugator.append(generator * sign)
        generator = rng.randrange(1, n)
        draw = rng.choice((-1, 1))
        sign = 1 if positive else draw
        pure_blocks.append({"conjugator": conjugator, "generator": generator, "sign": sign})
        word += conjugator
        center.append(len(word))
        word += [sign * generator, sign * generator]
        word += [-g for g in conjugator[::-1]]
    labels = list(range(n))
    for g in word:
        p = abs(g) - 1
        labels[p : p + 2] = labels[p : p + 2][::-1]
    visited, head = set(), 0
    while head not in visited:
        visited.add(head)
        head = labels[head]
    if len(visited) != n:
        raise ValueError("reference permutation has more than one cycle")
    k = index % 3
    source = {
        "positive_centers": positive,
        "blocks": pure_blocks,
        "reference_strands": n,
        "crossing_changes": k,
        "rotation_from_reference": None,
        "selected_blocks": [],
    }
    if k == 0:
        offset = rng.randrange(len(word))
        source["rotation_from_reference"] = offset
        return (
            source,
            [{"word": word[offset:] + word[:offset], "strands": n}, {"word": word, "strands": n}],
            [],
        )
    chosen = rng.sample(range(count), k)
    source["selected_blocks"] = chosen
    positions = [center[b] for b in chosen]
    states = [{"word": word, "strands": n}]
    for p in positions:
        next_word = states[-1]["word"][:p] + [-states[-1]["word"][p]] + states[-1]["word"][p + 1 :]
        states.append({"word": next_word, "strands": n})
    return source, states[::-1], positions[::-1]


def check_row(row, index, fingerprint=invariants, determinant_check=determinant):
    expected = reconstruct(index)
    if row["index"] != index or (row["source"], row["states"], row["cc_positions"]) != expected:
        raise ValueError("construction/draw differs from fixed experiment")
    if any(type(row[k]) is not int for k in ("lower", "upper")):
        raise ValueError("distance bounds must be integer CC counts")
    if row["cc_positions"]:
        trace_bounds(row, fingerprint, determinant_check)
    else:
        fps = []
        for s in row["states"]:
            fp = fingerprint(tuple(s["word"]), s["strands"])
            if fp[0] != determinant_check(tuple(s["word"]), s["strands"]):
                raise ValueError("independent determinant differs")
            fps.append(list(fp))
        if fps != row["fingerprints"] or row["lower"] != 0 or row["upper"] != 0:
            raise ValueError("zero-coordinate bound/fingerprint differs")
    left, right = row["states"][0], row["states"][-1]
    a, b = left["word"], right["word"]
    shifts = (
        [k for k in range(len(a)) if a[k:] + a[:k] == b]
        if left["strands"] == right["strands"]
        else []
    )
    expected_shift = min(shifts) if shifts else None
    if row["cyclic_rotation"] != expected_shift or (row["upper"] == 0) != bool(shifts):
        raise ValueError("cyclic-zero certificate differs")


def selection(rows, training, excluded):
    selected, decisions, groups, classes, seen = [], [], Counter(), Counter(), set()
    for r in rows:
        a, b = r["fingerprints"][0], r["fingerprints"][-1]
        g = a[0], abs(a[1])
        key = tuple(sorted(endpoint_key(s) for s in (r["states"][0], r["states"][-1])))
        category = str(r["lower"]) if r["lower"] == r["upper"] else "interval"
        if owner(a) != "test":
            reason = "other_owner"
        elif g in training:
            reason = "optimizer_collision"
        elif g in excluded:
            reason = "excluded_query"
        elif owner(b) == "quarantine":
            reason = "quarantine_reference"
        elif b == [1, 0]:
            reason = "uncertified_nontrivial_reference"
        elif key in seen:
            reason = "duplicate_pair"
        elif groups[g] >= CONFIG["per_query_group"]:
            reason = "group_cap"
        elif classes[category] >= CONFIG["per_category"]:
            reason = "class_cap"
        else:
            reason = "accepted"
        decisions.append(reason)
        if reason == "accepted":
            selected.append(
                {
                    **r,
                    "split": "test",
                    "left_fingerprint": a,
                    "right_fingerprint": b,
                    "interval": {
                        "metric": "cc",
                        "left": r["states"][0],
                        "right": r["states"][-1],
                        "lower": r["lower"],
                        "upper": r["upper"],
                        "witness_semantic_length": len(r["cc_positions"]),
                    },
                }
            )
            groups[g] += 1
            classes[category] += 1
            seen.add(key)
    adequate = (
        len(selected) >= 32 and len(groups) >= 8 and all(classes[c] >= 8 for c in ("0", "1", "2"))
    )
    return selected, decisions, dict(classes), groups, adequate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite independent audit")
    panel = json.loads(args.panel.read_text())
    pp = args.panel.with_name("progress.json")
    progress = json.loads(pp.read_text())
    pins, by_source, training, excluded = inputs()
    merge_pins(pins, panel["contract"]["inputs"])
    merge_pins(
        pins,
        {
            str(p): sha(p)
            for p in (
                args.panel,
                pp,
                Path(__file__).resolve(),
                Path("scripts/audit_graph_mixture_short_u_panel.py"),
            )
        },
    )
    check_pins(pins)
    if (
        panel["schema"] != SCHEMA
        or panel["partition"] != "test"
        or not panel["construction_complete"]
        or panel["size_shift"] != []
        or panel["contract"]["config"] != CONFIG
        or progress["contract"] != panel["contract"]
        or progress["status"] != "completed"
        or progress["panel_sha256"] != sha(args.panel)
        or progress["next_index"] != CONFIG["attempts"]
        or len(progress["rows"]) != CONFIG["attempts"]
        or panel["training_groups"]
        != {k: [list(g) for g in sorted(v)] for k, v in by_source.items()}
        or panel["excluded_query_groups"] != [list(g) for g in sorted(excluded)]
    ):
        raise ValueError("construction/partition/coverage contract differs")

    @cache
    def fingerprint(word, n):
        return invariants(word, n)

    @cache
    def independent_det(word, n):
        return determinant(word, n)

    for i, r in enumerate(progress["rows"]):
        check_row(r, i, fingerprint, independent_det)
        if (i + 1) % 128 == 0:
            print(
                json.dumps({"attempts_checked": i + 1, "total": len(progress["rows"])}), flush=True
            )
    records, decisions, classes, groups, adequate = selection(progress["rows"], training, excluded)
    if (
        records != panel["validation"]
        or decisions != panel["selection_decisions"]
        or dict(Counter(decisions)) != panel["selection_counts"]
        or classes != panel["classes"]
        or len(groups) != panel["groups"]
        or adequate != panel["complete"]
        or (
            progress["selected"],
            progress["classes"],
            progress["groups"],
            progress["adequate"],
            progress["selection_counts"],
        )
        != (len(records), classes, len(groups), adequate, dict(Counter(decisions)))
    ):
        raise ValueError("first-eligible selection/caps/adequacy differs")
    sizes = {}
    for endpoint in ("left", "right"):
        states = [r["interval"][endpoint] for r in records]
        sizes[endpoint] = {
            "min_length": min((len(s["word"]) for s in states), default=None),
            "max_length": max((len(s["word"]) for s in states), default=None),
            "strands": dict(Counter(str(s["strands"]) for s in states)),
            "metric_train_strata": dict(
                Counter(stratum(len(s["word"]), s["strands"]) for s in states)
            ),
        }
    check_pins(pins)
    result = {
        "status": "passed",
        "evaluation_ready": adequate,
        "inputs": pins,
        "attempts_checked": len(progress["rows"]),
        "selected_pairs": len(records),
        "query_groups": sorted(groups),
        "classes": classes,
        "raw_sizes": sizes,
        "query_train_intersection": [],
        "query_historical_diagnostic_intersection": [],
        "scope": "Independent deterministic pure-block construction/closure, all ordinary "
        "CC and cyclic-zero certificates, signed bounds and first-eligible selection. "
        "Synthetic family only; nontrivial references, not unknotting-number labels.",
    }
    atomic(args.out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"inputs", "query_groups", "raw_sizes"}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
