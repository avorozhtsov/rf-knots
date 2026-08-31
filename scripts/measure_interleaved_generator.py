#!/usr/bin/env python3
"""Throughput and shape of the interleaved backward generator.

Two numbers matter for §4.1 to be worth building: how many instances a core
produces per second, and what `(certified upper bound, word length, strand
count)` actually comes out -- as opposed to what was asked for.  Both are
measured here, on the generator alone, with witness verification included
because a witness nobody verified is not evidence.

Knot identification is opt-in and bounded, for one reason that is worth stating
because it is the opposite of the usual warning.  `rf_knots.invariants`'
Kauffman bracket is a Temperley-Lieb state sum: exponential in the number of
**strands** (Catalan(n) basis elements) and linear in word length.  On these
words it costs about a millisecond.  What is expensive is the *Alexander*
polynomial, whose Burau determinant grows with the Laurent span and therefore
with word length -- 200 ms on a 40-letter five-strand word against 0.8 ms for
the bracket.  So identification runs on the **free reduction** of the emitted
word, which is the same element of the braid group and the same knot, and is
capped by `--identify`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any

from rf_knots.interleaved_generator import InterleavedConfig, WalkWeights, generate_many
from rf_knots.reference import free_reduce


def histogram(counter: Counter) -> dict[str, Any]:
    if not counter:
        return {}
    keys = sorted(counter)
    total = sum(counter.values())
    running = 0
    median = keys[-1]
    for key in keys:
        running += counter[key]
        if running * 2 >= total:
            median = key
            break
    return {
        "min": keys[0],
        "median": median,
        "max": keys[-1],
        "mean": round(sum(k * v for k, v in counter.items()) / total, 2),
        "counts": {str(k): counter[k] for k in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("-k", "--crossing-changes", type=int, default=5)
    parser.add_argument("-l", "--walk-length", type=int, default=24)
    parser.add_argument("--target-strands", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--max-strands", type=int, default=6)
    parser.add_argument("--strand-band", type=int, default=0)
    parser.add_argument("--settle-walk-length", type=int, default=0)
    parser.add_argument("--cyclic-band-generators", action="store_true")
    parser.add_argument(
        "--identify",
        type=int,
        default=0,
        help="identify this many instances against the bundled knot table",
    )
    parser.add_argument(
        "--identify-max-length",
        type=int,
        default=24,
        help="skip identification when the free reduction is longer than this",
    )
    parser.add_argument(
        "--signature-bound",
        action="store_true",
        help="also report the Murasugi lower bound distribution (needs spherogram)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    config = InterleavedConfig(
        crossing_changes=args.crossing_changes,
        walk_length=args.walk_length,
        target_strands=args.target_strands,
        target_length=args.target_length,
        max_len=args.max_len,
        max_strands=args.max_strands,
        strand_band=args.strand_band,
        settle_walk_length=args.settle_walk_length,
        cyclic_band_generators=args.cyclic_band_generators,
        weights=WalkWeights(),
    )

    counters = {
        "certified_upper_bound": Counter(),
        "word_length": Counter(),
        "free_reduced_length": Counter(),
        "strands": Counter(),
        "witness_moves": Counter(),
        "signature_lower_bound": Counter(),
    }
    erased = 0
    erased_changes = 0
    skipped = 0
    moves = 0
    sample: list[tuple[tuple[int, ...], int, int]] = []

    started = time.perf_counter()
    for instance in generate_many(config, seed=args.seed, count=args.count):
        instance.witness.verify()
        reduced = free_reduce(instance.word)
        counters["certified_upper_bound"][instance.certified_upper_bound] += 1
        counters["word_length"][len(instance.word)] += 1
        counters["free_reduced_length"][len(reduced)] += 1
        counters["strands"][instance.strands] += 1
        counters["witness_moves"][instance.witness.moves] += 1
        erased += instance.erased_steps
        erased_changes += instance.erased_crossing_changes
        skipped += instance.skipped_crossing_changes
        moves += instance.witness.moves
        if args.signature_bound:
            from rf_knots import seifert

            try:
                sigma = seifert.signature(reduced, instance.strands)
                counters["signature_lower_bound"][abs(sigma) // 2] += 1
            except (ValueError, seifert.BackendUnavailable):
                pass
        if len(sample) < args.identify:
            sample.append((reduced, instance.strands, instance.certified_upper_bound))
    elapsed = time.perf_counter() - started

    report: dict[str, Any] = {
        "config": config.provenance(),
        "seed": args.seed,
        "count": args.count,
        "seconds": round(elapsed, 3),
        "instances_per_second": round(args.count / elapsed, 1),
        "witness_moves_per_second": round(moves / elapsed, 1),
        "erased_steps_total": erased,
        "erased_crossing_changes_total": erased_changes,
        "skipped_crossing_changes_total": skipped,
        "distribution": {
            name: histogram(counter)
            for name, counter in counters.items()
            if counter
        },
    }

    if sample:
        from rf_knots.invariants import invariants

        names: Counter = Counter()
        exact = 0
        identified_started = time.perf_counter()
        attempted = 0
        for word, strands, upper in sample:
            if len(word) > args.identify_max_length:
                names["(skipped: longer than --identify-max-length)"] += 1
                continue
            attempted += 1
            result = invariants(word, strands)
            if result.name is not None:
                label = result.name + (" mirror" if result.mirror else "")
            elif result.summands:
                label = " # ".join(result.summands)
            else:
                label = "(unidentified)"
            names[label] += 1
            lower = result.unknotting_lower
            if lower is not None and lower == upper:
                exact += 1
        report["identification"] = {
            "attempted": attempted,
            "seconds": round(time.perf_counter() - identified_started, 3),
            "exact_at_k": exact,
            "note": "identified on the free reduction of the emitted word, which "
            "is the same braid-group element and the same knot",
            "types": dict(names.most_common()),
        }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"config          {json.dumps(config.provenance(), sort_keys=True)}")
    print(f"seed            {args.seed}   count {args.count}")
    print(
        f"throughput      {report['instances_per_second']} instances/s, "
        f"{report['witness_moves_per_second']} witness moves/s "
        f"({report['seconds']} s)"
    )
    print(
        f"loop erasure    {erased} steps, {erased_changes} crossing changes; "
        f"{skipped} changes never placed"
    )
    for name, stats in report["distribution"].items():
        print(
            f"{name:22s} min {stats['min']:4d}  median {stats['median']:4d}  "
            f"max {stats['max']:4d}  mean {stats['mean']}"
        )
    if "identification" in report:
        ident = report["identification"]
        print(
            f"identification  {ident['attempted']} attempted in {ident['seconds']} s; "
            f"{ident['exact_at_k']} exact (lower bound meets the witness)"
        )
        for label, count in ident["types"].items():
            print(f"    {count:5d}  {label}")


if __name__ == "__main__":
    main()
