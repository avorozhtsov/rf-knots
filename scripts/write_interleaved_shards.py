#!/usr/bin/env python3
"""Write interleaved backward-generated instances to replayable JSONL shards.

Each row is an `rf_knots.evidence.EvidenceRecord`, so the witness is verified
twice before it reaches disk -- once by `UnknotWitness.from_actions` inside the
generator and once by `EvidenceRecord.__post_init__` -- and a third time by
`EvidenceStore.records()` on every read, along with the record hash.  Nothing
about a shard has to be trusted.

The claim carried by a row is `u <= metrics.certified_upper_bound`, which is at
most the requested `k`.  `outcome` is `"constructed"` rather than `"solved"`:
no search ran, the path was built and then read backwards.  A row is *exact*
only where `--lower-bounds` found a theorem that meets the witness, and then
`metrics.exact_unknotting_number` holds the value and the theorem is in
`lower_bounds` with its provenance.  Everywhere else that field is `null`.

Usage::

    python scripts/write_interleaved_shards.py \\
        --output-dir artifacts/interleaved/k5-l24 \\
        --count 20000 -k 5 -l 24 --target-strands 4 --target-length 32 \\
        --max-len 64 --max-strands 6 --seed 20260830
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rf_knots.evidence import (
    EvidenceRecord,
    EvidenceStore,
    LowerBoundClaim,
    braid_knot_key,
)
from rf_knots.interleaved_generator import (
    InterleavedConfig,
    InterleavedInstance,
    Quarantine,
    SeedRepresentation,
    WalkWeights,
    default_seed,
    generate_many,
    gordian_interval,
    inherited_claim,
    lipschitz_values,
    load_quarantine,
    positive_braid_seed,
    torus_seed,
)
from rf_knots.reference import free_reduce

MANIFEST_SCHEMA = "rf-knots-interleaved-corpus-v0"
SOLVER = "interleaved-backward-generator"
OUTCOME = "constructed"
UNKNOTDB = "/Users/artemvorozhtsov/projects/unknotdb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_record(
    instance: InterleavedInstance,
    config: InterleavedConfig,
    seed: int,
    lower_bounds: bool,
) -> EvidenceRecord:
    """One evidence row, with every dial that produced it attached.

    Two independent lower bounds can appear.  The *inherited* one is the seed's
    theorem carried across the walk by the 1-Lipschitz property (note 22 §4.1b);
    it costs nothing and is present for every seed with a certified `L`.  The
    *computed* ones come from `lower_bounds.computed_claims` on the emitted word
    and cost a Seifert matrix each.  `EvidenceRecord.certified_lower_bound` takes
    the strongest of whatever is present, which is the correct fold: they are
    separate theorems about the same knot.
    """
    claims: tuple[LowerBoundClaim, ...] = ()
    if lower_bounds:
        from rf_knots.lower_bounds import computed_claims

        claims = computed_claims(instance.word, instance.strands)
    inherited = inherited_claim(instance)
    if inherited is not None:
        claims = (inherited, *claims)
    upper = instance.certified_upper_bound
    lower = max((claim.value for claim in claims), default=None)
    metrics: dict[str, Any] = instance.metrics()
    metrics["free_reduced_length"] = len(free_reduce(instance.word))
    metrics["certified_lower_bound"] = lower
    # `u` only where a theorem meets the witness.  `None` is the honest answer
    # everywhere else, and it is the answer for almost every row: the label this
    # generator produces is an upper bound and calling it `u` would be a lie.
    metrics["exact_unknotting_number"] = upper if lower == upper else None
    return EvidenceRecord(
        instance_id=instance.instance_id,
        knot_key=braid_knot_key(instance.word, instance.strands),
        solver=SOLVER,
        benchmark=None,
        checkpoint=None,
        search_budget={
            "seed": seed,
            "index": instance.index,
            "seed_representation": instance.seed_representation.provenance(),
            **config.provenance(),
        },
        outcome=OUTCOME,
        witness=instance.witness,
        lower_bounds=claims,
        metrics=metrics,
    )


def build_seed(
    config: InterleavedConfig,
    family: str,
    strength: int,
    strands: int,
    rng_seed: int,
) -> SeedRepresentation:
    """The §4.1b seed for one shard family.  Every program is verified on build."""
    if family == "unknot":
        if strength:
            raise SystemExit("the unknot seed certifies nothing; --seed-strength 0")
        return default_seed(config)
    spec = config.spec()
    if family == "torus":
        return torus_seed(strength, spec)
    if family == "positive-braid":
        return positive_braid_seed(
            spec, strands=strands, genus=strength, seed=rng_seed
        )
    raise SystemExit(f"unknown seed family {family!r}")


def summarise(counters: dict[str, Counter]) -> dict[str, Any]:
    return {
        name: {str(key): int(value) for key, value in sorted(counter.items())}
        for name, counter in counters.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=5_000)
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
    parser.add_argument("--no-erase-loops", action="store_true")
    parser.add_argument(
        "--keep-trivial",
        action="store_true",
        help="emit instances whose word is the empty one-braid (u <= 0)",
    )
    parser.add_argument(
        "--min-certified-lower-bound",
        type=int,
        default=0,
        help="reject instances whose Murasugi signature bound falls below this. "
        "Measured at k=5, l=24: three quarters of instances have a bound of 0 or "
        "1, so an unfiltered corpus labels trefoils `u <= 5`. Implies "
        "--lower-bounds and needs spherogram.",
    )
    parser.add_argument(
        "--attempts-per-instance",
        type=int,
        default=16,
        help="consecutive rejections tolerated before giving up; raise it when "
        "--min-certified-lower-bound is aggressive",
    )
    parser.add_argument(
        "--lower-bounds",
        action="store_true",
        help="compute certified lower bounds per instance (spherogram and knot "
        "Floer homology; two orders of magnitude slower than generating)",
    )
    parser.add_argument(
        "--seed-family",
        choices=["unknot", "torus", "positive-braid"],
        default="unknot",
        help="where the walk starts (note 22 §4.1b). `unknot` is §4.1 and "
        "certifies nothing; the other two carry a certified L from the word.",
    )
    parser.add_argument(
        "--seed-strength",
        type=int,
        default=0,
        help="certified u of the seed: m for T(2,2m+1), the genus for a "
        "positive braid",
    )
    parser.add_argument("--braid-seed-strands", type=int, default=4)
    parser.add_argument(
        "--no-quarantine",
        action="store_true",
        help="skip the §4.3 screen. Only for corpora that will never be trained "
        "on; the manifest records that it was skipped.",
    )
    parser.add_argument("--unknotdb", type=str, default=UNKNOTDB)
    parser.add_argument(
        "--gordian-pairs",
        action="store_true",
        help="also write gordian-intervals.jsonl: the (R', seed) pair with "
        "upper = d by construction and a lower bound from sigma/2 and tau",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.count <= 0 or args.shard_size <= 0:
        raise SystemExit("count and shard size must be positive")
    if args.min_certified_lower_bound < 0:
        raise SystemExit("--min-certified-lower-bound cannot be negative")
    lower_bounds = args.lower_bounds or args.min_certified_lower_bound > 0

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
        erase_loops=not args.no_erase_loops,
        weights=WalkWeights(),
    )
    provenance = config.provenance()
    representation = build_seed(
        config,
        args.seed_family,
        args.seed_strength,
        args.braid_seed_strands,
        args.seed,
    )
    quarantine = (
        Quarantine() if args.no_quarantine else load_quarantine(args.unknotdb)
    )
    seed_reason = quarantine.screen_seed(representation)
    if seed_reason is not None:
        raise SystemExit(
            f"seed {representation.name} is quarantined ({seed_reason}); §4.1b "
            "moves the quarantine to seed selection and this is it firing"
        )
    seed_values = lipschitz_values(representation.word, representation.strands)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] | None = None
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("generator") != provenance or manifest.get("seed") != args.seed:
            raise SystemExit(
                "refusing to resume: the manifest was written with different dials"
            )
        # `status: complete` means complete *at the count it was asked for*.  A
        # resume that raises `--count` extends the corpus rather than no-opping.
        if int(manifest.get("examples", 0)) >= args.count:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return

    screened: Counter = Counter()
    floor = args.min_certified_lower_bound

    def accept(instance: InterleavedInstance) -> bool:
        """The §4.3 screen on the emitted word, then the difficulty filter.

        Order matters only for the counters: a word that is both quarantined and
        too easy is recorded as quarantined, which is the reason a reader cares
        about.
        """
        reason = quarantine.screen_word(instance.word, instance.strands)
        if reason is not None:
            screened[reason] += 1
            return False
        if floor <= 0:
            return True
        from rf_knots import seifert

        reduced = free_reduce(instance.word)
        try:
            sigma = seifert.signature(reduced, instance.strands)
        except (ValueError, seifert.BackendUnavailable):
            screened["signature-unavailable"] += 1
            return False
        if sigma == 0:
            # The target's fingerprint has sigma = 0, so this is the only branch
            # on which the determinant can match and the only one that pays for it.
            try:
                determinant = seifert.determinant(reduced, instance.strands)
            except (ValueError, seifert.BackendUnavailable):
                determinant = None
            if determinant is not None:
                reason = quarantine.screen_fingerprint(determinant, 0)
                if reason is not None:
                    screened[reason] += 1
                    return False
        # The inherited bound is free and is a theorem; the strongest of the two
        # is what the filter should be reading.
        best = max(abs(sigma) // 2, instance.u_lower_inherited)
        if best < floor:
            screened["below-min-certified-lower-bound"] += 1
            return False
        return True

    written = int(manifest.get("examples", 0)) if manifest else 0
    stream_index = int(manifest.get("next_stream_index", 0)) if manifest else 0
    shards = list(manifest.get("shards", [])) if manifest else []
    counters = {
        "certified_upper_bound": Counter(),
        "word_length": Counter(),
        "free_reduced_length": Counter(),
        "strands": Counter(),
        "witness_moves": Counter(),
        "d_upper": Counter(),
        "u_lower_inherited": Counter(),
    }
    intervals: list[dict[str, Any]] = []
    if manifest:
        for name, stored in manifest.get("distribution", {}).items():
            counters[name].update({int(k): int(v) for k, v in stored.items()})
    rejected = int(manifest.get("rejected", 0)) if manifest else 0

    started = time.perf_counter()
    while written < args.count:
        size = min(args.shard_size, args.count - written)
        path = args.output_dir / f"shard-{len(shards):06d}.jsonl"
        if path.exists():
            path.unlink()  # a torn shard from a killed run is rewritten, not appended
        store = EvidenceStore(path)
        stream = generate_many(
            config,
            seed=args.seed,
            count=size,
            start=stream_index,
            skip_trivial=not args.keep_trivial,
            accept=accept,
            max_attempts_per_instance=args.attempts_per_instance,
            seed_representation=representation,
        )
        first_index = stream_index
        for instance in stream:
            store.append(build_record(instance, config, args.seed, lower_bounds))
            if args.gordian_pairs:
                intervals.append(
                    gordian_interval(instance, right=seed_values).to_dict()
                )
            rejected += instance.index - stream_index
            stream_index = instance.index + 1
            counters["certified_upper_bound"][instance.certified_upper_bound] += 1
            counters["d_upper"][instance.d_upper] += 1
            counters["u_lower_inherited"][instance.u_lower_inherited] += 1
            counters["word_length"][len(instance.word)] += 1
            counters["strands"][instance.strands] += 1
            counters["witness_moves"][instance.witness.moves] += 1
            counters["free_reduced_length"][len(free_reduce(instance.word))] += 1
        written += size
        record = {
            "index": len(shards),
            "path": path.name,
            "count": size,
            "first_stream_index": first_index,
            "next_stream_index": stream_index,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        shards.append(record)
        elapsed = time.perf_counter() - started
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "record_schema": "rf-knots-evidence-v1",
            "solver": SOLVER,
            "outcome": OUTCOME,
            "label": "u <= metrics.certified_upper_bound",
            "status": "complete" if written == args.count else "in_progress",
            "seed": args.seed,
            "generator": provenance,
            "config_sha256": sha256_payload(provenance),
            "skip_trivial": not args.keep_trivial,
            "lower_bounds": lower_bounds,
            "min_certified_lower_bound": args.min_certified_lower_bound,
            "seed_representation": representation.provenance(),
            "witness_provenance": (
                "replay-complete" if representation.replay_complete
                else "catalogue-composed"
            ),
            "quarantine": (
                {"screened": False, "reason": "--no-quarantine"}
                if args.no_quarantine
                else {"screened": True, **quarantine.provenance()}
            ),
            "quarantine_rejections": dict(screened),
            "examples": written,
            "target_examples": args.count,
            "shard_size": args.shard_size,
            "rejected": rejected,
            "next_stream_index": stream_index,
            "shards": shards,
            "total_bytes": sum(row["bytes"] for row in shards),
            "distribution": summarise(counters),
            "instances_per_second": round(written / elapsed, 2) if elapsed else None,
        }
        atomic_json(manifest_path, manifest)
        if args.gordian_pairs:
            with (args.output_dir / "gordian-intervals.jsonl").open("w") as handle:
                for row in intervals:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    assert manifest is not None
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
