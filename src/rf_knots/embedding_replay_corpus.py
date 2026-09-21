"""Replay-first, chirality-sensitive embedding data with invariant-group splits.

A fingerprint is NOT an identity certificate. Assigning every equal fingerprint
to one split is nevertheless conservative for leakage: isotopic knots have the
same invariants. Mirrored knots are grouped together for splitting, not labelled
as distance zero. All train witness states must belong to train, not just roots.
Validation query types are absent from training; reference endpoints may be known.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from functools import lru_cache

from rf_knots import reference, seifert
from rf_knots.actions import INSERT, STABILIZE_POS, ActionSpec
from rf_knots.evidence import BraidState, UnknotWitness, WitnessStep
from rf_knots.interleaved_generator import (
    InterleavedConfig,
    generate,
    inverse_action,
    positive_braid_seed,
)


def canonical(word, strands):
    word = tuple(word)
    return strands, min(word[i:] + word[:i] for i in range(len(word))) if word else ()


@lru_cache(maxsize=65536)
def invariants(word: tuple[int, ...], strands: int) -> tuple[int, int]:
    """Exact determinant and signed signature from one integral Seifert matrix."""
    if reference.num_components(word, strands) != 1:
        raise ValueError("the corpus accepts knots, not multi-component links")
    if not word:
        return 1, 0
    form = seifert.symmetrised(seifert.seifert_matrix(word, strands))
    determinant = abs(seifert.integer_determinant(form))
    signature = -seifert.signature_of(form)
    if determinant <= 0 or determinant % 2 != 1 or signature % 2:
        raise ValueError("invalid knot invariant parity")
    return determinant, signature


def owner(fingerprint: tuple[int, int]) -> str:
    det, sigma = fingerprint
    key = (det, abs(sigma))
    if key == (49, 0):
        return "quarantine"  # final BH target orbit, including collisions
    if key in {(1, 0), (3, 2)}:
        return "train"  # unknot reference and the explicit chiral trefoil sentinel
    digest = hashlib.sha256(f"rf-embedding-split-v1:{det}:{abs(sigma)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "test" if bucket == 0 else "validation" if bucket <= 2 else "train"


def mirror(witness: UnknotWitness) -> UnknotWitness:
    def state(s):
        return BraidState(tuple(-g for g in s.word), s.strands)

    def action(a):
        if a.kind == "INSERT":
            return replace(a, sign=-a.sign)
        if a.kind == "STABILIZE_POS":
            return replace(a, kind="STABILIZE_NEG")
        if a.kind == "STABILIZE_NEG":
            return replace(a, kind="STABILIZE_POS")
        return a

    result = replace(
        witness,
        start=state(witness.start),
        steps=tuple(WitnessStep(action(s.action), state(s.after)) for s in witness.steps),
    )
    result.verify()
    return result


def path_fingerprints(witness: UnknotWitness) -> tuple[tuple[int, int], ...]:
    """Only crossing changes can change the fingerprint along a replayed path."""
    witness.verify()
    states = [witness.start] + [
        s.after for s in witness.steps if s.action.kind == "CROSSING_CHANGE"
    ]
    return tuple(invariants(s.word, s.strands) for s in states)


def lift(witness: UnknotWitness, strands: int, length: int) -> tuple[UnknotWitness, int]:
    """A replayed zero-CC lift to strictly larger dimensions, plus prefix length."""
    spec = ActionSpec(max(witness.max_len, length + 4), max(witness.max_strands, strands + 1))
    word, n = witness.start.word, witness.start.strands
    inverse = []
    while n < strands or len(word) < length:
        action = spec.encode(STABILIZE_POS) if n < strands else spec.encode(INSERT, 0, 1, 1)
        after, after_n = reference.apply(spec, word, n, action)
        undo = inverse_action(spec, word, n, action, after, after_n)
        if undo is None:
            raise ValueError("size lift must be exactly reversible")
        inverse.append(undo)
        word, n = after, after_n
    actions = tuple(reversed(inverse)) + tuple(s.action.to_flat(spec) for s in witness.steps)
    result = UnknotWitness.from_actions(word, n, spec, actions)
    return result, len(inverse)


def verify_record(record: dict, *, check_invariants: bool = True) -> UnknotWitness:
    witness = UnknotWitness.from_dict(record["witness"])
    witness.verify()
    pair = record["interval"]
    if (tuple(pair["left"]["word"]), pair["left"]["strands"]) != (
        witness.start.word,
        witness.start.strands,
    ):
        raise ValueError("pair and witness roots differ")
    prefix = record["pair_prefix_length"]
    if not 0 <= prefix <= len(witness.steps):
        raise ValueError("invalid pair prefix")
    reached = witness.steps[prefix - 1].after if prefix else witness.start
    if (reached.word, reached.strands) != (tuple(pair["right"]["word"]), pair["right"]["strands"]):
        raise ValueError("pair prefix does not reach its claimed endpoint")
    upper = sum(s.action.kind == "CROSSING_CHANGE" for s in witness.steps[:prefix])
    if pair["upper"] != upper or pair["witness_semantic_length"] != prefix:
        raise ValueError("pair upper bound must match the replayed prefix")
    if check_invariants:
        left = invariants(witness.start.word, witness.start.strands)
        right = invariants(reached.word, reached.strands)
        if list(left) != record["left_fingerprint"] or list(right) != record["right_fingerprint"]:
            raise ValueError("stored fingerprints differ from recomputation")
        if pair["lower"] != abs(left[1] - right[1]) // 2 or pair["lower"] > upper:
            raise ValueError("invalid signed-signature lower bound")
        expected_owner = "validation" if record["split"] == "size_shift" else record["split"]
        if owner(left) != expected_owner:
            raise ValueError("query fingerprint is in the wrong split")
        if owner(right) not in {"train", "validation"}:
            raise ValueError("reference endpoint is quarantined or reserved for test")
        if record["split"] == "train":
            if owner(right) != "train" or any(
                owner(f) != "train" for f in path_fingerprints(witness)
            ):
                raise ValueError("a train path enters a held-out fingerprint group")
    return witness


def build_corpus(count: int, validation_count: int, seed: int, progress=None) -> dict:
    """Balance exact 0/1/2 and interval rows; never inspect the test bucket's scores."""
    if count % 4 or validation_count % 4 or count < 4 or validation_count < 4:
        raise ValueError("corpus sizes must be positive multiples of four")
    groups = {"train": [], "validation": []}
    targets = {"train": count // 4, "validation": validation_count // 4}
    totals = {name: Counter() for name in groups}
    rejects = Counter()
    seen = set()
    fingerprint_counts = {name: Counter() for name in groups}
    for index in range((count + validation_count) * 200):
        changes = index % 3
        config = InterleavedConfig(
            crossing_changes=changes,
            walk_length=8,
            settle_walk_length=6,
            target_strands=3 + (index // 3) % 3,
            target_length=18 + (index // 9) % 8,
            max_len=40,
            max_strands=7,
            strand_band=1,
        )
        root = positive_braid_seed(
            config.spec(),
            strands=2 + (index // 27) % 4,
            genus=1 + (index // 108) % 6,
            seed=seed + index,
            churn=4,
        )
        instance = generate(config, seed, index, root)
        witness = instance.witness
        assert witness is not None
        if (index // 2) % 2:
            witness = mirror(witness)
        prefix = len(instance.reverse_actions)
        right = witness.steps[prefix - 1].after if prefix else witness.start
        left_fp = invariants(witness.start.word, witness.start.strands)
        right_fp = invariants(right.word, right.strands)
        split = owner(left_fp)
        if split not in groups:
            rejects[split] += 1
            continue
        if owner(right_fp) not in {"train", "validation"}:
            rejects["reserved_reference"] += 1
            continue
        lower, upper = abs(left_fp[1] - right_fp[1]) // 2, instance.d_upper
        if lower > upper:
            raise ValueError("signature and replay disagree")
        category = str(lower) if lower == upper else "interval"
        if totals[split][category] >= targets[split]:
            rejects["filled_category"] += 1
            continue
        fp_key = (left_fp[0], abs(left_fp[1]))
        if fingerprint_counts[split][fp_key] >= max(16, targets[split] // 2):
            rejects["fingerprint_cap"] += 1
            continue
        key = canonical(witness.start.word, witness.start.strands)
        if key in seen:
            rejects["duplicate_root"] += 1
            continue
        if split == "train" and (
            owner(right_fp) != "train"
            or any(owner(f) != "train" for f in path_fingerprints(witness))
        ):
            rejects["heldout_on_train_path"] += 1
            continue
        record = {
            "split": split,
            "config": config.provenance(),
            "seed": seed,
            "index": index,
            "seed_family": root.provenance(),
            "mirrored": bool((index // 2) % 2),
            "pair_prefix_length": prefix,
            "left_fingerprint": list(left_fp),
            "right_fingerprint": list(right_fp),
            "interval": {
                "metric": "cc",
                "left": witness.start.to_dict(),
                "right": right.to_dict(),
                "lower": lower,
                "upper": upper,
                "witness_semantic_length": prefix,
            },
            "witness": witness.to_dict(),
        }
        verify_record(record)
        groups[split].append(record)
        seen.add(key)
        totals[split][category] += 1
        fingerprint_counts[split][fp_key] += 1
        if progress and sum(map(len, groups.values())) % 64 == 0:
            progress(
                {
                    "attempts": index + 1,
                    "counts": {k: len(v) for k, v in groups.items()},
                    "categories": {k: dict(v) for k, v in totals.items()},
                    "rejects": dict(rejects),
                }
            )
        if len(groups["train"]) == count and len(groups["validation"]) == validation_count:
            break
    else:
        raise RuntimeError(f"could not fill balanced corpus: {totals}; rejects {rejects}")
    max_length = max(len(r["witness"]["start"]["word"]) for r in groups["train"])
    max_strands = max(r["witness"]["start"]["strands"] for r in groups["train"])
    # Guarantee an actual out-of-size panel, not merely a higher generator target.
    large = []
    for original in groups["validation"]:
        witness = UnknotWitness.from_dict(original["witness"])
        augmented, prefix = lift(witness, max_strands + 2, max_length + 16)
        row = {
            **original,
            "split": "size_shift",
            "witness": augmented.to_dict(),
            "pair_prefix_length": original["pair_prefix_length"] + prefix,
            "interval": {
                **original["interval"],
                "left": augmented.start.to_dict(),
                "witness_semantic_length": original["pair_prefix_length"] + prefix,
            },
        }
        verify_record(row)
        large.append(row)
    return {
        "schema": "rf-knots-replay-fingerprint-corpus-v1",
        **groups,
        "size_shift": large,
        "split_contract": "validation query fingerprint groups absent from ALL train path states; "
        "reference endpoints may be known; mirror orbits share split; "
        "fingerprint collision merges distinct types conservatively",
        "split_key": "sha256-v1(det,abs(signature)); unknown identities are not claimed equal",
        "counts": {k: len(v) for k, v in groups.items()},
        "categories": {k: dict(v) for k, v in totals.items()},
        "distinct_query_fingerprints": {k: len(v) for k, v in fingerprint_counts.items()},
        "attempts": index + 1,
        "rejects": dict(rejects),
        "quarantine": "det=49,abs(signature)=0; separate hash-test bucket is never scored",
    }
