"""Isolated zero-CC view generation with strict per-operation forward validation."""

import hashlib
from dataclasses import replace

from rf_knots.embedding_augmentation import (
    AugmentedRepresentation,
    ZeroCCOperation,
    augment_markov_zero_cc,
    reverse_markov_zero_cc,
)


def seed_for(seed, *parts):
    value = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def validate_knot(word, strands):
    if type(strands) is not int or strands < 1:
        raise ValueError("positive integer strand count required")
    if any(type(g) is not int or g == 0 or abs(g) >= strands for g in word):
        raise ValueError("generator outside braid group")
    permutation = list(range(strands))
    for g in word:
        i = abs(g) - 1
        permutation[i], permutation[i + 1] = permutation[i + 1], permutation[i]
    visited, i = set(), 0
    while i not in visited:
        visited.add(i)
        i = permutation[i]
    if len(visited) != strands:
        raise ValueError("require a one-component braid closure")


def forward_replay(word, strands, operations):
    word = tuple(word)
    validate_knot(word, strands)
    for op in operations:
        if any(type(v) is not int for v in (op.position, op.generator, op.sign, op.amount)):
            raise ValueError("operation fields must be integers")
        if op.kind == "cyclic_shift":
            if op.position or op.generator or op.sign or not 0 <= op.amount <= len(word):
                raise ValueError("invalid origin shift")
            word = word[op.amount :] + word[: op.amount]
        elif op.kind in ("insert_inverse_pair", "conjugate"):
            if not 1 <= op.generator < strands or op.sign not in (-1, 1) or op.amount:
                raise ValueError("invalid insertion/conjugation generator")
            g = op.sign * op.generator
            if op.kind == "insert_inverse_pair":
                if not 0 <= op.position <= len(word):
                    raise ValueError("invalid insertion position")
                word = word[: op.position] + (g, -g) + word[op.position :]
            else:
                if op.position:
                    raise ValueError("conjugation must wrap the whole word")
                word = (g, *word, -g)
        elif op.kind == "stabilize":
            if (
                op.generator != strands
                or op.position != len(word)
                or op.sign not in (-1, 1)
                or op.amount
            ):
                raise ValueError("invalid Markov stabilization")
            word = (*word, op.sign * strands)
            strands += 1
        else:
            raise ValueError("unsupported/non-zero-CC operation")
        validate_knot(word, strands)
    return tuple(word), strands


def verify_view(source, augmented):
    endpoint = forward_replay(source.word, source.strands, augmented.operations)
    if endpoint != (augmented.word, augmented.strands):
        raise ValueError("forward certificate endpoint changed")
    if reverse_markov_zero_cc(augmented) != (source.word, source.strands):
        raise ValueError("reverse certificate differs from source")
    return dict(
        operations=len(augmented.operations),
        crossing_changes=0,
        forward_reverse_exact=True,
        intermediate_braid_and_closure_checks=True,
    )


def make_view(source, seed, rounds):
    if source.split != "train" or type(rounds) is not int or not 1 <= rounds <= 64:
        raise ValueError("TRAIN source and bounded positive round count required")
    validate_knot(source.word, source.strands)
    current, operations = source, []
    if current.strands == 1:
        # This must precede any insertion/conjugation involving generator1.
        operations.append(ZeroCCOperation("stabilize", 0, 1, 1))
        current = replace(current, word=(1,), strands=2)
    for index in range(rounds):
        next_view = augment_markov_zero_cc(current, seed_for(seed, index), simple=False)
        verify_view(current, next_view)
        operations.extend(next_view.operations)
        current = replace(current, word=next_view.word, strands=next_view.strands)
    result = AugmentedRepresentation(current.word, current.strands, tuple(operations))
    verify_view(source, result)
    return result
