"""Constructive CC<=2 knots with explicit cancellation witnesses, no model oracle."""

import random

from rf_knots.actions import ActionSpec, CROSSING_CHANGE, REDUCE, DESTABILIZE
from rf_knots.evidence import UnknotWitness


def conjugate_witness(seed, index, *, max_strands=8, min_depth=4, max_depth=16):
    if max_strands < 3 or min_depth < 0 or max_depth < min_depth or index < 0:
        raise ValueError("invalid constructive size range")
    rng = random.Random((seed << 32) + index)
    strand_span = max_strands - 2
    n = 3 + index % strand_span
    changes = 1 + (index // strand_span) % 2
    depth = min_depth + index % (max_depth - min_depth + 1)
    placements = sorted(rng.randrange(n - 1) for _ in range(changes))
    word, blocks = [], []
    for position in range(n - 1):
        for _ in range(placements.count(position)):
            prefix = [rng.choice((-1, 1)) * rng.randrange(1, n) for _ in range(depth)]
            g = rng.choice((-1, 1)) * rng.randrange(1, n)
            start = len(word)
            word.extend(prefix + [g, g] + [-x for x in reversed(prefix)])
            blocks.append((start, depth))
        word.append(position + 1)
    spec = ActionSpec(len(word) + 4, n + 1)
    # Crossing changes preserve length, so all recorded centres remain valid.
    actions = [spec.encode(CROSSING_CHANGE, start + d + 1) for start, d in blocks]
    # Remove rightmost blocks first so offsets of earlier blocks do not move.
    for start, d in reversed(blocks):
        actions.extend(spec.encode(REDUCE, start + j) for j in range(d, -1, -1))
    actions.extend(spec.encode(DESTABILIZE) for _ in range(n - 1))
    witness = UnknotWitness.from_actions(tuple(word), n, spec, actions)
    witness.verify()
    if witness.crossing_changes != changes:
        raise ValueError("construction crossing count changed")
    return witness
