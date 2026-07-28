"""Action-space layout for the braid environments.

The action space is a flat integer range partitioned into blocks. Its size depends
only on `(max_len, max_strands)` -- never on `allow_crossing_change`, so the two
modes share one policy head.

Layout, with ``L = max_len`` and ``N = max_strands``::

    [0,          L)          REDUCE(p)          sigma^e sigma^-e -> empty, at p
    [L,         2L)          COMMUTE(p)         far commutation, at p
    [2L,        3L)          BRAID(p)           (a,b,a) -> (b,a,b), at p
    [3L, 3L+2L(N-1))         INSERT(p,g,s)      insert (s*g, -s*g) at p
    +0                       DESTABILIZE        drop the lone top generator
    +1                       STABILIZE_POS      n -> n+1, append +sigma_n
    +2                       STABILIZE_NEG      n -> n+1, append -sigma_n
    +3                       PASS               end the phase early
    [.., ..+L)               CROSSING_CHANGE(p) sigma_i <-> sigma_i^-1, at p

The word is **cyclic**: position `p+1` means `(p+1) mod length`. That is not a
storage convenience, it is the geometry -- the braid word is read by sweeping a
half-plane once around the braid axis, so it closes back on itself. Making the
representation a necklace rather than a list means conjugation is free, so the
two rotation moves that used to exist have been removed: they only ever moved an
arbitrary seam. Every local move now works across the seam, and `DESTABILIZE` no
longer needs the top generator rotated into last place.

INSERT is indexed as ``3L + ((g-1)*2 + s)*L + p`` with ``g in 1..N-1`` and
``s in {0: positive-first, 1: negative-first}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REDUCE = 0
COMMUTE = 1
BRAID = 2
INSERT = 3
DESTABILIZE = 4
STABILIZE_POS = 5
STABILIZE_NEG = 6
PASS = 7
CROSSING_CHANGE = 8

NUM_KINDS = 9

KIND_NAMES = [
    "REDUCE",
    "COMMUTE",
    "BRAID",
    "INSERT",
    "DESTABILIZE",
    "STABILIZE_POS",
    "STABILIZE_NEG",
    "PASS",
    "CROSSING_CHANGE",
]


@dataclass(frozen=True)
class ActionSpec:
    """Block offsets for one `(max_len, max_strands)` pair."""

    max_len: int
    max_strands: int

    @property
    def num_generators(self) -> int:
        return self.max_strands - 1

    @property
    def starts(self) -> np.ndarray:
        """Start index of each kind's block, length ``NUM_KINDS``."""
        length = self.max_len
        insert_block = 2 * length * self.num_generators
        sizes = [
            length,  # REDUCE
            length,  # COMMUTE
            length,  # BRAID
            insert_block,  # INSERT
            1,  # DESTABILIZE
            1,  # STABILIZE_POS
            1,  # STABILIZE_NEG
            1,  # PASS
            length,  # CROSSING_CHANGE
        ]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)

    @property
    def num_actions(self) -> int:
        return int(self.starts[-1])

    def start_of(self, kind: int) -> int:
        return int(self.starts[kind])

    # -- encoding -------------------------------------------------------------

    def encode(self, kind: int, position: int = 0, generator: int = 1, sign: int = 1) -> int:
        """Build a flat action index. `sign` is +1 or -1, `generator` is 1-based."""
        base = self.start_of(kind)
        if kind == INSERT:
            s = 0 if sign > 0 else 1
            return base + ((generator - 1) * 2 + s) * self.max_len + position
        if kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
            return base + position
        return base

    def decode(self, action: int) -> tuple[int, int, int, int]:
        """Inverse of `encode`: returns `(kind, position, generator, sign)`.

        Pure Python, for tests and the reference implementation. The jitted
        environment decodes with the same arithmetic on traced values.
        """
        starts = self.starts
        kind = int(np.searchsorted(starts, action, side="right") - 1)
        if kind == INSERT:
            rem = action - self.start_of(INSERT)
            position = rem % self.max_len
            gs = rem // self.max_len
            generator = gs // 2 + 1
            sign = 1 if gs % 2 == 0 else -1
            return kind, int(position), int(generator), int(sign)
        position = action - self.start_of(kind)
        if kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
            return kind, int(position), 1, 1
        return kind, 0, 1, 1

    def describe(self, action: int) -> str:
        kind, position, generator, sign = self.decode(action)
        name = KIND_NAMES[kind]
        if kind == INSERT:
            letter = sign * generator
            return f"{name}(p={position}, {letter:+d}{-letter:+d})"
        if kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
            return f"{name}(p={position})"
        return name
