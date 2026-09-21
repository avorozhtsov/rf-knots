"""Conservative zero-CC preprocessing for a GLOBAL knot-embedding view.

Not a canonical knot identifier, not a solver, and not a policy-state transform:
word positions change. Every deletion is a replayed REDUCE or DESTABILIZE in the
existing Artin environment. Raw site/layout tokens must keep their own coordinates.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache

from rf_knots import reference
from rf_knots.actions import DESTABILIZE, REDUCE, ActionSpec


@dataclass(frozen=True)
class NormalizedBraid:
    word: tuple[int, ...]
    strands: int
    actions: tuple[int, ...]
    spec: ActionSpec


@lru_cache(maxsize=8192)
def fast_normalize_global_view(word: tuple[int, ...], strands: int) -> tuple[tuple[int, ...], int]:
    """Same endpoint as the replay reference, O(L log L) time / O(L) space.

    Original-index order is preserved by both deletions. A minimum heap thus
    chooses exactly the reference's first cancellable adjacent pair, including
    the cyclic seam. Linked-list deletion only changes adjacency at its two
    ends. Per-generator live index sets implement top-strand counts without
    allocating an array of size N or scanning N unused strands. No trace is
    materialized here; the slower reference supplies differential replay tests.
    """
    if strands < 1 or any(g == 0 or abs(g) >= strands for g in word):
        raise ValueError("invalid Artin braid")
    size = len(word)
    if not size:
        return word, strands
    prev = [(i - 1) % size for i in range(size)]
    nxt = [(i + 1) % size for i in range(size)]
    alive = [True] * size
    indices: dict[int, set[int]] = {}
    for i, g in enumerate(word):
        indices.setdefault(abs(g), set()).add(i)
    pending = []

    def candidate(i):
        if alive[i] and nxt[i] != i and word[i] == -word[nxt[i]]:
            heapq.heappush(pending, i)

    def remove(i):
        alive[i] = False
        indices[abs(word[i])].remove(i)
        nxt[prev[i]] = nxt[i]
        prev[nxt[i]] = prev[i]
        candidate(prev[i])

    for i in range(size):
        candidate(i)
    while True:
        while pending:
            i = heapq.heappop(pending)
            j = nxt[i]
            if alive[i] and j != i and word[i] == -word[j]:
                remove(j)
                remove(i)
        top = indices.get(strands - 1, ())
        if strands <= 1 or len(top) != 1:
            return tuple(g for i, g in enumerate(word) if alive[i]), strands
        remove(next(iter(top)))
        strands -= 1


@lru_cache(maxsize=32768)
def normalize_global_view(word: tuple[int, ...], strands: int) -> NormalizedBraid:
    """Greedy cyclic free reduction and top-strand destabilisation.

    Intentionally transparent reference implementation, quadratic worst case;
    benchmark and replace by a proven equivalent linear stack before long-word
    production use. It never flips a crossing or identifies mirror images.
    """
    if strands < 1 or any(g == 0 or abs(g) >= strands for g in word):
        raise ValueError("invalid Artin braid")
    spec = ActionSpec(max(len(word), 1), max(strands, 2))
    actions = []
    while True:
        length = len(word)
        position = next(
            (p for p in range(length) if length > 1 and word[p] == -word[(p + 1) % length]), None
        )
        if position is not None:
            action = spec.encode(REDUCE, position)
        elif strands > 1 and sum(abs(g) == strands - 1 for g in word) == 1:
            action = spec.encode(DESTABILIZE)
        else:
            return NormalizedBraid(word, strands, tuple(actions), spec)
        if not reference.is_legal(spec, word, strands, action, False):
            raise AssertionError("normalisation attempted an illegal zero-CC step")
        word, strands = reference.apply(spec, word, strands, action)
        actions.append(action)
