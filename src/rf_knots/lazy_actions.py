"""Exact sparse catalogue and factorized proposals for reference actions.

O(L) setup/storage; each insertion is indexed arithmetically in O(1), so taking
k candidates costs O(L+k), not O(L*N). Materialising k successors still copies
their words and costs O(k*L). This catalogue proposes, it does not prune.

``factorized_action_candidates`` is the opt-in neural-search interface: a model
supplies scores for action kind, position, generator, and sign separately. Only
the top ``k`` factors are combined, then every resulting flat action is checked
by the exact reference legality predicate. It never treats an unproposed action
as a negative label, and it does not change the historical catalogue/MCTS path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rf_knots import reference
from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)


def _top_indices(scores: Sequence[float], limit: int) -> list[int]:
    """Stable finite top-k indices without sorting an L*N Cartesian product."""
    if limit < 1:
        return []
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("factor scores must be one-dimensional")
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return []
    order = finite[np.argsort(-values[finite], kind="stable")[:limit]]
    return [int(index) for index in order]


def factorized_action_candidates(
    spec: ActionSpec,
    word: tuple[int, ...],
    strands: int,
    *,
    kind_scores: Sequence[float],
    position_scores: Sequence[Sequence[float]],
    generator_scores: Sequence[float],
    sign_scores: Sequence[float] = (0.0, 0.0),
    top_positions: int = 8,
    top_generators: int = 8,
    top_signs: int = 2,
    allow_crossing: bool = False,
) -> tuple[tuple[int, float], ...]:
    """Return exact legal actions from a factorized top-k proposal.

    ``position_scores`` has one row per action kind and at least ``len(word)``
    columns.  ``generator_scores`` is ordered for the active generators
    ``1..strands-1`` (or ``1..strands`` for a cyclic band with ``strands>=3``),
    while ``sign_scores`` is ordered ``(+1, -1)``.  The returned pairs are
    sorted by the additive factor score and contain no duplicate actions.

    The model may score all positions and all generators independently, which is
    O(L+N).  The Cartesian product is formed only for the selected top-k factors,
    so at most ``2*top_positions*top_generators*top_signs`` INSERT proposals are
    checked.  Exact legality is always applied after combination; this helper is
    therefore a proposal policy, never a proof that omitted actions are bad.
    """
    kinds = np.asarray(kind_scores, dtype=np.float64)
    positions = np.asarray(position_scores, dtype=np.float64)
    if kinds.ndim != 1 or kinds.shape[0] <= INSERT:
        raise ValueError("kind_scores must contain all action kinds")
    if positions.ndim != 2 or positions.shape[0] != kinds.shape[0]:
        raise ValueError("position_scores must be [num_kinds, positions]")
    if positions.shape[1] < max(1, len(word)):
        raise ValueError("position_scores has fewer columns than the word")
    generators = np.asarray(generator_scores, dtype=np.float64)
    signs = np.asarray(sign_scores, dtype=np.float64)
    if signs.ndim != 1 or signs.shape[0] != 2:
        raise ValueError("sign_scores must be (+1, -1)")
    expected_generators = strands if spec.cyclic_band_generators and strands >= 3 else strands - 1
    if expected_generators < 0 or generators.ndim != 1 or generators.shape[0] < expected_generators:
        raise ValueError("generator_scores does not cover the active generators")

    candidates: dict[int, float] = {}
    # Positionless actions have no meaningful position factor.
    for kind in (DESTABILIZE, STABILIZE_POS, STABILIZE_NEG):
        if kind < kinds.shape[0] and np.isfinite(kinds[kind]):
            action = spec.encode(kind)
            if reference.is_legal(spec, word, strands, action, allow_crossing):
                candidates[action] = float(kinds[kind])

    position_limit = min(top_positions, max(1, len(word)))
    for kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
        if kind >= kinds.shape[0] or not np.isfinite(kinds[kind]):
            continue
        for position in _top_indices(positions[kind, : max(1, len(word))], position_limit):
            action = spec.encode(kind, position)
            if reference.is_legal(spec, word, strands, action, allow_crossing):
                candidates[action] = float(kinds[kind] + positions[kind, position])

    insert_positions = _top_indices(
        positions[INSERT, : max(1, len(word))], position_limit
    )
    insert_generators = _top_indices(generators[:expected_generators], top_generators)
    insert_signs = _top_indices(signs, min(top_signs, 2))
    for position in insert_positions:
        for generator_index in insert_generators:
            generator = generator_index + 1
            for sign_index in insert_signs:
                sign = 1 if sign_index == 0 else -1
                action = spec.encode(INSERT, position, generator, sign)
                if reference.is_legal(spec, word, strands, action, allow_crossing):
                    candidates[action] = float(
                        kinds[INSERT]
                        + positions[INSERT, position]
                        + generators[generator_index]
                        + signs[sign_index]
                    )

    # Stable ordering makes replay and manifests deterministic when scores tie.
    return tuple(sorted(candidates.items(), key=lambda item: (-item[1], item[0])))


class LazyActionCatalogue:
    def __init__(
        self, spec: ActionSpec, word: tuple[int, ...], strands: int, allow_crossing: bool = False
    ):
        if strands < 1 or strands > spec.max_strands or len(word) > spec.max_len:
            raise ValueError("state outside action resource caps")
        largest = strands if spec.cyclic_band_generators else strands - 1
        if any(g == 0 or abs(g) > largest for g in word):
            raise ValueError("invalid active generator")
        self.spec, self.word, self.strands = spec, word, strands
        ordinary = []
        # Global simplifications first; then all local non-growing proposals.
        for kind in (DESTABILIZE, STABILIZE_POS, STABILIZE_NEG):
            action = spec.encode(kind)
            if reference.is_legal(spec, word, strands, action, allow_crossing):
                ordinary.append(action)
        for position in range(len(word)):
            for kind in (REDUCE, COMMUTE, BRAID, CROSSING_CHANGE):
                action = spec.encode(kind, position)
                if reference.is_legal(spec, word, strands, action, allow_crossing):
                    ordinary.append(action)
        self.ordinary = tuple(ordinary)
        self.positions = max(1, len(word))
        self.generators = strands if spec.cyclic_band_generators and strands >= 3 else strands - 1
        self.insertions = (
            2 * self.positions * self.generators if len(word) + 2 <= spec.max_len else 0
        )

    def __len__(self):
        return len(self.ordinary) + self.insertions

    def action_at(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Interleave both sources so neither waits for a huge other block.
        common = min(len(self.ordinary), self.insertions)
        if index < 2 * common:
            if index % 2 == 0:
                return self.ordinary[index // 2]
            insertion = index // 2
        elif len(self.ordinary) > common:
            return self.ordinary[index - common]
        else:
            insertion = index - common
        position = insertion % self.positions
        shell = insertion // self.positions
        sign = 1 if shell % 2 == 0 else -1
        # A cyclic permutation of ALL generators at each position. Nearby
        # generator first is a proposal ordering, not a restricted move set.
        anchor = abs(self.word[position]) - 1 if self.word else 0
        generator = (anchor + shell // 2) % self.generators + 1
        return self.spec.encode(INSERT, position, generator, sign)

    def materialize(self, start: int, count: int):
        if start < 0 or count < 0:
            raise ValueError("negative catalogue slice")
        for i in range(start, min(start + count, len(self))):
            action = self.action_at(i)
            word, strands = reference.apply(self.spec, self.word, self.strands, action)
            yield action, word, strands
