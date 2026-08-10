r"""A floor on the objective that no policy can beat, because it is a theorem.

## What this is for

`raster-bounded` is a candidate that differs from the roster in
[research/16](../../research/16-scientists-collaboration.md) in one way: it is not
a new encoder. `window-local`, `raster-axial`, `cyclic-memory` and `strand-graph`
all vary how the diagram is *read*. This varies what the value head is *allowed to
say*.

The reason is a measurement. In the probe sweep behind
[research/18](../../research/18-raster-representation.md), predicting the knot
determinant was a **null for every encoder** -- no arm reached `R^2 = 0.5` in
distribution and every arm went negative out of it. Networks over these diagrams
learn diagram manipulation, not knot recognition. But the unknotting number is a
property of the knot, not of the diagram. So a value head is being asked for
something the encoder demonstrably cannot compute, and it will guess.

It does not have to guess, because the quantity has a computable lower bound:

```
u(K) >= max( |sigma(K)|/2 ,  |tau(K)| ,  2 if H_1(Sigma_2(K)) is not cyclic )
```

All three are theorems (Murasugi; Ozsvath-Szabo; Montesinos/Lickorish) and all
three are computed in `rf_knots.seifert`. This module turns them into a floor on
the scientific objective

```
L_AB = A * crossing_changes + B * semantic_moves
```

so the value head can be clamped to it and a search can prune against it.

## Why a floor and not a feature

Handing the bound to the network as an input channel makes it a *hint*: the
network may still predict below it, and usually will early in training when the
value head is near its initialisation. Clamping makes it a **constraint**. The
difference matters because the bound is exactly the regime where the network is
weakest -- hard knots, where `u` is large and the diagram is long.

The same number then does double duty in search: `spent + floor(state)` is an
admissible cost-to-go, so a branch that cannot beat the incumbent is cut. That is
the branch-and-bound in `rf_knots.unknot_search`, which pruned 53 of 55 branches
on `5_1` and settled `u(7_5) = 2` in five seconds against a standing agent record
of 6.

## The honest caveat

The floor is only useful where it is tight. On `6_3` all three bounds are zero or
inapplicable -- `sigma = 0`, `tau = 0`, `H_1` cyclic -- while `u = 1`, so the
floor contributes nothing there. Before betting on this candidate, measure what
fraction of the *target* knots the bound actually pins; `certified_floor_report`
exists for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

Word = tuple[int, ...]


@dataclass(frozen=True)
class Floor:
    """The certified minimum remaining objective cost, and where it came from."""

    crossing_changes: int
    cost: float
    method: str
    details: dict

    @property
    def informative(self) -> bool:
        """Whether the bound says anything a trivial `>= 0` does not."""
        return self.crossing_changes > 0


@lru_cache(maxsize=8192)
def _bound(word: Word, strands: int) -> tuple[int, str, tuple]:
    """`max` over the certified lower bounds, with the method that achieved it.

    Cached because it costs a Seifert matrix and a knot Floer homology, and
    because a search revisits the same knot through many different diagrams --
    every type-preserving move leaves this value unchanged.
    """
    if not word:
        return 0, "solved", ()
    from rf_knots.lower_bounds import computed_claims

    claims = computed_claims(word, strands)
    if not claims:
        return 0, "unavailable", ()
    best = max(claims, key=lambda claim: claim.value)
    return best.value, best.method, tuple(sorted(best.details.items(), key=str))


def certified_floor(
    word: Word,
    strands: int,
    *,
    ratio: float = 1.0,
    moves_floor: int = 0,
) -> Floor:
    """Least `A * crossing_changes + B * semantic_moves` still to be paid.

    `ratio` is `A / B`, matching the environment's `log(A/B)` conditioning; the
    cost is expressed in units of `B`. `moves_floor` is any separately known
    minimum on semantic moves -- zero unless a caller has one, since a bound on
    move count is not what the theorems give.

    Admissible by construction: every crossing change costs `A`, and no sequence
    can use fewer than `u(K)` of them, so `A * lower_bound(u)` can never exceed
    the true remaining cost.
    """
    letters = tuple(int(x) for x in word if int(x))
    changes, method, details = _bound(letters, strands)
    return Floor(
        crossing_changes=changes,
        cost=ratio * changes + moves_floor,
        method=method,
        details=dict(details),
    )


def clamp_cost_to_go(predicted: float, floor: Floor) -> float:
    """Raise a predicted remaining cost to the floor. Never lowers it."""
    return max(float(predicted), floor.cost)


def clamp_value(
    predicted: float,
    floor: Floor,
    cap: float,
    *,
    spent_cost: float = 0.0,
    solver_to_move: bool = True,
) -> float:
    """Clamp the environment payoff using a certified total-cost lower bound.

    A successful serial episode pays ``1 - 2 * total_cost / cap``; failure pays
    ``-1``.  Therefore ``spent_cost + floor.cost`` gives an upper bound on the
    solver's eventual value.  The opponent sees the negated payoff and receives
    the symmetric lower bound.  This function intentionally does not assume the
    older ``-cost/cap`` convention: that is not the payoff used by the current
    semantic-move environment.
    """
    if cap <= 0:
        return float(predicted)
    normalized = min(max((float(spent_cost) + floor.cost) / cap, 0.0), 1.0)
    solver_ceiling = 1.0 - 2.0 * normalized
    if solver_to_move:
        return min(float(predicted), solver_ceiling)
    return max(float(predicted), -solver_ceiling)


def certified_floor_report(instances, ratio: float = 1.0) -> dict:
    """How often the floor says anything, over `(word, strands)` pairs.

    This is the pre-check the module docstring asks for: a candidate built on the
    bound is only as good as the fraction of its targets the bound actually pins.
    """
    floors = [certified_floor(word, strands, ratio=ratio) for word, strands in instances]
    informative = [f for f in floors if f.informative]
    methods: dict[str, int] = {}
    for floor in informative:
        methods[floor.method] = methods.get(floor.method, 0) + 1
    return {
        "instances": len(floors),
        "informative": len(informative),
        "informative_fraction": len(informative) / max(len(floors), 1),
        "mean_crossing_change_floor": (
            sum(f.crossing_changes for f in floors) / max(len(floors), 1)
        ),
        "max_crossing_change_floor": max((f.crossing_changes for f in floors), default=0),
        "by_method": methods,
    }
