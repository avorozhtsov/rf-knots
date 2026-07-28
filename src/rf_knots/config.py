from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BraidConfig:
    """Static shape and rule parameters for the braid environments.

    Every field is a Python int/bool and is baked into the jitted step function.
    Changing any of them changes the action space, so a network trained under one
    config cannot be loaded under another unless the policy head is positional.

    Attributes:
        max_len: array capacity `L` for the braid word. Simplification of a hard
            instance may need to *increase* the word length before it can shrink,
            so this must be generous: budget roughly 3x the longest instance you
            intend to pose. Instances that would need more than `L` letters become
            artificially unsolvable; `EnvStats.overflow_rate` tracks that.
        max_strands: strand capacity `N`. Generators are sigma_1 .. sigma_{n-1}
            for the currently active strand count `n <= N`.
        scramble_budget: `K`, the number of moves the Scrambler may spend.
            This is the difficulty dial.
        simplify_budget: `M`, the number of moves the Simplifier gets. Set
            `M = c*K` with c > 1 so the Simplifier is not forced to find the exact
            inverse path.
        allow_crossing_change: expose the crossing-change move to the Simplifier.
            Off for the Scrambler-vs-Simplifier game (it would break the
            by-construction ground truth); on for the unknotting-number mode.
    """

    max_len: int = 64
    max_strands: int = 8
    scramble_budget: int = 12
    simplify_budget: int = 48
    allow_crossing_change: bool = False
    simplifier_speed_bonus: float = 0.0

    def __post_init__(self) -> None:
        if self.max_len < 6:
            raise ValueError("max_len must be at least 6")
        if self.max_strands < 2:
            raise ValueError("max_strands must be at least 2")
        if self.scramble_budget < 1 or self.simplify_budget < 1:
            raise ValueError("budgets must be positive")
        if not 0.0 <= self.simplifier_speed_bonus < 1.0:
            raise ValueError("simplifier_speed_bonus must be in [0, 1)")


TIER0 = BraidConfig(max_len=32, max_strands=5, scramble_budget=6, simplify_budget=24)
TIER1 = BraidConfig(max_len=64, max_strands=8, scramble_budget=12, simplify_budget=48)
