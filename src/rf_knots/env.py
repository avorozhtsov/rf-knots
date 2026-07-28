"""Pgx environment: Scrambler vs. Simplifier on braid words.

    phase 0 (Scrambler, K plies):  start from the empty 1-braid, whose closure is
                                   the unknot. Apply K type-preserving moves. The
                                   closure is STILL the unknot, by construction.
    phase 1 (Simplifier, M plies): sees only the resulting word, not the move
                                   history. Wins iff it reaches the empty 1-braid.
    payoff:                        zero-sum, +1 / -1.

Ground truth is exact and free at every difficulty because the instance is
generated from the answer, not labelled after the fact. That is the property this
whole design is built around: no majority-vote pseudo-labels, no verifier drift.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pgx import core
from pgx._src.struct import dataclass

from rf_knots import braid
from rf_knots.actions import CROSSING_CHANGE, PASS, ActionSpec
from rf_knots.config import TIER1, BraidConfig

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    """Field defaults are placeholders; `BraidUnknot._init` sets real shapes."""

    current_player: jax.Array = jnp.int32(0)
    observation: jax.Array = jnp.zeros((1, 1), dtype=jnp.float32)
    rewards: jax.Array = jnp.zeros(2, dtype=jnp.float32)
    terminated: jax.Array = FALSE
    truncated: jax.Array = FALSE
    legal_action_mask: jax.Array = jnp.zeros(1, dtype=jnp.bool_)
    _step_count: jax.Array = jnp.int32(0)

    _word: jax.Array = jnp.zeros(1, dtype=jnp.int32)
    _n: jax.Array = jnp.int32(1)
    _phase: jax.Array = jnp.int32(0)
    _budget: jax.Array = jnp.int32(0)
    _scrambler: jax.Array = jnp.int32(0)
    _crossing_changes: jax.Array = jnp.int32(0)
    _log_ratio: jax.Array = jnp.float32(0.0)

    @property
    def env_id(self) -> core.EnvId:  # type: ignore[override]
        return "braid_unknot"  # type: ignore[return-value]

    def to_svg(self, **kwargs) -> str:  # type: ignore[override]
        raise NotImplementedError("braid_unknot has no Pgx visualiser; use braid_word_str")


class BraidUnknot(core.Env):
    """Two-player Scrambler vs. Simplifier game over braid words."""

    def __init__(self, config: BraidConfig = TIER1):
        super().__init__()
        self.config = config
        self.spec = ActionSpec(max_len=config.max_len, max_strands=config.max_strands)

    @property
    def id(self) -> core.EnvId:  # type: ignore[override]
        return "braid_unknot"  # type: ignore[return-value]

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return self.spec.num_actions

    # -- construction ---------------------------------------------------------

    def _init(self, key: jax.Array) -> State:
        key, ratio_key = jax.random.split(key)
        low, high = self.config.log_ratio_range
        log_ratio = jax.random.uniform(ratio_key, (), minval=low, maxval=high)
        scrambler = jax.random.bernoulli(key).astype(jnp.int32)
        word = braid.empty_word(self.config.max_len)
        n = jnp.int32(1)
        return State(  # type: ignore[call-arg]
            current_player=scrambler,
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=FALSE,
            truncated=FALSE,
            legal_action_mask=self._mask(word, n, jnp.int32(0)),
            _word=word,
            _n=n,
            _phase=jnp.int32(0),
            _budget=jnp.int32(self.config.scramble_budget),
            _scrambler=scrambler,
            _crossing_changes=jnp.int32(0),
            _log_ratio=log_ratio.astype(jnp.float32),
        )

    def init_from_word(
        self, word: list[int], n: int, budget: int | None = None, log_ratio: float = 0.0
    ) -> State:
        """Build a phase-1 (Simplifier-to-move) state from an externally supplied word.

        Used to point a trained agent at knot tables or at a stored hard-instance
        corpus, rather than at self-generated scrambles.
        """
        if len(word) > self.config.max_len:
            raise ValueError(f"word of length {len(word)} exceeds max_len={self.config.max_len}")
        if not 1 <= n <= self.config.max_strands:
            raise ValueError(f"n={n} outside 1..{self.config.max_strands}")
        if any(letter == 0 or abs(letter) > n - 1 for letter in word):
            raise ValueError(f"word {word} has letters outside sigma_1..sigma_{n - 1}")
        padded = jnp.asarray(
            list(word) + [0] * (self.config.max_len - len(word)), dtype=jnp.int32
        )
        n_arr = jnp.int32(n)
        phase = jnp.int32(1)
        state = State(  # type: ignore[call-arg]
            current_player=jnp.int32(1),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=FALSE,
            truncated=FALSE,
            legal_action_mask=self._mask(padded, n_arr, phase),
            _word=padded,
            _n=n_arr,
            _phase=phase,
            _budget=jnp.int32(budget if budget is not None else self.config.simplify_budget),
            _scrambler=jnp.int32(0),
            _crossing_changes=jnp.int32(0),
            _log_ratio=jnp.float32(log_ratio),
        )
        return state.replace(observation=self.observe(state))  # type: ignore[return-value]

    # -- transition -----------------------------------------------------------

    def _step(self, state: State, action: jax.Array, key: jax.Array | None = None) -> State:
        del key
        kind, _, _ = braid.decode(self.spec, action)
        word, n = braid.apply_action(self.spec, state._word, state._n, action)

        crossing_changes = state._crossing_changes + (kind == CROSSING_CHANGE).astype(jnp.int32)

        budget = state._budget - 1
        budget = jnp.where(kind == PASS, 0, budget)

        switching = (state._phase == 0) & (budget <= 0)
        phase = jnp.where(switching, 1, state._phase).astype(jnp.int32)
        budget = jnp.where(switching, self.config.simplify_budget, budget).astype(jnp.int32)

        solved = braid.is_trivial(word, n) & (phase == 1)
        exhausted = (phase == 1) & (budget <= 0) & ~solved
        terminated = solved | exhausted

        simplifier = 1 - state._scrambler
        # Optional speed bonus. Winning still dominates -- the bonus is bounded by
        # `simplifier_speed_bonus < 1` -- but among wins, shorter is better. This
        # matters because in unknotting-number mode the solution *length is the
        # mathematical output*: it is the upper bound on u(K). With the bonus at
        # 0 (the default) the game is exactly win/lose and nothing rewards speed.
        used = jnp.maximum(self.config.simplify_budget - budget, 0).astype(jnp.float32)
        # Multi-objective cost: lambda * crossing_changes + total_moves, scaled to
        # [-1, 1] so the value head keeps a fixed range whatever lambda is. With
        # lambda = 1 and no crossing changes this reduces to the speed bonus.
        if not self.config.multi_objective:
            # Win/lose, optionally graded by speed. The historical game.
            payoff = jnp.where(
                solved,
                1.0
                - self.config.simplifier_speed_bonus * (used / self.config.simplify_budget),
                -1.0,
            )
        else:
            ratio = jnp.exp(state._log_ratio)
            cost = ratio * crossing_changes.astype(jnp.float32) + used
            worst = (ratio + 1.0) * self.config.simplify_budget
            payoff = jnp.where(solved, 1.0 - 2.0 * jnp.clip(cost / worst, 0.0, 1.0), -1.0)
        rewards = jnp.zeros(2, dtype=jnp.float32)
        rewards = rewards.at[simplifier].set(payoff).at[state._scrambler].set(-payoff)
        rewards = jnp.where(terminated, rewards, jnp.zeros(2, dtype=jnp.float32))

        current_player = jnp.where(phase == 0, state._scrambler, simplifier).astype(jnp.int32)

        return state.replace(  # type: ignore[return-value]
            current_player=current_player,
            rewards=rewards,
            terminated=terminated,
            legal_action_mask=self._mask(word, n, phase),
            _word=word,
            _n=n,
            _phase=phase,
            _budget=budget,
            _crossing_changes=crossing_changes,
        )

    def _mask(self, word: jax.Array, n: jax.Array, phase: jax.Array) -> jax.Array:
        allow = jnp.asarray(self.config.allow_crossing_change) & (phase == 1)
        return braid.legal_action_mask(self.spec, word, n, allow)

    # -- observation ----------------------------------------------------------

    def _observe(self, state: State, player_id: jax.Array) -> jax.Array:
        del player_id  # both roles see the same word; the phase plane says who is to move
        max_len = self.config.max_len
        max_strands = self.config.max_strands
        word = state._word

        generators = jnp.arange(1, max_strands, dtype=word.dtype)[:, None]
        positive = word[None, :] == generators
        negative = word[None, :] == -generators
        empty = (word == 0)[None, :]

        # Normalise the budget by the *current phase's* budget, not by the larger
        # of the two. Sharing one scale compresses the Scrambler's clock into
        # [0, K/M] -- at tier 0 that is [0, 1/6], so the player who has to plan
        # against a deadline can barely see it.
        phase_budget = jnp.where(
            state._phase == 0, self.config.scramble_budget, self.config.simplify_budget
        ).astype(jnp.float32)
        length = braid.word_length(word)

        # Destabilisation is the only move that lowers the strand count, and
        # reaching n=1 is the win condition -- it is 54% of the moves in optimal
        # solutions while being 1.8% of the legal action set. Its legality is a
        # *global* predicate ("does +-(n-1) occur exactly once?") that a
        # convolution with an 11-letter receptive field cannot evaluate, so it is
        # supplied directly:
        #   * a positional plane marking every +-(n-1) letter, so a local window
        #     can see "this letter is the blocker";
        #   * a scalar count, so the agent knows how many must be cleared before
        #     destabilising becomes possible -- strictly more useful than a
        #     boolean, which only says "not yet".
        # The legal-action mask already forbids the move when it is illegal, but a
        # mask filters logits after the fact; the network cannot learn *why*
        # unless it can see the predicate.
        top_generator = (jnp.abs(word) == (state._n - 1)) & (word != 0)
        top_count = jnp.sum(top_generator).astype(jnp.float32)
        scalars = jnp.stack(
            [
                state._phase.astype(jnp.float32),
                state._n.astype(jnp.float32) / max_strands,
                state._budget.astype(jnp.float32) / phase_budget,
                length.astype(jnp.float32) / max_len,
                jnp.minimum(top_count, max_len) / max_len,
                (top_count == 1).astype(jnp.float32),  # destabilisation available
                # log(A/B): what the agent is being asked to optimise this episode
                state._log_ratio / 5.0,
                state._crossing_changes.astype(jnp.float32) / max(max_len, 1),
            ]
        )
        scalar_planes = jnp.broadcast_to(scalars[:, None], (8, max_len))

        letter_planes = jnp.concatenate(
            [positive, negative, empty, top_generator[None, :]], axis=0
        ).astype(jnp.float32)
        planes = jnp.concatenate([letter_planes, scalar_planes], axis=0)
        return planes.T  # (max_len, channels)

    @property
    def num_channels(self) -> int:
        # letter one-hot (+/- each generator), padding, the top-generator
        # marker, and six broadcast scalars
        return 2 * (self.config.max_strands - 1) + 1 + 1 + 8


def braid_word_str(state: State) -> str:
    """Human-readable word, e.g. ``B3: s1 s2^-1 s1``."""
    word = [int(x) for x in state._word if int(x) != 0]
    n = int(state._n)
    if not word:
        return f"B{n}: e"
    letters = " ".join(f"s{abs(x)}" + ("^-1" if x < 0 else "") for x in word)
    return f"B{n}: {letters}"
