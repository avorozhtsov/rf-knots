"""Helpers for random play, scramble generation, and instance statistics."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rf_knots.actions import PASS
from rf_knots.config import BraidConfig
from rf_knots.env import BraidUnknot, State


def random_legal_action(key: jax.Array, mask: jax.Array) -> jax.Array:
    """Uniform over legal actions."""
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits).astype(jnp.int32)


def scramble(
    env: BraidUnknot,
    key: jax.Array,
    moves: int | None = None,
    exclude_pass: bool = True,
) -> State:
    """Run the scramble phase with uniform random legal moves.

    Returns the state at the start of the Simplifier's turn. The closure of the
    resulting word is the unknot by construction.
    """
    moves = env.config.scramble_budget if moves is None else moves
    pass_action = env.spec.start_of(PASS)
    key, sub = jax.random.split(key)
    state = env.init(sub)

    step = jax.jit(env.step)
    for _ in range(moves):
        key, sub = jax.random.split(key)
        mask = state.legal_action_mask
        if exclude_pass:
            mask = mask.at[pass_action].set(False)
        action = random_legal_action(sub, mask)
        state = step(state, action)
        if bool(state.terminated):
            break
    return state


def anchor_instances(
    config: BraidConfig,
    count: int,
    seed: int = 10_000,
) -> list[tuple[tuple[int, ...], int]]:
    """A fixed, reproducible set of scrambled instances.

    Progress has to be reported against a *frozen* set of problems. Measuring an
    agent only against the instances it is currently generating conflates "the
    Simplifier improved" with "the Scrambler got easier", which is precisely the
    rating-inflation failure the league design warns about. These come from fixed
    seeds and a uniform-random Scrambler, so they never move.
    """
    env = BraidUnknot(config)
    instances: list[tuple[tuple[int, ...], int]] = []
    index = 0
    while len(instances) < count:
        state = scramble(env, jax.random.PRNGKey(seed + index))
        index += 1
        word, n = to_word(state)
        if (word, n) == ((), 1):
            continue  # already solved; carries no signal
        instances.append((word, n))
    return instances


@dataclass(frozen=True)
class InstanceStats:
    length: int
    strands: int
    writhe: int
    components: int


def instance_stats(state: State) -> InstanceStats:
    from rf_knots import reference

    word = tuple(int(x) for x in np.asarray(state._word) if int(x) != 0)
    n = int(state._n)
    return InstanceStats(
        length=len(word),
        strands=n,
        writhe=reference.writhe(word),
        components=reference.num_components(word, n),
    )


def to_word(state: State) -> tuple[tuple[int, ...], int]:
    word = tuple(int(x) for x in np.asarray(state._word) if int(x) != 0)
    return word, int(state._n)


def batched_random_rollout(
    config: BraidConfig, batch: int, key: jax.Array
) -> tuple[State, jax.Array]:
    """Play a whole batch of games out with uniform random legal moves.

    Used for throughput benchmarking and for a random-play baseline: the fraction
    of `K`-move scrambles that random play undoes within `M` moves.
    """
    env = BraidUnknot(config)
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))

    key, sub = jax.random.split(key)
    state = init(jax.random.split(sub, batch))

    max_plies = config.scramble_budget + config.simplify_budget + 2
    pass_action = env.spec.start_of(PASS)

    def choose(k, mask):
        # Never volunteer PASS; it is only there to guarantee a non-empty mask.
        without_pass = mask.at[pass_action].set(False)
        usable = jnp.where(jnp.any(without_pass), without_pass, mask)
        return random_legal_action(k, usable)

    choose = jax.jit(jax.vmap(choose))

    # Pgx zeroes `rewards` when a terminated state is stepped again, so the
    # per-episode return has to be accumulated rather than read at the end.
    returns = jnp.zeros((batch, 2), dtype=jnp.float32)
    for _ in range(max_plies):
        key, sub = jax.random.split(key)
        actions = choose(jax.random.split(sub, batch), state.legal_action_mask)
        state = step(state, actions)
        returns = returns + state.rewards
        if bool(jnp.all(state.terminated)):
            break

    simplifier = 1 - state._scrambler
    simplifier_return = jnp.take_along_axis(returns, simplifier[:, None], axis=1).squeeze(-1)
    return state, simplifier_return
