from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rf_knots import reference
from rf_knots.actions import DESTABILIZE, PASS, STABILIZE_NEG, STABILIZE_POS
from rf_knots.config import BraidConfig
from rf_knots.env import BraidUnknot
from rf_knots.rollout import batched_random_rollout, scramble, to_word

SMALL = BraidConfig(max_len=12, max_strands=4, scramble_budget=4, simplify_budget=10)
# +/- one-hot per generator, the padding plane, and the top-generator marker
_LETTER_PLANES = 2 * (SMALL.max_strands - 1) + 1 + 1


@pytest.fixture
def env() -> BraidUnknot:
    return BraidUnknot(SMALL)


def test_init_is_the_unknot(env: BraidUnknot) -> None:
    state = env.init(jax.random.PRNGKey(0))
    assert to_word(state) == ((), 1)
    assert int(state._phase) == 0
    assert int(state._budget) == SMALL.scramble_budget
    assert int(state.current_player) == int(state._scrambler)
    assert not bool(state.terminated)
    # From the empty 1-braid only stabilisation can be legal. PASS is masked:
    # it is a last resort, not an opening move.
    mask = np.asarray(state.legal_action_mask)
    legal = np.flatnonzero(mask)
    kinds = {env.spec.decode(int(a))[0] for a in legal}
    assert kinds == {STABILIZE_POS, STABILIZE_NEG}
    assert not mask[env.spec.start_of(PASS)]


def test_shapes_and_action_count(env: BraidUnknot) -> None:
    assert env.num_players == 2
    assert env.num_actions == env.spec.num_actions
    assert env.observation_shape == (SMALL.max_len, env.num_channels)
    state = env.init(jax.random.PRNGKey(0))
    assert state.observation.shape == (SMALL.max_len, env.num_channels)
    assert state.legal_action_mask.shape == (env.num_actions,)


def test_step_is_jittable_and_vmappable(env: BraidUnknot) -> None:
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    states = init(jax.random.split(jax.random.PRNGKey(0), 8))
    actions = jnp.full((8,), env.spec.encode(STABILIZE_POS), dtype=jnp.int32)
    states = step(states, actions)
    assert states._word.shape == (8, SMALL.max_len)
    assert (np.asarray(states._n) == 2).all()
    assert np.asarray(states._word)[:, 0].tolist() == [1] * 8


def test_phase_switches_after_the_scramble_budget(env: BraidUnknot) -> None:
    state = env.init(jax.random.PRNGKey(1))
    step = jax.jit(env.step)
    stabilize = env.spec.encode(STABILIZE_POS)
    for ply in range(SMALL.scramble_budget):
        assert int(state._phase) == 0
        assert int(state.current_player) == int(state._scrambler)
        state = step(state, jnp.int32(stabilize if ply == 0 else _any_legal(env, state)))
    assert int(state._phase) == 1
    assert int(state._budget) == SMALL.simplify_budget
    assert int(state.current_player) == 1 - int(state._scrambler)


def _any_legal(env: BraidUnknot, state) -> int:
    mask = np.asarray(state.legal_action_mask).copy()
    mask[env.spec.start_of(PASS)] = False
    return int(np.flatnonzero(mask)[0])


def test_scrambler_cannot_forfeit_on_the_first_ply(env: BraidUnknot) -> None:
    """Regression: PASS used to be always legal, so an untrained Scrambler ended
    roughly a third of self-play games on ply 1 with a trivial Simplifier win."""
    state = env.init(jax.random.PRNGKey(2))
    for _ in range(SMALL.scramble_budget):
        assert not bool(state.legal_action_mask[env.spec.start_of(PASS)])
        assert not bool(state.terminated)
        state = env.step(state, jnp.int32(_any_legal(env, state)))
    assert int(state._phase) == 1
    assert not bool(state.terminated)


def test_simplifier_wins_by_reaching_the_empty_braid(env: BraidUnknot) -> None:
    """Scramble with a single stabilisation, then destabilise back."""
    config = BraidConfig(max_len=12, max_strands=4, scramble_budget=1, simplify_budget=5)
    env = BraidUnknot(config)
    state = env.init(jax.random.PRNGKey(3))
    scrambler = int(state._scrambler)
    state = env.step(state, jnp.int32(env.spec.encode(STABILIZE_POS)))
    assert int(state._phase) == 1
    assert to_word(state) == ((1,), 2)

    state = env.step(state, jnp.int32(env.spec.encode(DESTABILIZE)))
    assert bool(state.terminated)
    assert to_word(state) == ((), 1)
    rewards = np.asarray(state.rewards)
    assert rewards[1 - scrambler] == 1.0
    assert rewards[scrambler] == -1.0


def test_scrambler_wins_when_the_budget_runs_out(env: BraidUnknot) -> None:
    config = BraidConfig(max_len=12, max_strands=4, scramble_budget=1, simplify_budget=2)
    env = BraidUnknot(config)
    state = env.init(jax.random.PRNGKey(4))
    scrambler = int(state._scrambler)
    state = env.step(state, jnp.int32(env.spec.encode(STABILIZE_POS)))
    # Waste both simplifier moves on a stabilise/destabilise cycle.
    state = env.step(state, jnp.int32(env.spec.encode(STABILIZE_POS)))
    assert not bool(state.terminated)
    state = env.step(state, jnp.int32(env.spec.encode(STABILIZE_POS)))
    assert bool(state.terminated)
    rewards = np.asarray(state.rewards)
    assert rewards[scrambler] == 1.0
    assert rewards[1 - scrambler] == -1.0


def test_rewards_are_zero_sum_and_only_paid_at_termination() -> None:
    config = BraidConfig(max_len=16, max_strands=4, scramble_budget=3, simplify_budget=6)
    state, simplifier_return = batched_random_rollout(config, 64, jax.random.PRNGKey(5))
    returns = np.asarray(simplifier_return)
    assert bool(np.all(state.terminated))
    assert set(np.unique(returns)).issubset({-1.0, 1.0})


def test_scrambled_instances_are_always_unknots(env: BraidUnknot) -> None:
    """The property the whole design rests on: ground truth for free."""
    for seed in range(30):
        state = scramble(env, jax.random.PRNGKey(seed))
        word, n = to_word(state)
        assert reference.num_components(word, n) == 1, reference.format_word(word, n)
        assert all(1 <= abs(x) <= n - 1 for x in word)


def test_scrambled_instances_are_solvable_by_bfs() -> None:
    """End-to-end: generate, then prove solvability with an exhaustive search."""
    config = BraidConfig(max_len=12, max_strands=4, scramble_budget=3, simplify_budget=12)
    env = BraidUnknot(config)
    solved = 0
    for seed in range(12):
        state = scramble(env, jax.random.PRNGKey(100 + seed))
        word, n = to_word(state)
        if (word, n) == ((), 1):
            solved += 1
            continue
        path = reference.bfs_unknot(env.spec, word, n, max_depth=4, max_nodes=200_000)
        assert path is not None, f"no path back from {reference.format_word(word, n)}"
        # replay the path through the JAX environment
        for action in path:
            state = env.step(state, jnp.int32(action))
        assert bool(state.terminated)
        assert to_word(state) == ((), 1)
        solved += 1
    assert solved == 12


def test_init_from_word_round_trips(env: BraidUnknot) -> None:
    state = env.init_from_word([1, 2, 2, -2], n=3)
    assert to_word(state) == ((1, 2, 2, -2), 3)
    assert int(state._phase) == 1
    assert state.observation.shape == (SMALL.max_len, env.num_channels)

    path = reference.bfs_unknot(env.spec, (1, 2, 2, -2), 3, max_depth=4)
    assert path is not None
    for action in path:
        state = env.step(state, jnp.int32(action))
    assert bool(state.terminated)
    assert to_word(state) == ((), 1)


def test_init_from_word_rejects_malformed_input(env: BraidUnknot) -> None:
    with pytest.raises(ValueError):
        env.init_from_word([1, 0, 2], n=3)
    with pytest.raises(ValueError):
        env.init_from_word([3], n=3)  # sigma_3 does not exist in B_3
    with pytest.raises(ValueError):
        env.init_from_word([1] * (SMALL.max_len + 1), n=2)


def test_observation_encodes_the_word(env: BraidUnknot) -> None:
    state = env.init_from_word([1, -2, 3], n=4)
    obs = np.asarray(state.observation)
    generators = SMALL.max_strands - 1
    # channel g-1 is "letter == +g", channel (N-1)+(g-1) is "letter == -g"
    assert obs[0, 0] == 1.0  # +1 at position 0
    assert obs[1, generators + 1] == 1.0  # -2 at position 1
    assert obs[2, 2] == 1.0  # +3 at position 2
    assert obs[3, 2 * generators] == 1.0  # padding at position 3
    assert obs[:, : 2 * generators + 1].sum() == SMALL.max_len  # one-hot everywhere


def test_illegal_action_terminates_with_a_penalty(env: BraidUnknot) -> None:
    """Pgx contract: an illegal action ends the game against the player to move."""
    state = env.init(jax.random.PRNGKey(6))
    illegal = env.spec.encode(0, position=0)  # REDUCE on an empty word
    assert not bool(state.legal_action_mask[illegal])
    mover = int(state.current_player)
    state = env.step(state, jnp.int32(illegal))
    assert bool(state.terminated)
    assert np.asarray(state.rewards)[mover] == -1.0


def test_budget_plane_uses_the_current_phase_scale(env: BraidUnknot) -> None:
    """The Scrambler must be able to see its own clock.

    Normalising both phases by the larger budget squeezed the scramble phase into
    [0, K/M] -- a sixth of the range at tier 0 -- so the deadline the Scrambler
    plans against was nearly invisible.
    """
    budget_channel = _LETTER_PLANES + 2  # phase, n/N, budget, ...
    state = env.init(jax.random.PRNGKey(0))
    obs = np.asarray(state.observation)
    assert obs[0, budget_channel] == pytest.approx(1.0)  # full scramble budget

    for _ in range(SMALL.scramble_budget - 1):
        state = env.step(state, jnp.int32(_any_legal(env, state)))
    obs = np.asarray(state.observation)
    assert obs[0, budget_channel] == pytest.approx(1.0 / SMALL.scramble_budget)

    state = env.step(state, jnp.int32(_any_legal(env, state)))
    assert int(state._phase) == 1
    obs = np.asarray(state.observation)
    assert obs[0, budget_channel] == pytest.approx(1.0)  # full simplify budget


def test_speed_bonus_rewards_shorter_solutions() -> None:
    """In unknotting-number mode the solution length IS the mathematical output:
    it is the upper bound on u(K). Nothing in the plain win/lose game rewards it."""
    config = BraidConfig(
        max_len=12,
        max_strands=4,
        scramble_budget=1,
        simplify_budget=8,
        simplifier_speed_bonus=0.4,
    )
    env = BraidUnknot(config)
    stabilize = env.spec.encode(STABILIZE_POS)
    destabilize = env.spec.encode(DESTABILIZE)

    def play(waste: int) -> tuple[float, float]:
        state = env.init(jax.random.PRNGKey(0))
        scrambler = int(state._scrambler)
        state = env.step(state, jnp.int32(stabilize))
        for _ in range(waste):  # stabilise then destabilise: a wasted round trip
            state = env.step(state, jnp.int32(stabilize))
            state = env.step(state, jnp.int32(destabilize))
        state = env.step(state, jnp.int32(destabilize))
        assert bool(state.terminated)
        rewards = np.asarray(state.rewards)
        return float(rewards[1 - scrambler]), float(rewards[scrambler])

    fast, fast_scrambler = play(0)
    slow, slow_scrambler = play(2)
    assert fast > slow > 0.0, "a faster win must score higher, and both are still wins"
    assert fast == pytest.approx(-fast_scrambler)  # still zero-sum
    assert slow == pytest.approx(-slow_scrambler)


def test_speed_bonus_is_off_by_default(env: BraidUnknot) -> None:
    assert env.config.simplifier_speed_bonus == 0.0
    state, returns = batched_random_rollout(SMALL, 32, jax.random.PRNGKey(0))
    assert set(np.unique(np.asarray(returns))).issubset({-1.0, 1.0})


def test_destabilize_hint_is_visible_locally_and_globally(env: BraidUnknot) -> None:
    """Destabilisation is 54% of optimal solutions but its legality is a *global*
    predicate, which an 11-letter receptive field cannot evaluate on its own."""
    top_marker = _LETTER_PLANES - 1
    count_channel = _LETTER_PLANES + 4
    available_channel = _LETTER_PLANES + 5

    # Two occurrences of the top generator (sigma_2 in B_3): blocked.
    state = env.init_from_word([1, 2, 1, 2], n=3)
    obs = np.asarray(state.observation)
    assert obs[:, top_marker].tolist()[:4] == [0.0, 1.0, 0.0, 1.0]
    assert obs[0, count_channel] == pytest.approx(2 / SMALL.max_len)
    assert obs[0, available_channel] == 0.0
    assert not bool(state.legal_action_mask[env.spec.start_of(DESTABILIZE)])

    # One occurrence: available, and the mask agrees.
    state = env.init_from_word([1, 2, 1], n=3)
    obs = np.asarray(state.observation)
    assert obs[:, top_marker].tolist()[:3] == [0.0, 1.0, 0.0]
    assert obs[0, count_channel] == pytest.approx(1 / SMALL.max_len)
    assert obs[0, available_channel] == 1.0
    assert bool(state.legal_action_mask[env.spec.start_of(DESTABILIZE)])
