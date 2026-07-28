from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import typer

from rf_knots import reference
from rf_knots.actions import PASS
from rf_knots.config import TIER0, TIER1, BraidConfig
from rf_knots.env import BraidUnknot
from rf_knots.rollout import batched_random_rollout, scramble, to_word

app = typer.Typer(add_completion=False, help="Braid-word unknotting environments for Pgx")


def _config(tier: str, max_len: int | None, max_strands: int | None, k: int | None, m: int | None):
    base = {"tier0": TIER0, "tier1": TIER1}[tier]
    return BraidConfig(
        max_len=max_len if max_len is not None else base.max_len,
        max_strands=max_strands if max_strands is not None else base.max_strands,
        scramble_budget=k if k is not None else base.scramble_budget,
        simplify_budget=m if m is not None else base.simplify_budget,
        allow_crossing_change=base.allow_crossing_change,
    )


@app.command()
def rules(tier: str = "tier1") -> None:
    """Print the action-space layout and environment shapes."""
    config = {"tier0": TIER0, "tier1": TIER1}[tier]
    env = BraidUnknot(config)
    spec = env.spec
    typer.echo(f"config           : {config}")
    typer.echo(f"num_actions      : {env.num_actions}")
    typer.echo(f"observation shape: {env.observation_shape}")
    typer.echo(f"channels         : {env.num_channels}")
    typer.echo("action blocks:")
    starts = spec.starts
    from rf_knots.actions import KIND_NAMES

    for kind, name in enumerate(KIND_NAMES):
        typer.echo(f"  {name:<16} [{starts[kind]:>5}, {starts[kind + 1]:>5})")


@app.command()
def demo(
    tier: str = "tier0",
    seed: int = 0,
    k: int | None = None,
    depth: int = 6,
    growth: int | None = None,
) -> None:
    """Scramble from the unknot, then solve it exactly by breadth-first search."""
    config = _config(tier, None, None, k, None)
    env = BraidUnknot(config)
    state = scramble(env, jax.random.PRNGKey(seed))
    word, n = to_word(state)

    typer.echo(f"scrambled with K={config.scramble_budget} moves")
    typer.echo(f"  {reference.format_word(word, n)}")
    typer.echo(f"  length={len(word)}  strands={n}  writhe={reference.writhe(word)}")
    typer.echo(f"  components={reference.num_components(word, n)} (must be 1)")

    started = time.monotonic()
    path = reference.bfs_unknot(
        env.spec,
        word,
        n,
        max_depth=depth,
        max_growth=config.scramble_budget if growth is None else growth,
    )
    elapsed = time.monotonic() - started
    if path is None:
        typer.echo(f"  BFS found no solution within depth {depth} ({elapsed:.1f}s)")
        typer.echo("  -> the instance is still an unknot; it is just deeper than the cutoff")
    else:
        typer.echo(f"  BFS optimal solution: {len(path)} moves ({elapsed:.1f}s)")
        for action in path:
            typer.echo(f"    {env.spec.describe(action)}")


@app.command()
def selfcheck(tier: str = "tier0", episodes: int = 200, seed: int = 0) -> None:
    """Verify the invariants that make the generated instances trustworthy.

    For every move taken in a random rollout:
      * the JAX kernel and the Python reference produce the same word;
      * the number of closure components stays 1;
      * braid-group moves preserve the Artin image (exact equality in B_n);
      * Markov moves change the group element but keep the closure's components.
    """
    config = _config(tier, None, None, None, None)
    env = BraidUnknot(config)
    spec = env.spec
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)

    from rf_knots.actions import BRAID, COMMUTE, INSERT, REDUCE

    group_moves = {REDUCE, COMMUTE, BRAID, INSERT}
    seam_moves = {REDUCE, COMMUTE, BRAID}
    checked = {"steps": 0, "group": 0, "seam": 0, "markov": 0}

    for episode in range(episodes):
        key, sub = jax.random.split(key)
        state = env.init(sub)
        word, n = to_word(state)
        while not bool(state.terminated):
            key, sub = jax.random.split(key)
            mask = np.asarray(state.legal_action_mask).copy()
            mask[spec.start_of(PASS)] = False
            if not mask.any():
                break
            legal = np.flatnonzero(mask)
            action = int(jax.random.choice(sub, jnp.asarray(legal)))

            expected_word, expected_n = reference.apply(spec, word, n, action)
            assert reference.is_legal(spec, word, n, action, config.allow_crossing_change), (
                f"episode {episode}: JAX says {spec.describe(action)} is legal, reference disagrees"
            )

            state = step(state, action)
            got_word, got_n = to_word(state)
            assert (got_word, got_n) == (expected_word, expected_n), (
                f"episode {episode}: {spec.describe(action)} on {reference.format_word(word, n)}"
                f" -> jax {reference.format_word(got_word, got_n)}"
                f" != ref {reference.format_word(expected_word, expected_n)}"
            )

            kind, position, _, _ = spec.decode(action)
            wraps = kind in seam_moves and (
                position == len(word) - 1
                or (kind == BRAID and position >= len(word) - 2)
            )
            if kind in group_moves and not wraps:
                assert reference.equal_in_braid_group(word, expected_word, n), (
                    f"episode {episode}: {spec.describe(action)} changed the braid group element"
                )
                checked["group"] += 1
            elif wraps:
                # Across the seam a move is a conjugation composed with the
                # relation, so it must land on the same necklace as rotating it
                # into the interior and acting there.
                long_way = reference.seam_move_via_rotation(spec, word, n, kind, position)
                assert expected_word in reference.rotations(long_way), (
                    f"episode {episode}: seam {spec.describe(action)} is not "
                    f"rotate-then-move"
                )
                assert reference.writhe(expected_word) == reference.writhe(word)
                checked["seam"] += 1
            else:
                checked["markov"] += 1
            assert reference.num_components(expected_word, expected_n) == 1, (
                f"episode {episode}: {spec.describe(action)} made the closure a link"
            )

            checked["steps"] += 1
            word, n = got_word, got_n

    typer.echo(
        f"ok: {checked['steps']} steps over {episodes} episodes "
        f"({checked['group']} interior braid-group moves, {checked['seam']} seam moves, "
        f"{checked['markov']} Markov moves)"
    )


@app.command()
def bench(tier: str = "tier1", batch: int = 1024, seed: int = 0) -> None:
    """Throughput of vmapped random self-play, and the random-play baseline."""
    config = _config(tier, None, None, None, None)
    key = jax.random.PRNGKey(seed)

    key, sub = jax.random.split(key)
    batched_random_rollout(config, 8, sub)  # warm the compilation cache

    key, sub = jax.random.split(key)
    started = time.monotonic()
    state, simplifier_return = batched_random_rollout(config, batch, sub)
    elapsed = time.monotonic() - started

    plies = int(np.asarray(state._step_count).max())
    steps = batch * plies
    win_rate = float((np.asarray(simplifier_return) > 0).mean())
    typer.echo(f"config        : {config}")
    typer.echo(f"batch         : {batch}")
    typer.echo(f"plies (max)   : {plies}")
    typer.echo(f"wall clock    : {elapsed:.2f}s")
    typer.echo(f"env steps/s   : {steps / elapsed:,.0f}")
    typer.echo(f"random-play simplifier win rate: {win_rate:.3f}")
    typer.echo("  (this is the baseline a trained Simplifier has to beat)")


@app.command()
def calibrate(
    tier: str = "tier0",
    samples: int = 32,
    depth: int = 6,
    growth: int | None = None,
    k: int | None = None,
    seed: int = 0,
) -> None:
    """How hard is a K-move scramble really? Exact optimal depths by BFS.

    The scrambler is uniform random here, so this measures the difficulty floor:
    a trained Scrambler should push the optimal depth well above these numbers at
    the same K.
    """
    config = _config(tier, None, None, k, None)
    env = BraidUnknot(config)
    max_growth = config.scramble_budget if growth is None else growth
    key = jax.random.PRNGKey(seed)

    paths: list[list[int] | None] = []
    lengths: list[int] = []
    started = time.monotonic()
    for _ in range(samples):
        key, sub = jax.random.split(key)
        state = scramble(env, sub)
        word, n = to_word(state)
        lengths.append(len(word))
        paths.append(reference.bfs_unknot(env.spec, word, n, depth, max_growth=max_growth))
    elapsed = time.monotonic() - started

    solved = [len(path) for path in paths if path is not None]
    typer.echo(f"K = {config.scramble_budget}, BFS cutoff = {depth}, growth = {max_growth}")
    typer.echo(f"mean word length : {np.mean(lengths):.1f}  max {max(lengths)}")
    typer.echo(f"solved           : {len(solved)}/{samples} ({elapsed:.1f}s)")
    if solved:
        typer.echo(f"optimal depth    : mean {np.mean(solved):.2f}  max {max(solved)}")
    typer.echo(f"beyond cutoff    : {sum(p is None for p in paths)}/{samples}")


if __name__ == "__main__":
    app()
