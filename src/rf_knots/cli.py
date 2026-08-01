from __future__ import annotations

import time
from pathlib import Path

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


@app.command()
def knot(word: str, strands: int = 0) -> None:
    """Invariants of a braid word, and the name of the knot it closes to.

        uv run rf-knots knot "1,1,2,2,1,1,2,1,-2,-1,2,1,-2,-2,-1,2,1,-2"

    The word is comma-separated Artin generators: `2` is sigma_2, `-2` its
    inverse. `--strands` defaults to the smallest braid group the word fits in.
    """
    from rf_knots.invariants import format_polynomial, invariants

    letters = tuple(int(x) for x in word.replace(" ", "").split(",") if x)
    n = strands or (max((abs(x) for x in letters), default=0) + 1)
    inv = invariants(letters, n)

    typer.echo(f"braid            : {list(inv.word)} on {inv.strands} strands")
    typer.echo(f"word length      : {inv.crossings}   writhe {inv.writhe}")
    typer.echo(f"Alexander        : {format_polynomial(inv.alexander_polynomial)}")
    typer.echo(f"determinant      : {inv.determinant}")
    typer.echo(f"Jones            : {format_polynomial(inv.jones_polynomial)}")
    sigma = inv.signature if inv.signature is not None else "(needs spherogram)"
    typer.echo(f"signature        : {sigma}")
    genus = (str(inv.genus_lower) if inv.genus_lower == inv.genus_upper
             else f"{inv.genus_lower}..{inv.genus_upper}")
    typer.echo(f"genus            : {genus}")
    if inv.name:
        mirror = " (mirror image)" if inv.mirror else ""
        typer.echo(f"identified as    : {inv.name}{mirror}, {inv.identified_crossings} crossings")
        # The distinction that matters: the word is as long as it is, but the
        # knot is as big as it is, and only the second one bounds u.
        if inv.identified_crossings is not None and inv.identified_crossings < inv.crossings:
            typer.echo(f"                   the word is {inv.crossings} letters for a "
                       f"{inv.identified_crossings}-crossing knot")
    if inv.unknotting is not None:
        typer.echo(f"unknotting number: {inv.unknotting}  (published)")
    elif inv.unknotting_lower is not None:
        typer.echo(f"unknotting number: >= {inv.unknotting_lower}  (from |sigma|/2)")
    from rf_knots.lower_bounds import claims_for, strongest

    lower_claims = claims_for(inv)
    lower = strongest(lower_claims)
    if lower is not None:
        methods = ", ".join(claim.method for claim in lower_claims if claim.value == lower)
        typer.echo(f"certified lower   : {lower}  ({methods})")
    for note in inv.notes:
        typer.echo(f"note             : {note}")


@app.command("evidence-verify")
def evidence_verify(path: Path) -> None:
    """Replay and hash-check every record in an evidence JSONL store."""
    from rf_knots.evidence import EvidenceStore

    records = EvidenceStore(path).records(skip_torn_last_line=False)
    witnesses = sum(record.witness is not None for record in records)
    exact = sum(record.exact_unknotting_number is not None for record in records)
    typer.echo(f"ok: {len(records)} records, {witnesses} replayable witnesses, {exact} exact u")


@app.command("benchmark-check")
def benchmark_check(path: Path) -> None:
    """Validate a frozen benchmark manifest and print its split counts."""
    from rf_knots.benchmarks import BenchmarkManifest

    manifest = BenchmarkManifest.read(path)
    counts = {split: sum(x.split == split for x in manifest.instances)
              for split in ("train", "validation", "test")}
    typer.echo(f"ok: {manifest.name} v{manifest.version}, {len(manifest.instances)} instances")
    typer.echo(f"splits: {counts}")


@app.command("baseline")
def baseline(
    name: str,
    word: str,
    strands: int = 0,
    timeout: float = 60.0,
) -> None:
    """Run one fixed baseline: snappy, reapr, or regina."""
    from rf_knots.baselines import BaselineUnavailable, run_reapr, run_regina, run_snappy

    letters = tuple(int(x) for x in word.replace(" ", "").split(",") if x)
    n = strands or (max((abs(x) for x in letters), default=0) + 1)
    runners = {
        "snappy": lambda: run_snappy(letters, n),
        "reapr": lambda: run_reapr(letters, n, timeout=timeout),
        "regina": lambda: run_regina(letters, n),
    }
    if name not in runners:
        raise typer.BadParameter("name must be snappy, reapr, or regina")
    try:
        result = runners[name]()
    except BaselineUnavailable as error:
        typer.echo(f"unavailable: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"baseline         : {result.name}")
    typer.echo(f"status           : {result.status}")
    typer.echo(f"wall clock       : {result.elapsed_seconds:.6f}s")
    typer.echo(f"crossings        : {result.input_crossings} -> {result.output_crossings}")
    typer.echo(f"detail           : {result.detail}")


if __name__ == "__main__":
    app()
