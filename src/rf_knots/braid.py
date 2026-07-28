"""JAX kernels for cyclic braid words under Markov moves.

Representation
--------------
A braid word on `n` strands is a left-compacted ``int32[L]`` array. Entry ``+g``
means sigma_g, ``-g`` means sigma_g^-1, and ``0`` is padding. The invariant
maintained everywhere is:

    word[k] != 0  for k < length,   word[k] == 0  for k >= length,
    1 <= |word[k]| <= n - 1.

`length` is never stored: it is ``sum(word != 0)``, which is exact because the
word is compacted. Storing it would only create a way for it to desynchronise.

The word is **cyclic**. Position ``p+1`` means ``(p+1) mod length``, and the
storage seam at index 0 carries no meaning. This is the geometry rather than a
convenience: a braid word is read by sweeping a half-plane once around the braid
axis, so it closes back on itself, and conjugation ``w -> g w g^-1`` -- a Markov
move that does not change the closure -- is exactly a rotation of the seam.
Working with necklaces instead of lists buys three things:

* the two rotation moves disappear, since they only ever moved the seam;
* free reduction, commutation and the braid relation apply across the seam,
  which previously needed a rotation first;
* `DESTABILIZE` no longer requires the lone top generator to sit in last place.

Moves
-----
Moves in the first two tiers preserve the *link type of the closure*: the
braid-group relations (free reduction/insertion, far commutation, the braid
relation) and the Markov (de)stabilisations. A tier-1 move applied away from the
seam preserves the braid group element exactly; applied across the seam it
preserves it up to conjugacy, which is the same thing for the closure.

The crossing change is the only move that changes the knot; it is the unknotting
move, masked off during the scramble phase so that instances generated from the
unknot are still the unknot by construction.

The braid relation is implemented for triples ``(a, b, a)`` with
``| |a| - |b| | == 1`` and ``sign(a) == sign(b)``, rewritten to ``(b, a, b)``.
That covers the all-positive relation and its inverse, the all-negative one. The
mixed-sign forms are derivable: e.g.
``(-b, a, b) -> (-b, a, b, a, -a) -> (-b, b, a, b, -a) -> (a, b, -a)``
using one insertion, one braid move and one reduction, so nothing is lost.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    NUM_KINDS,
    PASS,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)

DTYPE = jnp.int32


def empty_word(max_len: int) -> jax.Array:
    """The unknot: the empty braid word on one strand."""
    return jnp.zeros((max_len,), dtype=DTYPE)


def word_length(word: jax.Array) -> jax.Array:
    return jnp.sum(word != 0).astype(jnp.int32)


def is_trivial(word: jax.Array, n: jax.Array) -> jax.Array:
    """True iff this is the empty 1-braid, whose closure is the unknot."""
    return (word_length(word) == 0) & (n == 1)


# -- cyclic access -------------------------------------------------------------


def _wrap(index: jax.Array, length: jax.Array) -> jax.Array:
    """``index mod length``, safe when ``length == 0``."""
    return jnp.where(length > 0, jnp.mod(index, jnp.maximum(length, 1)), 0)


def cyclic_shift(word: jax.Array, offset: int) -> jax.Array:
    """``out[k] = word[(k + offset) mod length]`` inside the word, 0 outside."""
    length = word_length(word)
    idx = jnp.arange(word.shape[0])
    gathered = jnp.take(word, _wrap(idx + offset, length), fill_value=0)
    return jnp.where(idx < length, gathered, 0).astype(word.dtype)


def _compact_after_deleting(word: jax.Array, drop: jax.Array) -> jax.Array:
    """Remove the flagged positions and re-close the gaps.

    Scatter-based so it works for any deletion pattern, including one that wraps
    the seam. Dropped entries are written to a scratch slot past the end.
    """
    max_len = word.shape[0]
    idx = jnp.arange(max_len)
    length = word_length(word)
    keep = (idx < length) & ~drop
    destination = jnp.cumsum(keep) - 1
    scratch = jnp.zeros((max_len + 1,), dtype=word.dtype)
    scattered = scratch.at[jnp.where(keep, destination, max_len)].set(
        jnp.where(keep, word, 0)
    )
    return scattered[:max_len]


# -- primitive rewrites --------------------------------------------------------


def reduce_at(word: jax.Array, p: jax.Array) -> jax.Array:
    """Delete the cancelling pair at positions p and p+1 (mod length)."""
    length = word_length(word)
    idx = jnp.arange(word.shape[0])
    drop = (idx == _wrap(p, length)) | (idx == _wrap(p + 1, length))
    return _compact_after_deleting(word, drop)


def delete_at(word: jax.Array, p: jax.Array) -> jax.Array:
    """Delete the single letter at position p."""
    idx = jnp.arange(word.shape[0])
    return _compact_after_deleting(word, idx == p)


def insert2(word: jax.Array, p: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
    """Insert `first`, `second` so that they land at positions p, p+1.

    Never wraps: the seam is arbitrary, so inserting "across" it is the same
    necklace as inserting at position 0.
    """
    idx = jnp.arange(word.shape[0])
    shifted = jnp.concatenate([jnp.zeros((2,), dtype=word.dtype), word[:-2]])
    out = jnp.where(idx < p, word, shifted)
    out = jnp.where(idx == p, first, out)
    return jnp.where(idx == p + 1, second, out)


def swap_at(word: jax.Array, p: jax.Array) -> jax.Array:
    """Exchange the letters at p and p+1 (mod length)."""
    length = word_length(word)
    idx = jnp.arange(word.shape[0])
    left, right = _wrap(p, length), _wrap(p + 1, length)
    a, b = word[left], word[right]
    out = jnp.where(idx == left, b, word)
    return jnp.where(idx == right, a, out)


def braid_at(word: jax.Array, p: jax.Array) -> jax.Array:
    """``(a, b, a) -> (b, a, b)`` at positions p, p+1, p+2 (mod length)."""
    length = word_length(word)
    idx = jnp.arange(word.shape[0])
    first, second, third = (_wrap(p + k, length) for k in range(3))
    a, b = word[first], word[second]
    out = jnp.where(idx == first, b, word)
    out = jnp.where(idx == second, a, out)
    return jnp.where(idx == third, b, out)


def append_letter(word: jax.Array, length: jax.Array, letter: jax.Array) -> jax.Array:
    idx = jnp.arange(word.shape[0])
    return jnp.where(idx == length, letter, word)


def flip_at(word: jax.Array, p: jax.Array) -> jax.Array:
    idx = jnp.arange(word.shape[0])
    return jnp.where(idx == p, -word[p], word)


def top_generator_position(word: jax.Array, n: jax.Array) -> jax.Array:
    """Index of the (assumed unique) occurrence of ``+-(n-1)``."""
    return jnp.argmax(jnp.abs(word) == (n - 1)).astype(jnp.int32)


# -- action decoding and dispatch ---------------------------------------------


def decode(spec: ActionSpec, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Traced version of `ActionSpec.decode`, returning `(kind, position, letter)`.

    `letter` is the signed generator inserted by INSERT and is meaningless for
    the other kinds.
    """
    starts = jnp.asarray(spec.starts)
    kind = jnp.searchsorted(starts, action, side="right") - 1
    kind = jnp.clip(kind, 0, len(spec.starts) - 2).astype(jnp.int32)

    rem = action - spec.start_of(INSERT)
    ins_position = jnp.mod(rem, spec.max_len)
    gs = jnp.floor_divide(rem, spec.max_len)
    generator = jnp.floor_divide(gs, 2) + 1
    sign = jnp.where(jnp.mod(gs, 2) == 0, 1, -1)

    position = action - starts[kind]
    position = jnp.where(kind == INSERT, ins_position, position).astype(jnp.int32)
    letter = (sign * generator).astype(DTYPE)
    return kind, position, letter


def apply_action(
    spec: ActionSpec,
    word: jax.Array,
    n: jax.Array,
    action: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply one action. Assumes it is legal; illegality is handled by the mask."""
    kind, position, letter = decode(spec, action)
    length = word_length(word)

    def _reduce(_):
        return reduce_at(word, position), n

    def _commute(_):
        return swap_at(word, position), n

    def _braid(_):
        return braid_at(word, position), n

    def _insert(_):
        return insert2(word, position, letter, -letter), n

    def _destab(_):
        return delete_at(word, top_generator_position(word, n)), n - 1

    def _stab_pos(_):
        return append_letter(word, length, n.astype(DTYPE)), n + 1

    def _stab_neg(_):
        return append_letter(word, length, -n.astype(DTYPE)), n + 1

    def _pass(_):
        return word, n

    def _crossing(_):
        return flip_at(word, position), n

    # Indexed by kind constant rather than written out in order, so that adding a
    # move or renumbering the constants cannot silently misdispatch.
    branches: list = [None] * NUM_KINDS
    branches[REDUCE] = _reduce
    branches[COMMUTE] = _commute
    branches[BRAID] = _braid
    branches[INSERT] = _insert
    branches[DESTABILIZE] = _destab
    branches[STABILIZE_POS] = _stab_pos
    branches[STABILIZE_NEG] = _stab_neg
    branches[PASS] = _pass
    branches[CROSSING_CHANGE] = _crossing
    assert all(branch is not None for branch in branches)

    new_word, new_n = jax.lax.switch(kind, branches, None)
    return new_word.astype(DTYPE), new_n.astype(jnp.int32)


# -- legality ------------------------------------------------------------------


def legal_action_mask(
    spec: ActionSpec,
    word: jax.Array,
    n: jax.Array,
    allow_crossing_change: jax.Array | bool,
) -> jax.Array:
    """Boolean mask over the flat action space.

    PASS is always legal, so the mask is never empty -- Pgx requires that.
    """
    length = word_length(word)
    max_len = spec.max_len
    idx = jnp.arange(max_len)
    inside = idx < length

    nxt = cyclic_shift(word, 1)
    nxt2 = cyclic_shift(word, 2)

    abs_w = jnp.abs(word)
    abs_next = jnp.abs(nxt)

    # A cyclic pair needs two distinct positions, a cyclic triple needs three.
    reduce_mask = inside & (length >= 2) & (nxt == -word)
    commute_mask = (
        inside & (length >= 2) & (nxt != 0) & (jnp.abs(abs_w - abs_next) >= 2)
    )
    braid_mask = (
        inside
        & (length >= 3)
        & (nxt2 == word)
        & (jnp.sign(word) == jnp.sign(nxt))
        & (jnp.abs(abs_w - abs_next) == 1)
    )

    # INSERT: the seam is arbitrary, so positions 0..length-1 cover every distinct
    # necklace; position `length` would only repeat position 0.
    generators = jnp.arange(1, spec.num_generators + 1)
    gen_ok = (generators <= n - 1)[:, None, None]
    pos_ok = (idx < jnp.maximum(length, 1))[None, None, :]
    room_ok = length + 2 <= max_len
    insert_mask = jnp.broadcast_to(
        gen_ok & pos_ok & room_ok, (spec.num_generators, 2, max_len)
    ).reshape(-1)

    # DESTABILIZE is position-free now: the lone top generator may sit anywhere,
    # because rotating it into last place is a conjugation and conjugation is
    # built into the representation.
    top = n - 1
    top_count = jnp.sum(abs_w == top)
    destab_ok = (n >= 2) & (length >= 1) & (top_count == 1)

    stab_ok = (n < spec.max_strands) & (length + 1 <= max_len)

    crossing_mask = (word != 0) & jnp.asarray(allow_crossing_change)

    singletons = jnp.asarray([destab_ok, stab_ok, stab_ok])
    others = jnp.concatenate(
        [reduce_mask, commute_mask, braid_mask, insert_mask, singletons, crossing_mask]
    )
    # PASS exists only to guarantee a non-empty mask (Pgx requires one), so it is
    # legal only when nothing else is. Left always-legal it is a strictly
    # dominated instant-forfeit: an untrained Scrambler picks it on ply 1 in
    # roughly a third of games -- the same collapse this benchmark already guards
    # against in Go by masking pass for the opening plies.
    pass_ok = ~jnp.any(others)

    return jnp.concatenate(
        [
            reduce_mask,
            commute_mask,
            braid_mask,
            insert_mask,
            singletons,
            jnp.asarray([pass_ok]),
            crossing_mask,
        ]
    ).astype(jnp.bool_)


# -- diagnostics ---------------------------------------------------------------


def permutation(word: jax.Array, n: jax.Array, max_strands: int) -> jax.Array:
    """Underlying permutation of the braid, as an array of length `max_strands`.

    Strands beyond `n` are fixed points. The number of cycles of this permutation
    is the number of components of the closure, which is invariant under every
    move in this module except the crossing change (which also preserves it).
    """
    perm = jnp.arange(max_strands, dtype=jnp.int32)

    def body(carry, letter):
        i = jnp.abs(letter) - 1  # 0-based: swaps strands i and i+1
        swapped = jnp.where(letter == 0, carry, _swap_entries(carry, i))
        return swapped, None

    perm, _ = jax.lax.scan(body, perm, word)
    return perm


def _swap_entries(perm: jax.Array, i: jax.Array) -> jax.Array:
    idx = jnp.arange(perm.shape[0])
    a = perm[i]
    b = perm[jnp.minimum(i + 1, perm.shape[0] - 1)]
    out = jnp.where(idx == i, b, perm)
    return jnp.where(idx == i + 1, a, out)


def num_components(word: jax.Array, n: jax.Array, max_strands: int) -> jax.Array:
    """Number of components of the closure (1 means it is a knot)."""
    perm = permutation(word, n, max_strands)
    active = jnp.arange(max_strands) < n

    # Count cycles by counting, for each element, whether it is the minimum of its
    # own cycle. Bounded iteration: max_strands steps suffice.
    def orbit_min(start):
        def body(carry, _):
            current, best = carry
            current = perm[current]
            return (current, jnp.minimum(best, current)), None

        (_, best), _ = jax.lax.scan(body, (start, start), None, length=max_strands)
        return best

    mins = jax.vmap(orbit_min)(jnp.arange(max_strands))
    is_cycle_min = (mins == jnp.arange(max_strands)) & active
    return jnp.sum(is_cycle_min).astype(jnp.int32)


def writhe(word: jax.Array) -> jax.Array:
    """Exponent sum. Invariant under conjugation and every tier-1 move; changes
    by +-1 under (de)stabilisation. Cheap conjugacy-class check."""
    return jnp.sum(jnp.sign(word)).astype(jnp.int32)
