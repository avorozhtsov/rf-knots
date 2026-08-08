"""Slow, obvious, pure-Python reference implementations.

Nothing here runs during training. It exists so the JAX kernels in `braid.py` can
be differentially tested against something with no index arithmetic in it, and so
that small instances can be solved exactly by breadth-first search.

The strongest check available is `artin_image`: the Artin representation of the
braid group on the free group is *faithful*, so two braid words are equal in B_n
iff their Artin images agree on every generator. That turns "did this rewrite
preserve the braid group element?" into an exact, decidable test.
"""

from __future__ import annotations

from collections import deque

from rf_knots.actions import (
    BRAID,
    COMMUTE,
    CROSSING_CHANGE,
    DESTABILIZE,
    INSERT,
    PASS,
    REDUCE,
    STABILIZE_NEG,
    STABILIZE_POS,
    ActionSpec,
)

Word = tuple[int, ...]

# -- free group ----------------------------------------------------------------


def free_reduce(word: Word) -> Word:
    stack: list[int] = []
    for letter in word:
        if stack and stack[-1] == -letter:
            stack.pop()
        else:
            stack.append(letter)
    return tuple(stack)


def free_inverse(word: Word) -> Word:
    return tuple(-letter for letter in reversed(word))


def _substitute(word: Word, table: dict[int, Word]) -> Word:
    out: list[int] = []
    for letter in word:
        image = table[abs(letter)]
        out.extend(image if letter > 0 else free_inverse(image))
    return free_reduce(tuple(out))


def _generator_table(letter: int, n: int) -> dict[int, Word]:
    """Artin automorphism of the free group F_n induced by one braid letter."""
    i = abs(letter)
    table: dict[int, Word] = {k: (k,) for k in range(1, n + 1)}
    if letter > 0:
        table[i] = (i, i + 1, -i)
        table[i + 1] = (i,)
    else:
        table[i] = (i + 1,)
        table[i + 1] = (-(i + 1), i, i + 1)
    return table


def artin_image(word: Word, n: int) -> tuple[Word, ...]:
    """Image of x_1..x_n under the automorphism of F_n determined by `word`.

    Faithful, so equality of these tuples is equality in B_n. (The composition
    order makes this an anti-homomorphism, which is equally faithful and equally
    fine as an equality test.)
    """
    images = [(k,) for k in range(1, n + 1)]
    for letter in word:
        if letter == 0:
            continue
        if abs(letter) >= n + 1:
            raise ValueError(f"letter {letter} outside B_{n}")
        table = _generator_table(letter, n)
        images = [_substitute(image, table) for image in images]
    return tuple(images)


def equal_in_braid_group(left: Word, right: Word, n: int) -> bool:
    return artin_image(left, n) == artin_image(right, n)


def rotations(word: Word) -> list[Word]:
    """Every cyclic rotation -- the necklace this word represents."""
    return [word[k:] + word[:k] for k in range(max(len(word), 1))]


def equal_up_to_rotation(left: Word, right: Word, n: int) -> bool:
    """True if `right` equals some rotation of `left` as an element of B_n.

    Rotation is conjugation by a prefix, so this is *sufficient* for conjugacy but
    not necessary -- full conjugacy testing in a braid group needs summit sets and
    is not worth implementing here. For seam moves the exact justification is the
    decomposition in `seam_move_via_rotation` instead.
    """
    images = {artin_image(rotation, n) for rotation in rotations(left)}
    return artin_image(right, n) in images


def seam_move_via_rotation(
    spec: ActionSpec, word: Word, n: int, kind: int, position: int
) -> Word:
    """Apply a cyclic move the long way: rotate it into the interior, then act.

    This is *why* the cyclic moves are sound. Rotation is conjugation, which is a
    Markov move and preserves the closure; the interior move preserves the braid
    group element outright. So the composite preserves the closure, and the result
    must be the same necklace as the direct cyclic move.
    """
    rotated = word[position:] + word[:position]
    action = spec.encode(kind, position=0)
    if not is_legal(spec, rotated, n, action, allow_crossing=True):
        raise ValueError(f"{spec.describe(action)} not legal on the rotated word")
    result, _ = apply(spec, rotated, n, action)
    return result


# -- closure combinatorics -----------------------------------------------------


def permutation(
    word: Word, n: int, cyclic_band_generators: bool = False
) -> list[int]:
    perm = list(range(n))
    for letter in word:
        if letter == 0:
            continue
        generator = abs(letter)
        i, j = (
            (0, n - 1)
            if cyclic_band_generators and generator == n
            else (generator - 1, generator)
        )
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def num_components(
    word: Word, n: int, cyclic_band_generators: bool = False
) -> int:
    """Components of the closure. Invariant under every move in this project."""
    perm = permutation(word, n, cyclic_band_generators)
    seen = [False] * n
    cycles = 0
    for start in range(n):
        if seen[start]:
            continue
        cycles += 1
        current = start
        while not seen[current]:
            seen[current] = True
            current = perm[current]
    return cycles


def writhe(word: Word) -> int:
    return sum(1 if letter > 0 else -1 for letter in word if letter != 0)


# -- moves ---------------------------------------------------------------------


def is_legal(spec: ActionSpec, word: Word, n: int, action: int, allow_crossing: bool) -> bool:
    kind, p, generator, _ = spec.decode(action)
    length = len(word)
    if kind == PASS:
        # Only as a last resort, so that it is never a dominated instant-forfeit.
        return not successors(spec, word, n, allow_crossing)
    if kind == REDUCE:
        return length >= 2 and p < length and word[p] == -word[(p + 1) % length]
    if kind == COMMUTE:
        if length < 2 or p >= length:
            return False
        distance = abs(abs(word[p]) - abs(word[(p + 1) % length]))
        if spec.cyclic_band_generators:
            return (
                abs(word[p]) != abs(word[(p + 1) % length])
                and distance not in (1, n - 1)
            )
        return (
            abs(abs(word[p]) - abs(word[(p + 1) % length])) >= 2
        )
    if kind == BRAID:
        if length < 3 or p >= length:
            return False
        a, b, c = word[p], word[(p + 1) % length], word[(p + 2) % length]
        distance = abs(abs(a) - abs(b))
        adjacent = distance == 1 or (
            spec.cyclic_band_generators and distance == n - 1
        )
        return c == a and (a > 0) == (b > 0) and adjacent
    if kind == INSERT:
        largest = n if spec.cyclic_band_generators else n - 1
        return (
            n >= 2
            and generator <= largest
            and (generator != n or n >= 3)
            and p < max(length, 1)
            and length + 2 <= spec.max_len
        )
    if kind == DESTABILIZE:
        top = n - 1
        seam_absent = not spec.cyclic_band_generators or all(abs(x) != n for x in word)
        return (
            n >= 2
            and length >= 1
            and sum(abs(x) == top for x in word) == 1
            and seam_absent
        )
    if kind in (STABILIZE_POS, STABILIZE_NEG):
        seam_absent = not spec.cyclic_band_generators or all(abs(x) != n for x in word)
        return n < spec.max_strands and length + 1 <= spec.max_len and seam_absent
    if kind == CROSSING_CHANGE:
        return allow_crossing and p < length
    raise AssertionError(f"unknown kind {kind}")


def apply(spec: ActionSpec, word: Word, n: int, action: int) -> tuple[Word, int]:
    """Apply an action, assuming it is legal. Mirrors `braid.apply_action`."""
    kind, p, generator, sign = spec.decode(action)
    letters = list(word)
    length = len(letters)
    if kind == REDUCE:
        for index in sorted({p, (p + 1) % length}, reverse=True):
            del letters[index]
    elif kind == COMMUTE:
        q = (p + 1) % length
        letters[p], letters[q] = letters[q], letters[p]
    elif kind == BRAID:
        first, second, third = p, (p + 1) % length, (p + 2) % length
        a, b = letters[first], letters[second]
        letters[first], letters[second], letters[third] = b, a, b
    elif kind == INSERT:
        value = sign * generator
        letters[p:p] = [value, -value]
    elif kind == DESTABILIZE:
        top = n - 1
        del letters[next(i for i, x in enumerate(letters) if abs(x) == top)]
        n -= 1
    elif kind == STABILIZE_POS:
        letters.append(n)
        n += 1
    elif kind == STABILIZE_NEG:
        letters.append(-n)
        n += 1
    elif kind == CROSSING_CHANGE:
        letters[p] = -letters[p]
    elif kind == PASS:
        pass
    else:
        raise AssertionError(f"unknown kind {kind}")
    return tuple(letters), n


def legal_actions(spec: ActionSpec, word: Word, n: int, allow_crossing: bool = False) -> list[int]:
    return [
        action
        for action in range(spec.num_actions)
        if is_legal(spec, word, n, action, allow_crossing)
    ]


def successors(
    spec: ActionSpec, word: Word, n: int, allow_crossing: bool = False
) -> list[tuple[int, Word, int]]:
    """Every legal non-PASS move as `(action, new_word, new_n)`, generated directly.

    Equivalent to `legal_actions` followed by `apply`, but it enumerates candidates
    instead of testing all `num_actions` of them, which is what makes search
    tractable. `test_reference.py` checks the two agree.
    """
    length = len(word)
    out: list[tuple[int, Word, int]] = []

    if length >= 2:
        for p in range(length):
            q = (p + 1) % length
            left, right = word[p], word[q]
            if left == -right:
                drop = sorted({p, q}, reverse=True)
                letters = list(word)
                for index in drop:
                    del letters[index]
                out.append((spec.encode(REDUCE, p), tuple(letters), n))
            distance = abs(abs(left) - abs(right))
            commute = (
                abs(left) != abs(right) and distance not in (1, n - 1)
                if spec.cyclic_band_generators
                else distance >= 2
            )
            if commute:
                letters = list(word)
                letters[p], letters[q] = letters[q], letters[p]
                out.append((spec.encode(COMMUTE, p), tuple(letters), n))

    if length >= 3:
        for p in range(length):
            first, second, third = p, (p + 1) % length, (p + 2) % length
            a, b, c = word[first], word[second], word[third]
            distance = abs(abs(a) - abs(b))
            adjacent = distance == 1 or (
                spec.cyclic_band_generators and distance == n - 1
            )
            if c == a and (a > 0) == (b > 0) and adjacent:
                letters = list(word)
                letters[first], letters[second], letters[third] = b, a, b
                out.append((spec.encode(BRAID, p), tuple(letters), n))

    if length + 2 <= spec.max_len:
        generator_stop = n + 1 if spec.cyclic_band_generators and n >= 3 else n
        for generator in range(1, generator_stop):
            for sign in (1, -1):
                value = sign * generator
                for p in range(max(length, 1)):
                    action = spec.encode(INSERT, p, generator, sign)
                    out.append((action, word[:p] + (value, -value) + word[p:], n))

    seam_absent = not spec.cyclic_band_generators or all(abs(x) != n for x in word)
    if n >= 2 and length >= 1 and seam_absent:
        top = n - 1
        positions = [index for index, x in enumerate(word) if abs(x) == top]
        if len(positions) == 1:
            letters = list(word)
            del letters[positions[0]]
            out.append((spec.encode(DESTABILIZE), tuple(letters), n - 1))

    if n < spec.max_strands and length + 1 <= spec.max_len and seam_absent:
        out.append((spec.encode(STABILIZE_POS), word + (n,), n + 1))
        out.append((spec.encode(STABILIZE_NEG), word + (-n,), n + 1))

    if allow_crossing:
        for p in range(length):
            out.append(
                (spec.encode(CROSSING_CHANGE, p), word[:p] + (-word[p],) + word[p + 1 :], n)
            )

    return out


def compile_cyclic_bands(word: Word, n: int) -> Word:
    """Compile seam generator ``+/-n`` to the BKL band ``a_{1,n}`` in Artin form.

    With ``w = sigma_{n-1} ... sigma_2``, the positive band is
    ``w sigma_1 w^-1`` and the negative band is ``w sigma_1^-1 w^-1``.
    Ordinary letters pass through unchanged.  The result is accepted by the
    historical verifier and makes every B-star witness proof-carrying.
    """
    if n < 2:
        if word:
            raise ValueError("a one-strand braid cannot contain generators")
        return ()
    prefix = tuple(range(n - 1, 1, -1))
    suffix = tuple(-value for value in reversed(prefix))
    out: list[int] = []
    for letter in word:
        if abs(letter) > n:
            raise ValueError(f"generator {letter} is invalid for B*_{n}")
        if abs(letter) == n:
            out.extend(prefix)
            out.append(1 if letter > 0 else -1)
            out.extend(suffix)
        else:
            out.append(letter)
    return tuple(out)


# -- exact search --------------------------------------------------------------


def bfs_unknot(
    spec: ActionSpec,
    word: Word,
    n: int,
    max_depth: int = 6,
    max_nodes: int = 400_000,
    max_growth: int = 2,
) -> list[int] | None:
    """Shortest move sequence reducing `(word, n)` to the empty 1-braid.

    Sound but deliberately incomplete: a returned path is always valid, but `None`
    only means "no path within `max_depth` moves, `max_growth` extra letters and
    `max_nodes` states" -- never "unsolvable". Simplification of a hard instance
    genuinely may require growing the word first, and the search space grows by
    roughly `(2n)^2` per unit of `max_growth`, so this is the knob that decides
    whether the oracle returns today.

    Only for tiny instances: as a ground-truth oracle for tests, and to calibrate
    how hard a `K`-move scramble really is.
    """
    start = (word, n)
    if start == ((), 1):
        return []
    length_cap = len(word) + max_growth
    seen = {start}
    queue: deque[tuple[Word, int, list[int]]] = deque([(word, n, [])])
    nodes = 0
    while queue:
        current, strands, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action, next_word, next_n in successors(spec, current, strands):
            if len(next_word) > length_cap:
                continue
            nxt = (next_word, next_n)
            if nxt in seen:
                continue
            nodes += 1
            if nodes > max_nodes:
                return None
            if nxt == ((), 1):
                return [*path, action]
            seen.add(nxt)
            queue.append((next_word, next_n, [*path, action]))
    return None


def format_word(word: Word, n: int) -> str:
    if not word:
        return f"B{n}: e"
    letters = " ".join(f"s{abs(x)}" + ("^-1" if x < 0 else "") for x in word)
    return f"B{n}: {letters}"
