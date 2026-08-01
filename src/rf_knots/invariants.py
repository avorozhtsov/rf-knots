"""Knot invariants of a braid closure, and identification against a knot table.

Every rung of the ladder is a braid word, and until now the only thing recorded
about the knot it closes to was the word's length. That is not an invariant: a
word can be long and the knot small. `R(3,18)#0` is eighteen letters and is the
seven-crossing knot `7_5`, whose unknotting number is a theorem -- which the
generator, whose only knottedness filter is a depth-4 breadth-first search, had
no way to notice. This module computes what the word actually determines.

What is here, and why each one is trusted:

* **Alexander polynomial**, from the reduced Burau representation. Exact integer
  arithmetic, no fitting, and it reproduces the textbook value for every torus
  knot and for the figure-eight.
* **Jones polynomial**, from the Kauffman bracket evaluated in the
  Temperley-Lieb algebra. The state sum over `2^c` states is correct and hopeless
  past twenty crossings; TL_n has only Catalan(n) basis elements, so a 26-letter
  word on 5 strands costs the same as a 4-letter one.
* **determinant**, as `|Delta(-1)|`.
* **genus bounds**: `deg Delta / 2 <= g_3 <= (c - n + 1) / 2`, the lower bound
  from the Alexander polynomial and the upper from the Bennequin surface of the
  braid.
* **signature**, only if `spherogram` is importable -- see below.
* **identification**, by matching the fingerprint against a bundled table.

**On the signature.** It is not computed here from the braid word, and the reason
is worth recording. A local rule on the obvious generators of the Bennequin
surface -- one loop per consecutive pair of crossings in a column -- was fitted
against 50 reference matrices and reproduces the signature on all 50, including
every torus knot. It then fails to reproduce the Alexander polynomial on 18 of
them, so whatever it is, it is not a Seifert matrix; its symmetrisation merely
has the right inertia on the sample. Shipping it would have been shipping a
coincidence. `spherogram` computes a real Seifert matrix without needing Sage, so
`signature()` defers to it and returns `None` when it is absent. The signatures
in the committed rung data were computed with it and do not need it to be read.

`spherogram` is deliberately not a declared optional dependency: it drags in
IPython, matplotlib and twenty other packages, and adding it to `pyproject.toml`
would put all of that in `uv.lock` for every image that builds from this
repository. Reach for it explicitly when it is wanted:

    uv run --with snappy python scripts/rung_invariants.py
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

Word = tuple[int, ...]
Laurent = dict[int, int]
"""A Laurent polynomial as {exponent: coefficient}, with no zero coefficients."""


# --------------------------------------------------------------------------- #
# Laurent polynomial arithmetic
# --------------------------------------------------------------------------- #

def _mul(a: Laurent, b: Laurent) -> Laurent:
    out: Laurent = {}
    for e1, c1 in a.items():
        for e2, c2 in b.items():
            out[e1 + e2] = out.get(e1 + e2, 0) + c1 * c2
    return {e: c for e, c in out.items() if c}


def _add(a: Laurent, b: Laurent) -> Laurent:
    out = dict(a)
    for e, c in b.items():
        out[e] = out.get(e, 0) + c
    return {e: c for e, c in out.items() if c}


def _scale(a: Laurent, factor: int) -> Laurent:
    return {e: c * factor for e, c in a.items()} if factor else {}


def _divide(num: Laurent, den: Laurent) -> Laurent:
    """Exact division. Raises if the division leaves a remainder."""
    num = dict(num)
    quotient: Laurent = {}
    top = max(den)
    while num:
        e = max(num) - top
        c, rem = divmod(num[max(num)], den[top])
        if rem:
            raise ValueError("not divisible")
        quotient[e] = c
        for de, dc in den.items():
            key = e + de
            num[key] = num.get(key, 0) - c * dc
            if not num[key]:
                del num[key]
    return {e: c for e, c in quotient.items() if c}


def _canonical(p: Laurent) -> Laurent:
    """Shift to lowest degree zero and force a positive leading coefficient.

    The Alexander polynomial is only defined up to `+- t^k`, so a canonical form
    is what makes two of them comparable -- which is the whole point of computing
    it here. The Jones polynomial is *not* canonicalised: its exponents carry the
    chirality, and shifting them would throw away the one invariant that tells a
    knot from its mirror image.
    """
    if not p:
        return {}
    low = min(p)
    shifted = {e - low: c for e, c in p.items()}
    return shifted if shifted[max(shifted)] > 0 else {e: -c for e, c in shifted.items()}


def format_polynomial(p: Laurent, var: str = "t") -> str:
    """`2 - 4t + 5t^2 - 4t^3 + 2t^4`, for tables meant to be read."""
    if not p:
        return "0"
    parts = []
    for e in sorted(p):
        c = p[e]
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if e == 0:
            body = str(mag)
        else:
            power = var if e == 1 else f"{var}^{e}"
            body = power if mag == 1 else f"{mag}{power}"
        parts.append((sign, body))
    head = ("-" + parts[0][1]) if parts[0][0] == "-" else parts[0][1]
    return " ".join([head] + [f"{s} {b}" for s, b in parts[1:]])


def to_pairs(p: Laurent) -> tuple[tuple[int, int], ...]:
    """JSON-friendly `((exponent, coefficient), ...)`, sorted by exponent."""
    return tuple((e, p[e]) for e in sorted(p))


def from_pairs(pairs) -> Laurent:
    return {int(e): int(c) for e, c in pairs}


# --------------------------------------------------------------------------- #
# Alexander polynomial, via the reduced Burau representation
# --------------------------------------------------------------------------- #

def _burau(word: Word, strands: int) -> list[list[Laurent]]:
    """The reduced Burau matrix of the braid word, over Z[t, t^-1].

    A generator differs from the identity in a single row, so right-multiplying
    by it leaves every column alone except `r-1`, `r` and `r+1`:

        col[r]   <- col[r] * g[r][r]
        col[r+-1] <- col[r+-1] + col[r] * g[r][r+-1]

    Doing the full matrix product instead costs `size^3` polynomial
    multiplications per letter rather than `3 * size`, which at nine strands and
    forty letters is the difference between seconds and milliseconds.
    """
    size = strands - 1
    if size == 0:
        return []

    columns = [[{0: 1} if r == c else {} for r in range(size)] for c in range(size)]
    for letter in word:
        r = abs(letter) - 1
        if letter > 0:
            diagonal, left, right = {1: -1}, {1: 1}, {0: 1}
        else:
            diagonal, left, right = {-1: -1}, {0: 1}, {-1: 1}
        middle = columns[r]
        if r - 1 >= 0:
            columns[r - 1] = [_add(v, _mul(m, left))
                              for v, m in zip(columns[r - 1], middle, strict=True)]
        if r + 1 < size:
            columns[r + 1] = [_add(v, _mul(m, right))
                              for v, m in zip(columns[r + 1], middle, strict=True)]
        columns[r] = [_mul(m, diagonal) for m in middle]
    return [[columns[c][r] for c in range(size)] for r in range(size)]


def _integer_determinant(matrix: list[list[int]]) -> int:
    """Exact determinant of an integer matrix, fraction-free (Bareiss)."""
    size = len(matrix)
    if size == 0:
        return 1
    rows = [row[:] for row in matrix]
    sign, previous = 1, 1
    for k in range(size - 1):
        if rows[k][k] == 0:
            swap = next((i for i in range(k + 1, size) if rows[i][k] != 0), None)
            if swap is None:
                return 0
            rows[k], rows[swap] = rows[swap], rows[k]
            sign = -sign
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                rows[i][j] = (rows[i][j] * rows[k][k] - rows[i][k] * rows[k][j]) // previous
        previous = rows[k][k]
    return sign * rows[size - 1][size - 1]


def _interpolate(xs: list[int], ys: list[int]) -> Laurent:
    """The unique polynomial through the points, with integer coefficients.

    Exact throughout: the determinant of an integer matrix is an integer, so the
    interpolant has integer coefficients and a `Fraction` that fails to reduce is
    a bug rather than a rounding artefact.
    """
    from fractions import Fraction

    degree = len(xs) - 1
    coefficients = [Fraction(0)] * (degree + 1)
    for i, xi in enumerate(xs):
        denominator = 1
        for j, xj in enumerate(xs):
            if j != i:
                denominator *= xi - xj
        basis = [Fraction(1)] + [Fraction(0)] * degree
        span = 0
        for j, xj in enumerate(xs):
            if j == i:
                continue
            nxt = [Fraction(0)] * (degree + 1)
            for k in range(span + 1):
                nxt[k + 1] += basis[k]
                nxt[k] -= basis[k] * xj
            basis, span = nxt, span + 1
        weight = Fraction(ys[i], denominator)
        for k in range(degree + 1):
            coefficients[k] += weight * basis[k]
    out: Laurent = {}
    for exponent, value in enumerate(coefficients):
        if value.denominator != 1:
            raise ValueError("interpolant is not integral")
        if value:
            out[exponent] = int(value)
    return out


def _determinant(matrix: list[list[Laurent]], span_bound: int) -> Laurent:
    """Determinant of a matrix of Laurent polynomials, by evaluation.

    The obvious Leibniz expansion is `(n-1)!` terms, which is 24 at five strands
    and 40320 at nine -- and the knot table has a hundred nine-strand braids in
    it. Evaluating at enough integer points and interpolating is exact, needs
    only integer determinants, and does not care how wide the braid is.

    `span_bound` is how many points are needed, and it has to be a theorem rather
    than a guess: interpolating through too few points is silently wrong. Bound
    the entries' degrees and you get `size * c`, which is honest but costs
    hundreds of evaluations at integers raised to the fortieth power. For the
    matrix this is actually called on, `det = +- t^k Delta(t) (1 + ... + t^(n-1))`,
    and `deg Delta <= 2g <= c - n + 1` because the braid's own Seifert surface
    realises some genus -- so the span is at most `c`, the length of the word.
    """
    size = len(matrix)
    if size == 0:
        return {0: 1}
    lowest = min((min(p) for row in matrix for p in row if p), default=0)
    shift = -lowest if lowest < 0 else 0
    # Clearing the negative exponents multiplies the determinant by t^(shift*size);
    # it is divided out again at the end.
    cleared = [[{e + shift: c for e, c in p.items()} for p in row] for row in matrix]
    points = list(range(shift * size + span_bound + 1))
    values = [
        _integer_determinant(
            [[sum(c * x**e for e, c in p.items()) for p in row] for row in cleared]
        )
        for x in points
    ]
    return {e - shift * size: c for e, c in _interpolate(points, values).items()}


def alexander_polynomial(word: Word, strands: int) -> Laurent:
    """`Delta(t)`, canonicalised to lowest degree 0 with a positive leading term.

    From `det(Burau(beta) - I) * (1 - t) / (1 - t^n)`. The division is exact for
    any braid whose closure is a knot, and raising on a remainder is a real check
    rather than a formality -- it fires if the word is not what it claims to be.
    """
    if strands <= 1 or not word:
        return {0: 1}
    matrix = _burau(tuple(word), strands)
    for r in range(strands - 1):
        matrix[r][r] = _add(matrix[r][r], {0: -1})
    numerator = _determinant(matrix, span_bound=len(word))
    if not numerator:
        return {}
    return _canonical(_divide(numerator, {e: 1 for e in range(strands)}))


def determinant(word: Word, strands: int) -> int:
    """`|Delta(-1)|`, the order of the first homology of the double branched cover."""
    poly = alexander_polynomial(word, strands)
    # `e % 2` rather than `e`: negative exponents would make `(-1) ** e` a float.
    return abs(sum(c * (-1) ** (e % 2) for e, c in poly.items()))


# --------------------------------------------------------------------------- #
# Jones polynomial, via Temperley-Lieb
# --------------------------------------------------------------------------- #

class _Components:
    """Union-find, used wherever "how many circles is this" has to be answered."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@functools.lru_cache(maxsize=1 << 17)
def _compose(upper: tuple[int, ...], lower: tuple[int, ...], n: int) -> tuple[tuple[int, ...], int]:
    """Stack two planar matchings, returning the composite and the loop count.

    Cached, and the cache is why this is usable at all past seven strands. The
    second argument is always one of the `n-1` generators, and the first ranges
    over at most Catalan(n) matchings, so the whole composition table for a given
    strand count is small and fills in once -- while a table of three thousand
    knots would otherwise recompute it from scratch for every word.

    A matching on `2n` points is stored as a tuple where entry `i` is the partner
    of point `i`; points `0..n-1` are the top row and `n..2n-1` the bottom. The
    bottom of `upper` is glued to the top of `lower`. A walk that starts at a
    free end alternates between the two diagrams until it reaches another free
    end; whatever is left in the glued middle closes up on itself, and each such
    circle is a factor of `d`.
    """
    result = [-1] * (2 * n)
    for start in range(2 * n):
        if result[start] != -1:
            continue
        # `in_upper` says which diagram the next hop is taken in; `point` is an
        # index into that diagram's own numbering.
        in_upper, point = (start < n), (start if start < n else start)
        while True:
            if in_upper:
                partner = upper[point]
                if partner < n:                       # a free end on the top row
                    end = partner
                    break
                point, in_upper = partner - n, False  # cross the glue, downwards
            else:
                partner = lower[point]
                if partner >= n:                      # a free end on the bottom row
                    end = partner
                    break
                point, in_upper = partner + n, True   # cross the glue, upwards
        result[start], result[end] = end, start

    # A middle point is "open" when one of its two sides runs to a free end; a
    # component of the middle with no open point never escapes, so it is a loop.
    middle = _Components(n)
    open_middle = [False] * n
    for i in range(n):
        up, down = upper[n + i], lower[i]
        if up < n:
            open_middle[i] = True
        else:
            middle.union(i, up - n)
        if down >= n:
            open_middle[i] = True
        else:
            middle.union(i, down)
    closed = {middle.find(i) for i in range(n)} - {
        middle.find(i) for i in range(n) if open_middle[i]
    }
    return tuple(result), len(closed)


def _identity_matching(n: int) -> tuple[int, ...]:
    return tuple(list(range(n, 2 * n)) + list(range(n)))


def _cup_cap(n: int, index: int) -> tuple[int, ...]:
    """`e_i`: the two strands at `i, i+1` turn back on themselves."""
    m = list(_identity_matching(n))
    i = index - 1
    m[i], m[i + 1] = i + 1, i
    m[n + i], m[n + i + 1] = n + i + 1, n + i
    return tuple(m)


def kauffman_bracket(word: Word, strands: int) -> Laurent:
    """`<D>` for the closed braid, normalised so the unknot diagram is 1.

    `sigma_i -> A + A^-1 e_i` and `sigma_i^-1 -> A^-1 + A e_i` in TL_n, then the
    Markov trace closes the diagram. This is exponential in the number of
    *strands* -- Catalan(n) basis elements -- and linear in the length of the
    word, which is the right way round: the ladder's words are long and narrow.
    """
    n = max(strands, 1)
    delta: Laurent = {2: -1, -2: -1}
    state: dict[tuple[int, ...], Laurent] = {_identity_matching(n): {0: 1}}
    for letter in word:
        index = abs(letter)
        straight = {1: 1} if letter > 0 else {-1: 1}
        turned = {-1: 1} if letter > 0 else {1: 1}
        e = _cup_cap(n, index)
        nxt: dict[tuple[int, ...], Laurent] = {}
        for matching, coeff in state.items():
            for factor, gen in ((straight, None), (turned, e)):
                if gen is None:
                    key, loops = matching, 0
                else:
                    key, loops = _compose(matching, gen, n)
                weight = _mul(coeff, factor)
                for _ in range(loops):
                    weight = _mul(weight, delta)
                nxt[key] = _add(nxt.get(key, {}), weight)
        state = {k: v for k, v in nxt.items() if v}

    total: Laurent = {}
    for matching, coeff in state.items():
        loops = _trace_loops(matching, n)
        term = coeff
        for _ in range(loops - 1):
            term = _mul(term, delta)
        total = _add(total, term)
    return total


def _trace_loops(matching: tuple[int, ...], n: int) -> int:
    """Closing top `i` to bottom `i` round the back, how many circles result.

    Every point then has degree two -- one edge along the diagram, one round the
    closure -- so the circles are exactly the connected components.
    """
    circles = _Components(2 * n)
    for i in range(2 * n):
        circles.union(i, matching[i])          # along the diagram
    for i in range(n):
        circles.union(i, i + n)                # round the back
    return len({circles.find(i) for i in range(2 * n)})


def jones_polynomial(word: Word, strands: int) -> Laurent:
    """`V(t)`, with `A = t^(-1/4)` and the writhe correction applied.

    Not canonicalised: `V(1/t)` is the mirror image, and that difference is the
    reason to compute it at all.
    """
    bracket = kauffman_bracket(tuple(word), strands)
    if not bracket:
        return {}
    writhe = sum(1 if x > 0 else -1 for x in word)
    factor: Laurent = {0: 1}
    unit = {-3: -1} if writhe > 0 else {3: -1}
    for _ in range(abs(writhe)):
        factor = _mul(factor, unit)
    normalised = _mul(factor, bracket)
    if any(e % 4 for e in normalised):
        raise ValueError("bracket exponents are not divisible by 4; word is not a knot")
    return {-e // 4: c for e, c in normalised.items()}


# --------------------------------------------------------------------------- #
# Signature, deferred to spherogram
# --------------------------------------------------------------------------- #

def signature(word: Word, strands: int) -> int | None:
    """`sigma(K)`, or `None` if `spherogram` is not installed.

    Sign convention: positive braids have negative signature, so the right-handed
    trefoil is `-2`. `spherogram` builds the mirror of our closure, hence the
    flip. See the module docstring for why this is not computed here directly.
    """
    letters = [int(x) for x in word if int(x) != 0]
    if not letters:
        return 0                                   # the unknot; spherogram wants a crossing
    try:
        import numpy as np
        import spherogram
    except ImportError:
        return None
    matrix = spherogram.Link(braid_closure=letters).seifert_matrix()
    if not matrix:
        return 0
    form = np.array(matrix, dtype=float)
    form = form + form.T
    eigenvalues = np.linalg.eigvalsh(form)
    tolerance = 1e-7 * max(1.0, float(abs(eigenvalues).max()))
    return -int((eigenvalues > tolerance).sum() - (eigenvalues < -tolerance).sum())


# --------------------------------------------------------------------------- #
# The whole fingerprint
# --------------------------------------------------------------------------- #

def connected_summands(word: Word, strands: int) -> tuple[Word, ...] | None:
    """Braid words for the prime summands, or `None` without `spherogram`.

    Knot tables list prime knots, so a composite knot fails to identify no matter
    how small its pieces are -- six of the ladder's rungs are connected sums of
    trefoils and figure-eights and looked like unlabelled frontier knots. An
    empty tuple means the closure is the unknot; a single entry means prime.
    """
    letters = [int(x) for x in word if int(x) != 0]
    if not letters:
        return ()
    try:
        import spherogram
    except ImportError:
        return None
    link = spherogram.Link(braid_closure=letters)
    link.simplify("global")
    if not link.crossings:
        return ()
    pieces = []
    for piece in link.deconnect_sum():
        braid = tuple(int(x) for x in piece.braid_word())
        if braid:
            pieces.append(braid)
    return tuple(pieces)


@dataclass(frozen=True)
class Invariants:
    """Everything this module can say about one braid word.

    `crossings` is the length of the *word*, which is an upper bound on the
    crossing number of the knot and frequently a poor one. Where the knot is
    identified, `identified_crossings` is the real thing.
    """

    word: Word
    strands: int
    crossings: int
    writhe: int
    alexander: tuple[tuple[int, int], ...]
    determinant: int
    jones: tuple[tuple[int, int], ...]
    signature: int | None
    genus_lower: int
    genus_upper: int
    unknotting_lower: int | None
    name: str | None = None
    identified_crossings: int | None = None
    mirror: bool = False
    unknotting: int | None = None
    summands: tuple[str, ...] = field(default_factory=tuple)
    unknotting_upper: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def alexander_polynomial(self) -> Laurent:
        return from_pairs(self.alexander)

    @property
    def jones_polynomial(self) -> Laurent:
        return from_pairs(self.jones)

    @property
    def unknotting_known(self) -> int | None:
        """`u` where the bounds meet, whether or not anyone tabulated it."""
        if self.unknotting is not None:
            return self.unknotting
        if (self.unknotting_upper is not None and self.unknotting_lower is not None
                and self.unknotting_upper == self.unknotting_lower):
            return self.unknotting_upper
        return None


def _decompose(result: Invariants) -> Invariants:
    """Name the prime summands of a composite knot, and bound `u` from them.

    Two theorems do the work. Unknotting number is subadditive under connected
    sum, so the summands' numbers add to an upper bound -- and only an upper
    bound, since additivity itself was disproved in 2025. Scharlemann's theorem
    says an unknotting-number-one knot is prime, so any composite knot has
    `u >= 2`. Together with `|sigma|/2` those frequently meet, which is how a
    rung with no name at all still ends up with an exact unknotting number.
    """
    import dataclasses

    pieces = connected_summands(result.word, result.strands)
    if pieces is None or len(pieces) < 2:
        return result
    names, upper = [], 0
    for braid in pieces:
        strands = max((abs(x) for x in braid), default=0) + 1
        piece = invariants(braid, strands, decompose=False)
        names.append(piece.name or "?")
        if upper is not None and piece.unknotting is not None:
            upper += piece.unknotting
        else:
            upper = None
    lower = max(result.unknotting_lower or 0, 2)   # Scharlemann
    notes = result.notes + (
        f"composite: {' # '.join(names)}. Knot tables list prime knots, which is "
        "why the fingerprint found nothing.",
    )
    if upper is not None and upper == lower:
        notes = notes + (f"u = {upper} exactly: the summands give u <= {upper}, and "
                         "|sigma|/2 with Scharlemann's theorem gives the same lower bound.",)
    return dataclasses.replace(
        result, summands=tuple(names), unknotting_upper=upper, unknotting_lower=lower,
        notes=notes,
    )


def invariants(word, strands: int, identify_knot: bool = True,
               decompose: bool = True) -> Invariants:
    """Compute the fingerprint of a braid word, and try to name the knot."""
    word = tuple(int(x) for x in word if int(x) != 0)
    alexander = alexander_polynomial(word, strands)
    jones = jones_polynomial(word, strands)
    sigma = signature(word, strands)
    # deg Delta <= 2g, and the Bennequin surface of the braid realises some
    # genus, so the truth is caught between them. For the unknot both are 0.
    genus_lower = max(alexander) // 2 if alexander else 0
    genus_upper = max(0, (len(word) - strands + 1)) // 2
    result = Invariants(
        word=word,
        strands=strands,
        crossings=len(word),
        writhe=sum(1 if x > 0 else -1 for x in word),
        alexander=to_pairs(alexander),
        determinant=abs(sum(c * (-1) ** (e % 2) for e, c in alexander.items())),
        jones=to_pairs(jones),
        signature=sigma,
        genus_lower=genus_lower,
        genus_upper=genus_upper,
        unknotting_lower=None if sigma is None else abs(sigma) // 2,
    )
    if identify_knot:
        from rf_knots.knot_table import identify

        result = identify(result)
        if result.unknotting is not None:
            result = dataclasses_replace(result, unknotting_upper=result.unknotting)
    # Only worth decomposing what the table could not name: a prime knot that
    # identified is already answered, and deconnect_sum is not free.
    if decompose and result.name is None:
        result = _decompose(result)
    return result


def dataclasses_replace(inv: Invariants, **changes) -> Invariants:
    import dataclasses

    return dataclasses.replace(inv, **changes)
