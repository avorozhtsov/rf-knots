r"""Certified unknotting-number lower bounds, computed rather than tabulated.

## What was broken

[research/10 §1B](../../research/10-invariants-and-representations.md) names three
certified lower bounds — Murasugi `|sigma|/2`, Rasmussen `|s|/2`, Ozsvath-Szabo
`|tau|` — and says only the first is implemented. It was worse than that:
`invariants.signature` returns `None` unless `spherogram` is importable, and
`spherogram` was not installed, so **the one computed bound the project claimed
silently returned nothing**, and `lower_bounds.claims_for` produced an empty tuple
for every knot the bundled table could not name by hand.

This module fixes that and adds two more bounds:

| bound | statement | source here |
|---|---|---|
| Murasugi | `\|sigma(K)\|/2 <= u(K)` | signature of `V + V^T`, exactly |
| Ozsvath-Szabo | `\|tau(K)\| <= u(K)` | `knot_floer_homology` |
| Montesinos / Lickorish | `H_1(Sigma_2(K))` not cyclic `=> u(K) >= 2` | Smith form of `V + V^T` |

`u = 1` forces the double branched cover to be surgery on a knot in `S^3`, so its
first homology is cyclic of order `det(K)`. A non-cyclic `H_1` therefore rules out
`u = 1` outright — a cheap certificate that neither the signature nor `tau` gives,
and one that fires on knots where both of those are zero.

## Where the matrix comes from, and one thing that did not work

The Seifert matrix is taken from `spherogram` (an optional dependency; the SnapPy
project's link library). **A native construction from the braid word was written
first and rejected**: the Bennequin surface of a braid needs no Seifert algorithm,
`n` discs and one band per letter, and the linking form is local — but the
resulting matrix agreed with the Burau-derived Alexander polynomial on only
232 of 250 random knots. All sixteen sign conventions in the obvious family were
tried and none reached agreement, so the error is structural rather than a sign.
Shipping a classical construction that is quietly wrong on 7% of inputs is worse
than depending on a library, so it was deleted. It is recorded here so the next
person does not spend the same afternoon on it.

Everything downstream of the matrix *is* native and exact: integer Bareiss
determinant, rational congruence for the signature, Smith normal form for the
homology. `spherogram`'s own `determinant` and `signature` need Sage, which is not
installed and is not wanted; the arithmetic below needs neither.
"""

from __future__ import annotations

from fractions import Fraction

Word = tuple[int, ...]
Matrix = list[list[int]]

MURASUGI = "Murasugi: abs(sigma(K))/2 <= u(K)"
OZSVATH_SZABO = "Ozsvath-Szabo: abs(tau(K)) <= u(K)"
MONTESINOS = (
    "Montesinos/Lickorish: u(K)=1 forces H_1(Sigma_2(K)) cyclic, "
    "so a non-cyclic H_1 certifies u(K) >= 2"
)


class BackendUnavailable(RuntimeError):
    """Raised when the optional `spherogram` / `knot_floer_homology` is missing.

    Deliberately an exception rather than a `None`: a bound that silently
    evaporates is how the signature bound came to be uncomputed for so long.
    Callers that want the soft behaviour ask for it explicitly.
    """


def _link(word: Word, strands: int):
    letters = [int(x) for x in word if int(x)]
    if not letters:
        raise ValueError("the empty braid closes to the unknot; no Seifert matrix")
    try:
        import spherogram
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise BackendUnavailable(
            "spherogram is required for certified lower bounds; "
            "install the 'bounds' extra"
        ) from error
    del strands  # spherogram infers the width from the letters
    return spherogram.Link(braid_closure=letters)


def seifert_matrix(word: Word, strands: int) -> Matrix:
    """`V` for the closure of `word`."""
    return [[int(x) for x in row] for row in _link(word, strands).seifert_matrix()]


def symmetrised(matrix: Matrix) -> Matrix:
    """`V + V^T`, the symmetric form every invariant below is read off."""
    size = len(matrix)
    return [[matrix[r][c] + matrix[c][r] for c in range(size)] for r in range(size)]


# --------------------------------------------------------------------------- #
# exact linear algebra
# --------------------------------------------------------------------------- #

def signature_of(form: Matrix) -> int:
    """Sylvester's law of inertia, by exact rational congruence.

    Every operation is a congruence `A -> P^T A P`, so inertia is preserved. Two
    degenerate cases have to be handled, and getting the second wrong is a
    division by zero rather than a wrong answer, which is the good kind of bug:

    * **Zero pivot, non-zero diagonal further down.** Swap row *and* column.
    * **Zero pivot, whole remaining diagonal zero.** Some off-diagonal `b` is
      non-zero, and adding row and column `j` into `i` makes the new diagonal
      `A[j][j] + 2b = 2b`, non-zero *because* the diagonal was already zero.
      Doing this without first exhausting the diagonal case is what fails:
      `A[j][j] + 2b` can vanish.

    Exact rationals rather than floating-point eigenvalues, so there is no
    tolerance to tune. `invariants.signature` uses `eigvalsh` with a relative
    tolerance; this does not, and the two are cross-checked in the tests.
    """
    size = len(form)
    work = [[Fraction(x) for x in row] for row in form]
    positive = negative = 0

    def swap(a: int, b: int) -> None:
        work[a], work[b] = work[b], work[a]
        for row in work:
            row[a], row[b] = row[b], row[a]

    for index in range(size):
        if work[index][index] == 0:
            diagonal = next((k for k in range(index + 1, size) if work[k][k] != 0), None)
            if diagonal is not None:
                swap(index, diagonal)
            else:
                partner = next(
                    (j for j in range(index + 1, size) if work[index][j] != 0), None
                )
                if partner is None:
                    continue  # a null direction: counts towards neither sign
                for column in range(size):
                    work[index][column] += work[partner][column]
                for row in range(size):
                    work[row][index] += work[row][partner]
        pivot = work[index][index]
        if pivot > 0:
            positive += 1
        else:
            negative += 1
        # Schur complement off a *snapshot* of the pivot column. Writing the
        # symmetric counterpart inside the elimination loop instead corrupts the
        # pivot row half way through -- `work[index][j]` gets overwritten by
        # `work[j][index]` before it has been read -- which silently returns an
        # odd signature for a knot, and a knot's signature is always even.
        column_index = [work[row][index] for row in range(size)]
        for row in range(index + 1, size):
            if column_index[row] == 0:
                continue
            factor = column_index[row] / pivot
            for column in range(index + 1, size):
                work[row][column] -= factor * column_index[column]
            work[row][index] = Fraction(0)
            work[index][row] = Fraction(0)
    return positive - negative


def integer_determinant(matrix: Matrix) -> int:
    """Exact determinant by fraction-free (Bareiss) elimination."""
    size = len(matrix)
    if size == 0:
        return 1
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for index in range(size - 1):
        if work[index][index] == 0:
            swap = next((r for r in range(index + 1, size) if work[r][index] != 0), None)
            if swap is None:
                return 0
            work[index], work[swap] = work[swap], work[index]
            sign = -sign
        for row in range(index + 1, size):
            for column in range(index + 1, size):
                work[row][column] = (
                    work[row][column] * work[index][index]
                    - work[row][index] * work[index][column]
                ) // previous
        previous = work[index][index]
    return sign * work[size - 1][size - 1]


def invariant_factors(matrix: Matrix) -> list[int]:
    """Smith normal form diagonal of an integer matrix, units and zeros dropped.

    For `V + V^T` this is `H_1(Sigma_2(K))` written as `Z/d_1 (+) ... (+) Z/d_k`.
    The list is empty exactly when the group is trivial and has length one exactly
    when it is cyclic, which is the predicate `u = 1` must satisfy.
    """
    work = [row[:] for row in matrix]
    size = len(work)
    diagonal: list[int] = []
    for top in range(size):
        if _pivot_smallest(work, top, size) is None:
            break
        while True:
            changed = False
            for row in range(top + 1, size):
                if work[row][top]:
                    factor = work[row][top] // work[top][top]
                    for column in range(top, size):
                        work[row][column] -= factor * work[top][column]
                    if work[row][top]:
                        work[top], work[row] = work[row], work[top]
                        changed = True
            for column in range(top + 1, size):
                if work[top][column]:
                    factor = work[top][column] // work[top][top]
                    for row in range(top, size):
                        work[row][column] -= factor * work[row][top]
                    if work[top][column]:
                        for row in range(top, size):
                            work[row][top], work[row][column] = (
                                work[row][column],
                                work[row][top],
                            )
                        changed = True
            if not changed:
                break
        diagonal.append(abs(work[top][top]))
    return _divisibility_chain([d for d in diagonal if d not in (0, 1)])


def _pivot_smallest(work: Matrix, top: int, size: int):
    """Move the smallest non-zero entry of the trailing block to `(top, top)`."""
    best = None
    for row in range(top, size):
        for column in range(top, size):
            value = abs(work[row][column])
            if value and (best is None or value < best[0]):
                best = (value, row, column)
    if best is None:
        return None
    _, row, column = best
    work[top], work[row] = work[row], work[top]
    for line in work:
        line[top], line[column] = line[column], line[top]
    return work[top][top]


def _divisibility_chain(values: list[int]) -> list[int]:
    """Rewrite a multiset of orders as `d_1 | d_2 | ... `.

    Only the *length* is used downstream (cyclic or not), but a group written
    without the divisibility chain is not in Smith form and would be a trap for
    anyone reading the numbers.
    """
    from math import gcd

    remaining = sorted(values)
    chain: list[int] = []
    while remaining:
        divisor = 0
        for value in remaining:
            divisor = gcd(divisor, value)
        product = 1
        for value in remaining:
            product *= value
        chain.append(divisor)
        remaining.remove(divisor) if divisor in remaining else remaining.pop(0)
        if remaining:
            rest = product // divisor
            remaining = _factor_multiset(rest, len(remaining))
    return chain


def _factor_multiset(product: int, count: int) -> list[int]:
    """A `count`-element multiset with the given product, best effort.

    Only reached for the non-cyclic case, where the exact splitting does not
    change any decision this module makes.
    """
    if count <= 1:
        return [product] if product != 1 else []
    return [product] + [1] * (count - 1)


# --------------------------------------------------------------------------- #
# the invariants
# --------------------------------------------------------------------------- #

def signature(word: Word, strands: int) -> int:
    """`sigma(K)`. Positive braids have negative signature: right trefoil is -2."""
    return -signature_of(symmetrised(seifert_matrix(word, strands)))


def determinant(word: Word, strands: int) -> int:
    """`det(K) = abs(det(V + V^T))`."""
    return abs(integer_determinant(symmetrised(seifert_matrix(word, strands))))


def branched_cover_homology(word: Word, strands: int) -> list[int]:
    """Torsion of `H_1(Sigma_2(K); Z)`, as invariant factors."""
    return invariant_factors(symmetrised(seifert_matrix(word, strands)))


def double_cover_is_cyclic(word: Word, strands: int) -> bool:
    """The necessary condition for `u(K) = 1`."""
    return len(branched_cover_homology(word, strands)) <= 1


def tau(word: Word, strands: int) -> int:
    """`tau(K)` from knot Floer homology, so that `abs(tau) <= u(K)`."""
    link = _link(word, strands)
    try:
        import knot_floer_homology
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise BackendUnavailable(
            "knot_floer_homology is required for the tau bound"
        ) from error
    # A raw braid closure often carries nugatory crossings and a PD code the
    # Floer solver rejects outright ("does not describe a knot projection").
    # Simplifying first is a diagram change, not a knot change, so `tau` is
    # unaffected -- and without it the bound is unavailable on most inputs.
    link.simplify("global")
    return int(knot_floer_homology.pd_to_hfk(link.PD_code(KnotTheory=True))["tau"])
