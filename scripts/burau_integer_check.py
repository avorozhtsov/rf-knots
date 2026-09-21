"""Independent exact knot determinant from an unreduced Burau cofactor at t=-1.

No Seifert matrix, polynomial interpolation or floating-point arithmetic.
The Alexander cofactor identity is Lemma2.10 in
https://people.tamu.edu/~rowell/DRWv10.pdf (section2.4.4).
At t=-1 its Laurent-unit ambiguity has absolute value1. Unlike dividing a
reduced determinant by 1+t+...+t^(n-1), the cofactor works for even n as well.
This small independent checker allocates O(n^2), so it is for the bounded
training audit, not the size-independent neural inference path.
"""


def integer_det(matrix):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("require square matrix")
    if not n:
        return 1
    a = [list(row) for row in matrix]
    sign, previous = 1, 1
    for k in range(n - 1):
        if not a[k][k]:
            pivot = next((i for i in range(k + 1, n) if a[i][k]), None)
            if pivot is None:
                return 0
            a[k], a[pivot] = a[pivot], a[k]
            sign *= -1
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                value, rem = divmod(a[i][j] * a[k][k] - a[i][k] * a[k][j], previous)
                if rem:
                    raise ValueError("nonexact fraction-free elimination")
                a[i][j] = value
            a[i][k] = 0
        previous = a[k][k]
    return sign * a[-1][-1]


def determinant(word, strands):
    if strands < 1 or any(not isinstance(g, int) or g == 0 or abs(g) >= strands for g in word):
        raise ValueError("invalid Artin braid")
    braid = [[int(i == j) for j in range(strands)] for i in range(strands)]
    for generator in word:
        k = abs(generator) - 1
        for row in braid:
            a, b = row[k], row[k + 1]
            # Right multiply by [[2,-1],[1,0]] or its inverse [[0,1],[-1,2]].
            row[k], row[k + 1] = (2 * a + b, -a) if generator > 0 else (-b, a + 2 * b)
    minor = [[int(i == j) - braid[i][j] for j in range(1, strands)] for i in range(1, strands)]
    return abs(integer_det(minor))
