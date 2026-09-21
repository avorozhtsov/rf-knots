import importlib.util
import itertools
from pathlib import Path
import random

import pytest

from rf_knots import reference
from rf_knots.invariants import determinant as polynomial_determinant


@pytest.fixture
def checker():
    path = Path(__file__).parents[1] / "scripts/burau_integer_check.py"
    spec = importlib.util.spec_from_file_location("burau_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_elimination_with_pivoting_and_singular_matrices(checker):
    rng = random.Random(8127)
    for n in range(5):
        for _ in range(12):
            a = [[rng.randrange(-3, 4) for _ in range(n)] for _ in range(n)]
            expected = 0
            for perm in itertools.permutations(range(n)):
                term = (-1) ** sum(perm[i] > perm[j] for i in range(n) for j in range(i + 1, n))
                for i in range(n):
                    term *= a[i][perm[i]]
                expected += term
            assert checker.integer_det(a) == expected


@pytest.mark.parametrize("word,n,expected", [((), 1, 1), ((1, 1, 1), 2, 3),
    ((-1, -1, -1), 2, 3), ((1, -2, 1, -2), 3, 5), ((1, 1, 1, 2, 3), 4, 3),
    ((1, 2, 3, 4, 5), 6, 1)])
def test_known_knots_and_even_strand_stabilizations(checker, word, n, expected):
    assert checker.determinant(word, n) == expected == polynomial_determinant(word, n)


def test_matches_existing_polynomial_algorithm_on_random_small_knots(checker):
    rng = random.Random(8128)
    checked = 0
    for n in range(2, 6):
        for _ in range(24):
            word = tuple(rng.choice((-1, 1)) * rng.randrange(1, n) for _ in range(7))
            if reference.num_components(word, n) != 1:
                continue
            assert checker.determinant(word, n) == polynomial_determinant(word, n)
            checked += 1
    assert checked >= 20


@pytest.mark.parametrize("word,n", [((0,), 2), ((2,), 2), ((1,), 1), ((), 0)])
def test_invalid_braid_rejected(checker, word, n):
    with pytest.raises(ValueError):
        checker.determinant(word, n)
