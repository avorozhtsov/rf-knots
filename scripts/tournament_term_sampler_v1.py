"""TRAIN-only bounded-work estimate of a cancelled tournament coefficient set.

The returned coefficients are unbiased for the full coefficient vector. A
nonlinear preference loss evaluated on the sample is not automatically an
unbiased estimate of the full loss; that requires a separate gradient gate.
"""

import random
from collections import defaultdict


def sample_coefficients(coefficients, *, terms, seed):
    if not coefficients or terms < 2 or terms % 2:
        raise ValueError("require coefficients and an even positive term budget")
    if len(coefficients) <= terms:
        return dict(coefficients)
    positive = [(key, value) for key, value in coefficients.items() if value > 0]
    negative = [(key, value) for key, value in coefficients.items() if value < 0]
    if not positive or not negative:
        raise ValueError("tournament signal lost one side of the comparison")
    rng = random.Random(seed)
    sampled = defaultdict(float)
    for sign, pool in ((1, positive), (-1, negative)):
        mass = sum(abs(value) for _, value in pool)
        keys = [key for key, _ in pool]
        weights = [abs(value) for _, value in pool]
        for key in rng.choices(keys, weights=weights, k=terms // 2):
            sampled[key] += sign * mass / (terms // 2)
    return {key: value for key, value in sampled.items() if abs(value) > 1e-12}
