r"""Labelled probes for comparing braid representations.

The question is not "which network is bigger" but "which *encoding* lets a
parameter-matched network compute the things this project actually needs". So
every probe is a quantity the environment or the agent already depends on, and
every label is exact rather than learned:

| probe | label | what it stresses |
|---|---|---|
| `destab` | is `+-sigma_{n-1}` used exactly once | a **global count** at the strand boundary |
| `isknot` | does the closure have one component | **tracing strands** through the whole diagram |
| `determinant` | `log(1 + abs(Delta(-1)))` | a genuine **knot invariant** |
| `distance` | exact BFS depth to the empty 1-braid | the **value function**, verbatim |

`isknot` replaced an earlier "how many components" probe, which was unusable: the
count is bounded by the strand number, so the narrow and wide splits do not even
share a label set and transfer would have measured label shift rather than
representation. "Is the underlying permutation an `n`-cycle" is the same strand
tracing with a target that means the same thing at every width -- and it is the
predicate the environment itself checks on every step.

Both binary probes are rejection-sampled to a 50/50 base rate in **every** split,
so accuracy is comparable across splits without a prevalence correction.

`destab` deserves a note. `DESTABILIZE` is legal when the top generator occurs
exactly once *and* sits at the end of the word, but the end of the word is not
intrinsic -- `ROTATE` moves it there for free. The rotation-invariant predicate
"occurs exactly once" is therefore the one an agent actually needs, and it is the
one used here. The environment currently hands this answer to the network as a
precomputed `top_generator` plane precisely because a convolution over one-hot
letters could not compute it; that plane is **withheld from every arm here**, so
the probe measures whether a representation can earn it back.

Splits are by strand count: train narrow, test both narrow (in distribution) and
wide (transfer). The transfer split is the whole point -- a one-hot letter
alphabet has a channel per generator, so a checkpoint trained on four strands has
simply never activated the channels an eight-strand braid uses.
"""

from __future__ import annotations

import numpy as np

from rf_knots.actions import ActionSpec
from rf_knots.invariants import determinant
from rf_knots.reference import bfs_unknot, num_components, successors

Word = tuple[int, ...]
Sample = tuple[Word, int, float]  # word, strands, label

NARROW = (2, 3, 4)
WIDE = (6, 7, 8)


def _random_word(rng: np.random.Generator, strands: int, length: int,
                 top_generator: int | None = None) -> Word:
    """A uniform word, optionally restricted to generators below `top_generator`."""
    cap = top_generator if top_generator is not None else strands
    if cap < 2:
        return ()
    return tuple(
        int(rng.choice((-1, 1))) * int(rng.integers(1, cap)) for _ in range(length)
    )


# --------------------------------------------------------------------------- #
# destab: is the top generator used exactly once?
# --------------------------------------------------------------------------- #

def destab_sample(rng: np.random.Generator, strands: int, max_len: int) -> Sample:
    """Deliberately hard negatives: the easy ones make this probe meaningless.

    A uniform random word almost never uses the top generator exactly once, so a
    lazily built dataset is separable by "does the top generator appear at all".
    Half the words here are built to have exactly one, and the negatives are drawn
    from the two families that are one step away: no occurrence at all, and
    several occurrences.
    """
    length = int(rng.integers(2, max_len + 1))
    kind = rng.integers(0, 3)
    if kind == 0:  # exactly one, at a random position
        word = list(_random_word(rng, strands, length - 1, top_generator=strands - 1))
        word.insert(int(rng.integers(0, len(word) + 1)),
                    int(rng.choice((-1, 1))) * (strands - 1))
        word = tuple(word)
    elif kind == 1:  # none at all
        word = _random_word(rng, strands, length, top_generator=strands - 1)
    else:  # uniform, which usually means several
        word = _random_word(rng, strands, length)
    top = sum(1 for letter in word if abs(letter) == strands - 1)
    return word, strands, float(top == 1)


# --------------------------------------------------------------------------- #
# isknot: is the closure a single circle?
# --------------------------------------------------------------------------- #

def isknot_sample(rng: np.random.Generator, strands: int, max_len: int) -> Sample:
    length = int(rng.integers(1, max_len + 1))
    word = _random_word(rng, strands, length)
    return word, strands, float(num_components(word, strands) == 1)


# --------------------------------------------------------------------------- #
# determinant: a real knot invariant, on genuine knots only
# --------------------------------------------------------------------------- #

def determinant_sample(rng: np.random.Generator, strands: int, max_len: int) -> Sample | None:
    """`None` when the closure is a link -- the determinant of a link is a
    different animal, and mixing the two would measure component counting again."""
    length = int(rng.integers(2, max_len + 1))
    word = _random_word(rng, strands, length)
    if num_components(word, strands) != 1:
        return None
    value = determinant(word, strands)
    return word, strands, float(np.log1p(abs(value)))


# --------------------------------------------------------------------------- #
# distance: exact optimal solution depth, which is the value head's target
# --------------------------------------------------------------------------- #

def _descending(strands: int) -> Word:
    """`sigma_1 ... sigma_{n-1}`: closes to the unknot at any strand count.

    Needed because a random walk from the empty 1-braid reaches `n` strands only
    by stabilising `n-1` times, so a short walk never produces a wide instance.
    """
    return tuple(range(1, strands))


def distance_sample(
    rng: np.random.Generator, spec: ActionSpec, strands: int, walk: int, max_depth: int,
    max_word: int,
) -> Sample | None:
    """Random-walk away from a known unknot, then solve it exactly.

    `None` when the walk lands outside the oracle's reach; those instances are
    dropped rather than guessed, so every label is a theorem about that word.

    Insertions are subsampled before the uniform draw. There are `2(n-1)L` of them
    against a handful of everything else, so an unweighted walk is a walk that only
    ever inserts, and the resulting words are long, trivially reducible, and all
    alike.
    """
    word: Word = _descending(strands)
    n = strands
    for _ in range(walk):
        moves = [m for m in successors(spec, word, n) if len(m[1]) <= max_word]
        lengthening = [m for m in moves if len(m[1]) > len(word)]
        other = [m for m in moves if len(m[1]) <= len(word)]
        pool = other + [
            lengthening[i]
            for i in rng.choice(len(lengthening), size=min(4, len(lengthening)),
                                replace=False)
        ] if lengthening else other
        if not pool:
            break
        _, word, n = pool[int(rng.integers(0, len(pool)))]
    path = bfs_unknot(spec, word, n, max_depth=max_depth, max_nodes=60_000, max_growth=1)
    if path is None:
        return None
    return word, n, float(len(path))


# --------------------------------------------------------------------------- #
# building a split
# --------------------------------------------------------------------------- #

def build(
    probe: str,
    *,
    strand_counts,
    count: int,
    seed: int,
    max_len: int = 20,
    max_strands: int = 8,
) -> list[Sample]:
    rng = np.random.default_rng(seed)
    spec = ActionSpec(max_len + 8, max_strands)
    kind = PROBE_KIND[probe][0]
    # Binary probes are filled to a 50/50 base rate by rejection, so accuracy on
    # the narrow and the wide split answer the same question.
    half = count // 2
    quota = {0.0: count - half, 1.0: half} if kind == "binary" else None
    samples: list[Sample] = []
    attempts = 0
    budget = count * 200
    while len(samples) < count and attempts < budget:
        attempts += 1
        strands = int(rng.choice(strand_counts))
        if probe == "destab":
            got = destab_sample(rng, strands, max_len)
        elif probe == "isknot":
            got = isknot_sample(rng, strands, max_len)
        elif probe == "determinant":
            got = determinant_sample(rng, strands, max_len)
        elif probe == "distance":
            walk = int(rng.integers(1, 6))
            got = distance_sample(
                rng, spec, strands, walk, max_depth=7, max_word=max_len
            )
        else:
            raise ValueError(f"unknown probe {probe!r}")
        if got is None:
            continue
        if quota is not None:
            if quota[got[2]] == 0:
                continue
            quota[got[2]] -= 1
        samples.append(got)
    if len(samples) < count:
        raise RuntimeError(f"{probe}: only {len(samples)}/{count} after {attempts} tries")
    return [samples[index] for index in rng.permutation(len(samples))]


#: Whether the probe is a regression or a classification, and how many classes.
PROBE_KIND = {
    "destab": ("binary", 1),
    "isknot": ("binary", 1),
    "determinant": ("regress", 1),
    "distance": ("class", 8),
}
