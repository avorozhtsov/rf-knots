r"""The braid diagram as a picture: a `position x strand` raster of route bits.

Companion to [docs/representation.md](../../docs/representation.md), which
describes the *word* encoding this module is an alternative to.

## What this is

A braid word is a sequence of letters over an alphabet that grows with the
strand count: `sigma_1 ... sigma_{n-1}` and their inverses, so `2(n-1)` symbols.
Every network in this project one-hot encodes that alphabet, which means the
input layer's width is a function of `max_strands` and a trained checkpoint
cannot be evaluated on a wider braid at all. Raising the strand capacity is
currently a *new network*, not a bigger input.

The raster removes that. Draw the diagram instead of naming its letters: a grid
with one column per strand position and one row per unit of time, where each
cell says only what the strand sitting there does locally.

```
    strand 0   strand 1   strand 2
      |          |          |
row 0 |          \----------/          sigma_2 : strands 1 and 2 cross
      |          /----------\
row 1 \----------/          |          sigma_1^-1
      /----------\          |
```

Each cell carries three bits -- `(left, over, right)`:

| bits | cell |
|---|---|
| `000` | no strand occupies this position |
| `010` | the strand goes straight down |
| `011` | it exchanges with the strand on its **right**, passing **over** |
| `001` | it exchanges with the strand on its right, passing **under** |
| `110` | it exchanges with the strand on its **left**, passing over |
| `100` | it exchanges with the strand on its left, passing under |

A crossing therefore occupies two adjacent cells and is written twice, once from
each participant's point of view: `sigma_g` is `011` at column `g-1` and `100` at
column `g`; `sigma_g^-1` is `001` and `110`. That redundancy is the point --
it is what makes the crossing a *local* pattern that a 3x3 kernel can read
without knowing which generator index it is looking at.

**The alphabet is now three bits wide whatever `n` is.** The grid gets wider with
more strands, exactly as an image gets wider with more pixels, and the
convolution that reads it does not change. That is the whole argument for this
representation.

## Which symmetries are real

Two of the environment's moves are pure relabelling, and the raster can be made
blind to them by construction rather than by training:

* **Cyclic in position.** The closure joins the bottom of the diagram to its top,
  so the word is a necklace: `ROTATE_LEFT` and `ROTATE_RIGHT` (conjugation, a
  Markov move) change the array and not the knot. Circular padding along the
  position axis makes a convolution exactly invariant to them.

* **Translation along strands.** The braid relations are stated for *any* `g`:
  `sigma_g sigma_{g+1} sigma_g = sigma_{g+1} sigma_g sigma_{g+1}` holds at every
  height. Sharing one kernel across the strand axis is therefore correct, and is
  what one-hot letter channels cannot express -- they have to learn the relation
  separately at every generator index.

**A third symmetry is not real, and this is worth stating plainly because it is
the tempting one.** Gluing the left edge of the grid to the right edge -- making
the diagram a torus rather than a cylinder -- asserts that strand `0` and strand
`n-1` are adjacent. They are not. In `B_n` there is no generator joining them;
adding one gives the braid group of the *annulus* (the affine braid group
`\tilde{A}_{n-1}`), whose closure lives in a thickened torus and is a different
object. Worse, the environment's win condition is anchored to that boundary:
`DESTABILIZE` fires only on `+-(n-1)`, the *last* strand. A horizontally periodic
network cannot tell the last strand from any other.

So the honest shape is a **cylinder**: periodic in time, bounded in strands, with
the two boundaries marked (`edges=True`). `wrap_strands=True` is provided anyway,
because the claim above is an argument and the experiment in
`research/18-raster-representation.md` measures it.

## Packing, and what it buys

`sigma_1` and `sigma_3` commute, so writing them in either order -- or at the
same time -- gives the same braid. A word puts one letter per position and makes
the agent shuffle them with `COMMUTE`; a grid can put both in one row. Greedy
layering (`pack=True`) does exactly that, and it is the only place where the
raster is *shorter* than the word rather than wider:

```
sigma_1 sigma_3 sigma_2      ->      row 0: sigma_1  sigma_3
                                     row 1: sigma_2
```

Every layering step is a far commutation, so the braid group element is
unchanged; `tests/test_torus.py` checks that against the Artin representation
rather than trusting the argument. The effect is that far commutation stops
being a move the agent has to make and becomes a symmetry the representation has
already quotiented out.
"""

from __future__ import annotations

import numpy as np

Word = tuple[int, ...]

#: `(left, over, right)` route bits, plus the occupancy mask.
RASTER_CHANNELS = 4

#: Optional boundary markers appended by `edges=True`: first strand, last active
#: strand. Both are needed. The first is where `sigma_1` lives; the second is the
#: only place `DESTABILIZE` can fire, and zero padding alone cannot distinguish
#: "past the last strand" from "outside the array".
EDGE_CHANNELS = 2

_LEFT, _OVER, _RIGHT, _ACTIVE = 0, 1, 2, 3

# The four half-crossings, indexed by (is_positive, is_left_half).
_STRAIGHT = (0.0, 1.0, 0.0)
_RIGHT_OVER = (0.0, 1.0, 1.0)  # 011
_RIGHT_UNDER = (0.0, 0.0, 1.0)  # 001
_LEFT_OVER = (1.0, 1.0, 0.0)  # 110
_LEFT_UNDER = (1.0, 0.0, 0.0)  # 100


def pack_layers(word: Word, strands: int) -> tuple[tuple[int, ...], ...]:
    """Greedy far-commutation layering: letters that commute share a row.

    The bookkeeping is per *strand*, not per generator, and that is the whole
    subtlety: `sigma_g` physically occupies columns `g-1` and `g`, and two letters
    commute exactly when those two-column footprints are disjoint. Tracking
    generator indices instead makes `sigma_1` and `sigma_3` look like a conflict
    because both are "about" `sigma_2`, which is false -- they commute.

    A letter therefore drops to the first row where both of its columns are free.
    This is sound by construction -- every displacement it performs is a
    `COMMUTE` -- and it is the canonical form for the far-commutation relation
    alone (not for the braid relation, which is not a commutation).
    """
    if strands < 1:
        raise ValueError(f"strands={strands} must be at least 1")
    # `depth[s]` is the first row in which strand column `s` is free again.
    depth = [0] * strands
    layers: list[list[int]] = []
    for raw in word:
        letter = int(raw)
        if letter == 0:
            continue
        generator = abs(letter)
        if not 1 <= generator <= strands - 1:
            raise ValueError(f"generator {generator} is invalid for {strands} strands")
        upper, lower = generator - 1, generator
        row = max(depth[upper], depth[lower])
        while len(layers) <= row:
            layers.append([])
        layers[row].append(letter)
        depth[upper] = depth[lower] = row + 1
    return tuple(tuple(sorted(layer, key=abs)) for layer in layers)


def packed_word(word: Word, strands: int) -> Word:
    """`word` re-ordered into layer order -- equal in `B_n`, never longer."""
    return tuple(letter for layer in pack_layers(word, strands) for letter in layer)


def raster(
    word: Word,
    strands: int,
    *,
    max_strands: int | None = None,
    rows: int | None = None,
    pack: bool = False,
    edges: bool = False,
    pad_mode: str = "identity",
) -> np.ndarray:
    """Draw `word` as a `(rows, max_strands, channels)` float array.

    `pad_mode` decides what an unused row looks like, and the choice is not
    cosmetic:

    * `"identity"` -- every strand goes straight down. Inserting such a row into
      a braid diagram changes nothing at all, so the padding is a *picture of the
      same knot on a taller canvas*. Every row of the array is then a legal braid
      picture.
    * `"zero"` -- the row is blank, matching how the word encoding pads with `0`.
      Cheaper to recognise, but it is not a diagram, and under circular padding
      the network wraps through a region that cannot occur in play.
    """
    if pad_mode not in ("identity", "zero"):
        raise ValueError(f"pad_mode={pad_mode!r} must be 'identity' or 'zero'")
    width = max_strands if max_strands is not None else strands
    if not 1 <= strands <= width:
        raise ValueError(f"strands={strands} outside 1..{width}")

    layers = pack_layers(word, strands) if pack else tuple(
        (int(letter),) for letter in word if int(letter) != 0
    )
    height = rows if rows is not None else len(layers)
    if len(layers) > height:
        raise ValueError(f"{len(layers)} layers do not fit in {height} rows")

    planes = np.zeros((height, width, RASTER_CHANNELS), dtype=np.float32)
    live = slice(0, len(layers)) if pad_mode == "zero" else slice(0, height)
    planes[live, :strands, :3] = _STRAIGHT
    planes[live, :strands, _ACTIVE] = 1.0

    for row, layer in enumerate(layers):
        for raw in layer:
            letter = int(raw)
            generator = abs(letter)
            if not 1 <= generator <= strands - 1:
                raise ValueError(f"generator {generator} is invalid for {strands} strands")
            upper, lower = generator - 1, generator
            if not np.array_equal(planes[row, upper, :3], _STRAIGHT) or not np.array_equal(
                planes[row, lower, :3], _STRAIGHT
            ):
                raise ValueError("two crossings in one row share a strand")
            if letter > 0:
                planes[row, upper, :3] = _RIGHT_OVER
                planes[row, lower, :3] = _LEFT_UNDER
            else:
                planes[row, upper, :3] = _RIGHT_UNDER
                planes[row, lower, :3] = _LEFT_OVER

    if not edges:
        return planes
    markers = np.zeros((height, width, EDGE_CHANNELS), dtype=np.float32)
    markers[live, 0, 0] = 1.0
    markers[live, strands - 1, 1] = 1.0
    return np.concatenate([planes, markers], axis=2)


def word_from_raster(planes: np.ndarray, strands: int) -> Word:
    """Exact inverse of `raster`, row by row. Raises on anything unphysical.

    Used by the round-trip test and as a validator: a representation that cannot
    be decoded is not lossless, whatever a training curve says.
    """
    if planes.ndim != 3 or planes.shape[2] < RASTER_CHANNELS:
        raise ValueError("planes must be (rows, strands, >=4)")
    letters: list[int] = []
    for row in planes:
        active = row[:, _ACTIVE] > 0.5
        expected = np.arange(row.shape[0]) < strands
        blank = not active.any()
        if not blank and not np.array_equal(active, expected):
            raise ValueError("the active mask is not a contiguous strand prefix")
        if blank:
            continue
        crossing = [
            column
            for column in range(strands)
            if not np.array_equal(row[column, :3], _STRAIGHT)
        ]
        if not crossing:
            continue
        if len(crossing) % 2:
            raise ValueError("a crossing half has no partner")
        for upper, lower in zip(crossing[::2], crossing[1::2], strict=True):
            if lower != upper + 1:
                raise ValueError("crossing halves are not on adjacent strands")
            pair = (tuple(row[upper, :3]), tuple(row[lower, :3]))
            if pair == (_RIGHT_OVER, _LEFT_UNDER):
                sign = 1
            elif pair == (_RIGHT_UNDER, _LEFT_OVER):
                sign = -1
            else:
                raise ValueError("crossing halves disagree on direction or sign")
            letters.append(sign * (upper + 1))
    return tuple(letters)


def channels(*, edges: bool = False) -> int:
    """Input width of the raster, which does **not** depend on the strand count."""
    return RASTER_CHANNELS + (EDGE_CHANNELS if edges else 0)
