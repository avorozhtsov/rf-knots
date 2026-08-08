# 18 — The braid diagram as a picture, and how far the torus goes

> **Companion code.** The representation is `rf_knots.torus`, tested in
> `tests/test_torus.py`. The experiments are `research/experiments/`, and their
> outputs are under `artifacts/` (untracked).
>
> **Prior art inside this repo.** An earlier session left an uncommitted
> implementation of the same three route bits in `pgx-mcts-bench`
> (`serial_raster`, `braid_raster_planes`, `CylinderResidualBlock`,
> `conv-window-128`), never run and with no artifacts. Everything here is an
> independent implementation; the two agree bit-for-bit on 500 random words,
> which is the useful thing to do with a duplicate.

---

## 1. The proposal, stated precisely

Draw the braid instead of naming its letters. Put the diagram on a grid with one
column per strand position and one row per unit of time. Each cell carries three
bits, `(left, over, right)`:

| bits | cell |
|---|---|
| `000` | no strand here |
| `010` | the strand goes straight down |
| `011` / `001` | it crosses with its **right** neighbour, over / under |
| `110` / `100` | it crosses with its **left** neighbour, over / under |

A crossing occupies two adjacent cells and is written twice, once from each
participant's side: `sigma_g` is `011` at column `g-1` and `100` at column `g`.
Then glue the grid into a torus — top to bottom, left to right — share
convolutional weights across blocks, and combine blocks hierarchically into
`2x2` and `4x4` groups.

The claim to test is that this makes a network **strand-count agnostic**: the
alphabet stays three bits wide however many strands there are, so a wider braid
is a wider picture rather than a new network.

## 2. What is right about it

### 2.1 The strand count really is the bottleneck, and it binds in two places

This is worth spelling out because the proposal is aimed at a problem the project
genuinely has — and because it is the missing half of a principle the project has
already adopted. [06 §0](06-network-growth.md) opens the growth operators with
"best option: make the net size-invariant, so you never have to grow it", and
then discusses only the **word-length** axis: a positional policy head makes the
network `L`-agnostic, and the curriculum changes the data distribution rather than
the architecture. That was done. The strand axis was left out, and it is the axis
that is still forcing a retrain.

The observation is `2(N-1) + 1 + 1 + 8` channels — 24 at `N = 8` — of which
`2(N-1)` are the one-hot letter planes. Raise `max_strands` and the input layer
changes shape, so a checkpoint does not transfer; growing the strand capacity is
a new network, exactly as the proposal says.

**The policy head has the same disease, and the proposal does not mention it.**
The serial controller's per-offset action block is `3 + 2(N-1) + 1` wide, because
`INSERT(p, g, s)` names an absolute generator index. In the parallel formulation
it is worse: at `L = 64, N = 8` the action space is 1158 actions of which **896
are insertions** ([docs/representation.md §7](../docs/representation.md)). Fixing
the input while leaving the head indexed by `g` buys a checkpoint that can *read*
a nine-strand braid and cannot *act* on it.

The raster fixes both, and the second fix is the one that matters more. In the
grid, "insert a cancelling pair of crossings between these two strands" is an
action at a **cell**, and the cell already says which strands. So the policy head
becomes a `1x1` convolution with a fixed handful of output channels — one per
move type — and its parameter count stops depending on `N` at all. Today's head
is already independent of `L` by the same trick (the action layout is block-major
precisely so a `1x1` convolution reproduces it); this extends the trick to the
other axis:

| `max_strands` | per-offset head (today) | cell-indexed head |
|---:|---:|---:|
| 5 | 1,548 | 645 |
| 8 | 2,322 | 645 |
| 16 | 4,386 | 645 |
| 32 | 8,514 | 645 |

*(trunk width 128; `3 + 2(N-1) + 1` output channels against a fixed five.)* The
absolute numbers are small next to a 100k-parameter trunk. The point is the
column that does not move: a checkpoint trained at five strands has a head with
the right *shape* for thirty-two, so growing the braid stops being a
reinitialisation. That is the whole scalability argument, and it is a good one.

**Measured on the real networks.** Building each candidate at `max_strands = 5`
and again at `8` and diffing the state dicts says exactly which tensors stop a
checkpoint from moving:

| | `s-window-128` | `conv-window-128` (raster) |
|---|---|---|
| observation channels, 5 → 8 | 18 → 24 | 38 → 56 |
| total parameters, 5 → 8 | 140,308 → 142,234 | 137,620 → **137,818** |
| tensors that change shape | **3** of 138 | **2** of 127 |
| which ones | `representation.net.0.weight` (18→24 input channels), `positional.weight`, `positional.bias` | `positional.weight`, `positional.bias` |

Three things this settles.

* **The raster trunk really is strand-agnostic.** Its observation *array* grows
  (38 → 56 channels, because the raster is `4 x max_strands` wide) while its
  *parameters* do not: the input convolution reads four channels whatever `N` is,
  and the metadata convolution reads the environment's eight broadcast scalars.
  The input layer drops out of the blocking set entirely.
* **The head does not, and it is now the only thing left.** Both survivors are the
  per-offset policy convolution, `12 → 18` output channels — exactly
  `3 + 2(N-1) + 1` at `N = 5` and `N = 8`. The entire parameter growth of the
  raster arm is those 198 numbers: `(18-12) x 32 + (18-12)`.
* **So §4.1 is not a nice-to-have.** Swapping the encoder takes a checkpoint from
  three blocking tensors to two, which is still "cannot be loaded". Making the
  action cell-indexed removes the last two, and only then does a five-strand
  checkpoint load into an eight-strand game.

### 2.2 Cyclic in time is conjugation, and it is exactly right

The closure joins the bottom of the diagram to its top. So the word is a
necklace, and `ROTATE_LEFT` / `ROTATE_RIGHT` — conjugation, one of the two Markov
moves — change the array without changing the knot. Circular padding along the
position axis makes a convolution *invariant* to them by construction instead of
having to learn the invariance.

Two independent confirmations that this is the right reading:

* The environment's exact oracle already works this way. `reference.successors`
  indexes with `(p + 1) % length` and lets `DESTABILIZE` delete the unique top
  generator **wherever it sits**, not only at the end. The breadth-first solver
  has been searching the necklace, not the string, all along.
* The serial adapter already gathers its window with wraparound
  (`serial_braid.py`), for the same reason.

So the vertical gluing is not a modelling assumption. It is the geometry the rest
of the codebase already assumes.

### 2.3 Sharing weights across strands is right too

The braid relations are stated for *every* generator index:

```
sigma_g sigma_{g+1} sigma_g = sigma_{g+1} sigma_g sigma_{g+1}      for all g
sigma_i sigma_j = sigma_j sigma_i                                  for |i - j| >= 2
```

One kernel, applied at every height, is the correct way to encode a law that
holds at every height. One-hot letter channels cannot say this: they have to
learn the braid relation separately at `(sigma_1, sigma_2)`, at `(sigma_2,
sigma_3)`, and so on, and they learn it only at the indices the training data
happened to exercise. Section 6 measures exactly this and it is where the raster
wins by the largest margin.

### 2.4 The infinite-tiling picture is free in one direction and has a price in the other

"The knot representation is an infinite plane tiled by blocks, and one block
repeats infinitely" is **unconditionally true down the time axis** — that is what
closing the braid means, and circular padding is precisely a convolution over that
infinite periodic repetition. It costs nothing and asserts nothing.

Across the strand axis the tiling is also available, but only by moving to a larger
group and paying for it. That is §3, and it turned out to be the most interesting
part of this note.

## 3. The torus: a valid object, a mismatched network, and a bug

An earlier draft of this note said flatly that gluing the strand axis is wrong.
That was too strong, and the correction is worth setting out carefully because it
separates three different claims that are easy to run together.

### 3.1 As a *representation*, the torus is valid and complete

Gluing strand `0` to strand `n-1` adds a generator `sigma_n` and gives the
**annular braid group** — the affine type-`A` Artin group, whose presentation is
the ordinary braid presentation *with the indices read modulo `n`*. Its closures
are links in a thickened torus `T^2 x I` rather than in `S^3`.

That does not make it useless for knots in `S^3`, for two reasons:

* `T^2 x I` embeds in `S^3` as a neighbourhood of the standard torus, so every
  annular closure *is* a link in `S^3` once that embedding is fixed;
* and completeness is inherited for free, because **an ordinary braid word is an
  affine braid word that never uses `sigma_n`**. Adding a generator adds diagrams;
  it removes none. Alexander's theorem still supplies a braid word for every link,
  so every knot is still reachable.

So the answer to "is a braid on a torus, monotone in one direction, a valid knot
representation?" is **yes**. The affine picture is a strictly larger presentation
of the same set of knots.

### 3.2 It also buys something real

The argument for it is better than "the picture looks nicer". In `B_n`, making
strand `0` cross strand `n-1` takes `n-1` far commutations to bring them together
and `n-1` to put things back. In the affine presentation it is **one move**. On
wide braids that is the difference between an `O(n)` detour and a single action —
exactly the kind of shortcut a search benefits from, and exactly the kind of thing
that is invisible when you only look at small `n`.

### 3.3 What is actually wrong, then

Two things, and neither is the mathematics.

**(a) A mismatch between the network and the environment.** Making the *network*
periodic in strands while the *environment* has no `sigma_n` asserts a symmetry the
state space does not have. The network would be forced to represent strand `0` and
strand `n-1` as neighbours in a world where no move ever relates them. Adding
`sigma_n` to the action set repairs this — the periodicity becomes honest — but
that is an environment change, not a padding change, and §3.4 lists what it costs.

**(b) The arm in §5 wraps at the wrong place, so its verdict is a verdict on a
bug.** `F.pad(mode="circular")` on a fixed eight-row canvas glues strand `0` to
strand `7` **always**, whatever `n` is. A braid on `n` strands lives on a circle of
circumference `n`, and `n` changes during play — stabilisation and destabilisation
are two of the moves. So on the narrow split, where `n` is 2 to 4, that arm glues
strand `0` to a row that is *inactive*. It is not the torus; it is a canvas edge.

`research/experiments/torus_probe.py` fixes this with a per-sample gather that
wraps at the live strand count, and §5.1 reports the corrected arm alongside the
flawed one. The gather's layout is unit-tested against "output row `i` sees
strands `i-1 mod n`, `i`, `i+1 mod n`" rather than eyeballed.

### 3.4 What an affine environment would have to answer first

If `sigma_n` is added as a real action, three things need designing, and none is
merely an implementation detail:

1. ~~**The win condition loses its anchor.**~~ **Withdrawn — this was wrong.** An
   earlier version of this note argued that `DESTABILIZE` fires on `+-sigma_{n-1}`,
   the *last* strand, that a circle has no last strand, and that a periodic network
   therefore cannot see the predicate deciding 54% of optimal moves.

   Destabilization never needed a distinguished last strand. It needs a **gap**: an
   adjacent pair on the circle that never crosses. If positions `0` and `k-1` are
   unentangled the seam is unused, the cyclic word *is* an ordinary linear word,
   and destabilization applies unchanged. More generally a gap at any `j` cuts the
   circle there, making `j` the first column and `j-1` the last. **That predicate
   is local and translation-invariant** — precisely the kind a periodic convolution
   reads well. The anchor is not lost; it is *discovered*.

   One refinement rather than a disagreement: the gap makes the circle
   **cuttable**. The move itself additionally needs that boundary strand to carry
   **exactly one** crossing — Markov's condition, which is the same "occurs exactly
   once" predicate used throughout this note, now stated without reference to a
   boundary. Gap alone gives the linearisation; gap plus single crossing gives the
   move.

   And the wrap gives *more* freedom here, not less. There can be several gaps,
   hence several valid linearisations, and a removable strand no longer has to be
   shuffled to position `n-1` before it can go — the same `O(n)`-to-`1` saving as
   §3.2, applied to the win condition instead of to a crossing.

   **This invalidates part of §5.3.** The `destab` probe labels
   "`+-sigma_{n-1}` occurs exactly once", which is a *linear* notion defined
   against a boundary the torus arm does not have. So `raster-torus-n` at 0.568 was
   scored on the wrong predicate, and `raster-torus-n-edge` at 0.977 partly just
   recovered the hardcoded boundary the label presupposes. **Those two numbers are
   not evidence against the wrap** and should not be quoted as such. The corrected
   probe labels "there is a gap, and the strand beside it has exactly one
   crossing", generated over the `B*` alphabet in `reference.py` — a different
   experiment, not a relabelling, because the generator has to emit cyclic words.
   Unresolved rather than measured, as of this writing.
2. **Soundness has to be re-proved for the new closure.** Every existing move must
   still preserve the knot type under the annular closure and its chosen embedding
   into `S^3`. Completeness is inherited (§3.1); soundness is not automatic.
3. **The branching factor goes up.** One more generator, at every position. Whether
   the `O(n)`-to-`1` shortcut of §3.2 pays for that is an empirical question and
   should be measured on wide braids, where the shortcut is worth the most — not on
   the two-to-four-strand instances the current ladder is made of.

Until those exist, the shape that matches the environment we have is a
**cylinder**: periodic in time, bounded in strands, with both boundaries marked.
Marking them costs two channels and recovers what a bounded axis would otherwise
hide — zero padding alone cannot distinguish "past the last strand" from "outside
the array".

## 4. Improvements worth making

Five, in descending order of how much they change.

**4.1 Make the action space cell-indexed too.** §2.1. This is the change that
turns "strand-agnostic input" into "strand-agnostic agent", and without it the
rest is half a fix.

**4.2 Pack far-commuting letters into one row.** `sigma_1` and `sigma_3` commute,
so they can share a row. Greedy layering (`rf_knots.torus.pack_layers`) does this,
and the bookkeeping has one subtlety worth recording: it must be per *strand*, not
per generator index. `sigma_g` occupies columns `g-1` and `g`, and two letters
commute exactly when those footprints are disjoint. Tracking generator indices
instead makes `sigma_1` and `sigma_3` look like a conflict because both are
"about" `sigma_2` — which is false, and was a real bug in the first version here.

The payoff is that **`COMMUTE` stops being a move and becomes a symmetry the
representation has already quotiented out**, and the canvas gets shorter rather
than taller. `tests/test_torus.py` checks the layering is a braid-group identity
against the Artin representation, which is faithful and therefore decides the
question rather than testing a proxy for it.

**4.3 Pad with identity rows, not blank ones.** An unused row can be drawn as
"every strand goes straight down". Inserting such a row into a braid diagram
changes nothing at all, so the padding is a picture of the *same knot on a taller
canvas*, and every row of the array remains a legal braid picture. Blank padding
introduces a symbol that cannot occur in play, and under circular padding the
network wraps through it. §5 measures the difference rather than assuming it.

**4.4 Call the hierarchical combiners what they are.** "Blocks, plus a vertical
interaction block, plus a horizontal interaction block, plus a `2x2` combiner,
plus a `4x4` combiner" describes a **feature pyramid**: pool to half resolution,
convolve, add back, twice. Two levels give a cell at the finest scale a view of
the `4x4` neighbourhood of blocks around it, which is the stated goal. Writing it
as a pyramid is a dozen lines instead of a new module hierarchy, and it inherits
the padding decisions above for free rather than needing them re-litigated at
each level.

**4.5 Do not confuse matched parameters with matched compute.** The raster's
canvas is `max_strands` times taller than the one-hot strip, so at equal
parameter count it does roughly eight times the arithmetic per step — measured,
not estimated: 484 ms/step against 119 ms/step at 100k parameters on one CPU
thread. Any comparison that omits this is measuring a bigger network. §5 carries
a `word-onehot-cyclic-8x` arm for exactly this reason.

## 5. The experiment

Ten arms, four probes, three seeds — 120 runs — parameter-matched to
`s-window-128`'s 102,439 parameters. Every arm is the same trunk: input
projection, four residual blocks, masked global pool, two-layer head. They differ
only in what they read and which edges of the picture they treat as circles.
Widths are solved per arm so the counts match to within a few percent; without
that step the comparison quietly becomes "which encoder got more parameters at a
fixed channel width". The one deliberate exception is
`word-onehot-cyclic-8x`, which is given eight times the budget because that is
what the raster costs in arithmetic (§4.5).

The ablation ladder is ordered so each arm adds exactly one claim to the one above
it: letters, then letters-as-a-necklace, then the picture, then the picture with
time cyclic, then identity padding, then the torus, then the cylinder with marked
boundaries, then the pyramid, then packing.

The probes are quantities the project already depends on, each labelled exactly:

| probe | label | what it stresses |
|---|---|---|
| `destab` | is `+-sigma_{n-1}` used exactly once | a **global count** at the strand boundary |
| `isknot` | does the closure have one component | **tracing strands** through the diagram |
| `determinant` | `log(1 + abs(Delta(-1)))` | a genuine **knot invariant** |
| `distance` | exact breadth-first depth to the empty 1-braid | the **value function**, verbatim |

`destab` deserves a note. `DESTABILIZE` needs the top generator to occur exactly
once *and* to sit at the end of the word, but the end of a necklace is not
intrinsic — `ROTATE` puts it there for free, and the reference solver already
treats it that way. The rotation-invariant predicate "occurs exactly once" is
therefore the one an agent needs, and it is the one scored here. **The
environment currently hands this answer to the network as a precomputed
`top_generator` plane** because a convolution over one-hot letters could not
compute it; that plane is withheld from every arm, so the probe measures whether a
representation can earn it back.

That matters beyond this note. The zero-knowledge audit in
[12 §4](12-serial-formulation.md) marks `top_generator`, `top_count` and
`top_count == 1` as **unfair** — human-computed answers to a question the network
was supposed to work out — covering the move that is 54% of optimal solutions. If
a representation computes the predicate itself, those three channels can be
dropped and the zero-knowledge standard stops being applied unevenly.

**One caveat, stated up front so the result is not over-read.** This probe trains
directly on the predicate. An agent gets game outcomes, not labels, so a high
score here shows the encoder can *express and learn* the predicate at this
parameter budget — a necessary condition for dropping the channel, not a
sufficient one. The sufficient test is the ablation the audit asks for, run with
whichever encoder wins below.

Each probe is scored on two splits: **in**, held out from the two-to-four-strand
range trained on, and **wide**, six to eight strands, never seen. The wide split
is the whole claim.

### 5.1 The floor: what two scalars already answer

Before reading any arm, the probes have to be checked against the cheapest
possible predictor. **Word length and strand count are already broadcast into
every observation**, so any arm gets them for free, and a lookup table on that pair
alone is the floor a representation has to clear to have contributed anything
(`research/experiments/trivial_baseline.py`, fitted on the same training split,
scored on the same instances):

| probe | in | wide |
|---|---:|---:|
| `destab` | 0.702 ± 0.023 | 0.500 ± 0.000 |
| `isknot` | **0.891 ± 0.013** | 0.500 ± 0.000 |
| `determinant` | 0.123 ± 0.085 | −0.014 ± 0.010 |
| `distance` | **0.953 ± 0.005** | — |

Two of the four probes are largely decided by those two numbers, and it is worth
saying why rather than just discounting them:

* **`isknot` is a parity question in disguise.** The closure's permutation is a
  product of `L` transpositions on `k` letters, and a `k`-cycle is even exactly
  when `k` is odd, so being a knot forces `L ≡ k - 1 (mod 2)`. That one rule is
  most of the label, which is why the floor sits at 0.891.
* **`distance` is nearly vacuous at the sizes the exact oracle can reach.** The
  optimal solution is mostly "destabilise `k-1` times, then undo the walk", so its
  length is close to a function of `k` and the word length. At 0.953 there is
  almost no headroom, and the probe should be read as a sanity check rather than
  as evidence.

So the honest reading of the tables below is: **`destab` is the probe that
discriminates**, `isknot` discriminates only on the wide split, and `distance` and
`determinant` are reported for completeness. Had this baseline not been computed,
`distance` at 0.98 would have looked like a result.

### 5.2 What the sweep found

Full tables in `artifacts/raster-probe-20260808/report.md`. The two columns that
carry the argument, against the floors from §5.1:

**`destab` — floor 0.702 in, 0.500 wide**

| arm | params | in | wide |
|---|---:|---:|---:|
| `word-onehot` | 99,165 | 0.989 ± 0.006 | **0.509 ± 0.009** |
| `word-onehot-cyclic` | 99,165 | 0.988 ± 0.004 | **0.502 ± 0.004** |
| `word-onehot-cyclic-8x` | 825,975 | 0.924 ± 0.078 | **0.522 ± 0.012** |
| `raster-flat` | 98,769 | 0.999 ± 0.001 | 0.872 ± 0.052 |
| `raster-cyclic` | 98,769 | 0.999 ± 0.001 | 0.862 ± 0.002 |
| `raster-cyclic-idpad` | 98,769 | 0.999 ± 0.001 | **0.928 ± 0.066** |
| `raster-torus` | 98,769 | 1.000 ± 0.000 | 0.756 ± 0.086 |
| `raster-cylinder-edge` | 98,841 | 1.000 ± 0.001 | 0.917 ± 0.056 |
| `raster-cylinder-pyramid` | 102,069 | 1.000 ± 0.000 | 0.905 ± 0.111 |
| `raster-cylinder-packed` | 102,069 | 1.000 ± 0.000 | 0.870 ± 0.109 |

**`isknot` — floor 0.891 in, 0.500 wide**

| arm | params | in | wide |
|---|---:|---:|---:|
| `word-onehot` | 99,165 | 0.770 ± 0.026 | 0.504 ± 0.042 |
| `word-onehot-cyclic` | 99,165 | 0.772 ± 0.015 | 0.510 ± 0.069 |
| `word-onehot-cyclic-8x` | 825,975 | 0.760 ± 0.043 | 0.440 ± 0.014 |
| `raster-flat` | 98,769 | 0.899 ± 0.031 | 0.607 ± 0.061 |
| `raster-cyclic` | 98,769 | 0.906 ± 0.031 | **0.665 ± 0.021** |
| `raster-cyclic-idpad` | 98,769 | 0.840 ± 0.041 | 0.553 ± 0.053 |
| `raster-torus` | 98,769 | 0.816 ± 0.022 | 0.585 ± 0.073 |
| `raster-cylinder-edge` | 98,841 | 0.858 ± 0.084 | 0.568 ± 0.055 |
| `raster-cylinder-pyramid` | 102,069 | 0.849 ± 0.078 | 0.505 ± 0.005 |
| `raster-cylinder-packed` | 102,069 | 0.845 ± 0.045 | 0.573 ± 0.040 |

Six readings, in descending order of how confident I am in them.

**1. Strand transfer is a representation problem, not a capacity problem.** On
`destab`, every one-hot arm sits at the 0.500 floor out of distribution while every
raster arm clears it by 0.26 to 0.43. The decisive row is
`word-onehot-cyclic-8x`: **eight times the parameters and eight times the
arithmetic buys 0.522** — thirteen thousandths above chance. This is the result the
whole note was written to test, and it is unambiguous. It also cannot be explained
by compute, which is why that arm exists.

**2. The one-hot encoder is beaten by two scalars on `isknot`.** All three one-hot
arms score 0.76–0.77 in distribution against a floor of 0.891. A network reading
`2(n-1)` letter channels does *worse* at "is this closure a knot" than a lookup
table on word length and strand count — because the parity rule lives in exactly
those two scalars and the one-hot arm has to rediscover it through fourteen
channels. Every raster arm at least reaches the floor.

**3. The elaborate parts of the proposal do not pay.** The hierarchical `2x2`/`4x4`
combiners and far-commutation packing are the two most distinctive ideas in the
original design, and neither survives contact:

* the pyramid is best-in-family on nothing, and on `isknot` wide it collapses to
  0.505 ± 0.005 — the floor, with no seed spread, i.e. it reliably learned nothing;
* packing is likewise unremarkable on the discriminating probes, and on
  `determinant` wide it produces `-3.913 ± 4.965`, the worst number in the sweep.

Both add receptive field or canonicalisation, and both appear to trade that for
overfitting at this budget. **The cheap symmetry — cyclic in position — is the one
that pays.**

**4. If one arm has to be picked, it is `raster-cyclic`.** Plain raster, position
axis circular, strand axis bounded, no edge channels, no pyramid, no packing. It is
best on `isknot` wide (0.665 ± 0.021, and the tightest spread there), it has the
tightest spread of any arm on `destab` wide (0.862 ± 0.002), and it is the simplest
thing in the family. The arms that beat it on `destab` — `idpad` at 0.928 ± 0.066,
`edge` at 0.917 ± 0.056 — do so inside their own seed spread and lose on `isknot`.
With three seeds those differences are not resolvable, and choosing the simplest
arm is the honest response to a tie.

**5. Identity padding and edge marking are not settled.** §4.3 argued for identity
padding from first principles and it is the best `destab` arm and among the worst
`isknot` arms. The argument was not wrong so much as too small to dominate. This
one needs the eight seeds `docs/lessons.md` asks for, and it does not have them.

**6. `determinant` is a null result, and worth stating loudly.** No arm reaches
`R^2 = 0.5` in distribution, and **every arm is negative out of distribution** —
several of them far worse than the constant predictor. The raster makes
combinatorial predicates about the *diagram* learnable and does nothing for a
genuine invariant of the *knot* at 100k parameters. That is a real limit on the
idea. It is also consistent with [10 §4](10-invariants-and-representations.md)'s
warning against expecting a 2-D inductive bias to deliver knot theory, and with the
DeepMind/Oxford choice to feed invariants as features rather than hope a network
recovers them from a picture.

One caveat on that null: the `determinant` splits are unbalanced by construction —
one-component closures are common at two strands and rare at eight, so the narrow
set is 2-strand-heavy (943/2000 at `k=2`) and the wide set 6-strand-heavy
(240/400 at `k=6`). Part of the negative transfer is distribution shift rather than
representation. The in-distribution column has no such excuse.

### 5.3 The torus, tested properly

The `raster-torus` arm above wraps at the eight-row canvas, not at the live strand
count (§3.3b), so its verdict is a verdict on a bug. `torus_probe.py` reruns it
against a per-sample gather that wraps at `n`, on the same cached instances.
Correcting it changes the answer — in both directions.

| arm | `destab` in | `destab` wide | `isknot` in | `isknot` wide |
|---|---:|---:|---:|---:|
| `raster-cylinder-edge` | 1.000 ± 0.001 | 0.917 ± 0.056 | 0.858 ± 0.084 | 0.568 ± 0.055 |
| `raster-torus-canvas` (the bug) | 1.000 ± 0.000 | 0.756 ± 0.086 | 0.816 ± 0.022 | 0.585 ± 0.073 |
| `raster-torus-n` | 0.985 ± 0.012 | 0.568 ± 0.017 | 0.655 ± 0.069 | **0.500 ± 0.000** |
| `raster-torus-n-edge` | 1.000 ± 0.000 | **0.977 ± 0.034** | 0.868 ± 0.053 | **0.500 ± 0.001** |

Three things, and the first two point opposite ways.

**⚠ The first reading below is withdrawn — see §3.4.1.** `raster-torus-n` falls to
0.568 on `destab` transfer, and this note originally read that as the wrap erasing
the anchor `DESTABILIZE` needs. That conclusion does not follow, because **the
probe's label is the linear predicate** ("`+-sigma_{n-1}` occurs exactly once"),
defined against a boundary the torus arm does not have. The correct
translation-invariant predicate is "there is a gap, and the strand beside it has
exactly one crossing". So this arm was scored on a target biased toward the
cylinder, and 0.568 measures that bias, not the representation. The row is kept
rather than deleted because the run happened; it is not evidence.

**Wrap plus explicit boundary channels is still the best arm measured** —
`raster-torus-n-edge` at **0.977 ± 0.034** on `destab` transfer, ahead of the
cylinder's 0.917 — but read it narrowly. Given the label is the linear predicate,
what those two channels supply is the boundary the *label* presupposes, so this
mostly says "a marked origin lets a periodic trunk answer a boundary-relative
question". It does not say the wrap needs a marked origin in general; §3.4.1 argues
it does not, because the gap is discoverable.

**Both torus arms are dead at the floor on `isknot` transfer** — 0.500 ± 0.000
and 0.500 ± 0.001, a reliable failure rather than a noisy one. The most plausible
reading, and it is a reading rather than a measurement: training happens at `n = 2`
to `4`, where **a single 3-tap kernel wraps around the entire circle**. A network
can then learn a rule that is global-by-accident for tiny circles and has no
analogue at `n = 6` to `8`. The cylinder has no such hazard because its kernel
never closes up.

If that reading is right it has a concrete consequence: **a wrapped representation
must be trained on mixed and wide strand counts**, or it will learn
circumference-specific rules. That is exactly what
`pgx-mcts-bench/strand_architecture_gate.py`'s early mixed-strand curriculum was
built for, and it is the natural next experiment rather than a reason to drop the
torus.

**Verdict on the torus.** Valid as a representation (§3.1) and worth something
real (§3.2). The `destab` numbers do not settle it either way — the probe asked a
linear question (§3.4.1). What is left standing against it is the `isknot`
transfer floor, which is label-independent and therefore still a genuine finding.
As an *environment* the two remaining design questions are soundness and branching
factor; the win-condition objection is withdrawn, and `research/17`'s
Birman--Ko--Lee seam generator answers the group-theoretic half by keeping the
wrap inside `B_n`.

### 5.4 The selection

Two rules fall out, and they are the useful output of the whole sweep:

> **Draw the diagram, wrap the position axis, and keep the trunk plain.** The
> picture is what buys strand transfer; cyclic position is the one extra symmetry
> that pays for itself; the pyramid and the packing do not.
>
> **Mark the strand boundary for heads anchored to it, and not for heads that are
> not.** Boundary channels are worth +0.41 on `destab` under a torus wrap (§5.3)
> and −0.05 on local rewrite legality (§6). Whether to mark is a property of the
> question, not of the representation.

Concretely, for the next thing built:

| use | arm |
|---|---|
| general trunk, if one arm must be chosen | `raster-cyclic` |
| a head that must find the last strand | add the two edge channels; a live-`n` wrap then beats the cylinder |
| a head reading local rewrite legality | raster, **no** edge channels |
| anything justified by the pyramid or by packing | not supported by this data |

## 6. Swapping `s-window-128`'s input

`s-window-128` is the serial controller of [12](12-serial-formulation.md): a head
that sees seven cells of the word and may act at any of them. This section keeps
the geometry, the positional readout and the parameter budget, and changes only
what fills those seven cells — the same seven letters, drawn as a `7 x k` picture
(`research/experiments/window_probe.py`).

**What is measured, and why it is the right thing.** The controller's job is to
know which rewrite is available where, and the three local rewrites are decided by
arithmetic on generator indices:

| move | legal when |
|---|---|
| `REDUCE(p)` | `word[p] = -word[p+1]` |
| `COMMUTE(p)` | `abs(abs(word[p]) - abs(word[p+1])) >= 2` |
| `BRAID(p)` | `abs(abs(word[p]) - abs(word[p+1])) = 1`, same sign, `word[p+2] = word[p]` |

In the one-hot encoding each is a relation over pairs of symbols from a
`2(n-1)`-letter alphabet — something to be memorised pair by pair, and memorised
only at the indices training happened to contain. In the raster they are
statements about whether two crossings *touch*: `COMMUTE` is "their two-column
footprints are disjoint", `BRAID` is "they share exactly one column". Local
geometry, and index-free.

| arm | params | in acc | in recall | wide acc | wide recall |
|---|---:|---:|---:|---:|---:|
| `window-onehot` | 105,263 | 1.000 ± 0.000 | 1.000 ± 0.001 | 0.775 ± 0.027 | **0.284 ± 0.043** |
| `window-onehot-8x` | 812,599 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.804 ± 0.011 | **0.265 ± 0.083** |
| `window-raster` | 107,923 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.947 ± 0.069 | 0.919 ± 0.080 |
| `window-raster-noedge` | 107,847 | 1.000 ± 0.000 | 1.000 ± 0.000 | **0.996 ± 0.005** | **0.981 ± 0.024** |

`recall` is scored on the legal slots only. Most windows are legal at nothing, so
a network that answers "illegal" everywhere already scores 0.77 on plain accuracy;
recall is the column that cannot be reached by refusing to predict.

**Per move, on the wide split** — this is where the mechanism shows:

| arm | `REDUCE` | `COMMUTE` | `BRAID` |
|---|---:|---:|---:|
| `window-onehot` | 0.846 | **0.539** | 0.972 |
| `window-onehot-8x` | 0.922 | **0.534** | 0.987 |
| `window-raster` | 0.972 | 0.885 | 0.993 |
| `window-raster-noedge` | **1.000** | **0.992** | 0.996 |

Four readings.

**1. In distribution the swap costs nothing.** Every arm is at 1.000 on both
accuracy and recall. Whatever the raster is worth, it is not bought by giving up
anything the current encoder already does.

**2. Out of distribution the swap is the whole difference.** Wide recall goes from
**0.284 to 0.981** — the one-hot controller recovers barely a quarter of the legal
moves on braids wider than it trained on, and the raster controller recovers
essentially all of them. And `window-onehot-8x` settles the alternative
explanation: eight times the parameters moves recall from 0.284 to 0.265, i.e. not
at all.

**3. `COMMUTE` is the mechanism, exactly as predicted.** Its legality is
`abs(abs(i) - abs(j)) >= 2` — pure arithmetic on generator indices — and the
one-hot arms sit at **0.539**, which on a two-class slot is a coin flip. They never
learned the rule; they learned the pairs. The raster reads it as two crossings not
sharing a column and gets 0.992. `BRAID` is high everywhere because it is rare
enough for accuracy to flatter it; `REDUCE` (`a = -b`, a same-index relation) sits
in between at 0.846, which is the pattern you would expect if the one-hot encoder
generalises within an index and not across indices.

**4. Marking the boundary *hurts* here, and that is not a contradiction.**
`window-raster-noedge` beats `window-raster` (0.996 against 0.947). Local rewrite
legality is a translation-invariant predicate — it does not care where on the
strand axis it happens — so boundary channels are a positional cue with nothing to
explain, and the network overfits to them. Put next to §5.3, where the boundary
channels were worth 0.41 on `destab`, this gives a usable rule:

> **Mark the strand boundary for the heads that are anchored to it, and not for
> the heads that are not.** `DESTABILIZE` lives at the boundary and needs it;
> `REDUCE`/`COMMUTE`/`BRAID` are the same everywhere and are hurt by it.

**What this does and does not license.** It licenses swapping the encoder: same
heads, same budget, no in-distribution cost, and a checkpoint that keeps working on
wider braids. It does not license a claim about ladder performance — that needs a
curriculum run, and `pgx-mcts-bench/strand_architecture_gate.py` is the harness for
it (see §8).

## 7. Has anyone done this

Partly, and the part nobody seems to have done is the part worth doing.

**The picture is not new.** Knot mosaics (Lomonaco–Kauffman) and rectangular
Dynnikov diagrams are established grid representations, and
[arXiv:2011.03498](https://arxiv.org/abs/2011.03498) classifies knots from
rectangular diagrams with a convolutional network. [10
§4](10-invariants-and-representations.md) already recommended grid diagrams over
mosaics for this project, for reasons that still stand.

**The braid-word encoding with one-hot letters is the standard baseline.**
*Learning to Unknot* ([arXiv:2010.16263](https://arxiv.org/abs/2010.16263)) and
*Untangling braids* ([arXiv:2206.05373](https://arxiv.org/abs/2206.05373)) both
use it, and both fix the strand count — the latter works at width 4. The
DeepMind/Oxford unknotting agent
([arXiv:2409.09032](https://arxiv.org/abs/2409.09032)) sidesteps diagrams almost
entirely and feeds the network *invariants* of the diagram and of its
one-crossing-change neighbours.

**What I did not find anywhere** is the combination this proposal is really
about: a network made **equivariant to conjugation** by treating the closed braid
as periodic in time, with **weights shared along the strand axis** so that a
single checkpoint applies at any strand count. The affine/annular braid group is
well known to topologists — its presentation is exactly the braid presentation
with indices modulo `n` — but as far as these searches go, nobody has tried to
give a knot network that symmetry.

So: the representation is old, the equivariances are the contribution, and the
strand-transfer results in §5 and §6 are the evidence that they are worth having.

## 7b. The curriculum gate: setup, and what it can and cannot show

Running, as of this writing:

```
pgx-mcts-bench strand-architecture-gate \
    artifacts/strand-gate-raster-vs-window-20260808 \
    --only s-window-128,conv-window-128 --seeds 71,72 --workers 4 \
    --simulations 128 --max-iterations 12 --selfplay-games 6 \
    --eval-games 8 --eval-every 2 --stage-limit 4 --promote-at 0.8
```

**Why `conv-window-128` is the raster arm.** It is the `"joint"` raster variant,
whose trunk is `CylinderResidualBlock` — circular in word position, zero-padded in
strands, no edge channels — with masked row pooling back to one feature per
column. That is `raster-cyclic` from §5 and `window-raster-noedge` from §6, which
is the arm those two experiments selected. Same seven-column controller, same
heads, same 128 simulations; only the trunk differs.

**Why four stages.** The curriculum is `unknot → T(2,3) → P(3,4)#0 → P(4,5)#0`,
and stage 3 is the first **four-strand** rung — the whole reason the mixed-strand
gate exists, since it introduces four strands before a model can specialise on the
two-strand prefix. Stages 4 and 5 add a two-strand rung and a second four-strand
one; cutting them costs about an hour of wall clock and no four-strand evidence.

**Cost, measured rather than guessed.** A calibration run on stage 0 alone took
509 s (`s-window-128`) and 629 s (`conv-window-128`) for four iterations — about
142 s per iteration per stage — and **neither arm promoted stage 0 at four
iterations**, at solve rates of 0.42 and 0.25 against a 0.80 bar. Hence twelve
iterations, which is what the repository's own six-stage gate uses.

### The ceiling this run cannot cross

`max_strands = 5` in the ladder config. So:

* the widest instance the environment can present is **five strands**, and
* going wider is not a data change but an architecture change — rebuilding at
  `max_strands = 8` moves the action space from 98 to 140 and reshapes the tensors
  in the §2.1 table.

**A trained checkpoint therefore cannot be loaded past five strands today, in
either architecture.** That is the strand dependence of §2.1 showing up as an
operational wall rather than a design opinion, and it bounds every RL result in
this section.

Two consequences follow, and both should be read before the numbers are.

1. **The gate itself should show parity.** It trains *and* evaluates inside 2–4
   strands, and §6 already found both encoders at 1.000 there. A raster win would
   be a bonus; parity is the prediction, and parity is not a negative result.
2. **The discriminating measurement is transfer, and it has one strand of
   headroom.** `research/experiments/strand_transfer_eval.py` takes the gate's
   checkpoint — trained through `P(4,5)#0`, four strands — and evaluates it with
   no further training on `P(5,6)#0`, five strands, never seen. Four-to-five is
   the entire headroom the environment has. The claim §6 actually measured
   (0.284 → 0.981 at six to eight strands) is **not testable end-to-end until the
   policy head is cell-indexed**, which is §4.1.

### What the gate returned: nothing, and the reason is the budget

| candidate | seed | highest stage | per-stage solve rate |
|---|---:|---:|---|
| `conv-window-128` | 71 | −1 | `unknot` 0.38 |
| `conv-window-128` | 72 | **0** | `unknot` 0.83 ✓, `T(2,3)` 0.00 |
| `s-window-128` | 71 | −1 | `unknot` 0.38 |
| `s-window-128` | 72 | −1 | `unknot` 0.00 |

2.41 core-hours across four runs. **Three of the four never cleared stage 0**, and
**neither architecture reached four strands**, so the gate did not test the thing
it exists to test. This is weaker than the parity predicted above: parity would
have been a measurement, and this is no signal at all.

**It is not a result about the architectures, and should not be read as one.**
The single promotion anywhere in the run belongs to `conv-window-128`, which makes
the tally 0/-1 against -1/-1; with two seeds, a stage-0 solve rate that ranges
from 0.00 to 0.83 *within one architecture*, and a 0.80 promotion bar measured on
eight evaluation games, that difference is noise. Reporting it as a raster win
would be exactly the false positive `docs/lessons.md` warns about — three seeds
once gave this project a positive that survived two rounds of reporting.

**The diagnosis is training volume, not search and not capacity.** Twelve
iterations of six self-play games is 72 games of experience per stage. Stage 0 is
the unknot scrambled by two moves — it should be near-trivial — and the arms land
between 0.00 and 0.83 on it, which is the signature of networks that have not seen
enough data rather than networks that cannot represent the answer. Search is not
the constraint: 128 simulations is the setting the repository measured as
sufficient to reach stage 8 on the full ladder.

**What a decisive run would cost.** Stage 0 alone took 1,858–2,167 s per run at
twelve iterations. Getting both arms reliably through four stages needs roughly an
order of magnitude more self-play per iteration, which puts a two-architecture,
three-seed comparison in the region of **days**, not hours — consistent with this
project's own ladder history. That is a scheduling decision, not something to
sneak in behind a representation note.

**The transfer evaluation was not run, deliberately.** It was built to load a
checkpoint trained through `P(4,5)#0` and score it on `P(5,6)#0`. No run reached
`P(4,5)#0`; the best available checkpoint has cleared only `unknot+2` and scores
0.00 on `T(2,3)`. Evaluating it on a four-strand positive braid would return
0.000 for both arms and mean nothing. The script
(`research/experiments/strand_transfer_eval.py`) is verified end to end against the
untrained checkpoints and is ready for a run that gets far enough.

**So the standing evidence for the raster is §§5–6 and the §2.1 tensor count, and
none of it is an RL result.** That distinction is the honest summary of this note.

### Rerun with a real evaluation budget, and what it showed instead

The obvious objection to the above is the evaluation: eight games per ratio against
a 0.80 bar means one episode is worth 0.125, so promotion turns on a single game.
The rerun raised evaluation games 8 -> **32** and self-play 6 -> **12**, same
architectures, same two seeds, 3.0 core-hours:

| candidate | seed | highest stage | stage-0 solve |
|---|---:|---:|---:|
| `conv-window-128` | 71 | −1 | 0.46 |
| `conv-window-128` | 72 | −1 | 0.41 |
| `s-window-128` | 71 | −1 | 0.54 |
| `s-window-128` | 72 | −1 | 0.11 |

**The extra evaluation removed the signal rather than sharpening it.** The single
promotion in the first gate — `conv-window-128` at 0.83 — does not reappear;
nothing promotes at all. That is exactly what a noise-driven result does when it is
measured properly, and it retrospectively confirms that the 0/−1 against −1/−1
tally was worth nothing.

**And then the floor explains the whole thing.** Evaluating *untrained* networks on
the same stage, with the same 128-simulation search:

| | untrained | after 12 iterations |
|---|---:|---:|
| `s-window-128` | **0.750** | 0.54, 0.11 |
| `conv-window-128` | **0.333** | 0.46, 0.41 |

Stage 0 is the unknot scrambled by two moves, and **128-simulation MCTS solves much
of it with random weights**. The trained range (0.11–0.54) sits *inside* the
untrained range (0.33–0.75); on `s-window-128` twelve iterations of self-play left
it below where it started. The 0.80 promotion bar is barely above what search alone
delivers.

So the gate at this budget does not discriminate architectures because it barely
discriminates *trained from untrained*. The binding constraint is neither
evaluation noise nor representation: it is that 12 iterations of 12 self-play games
is 144 games, which is not enough to improve on the prior that search already
provides — and can be enough to damage it. This is the same lesson the repository
already recorded from the other direction ("search dominates capacity"), arriving
here as "search dominates *training*, at this budget".

*(Caveat on the floor numbers: one seed, 12 games per ratio, so the 0.750 and 0.333
are individually noisy. The claim that survives is the overlap of the ranges, not
the ordering within them.)*

**What a gate would have to look like to answer the original question.** Stage 0
has to be somewhere an untrained network fails, which means either many more
self-play games per iteration or a harder first rung — and then the strand-transfer
question still needs the cell-indexed head of §4.1 before any checkpoint can be
tested past five strands. Both are scheduling decisions, not experiments to slip in
behind a representation note.

## 8. What this does not settle

* **§§5–6 are supervised probes, not a ladder.** They answer "can the encoder see
  the thing" in minutes rather than "does the agent win more" in days. The RL run
  is §7b, and it is bounded by the `max_strands = 5` ceiling described there: it
  can show parity inside the trained range and one strand of transfer, and it
  cannot reach the six-to-eight-strand regime where §6 measured the effect.
* **Three seeds, where `docs/lessons.md` asks for eight.** The differences that
  survive their seed spread are: raster over one-hot (huge), boundary channels
  under a torus wrap (+0.41), and boundary channels on local legality (−0.05). The
  differences that do **not** survive are all the rankings *within* the cylinder
  family. Do not read the §5.2 ordering as a ranking.
* **Only two of the four probes discriminate.** `distance` is 0.953 from two
  scalars and `isknot` is 0.891 from two scalars in distribution (§5.1), so the
  weight is carried by `destab` and by the wide splits. A stronger suite would
  add probes with a lower floor.
* **`determinant` is a null, and it might be the informative one.** Nothing in the
  family learns a genuine knot invariant. If the goal is unknotting-number bounds
  rather than diagram manipulation, that null matters more than the transfer wins —
  see [19 §5](19-superseding-the-rl-unknotter.md), which argues the way to get an
  invariant out of a network is to constrain the architecture to *be* one rather
  than to hope a picture-reader recovers it.
* **`distance` has no wide split.** Its label is an exact search depth, and the
  shortest solution for an eight-strand instance already exceeds the depth the
  oracle can reach, so a wide set would have been selected for being unusually
  easy. The absent column is a limitation, not a result.
* **One machine, shared.** Everything ran on eight CPU cores alongside the
  project's own training jobs, so `ms/step` is an order of magnitude and not a
  benchmark. The 8× compute gap in §4.5 is real but would look different on a GPU,
  where a taller canvas is nearly free.
