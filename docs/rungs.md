# The ladder: every rung, and why it is where it is

The curriculum in `pgx-mcts-bench/src/pgx_mcts_bench/ladder.py` is a list of
`(source knot, scramble moves)` pairs called `STAGES`. This document says what
each rung is, why it was added, and -- for the first time -- **what knot it
actually contains**, as opposed to what the generator's naming suggests.

The invariants are in [rungs-invariants.md](rungs-invariants.md) (generated) and
[rungs.json](rungs.json) (machine-readable). Regenerate both with:

```bash
uv run --with snappy python scripts/rung_invariants.py
```

The rung list is parsed out of `ladder.py` rather than copied, because that
file's own docstring says the rung count "has changed twice and the prose did
not". This document is the prose. It has one job: not to be the third thing that
falls out of date.

## The shape

41 rungs, in three blocks:

| rungs | block | source family | reference for `u` |
|---|---|---|---|
| 0-16 | calibration | the unknot, then torus knots | a theorem |
| 17-30 | challenge | random mixed-sign braid words | the ratchet in `bounds.jsonl` |
| 31-40 | held-out continuation | longer random mixed-sign words | the ratchet |

Difficulty is meant to be monotone in two independent knobs at once: the
unknotting number of the source knot, and the number of scramble moves applied
on top of its canonical word. The generator orders the whole space with a single
integer, `crossing_weight * c + scramble`.

## Why the rungs are in this order

### Rungs 0-1: the unknot, scrambled

The empty word in `B_1` is the `c = 0` row of the same table as every other
rung, so the original game -- scramble the unknot, then undo it -- is not a
separate mode but the bottom of the ladder. Only the scramble depth varies
(`+2`, then `+6`). Nothing here can test unknotting: the answer is always zero
crossing changes.

### Rungs 2-16: torus knots, chosen so the gap to truth is measurable

`u(T(p,q)) = (p-1)(q-1)/2` is a theorem -- Milnor's conjecture, proved by
Kronheimer and Mrowka -- so on these rungs the distance between what an agent
achieves and what is optimal is a number, not a guess. That is the entire reason
this block exists, and it is why promotion here can exit on `objective`.

Within the block the order came out of measurement rather than taste. Every `+0`
rung in the first ladder promoted in two iterations at exactly the proved
unknotting number, and every `+4` rung overshot and none converged. The
conclusion recorded in `ladder.py` is that **the source knot is nearly free and
the scramble is the whole problem**, so the ladder is graded finely in scramble
depth (`+0, +2, +4, +8`) and coarsely in `u`.

Two consequences of that principle are visible in the list:

* **`T(2,7)` was dropped.** It has `u = 3`, the same as `T(3,4)`, which is
  strictly harder -- three strands and eight letters against two and seven. One
  knot per unknotting number is enough, and the plies freed paid for the `+2`
  and `+8` rungs.
* **`T(2,9)` and `T(3,5)` are adjacent on purpose.** `T(2,9)` is `u = 4` on two
  strands with nine crossings; `T(3,5)` is `u = 4` on three strands with ten. The
  pair separates "larger `u`" from "harder diagram at the same `u`", which are
  otherwise confounded all the way up the ladder.

### Rungs 17-30: random mixed-sign words, because the labelled families are all the same shape

Every torus knot and every positive braid is fibred, chiral, of positive
signature, and satisfies `u = g_3 = g_4`. An agent can learn "reduce
monotonically, crossing changes always pay", be right on every labelled rung, and
have learned nothing that transfers. Worse, an earlier version of this block used
positive braids, and ten of those rungs were torus knots under another name: on
two strands a reduced positive word is `sigma_1^c`, so `P(2,11)` *is* `T(2,11)`.

So the block was replaced with random **mixed-sign** words, filtered only to be
knots (one component) and to survive a shallow breadth-first unknotting search.
They carry `UNKNOWN_UNKNOTTING = -1`, and their reference is the ratcheting
best-known bound in `bounds.jsonl` rather than a theorem. Promotion here can end
only on plateau or at the cap.

### Rungs 31-40: the held-out continuation

Longer mixed-sign sources (22, 24 and 26 letters), each appearing at `+0` and
`+4` so that source-word length is separated from extra diagram scrambling. They
were added as a single block, at the end, so that identity-based resume keeps
every historical rung attached to the knot it actually measured -- inserting a
rung in the middle would silently re-point every index above it.

## What the invariants changed

The premise of the last two blocks is that the knots are unlabelled. Computing
the invariants shows that this is mostly false.

**The word length is not the crossing number.** The generator's only filters are
"one component" and "a depth-4 search failed to unknot it". Neither notices that
an eighteen-letter mixed-sign word on three strands can be a seven-crossing
knot. Every rung's advertised difficulty is the length of the word it was
generated from, and for the challenge blocks that number is frequently three
times the crossing number of the knot itself.

**19 of the 23 distinct knots now have an exact unknotting number.** Before this
they had one on 6 rungs. The detail is in
[rungs-invariants.md](rungs-invariants.md); the three findings that matter:

* **`R(3,18)#0` is `7_5`** -- seven crossings, determinant 17, signature -4, and
  `u = 2` since the knot tables. The ratchet's best recorded bound for it was 6.
  This is the case that prompted all of this.
* **`R(3,22)#0` is the unknot.** A twenty-two letter word that simplifies to zero
  crossings, with trivial Alexander and Jones polynomials and determinant 1.
  Rungs 31 and 32 are the unknot with extra decoration, sitting at the top of the
  ladder where the hardest problems are supposed to be. The depth-4 search that
  is supposed to catch exactly this ran out of depth.
* **Six rungs are composite knots** -- `3_1 # 4_1`, `3_1 # 5_2`,
  `3_1 # 3_1 # 3_1` and so on. These failed to identify at first for a reason
  that has nothing to do with being hard: knot tables list *prime* knots, so a
  connected sum of three trefoils is not in any of them. Their unknotting numbers
  are pinned exactly anyway, between subadditivity from above and `|sigma|/2`
  with Scharlemann's theorem -- an unknotting-number-one knot is prime, so every
  composite knot has `u >= 2` -- from below.

KnotInfo closes two gaps left by the older Knot Atlas data: `12n_647` has
`u = 4`, and `12n_570` has `u = 2`. What remains unresolved by this repository's
table is `R(5,24)#0` and `R(5,26)#0`, which are prime at 15 and 16 crossings and
so past its twelve-crossing limit.

This does not invalidate the ladder as a curriculum -- an agent still has to
solve what it is given, and a bloated diagram of a small knot is a legitimate
thing to be handed. It invalidates the *scoring*. A rung whose reference is "the
fewest crossing changes anyone has managed" is measuring the field's search
budget when the true answer has been in a table since the 1970s.

**What to do with it** is a separate decision, and the data is committed so it can
be made deliberately:

1. Attach the published `u` to every rung that identifies, converting most of the
   challenge set into extra calibration -- more rungs where the gap to truth is
   measurable, which is the scarce resource.
2. Or reject identified knots at generation time and search for genuinely
   unlabelled ones, which means filtering on the invariants rather than on a
   shallow unknotting search.

Option 1 costs nothing and is strictly more informative than what is there now.
Option 2 is the only way to get the challenge set the block was intended to be.

## Caveats

* `c(K)` is reported only where the knot was identified against the bundled
  table: 2870 knots, being every tabulated knot up to 12 crossings except 107
  whose braid is too wide for the build's strand cap. An unidentified knot is
  not a knot with unknown invariants; it is a knot outside the table, and the
  `notes` column says which.
* **Identification can be ambiguous, and says so.** 384 fingerprints in the
  table are shared by more than one knot -- `5_1` and `10_132` agree on both the
  Jones and the Alexander polynomial, and have different unknotting numbers, 2
  and 1. Where that happens the rung reports every candidate rather than picking
  the first. `T(2,5)` is one of them, and is only pinned because the generator
  built it from a formula and the Milnor conjecture supplies its `u`.
* The unknotting numbers marked as published were read from the 2026-08-01
  KnotInfo database snapshot, and are the only numbers in this repository not
  derived by its own code. They live in
  `src/rf_knots/data/unknotting_numbers.json` with the source URL and snapshot
  hash.
* `sigma` comes from a Seifert matrix computed by `spherogram`. The reasoning
  behind not computing it here is in the docstring of `rf_knots/invariants.py`,
  and is worth reading before anyone tries again: a locally-defined rule that
  reproduced the signature on 50 reference knots turned out not to be a Seifert
  matrix at all.
