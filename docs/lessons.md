# Lessons: what went wrong, and what to do instead

Written down because each of these cost real time to learn and none of them is
visible in the code that resulted. They are grouped, but the order within a group
is the order they were learned.

They come from `pgx-mcts-bench` training runs and from this repository's tooling.
Where a lesson names a file or an arm, the name is from the run that taught it and
may since have moved; the lesson is the durable part.

## Measurement

- **8 seeds minimum** for anything about instance difficulty. 3 gave a false
  positive that survived two rounds of reporting.
- **Read the trained checkpoint, not the design doc.** Every serial defect was
  found by probing weights and action histograms, none was visible in intent.
- **Diagnose before fixing.** I called `u1-puct`'s 14.00 crossing changes a FiLM
  conditioning bug. It was two different measurement artefacts — a stale recorded
  value, and a search-depth failure that 128 simulations fixed with the network
  untouched. Only 9 of 177 rungs inverted, which is what ruled out a systematic
  fault before any code was written.
- **Never let a report hardcode what it is reporting.** The ladder header claimed
  "Ten stages" through two changes to the stage list. Reading `len(STAGES)` is the
  same bug one level down — it relabelled a historical ten-rung run with today's
  rungs. Reports now derive everything from the results themselves.
- **Resume on stage identity, not index.** Inserting a rung silently repoints
  every index-keyed checkpoint at a different stage.
- **A promotion threshold makes the metric at promotion nearly meaningless.** It
  reports where the network happened to be when it crossed the bar.
- **`cc` is conditional on solving and anti-correlated with the solve rate.** An
  arm that abandons hard instances drops them from its own average, so failing
  more can look cheaper. `cc/sr` is the honest column.

## Computing an invariant

- **Project the cost of a sweep before starting it, and log progress inside it.**
  The knot table was launched three times before it finished: once with a
  factorial-time determinant, once probing past the end of the census, once with
  the cost concentrated in nine-strand braids nobody had measured. A two-minute
  scan of the strand distribution would have predicted all three.
- **A rule fitted to reproduce one invariant has not been derived.** A local rule
  on the Bennequin surface's generators reproduced the signature on all 50
  reference knots and then failed the Alexander polynomial on 18 of them. Fifty
  exact integer agreements were not evidence; they were the sample the rule was
  fitted to. Check a fitted rule against an invariant it was *not* fitted to.
- **`(-1) ** e` is a float when `e` is negative.** Jones exponents are routinely
  negative, so every determinant computed that way was `17.0` rather than `17`
  and no dictionary key ever matched. Nothing raised.
- **Identification by one invariant is ambiguous and must say so.** 384
  fingerprints in a 2870-knot table are shared by more than one knot; `5_1` and
  `10_132` agree on both the Jones and the Alexander polynomial and have
  different unknotting numbers. A lookup that keeps the first match answers
  confidently and wrongly.

## Running things

- **A gate on `getloadavg` cannot throttle a burst** — it is a one-minute decaying
  average and lags a launch by about a minute. Rate-limit starts instead.
- **`pkill` on a `uv run` wrapper orphans its child.** Two launches accumulated
  instead of replacing one another and load hit 45.
- **`grep "[m]y-pattern"` still matches your own shell command line**, which cost
  several rounds chasing three processes that were my own `ps` invocations.
- **List a shared host's running services before putting load on it.** `nproc`,
  `free` and `uptime` said "idle"; it was running production.
- Negative results belong in commit messages and notes, not in dead code.
