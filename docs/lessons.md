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
- **A classical construction can be wrong in a way that looks right.** A
  hand-derived Seifert matrix for a braid closure reproduced the trefoil, the
  figure-eight and every torus knot tried, and still disagreed with the
  Burau-derived Alexander polynomial on 7% of random knots. All sixteen sign
  conventions in the obvious family were tried; none reached agreement, so the
  error was structural. Differential-test against an independent implementation
  over thousands of inputs before trusting a construction, and delete it rather
  than ship it at 93%.
- **An invariant with a parity is a free correctness test.** A knot's signature
  is always even. That one assertion caught an elimination that wrote symmetric
  entries mid-pivot, corrupting the pivot row before it had been read — a bug that
  produced plausible numbers on every small example. The same assertion first
  fired on a *link*, where odd signatures are legitimate; the fix was to filter to
  knots, not to weaken the assertion.
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
- **A background launch whose redirect target does not exist fails silently, and
  the shell still reports success.** `nohup ... > artifacts/run.out &` with no
  `artifacts/` never started anything; the `&&` chain reported exit 0 from the
  `echo` after it. Forty minutes later there was no log to read, because there was
  no process. `mkdir -p` the output directory in the same command, and confirm the
  process exists with `pgrep` before walking away.
- **`torch.set_num_threads(1)` inside a pool worker is not enough.** Torch still
  sizes its pools from the environment, so seven workers ran 37 threads each and
  the load average hit 77 on eight cores. Export `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `OPENBLAS_NUM_THREADS` *before*
  Python starts. Everything still finishes, so the symptom is not a crash — it is
  wall-clock numbers that mean nothing.
- **Cache the labels, not the runs.** Exact labels (a breadth-first search, a
  Burau determinant) cost minutes and do not depend on which network is being
  trained. Building them once per `(probe, split, seed)` made a restart free and
  also made the experiment stronger: every arm is then scored on literally the
  same instances rather than on independently drawn ones.
- **Measure the untrained floor before reading a training curve.** An untrained
  network with 128-simulation MCTS solved 0.75 of the curriculum's first rung; the
  networks trained on it for twelve iterations scored 0.11 to 0.54. The whole gate
  was measuring how much 144 self-play games damage a prior that search was already
  carrying. A promotion bar only just above the untrained score cannot discriminate
  anything.
- Negative results belong in commit messages and notes, not in dead code.
