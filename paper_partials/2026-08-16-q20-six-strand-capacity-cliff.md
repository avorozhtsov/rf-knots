# Q20 six-strand capacity cliff

Date: 2026-08-16. Status: **preliminary gate snapshot at 72/80 durable
scientist-items; Q20 remains active**.

## Protocol and denominator

Q20 is the first gate of the frozen 4,000-representation strand-12 capacity
program. Four inherited scientists use the same static order, L10/L1000
objectives, paired seeds, no sharing, and twenty representations. Skipped and
unsolved rows remain in the coverage and capped-loss denominators. No scientist
has used its one-row skip ledger.

The first fifteen runtime rows have at most three strands and at most ten
presentation crossings. The last five are all six-strand presentations of
known unknotting-number-one knots:

| runtime round | knot | strands | presentation crossings | attribution queue |
|---:|---|---:|---:|---|
| 15 | `11a_121` | 6 | 17 | ACS10 |
| 16 | `11a_104` | 6 | 19 | ACS5 |
| 17 | `11a_11` | 6 | 19 | ACS10 |
| 18 | `11a_138` | 6 | 19 | ACS5 |
| 19 | `11a_167` | 6 | 19 | ACS10 |

## Current result

All four scientists solved every one of the first fifteen representations at
both objectives. All four then failed `11a_121` at both L10 and L1000. Raster
axial and strand graph have completed all five six-strand rows and failed every
one. Cyclic memory and combined dual each retain rounds 16--19, so the formal
gate verdict is still pending.

| scientist | committed rows | L10 coverage / capped mean / evaluation evals | L1000 coverage / capped mean / evaluation evals | six-strand successes |
|---|---:|---|---|---:|
| raster axial | 20/20 | 15/20 / 123.00 / 209,820 | 15/20 / 8289.30 / 208,260 | 0/5 |
| strand graph | 20/20 | 15/20 / 109.50 / 415,251 | 15/20 / 7037.95 / 418,089 | 0/5 |
| cyclic memory | 16/20 | 15/16 / 125.13 / 815,280 | 15/16 / 4986.94 / 954,600 | 0/1 |
| combined dual | 16/20 | 15/16 / 65.88 / 81,640 | 15/16 / 5016.81 / 69,940 | 0/1 |

The capped means include every committed six-strand failure. The current
evidence is therefore a sharp presentation-capacity cliff, not merely a fall
in conditional solution quality. It is not yet a failed gate: either remaining
scientist can still produce the required six-strand native success.

## First-ten compute control

The durable block-010 report provides a clean outcome-matched compute panel:
all four scientists solved all 80 evaluation attempts and retained all 20
rehearsal attempts. Native network evaluations nevertheless varied greatly:
356,590 raster axial; 514,800 combined dual; 1,661,649 strand graph; and
13,271,133 cyclic memory. Cyclic memory thus used 25.8 times the native
network evaluations of combined dual on an identical perfect-outcome panel.

## Gate rule

Q20 advances only if all four scientists commit all twenty rows, every
retention rate is at least 0.80, no capacity exception occurs, skipped rows are
included in all denominators, and at least one native success occurs on a
six-or-more-strand representation. If the gate fails, these resumable artifacts
must be preserved and Q40-1 must not start.

Artifact root:
`/srv/braid/artifacts/q4000-strand12-20260814/q20-static-no-sharing`.
Manifest SHA-256:
`f3cfb4384cd61bf1cdddf1e808def202fc370582afbd4f4b9bafa877f40a8917`.
Frozen bank SHA-256:
`e9a25901ee49373d10ec07ea5b1cccddc7556947f269f4e304fe6f06f8eb89ae`.
Block-010 SHA-256:
`f55a57eab15f51d71dde77aaf37c97553a4c676f28a7535c31183ea07a6f7ed6`.
The compact source table is
`data/2026-08-16-q20-capacity-cliff.csv`.
