# First committed raster MCTS shard: partial replay verified

2026-09-21, hourly inspection at20:43 UTC. This is a progress record for the first
8 of128 Q254-final conditions, NOT the complete protocol97 comparison or audit.
The next routine process inspection is no earlier than approximately21:43 UTC.

## Live state at the scheduled check

Controller98832 was evaluating `search-q254`, with9/13 jobs completed. Its actual
child17857 had the expected canonical RF/PGX command and parent98832, approximately
99% CPU; CPU time advanced from126:02.19 to126:13.70 over successive samples in
this inspection. The controller session60520 remained live. No restart, interrupt,
duplicate evaluation or protocol change was performed.

One durable shard appeared since the preceding hourly check. Search manifest
mtime1790022137.0415456 corresponds to20:22:17 UTC. The journal contains the
matching committed counts; its only warning is the pre-existing unavailable
plink tkinter GUI, not a training/search exception. The new shard, rather than
CPU activity alone, establishes actual saved search progress.

Directory: `artifacts/pretrained-raster-log-evaluation-v1/mcts/q254/`.
Shard `conditions-00000.json` SHA256:
`b93aee9b03f8f3a79d472239ba529491ea977c2235753841e680a7726b2481db`.
Roots SHA256:
`c1da996e1e6e95251986dd04bdad1467d45ad3f07055fc2b6ca79424131d7967`.
Q254-final checkpoint SHA256:
`c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f`.

## What was verified

A separate read-only replay process exited0. It checked the frozen input pins,
roots against the complete panel constructor and roots hash, and committed-shard
hashes. It called `inspect_rows` on the actual committed prefix with its unchanged
manifest counters: all8 conditions,47 episodes,25273 continuation moves,
25314 full-path moves and4235569 recorded policy action scores were checked.
This includes unsuccessful/censored paths, resource accounting, action validity,
query/seed identities and the successful witness. It does not re-execute neural
forwards or independently reproduce timing/tree-compute measurements.

The batch's recorded compute time was6336.97s (approximately1h45m37s). Do not
extrapolate a reliable final ETA from this one batch with varying root lengths.

## Partial search outcome

One success among8 conditions: condition7, panel root3, lambda1000 search.
The replayed witness has1 crossing change and93 moves; therefore its costs are
L10=103 and L1000=1093. Reporting its L10 cost does NOT mean the paired lambda10
search succeeded: only the lambda1000 condition was successful in this prefix.

The prefix covers four roots at two cost weights each, not eight independent
knot types. There is no claim of optimality, improvement over the database,
overall1/64 success, embedding utility or transfer advantage from this partial
result. No UnknotDB import/release or checkpoint promotion follows from it.

Continue the existing controller through both128-condition searches, their full
replay/cost audits and final summary. Keep hourly monitoring, preserve all dirty
work, and retain censored failures as censored observations rather than labels
that a state is impossible to solve.

## Hourly update: 21:46–21:47 UTC, prefix16 replay verified

The same controller98832 and child17857 remained live with unchanged commands.
Worker CPU time advanced187:30.41 to187:43.50 during this inspection. Controller
still had9/13 completed jobs. Next routine main-process check: no earlier than
22:47 UTC. No restart or backend/protocol change.

A second shard was committed at21:02:08 UTC (local22:02:08+0100):
`conditions-00008.json`, SHA256
`eb170ac66eaa5c105feb6b450e3eed41fb1645929d2a7c68506bda711c56a342`.
Its recorded batch time is2389.507122s. The matching journal advanced without
new errors. Total durable coverage is16/128 conditions,95 episodes,40497
continuation moves,40578 full-path moves and5596592 recorded policy action scores.
The solved-condition count remains1, the witness described above.

A separate read-only audit finished at approximately21:50 UTC, exit0. It checked
all input pins, full128-root constructor/hash identity and both shard hashes,
then replayed the complete committed16-condition prefix using its unchanged
manifest counters. All95 episodes, including censored ones, passed. This is a
partial replay audit, not the full protocol97 comparison or a success-rate claim
for the remaining roots. The eight additional conditions yielded no new success.

## Hourly update: 22:47–22:49 UTC, prefix24 and a second solution

No intervening main-process inspections were performed between the preceding
check and this hourly window. Controller98832 still had9/13 jobs complete, with
the expected child17857 running Q254-final search from the canonical RF checkout.
The worker's CPU time advanced247:47.21 to248:50.99, approximately100% CPU.
No duplicate evaluation workers or new errors were found. Random-arm search had
not started. Next routine main-process inspection: no earlier than23:49 UTC.

Third shard `conditions-00016.json` was committed at22:33:13.999215 UTC, SHA256:
`8ca6ddc8aad5be35f219575dae929799a6c0d0e5300fa84a1c688ff55d465f22`.
Its recorded batch time is5463.837401s. The journal grew to1403 bytes with matching
counts. Total committed coverage:24/128 conditions,141 episodes,64545
continuation moves,64667 full-path moves and8460385 recorded policy action scores.
Two conditions are solved. These24 conditions cover12 roots at two cost weights,
not24 independent knots or the complete64-root cohort.

A separate read-only prefix audit exited0. It verified all frozen input pins,
the full128-root constructor/hash identity, all three shard hashes, and replayed
all141 committed episodes with the original resource/RNG/count checks. Both
successful witnesses were additionally replayed. This remains a partial audit,
not the controller's full replay/cost audit or paired comparison.

The new success is condition16, panel root8, lambda10, starting with a word of
length128 on9 strands. It uses1 crossing change and71 native moves, giving
L10=81 and L1000=1071. Only the lambda10 search succeeded for this root in the
committed prefix; evaluating its witness under L1000 is not a lambda1000 search
success. The prior success remains condition7/root3/lambda1000,1CC/93moves.

### Independently checked cost gap for condition16

The reference and new witness start from exactly the same raw word. Both were
replayed through `witness_identity`, checking every prefix's native potential
identity. Independent determinant computation gives11227, proving the knot is
nontrivial and hence requires at least1CC. The initial potential is
q0=(128+9-1)/2=68. Therefore the fixed native move system gives cost lower bounds
68+11=79 for L10 and68+1001=1069 for L1000. The known reference achieves both,
with1CC and69 moves, and is certifiably optimal under these move semantics.

The new witness is exactly2 above those optima. Its action counts are60 REDUCE,
9 DESTABILIZE,1 STABILIZE_POS and1 CROSSING_CHANGE; its semantic excess is2.
The reference has60 REDUCE,8 DESTABILIZE and1 CROSSING_CHANGE, excess0. This does
not imply one may simply delete two actions from the found witness without
rechecking legality, and no such modification or database import was made.

This is a replay-verified near-optimal solution for one128-letter presentation,
not a new unknotting-number bound, an improvement over its existing reference,
or evidence that the model generalizes to arbitrary lengths/strand counts.
The input is longer than48 but within this run's actual training length range;
there is no claim of length extrapolation or causal embedding benefit. Full
paired evaluation remains necessary. Training and main MCTS sources/checkpoints
were neither changed nor restarted.

## Hourly update: 23:49–23:50 UTC, compute active without a new shard

The actual manifest still selects child17857 under controller98832. Exactly one
matching worker and one matching controller were found. Worker CPU time advanced
308:10.92 to308:43.09 across samples, approximately94–99% CPU, with the expected
command. This verifies live computation, not additional completed conditions.

Durable coverage remains24/128, two solutions,9/13 controller jobs. The last
manifest/journal update remains22:33:13.999215 UTC and the log remains1403 bytes,
without new errors. All three previously replayed shard hashes and the roots
hash remain unchanged. Random-arm search has not started. The worker commits only
whole batches of eight conditions; an unfinished batch does not update these
files. No restart or protocol change is justified by this observation alone.

No new trajectories existed to audit, so no duplicate full-prefix replay was
performed. Next routine process inspection: no earlier than2026-09-22 00:51 UTC.

## Hourly update: 2026-09-22 00:51–00:52 UTC, prefix32 verified

The controller remains98832 with its unique Q254 worker17857; expected commands,
no errors/duplicates. Worker CPU time advanced369:17.53 to369:43.70 at98–100% CPU.
Controller still has9/13 completed jobs; random-arm search has not started.
Next routine inspection: no earlier than01:53 UTC.

Fourth shard `conditions-00024.json` was committed at00:47:27.282307 UTC, SHA256
`ee6def67437fb0dcdf33c98c7bd8ee5d5b15b8c4e05db90381c4071bef202190`.
Recorded batch time8051.599078s; matching journal1793 bytes. Total durable coverage
is32/128 conditions (16 roots x two cost weights),187 episodes,89988 continuation
moves,90151 full-path moves,13935871 policy action scores and three solutions.
The89988 continuation-state neural evaluations are not13.9 million independent
states; multiple actions are scored for each state.

This inspection used an incremental replay audit, exit0. The prior24-condition
prefix remains bound to the exact three previously replayed shard hashes. All
input pins, full roots constructor/hash, four shard hashes and query/RNG
identities were rechecked. The NEW eight conditions were replayed in full:
46 episodes,25443 continuation moves,25484 full-path moves,5475486 action scores
and one success. Adding these to the previously verified prefix counts exactly
matches the new manifest. Neural-state and timing counters were checked as before;
no additional replay of the old141 episodes was claimed or required for this
incremental check. The controller's eventual full replay audit is still pending.

New success: condition31, panel root15, lambda1000. Its196-letter,9-strand input
is solved with2CC and106 native moves: L1000=2106, L10=126. Reference and found
witnesses were replayed and their starts matched exactly. Independent determinant
is2009, matching the frozen panel; the audited signature bound supplies lowerCC2.
The reference uses2CC/104moves, q0=102 and zero semantic excess, attaining certified
floors L1000=2104 and L10=124. The found path is2 above these optima, with one
INSERT,95 REDUCE, two CROSSING_CHANGE and eight DESTABILIZE actions, excess2.

Only the lambda1000 condition succeeded for this root. The new witness's L10
value is not another successful search. As with the prior example, length196 is
within this run's training length range, not a length-extrapolation result; the
reference was already optimal and no database bound was improved or imported.
This is a third valid solution, not a complete paired quality/promotion result.

## Hourly update: 2026-09-22 01:53–01:54 UTC, prefix40 verified

Unique controller98832/worker17857 remain on the expected canonical command.
Worker CPU time advanced430:54.57 to431:51.23 at98–100% CPU; no errors or duplicate
workers. Controller9/13 jobs; random search still not started. Next routine
process inspection: no earlier than02:55 UTC.

Fifth shard `conditions-00032.json` was committed at01:18:21.075536 UTC, SHA256
`d3343a792eb62daeebe4fb368f223ba78488c65cce6b390ebf47c14ec802c3c3`.
Recorded batch1852.203931s; journal2187 bytes. Total40/128 conditions,233 episodes,
104709 continuation moves,104914 full-path moves,15092154 action scores and six
successful conditions. These cover20 roots; the six successes concern five roots,
since one root now succeeds under both cost weights.

Incremental replay exited0: the first four shard hashes retain the previously
verified32-condition prefix; all pins, roots, five shard hashes and query/RNG
identities checked. All NEW eight conditions were replayed:46 episodes,14721
continuation moves,14763 full-path moves,1156283 action scores, three successes.
Their counters plus the prior prefix exactly match the current manifest. Each
new solution and its reference were additionally replayed with the prefix cost
identity and an independent determinant check.

| Condition / root | Search lambda | Input length / strands | Found CC / moves | Cost for this search | Reference cost | Reference certified optimal? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 36 / 18 | 10 | 142 / 7 | 2 / 78 | 98 | 96 | yes |
| 37 / 18 | 1000 | 142 / 7 | 2 / 78 | 2078 | 2076 | yes |
| 39 / 19 | 1000 | 217 / 22 | 2 / 125 | 2125 | 2121 | no |

Root18: determinant457049, audited lowerCC2, q0=74. The2CC/76-move reference has
zero excess and achieves the native cost floor. Both found paths have excess2
(one INSERT plus its extra reduction), hence certified regret2 for their
respective cost objectives. They are two successful conditions of ONE root.

Root19: determinant4347, lowerCC1, q0=119. Its2CC/121-move reference has zero
excess but is NOT known optimal because u is only bounded by[1,2]. The found
2CC/125-move path has excess4 (two INSERT actions and extra reductions). Its gap
to the reference is4; that is NOT a certified optimality regret. The L1000 lower
floor is1120, so the gap to that bound is1005, not proof a better path exists.
Only lambda1000 succeeded for this root. No database bound improvement/import.

### Bounded strand-count extrapolation, not arbitrary-size success

The217-letter/22-strand solution is outside the actual strand-count support of
protocol97 optimization:121's hash-verified complete census has metric maximum21
strands and positive-policy maximum18; reading ALL four full relative groups from
118 gives maxima17/5/4/4. The model therefore solved this one raw representation
with one more strand than the largest metric-training input. Word length217 is
above the positive-policy maximum208 but below the metric-training maximum254.
This is bounded strand-count extrapolation within a shared trained backbone,
not length extrapolation of the whole model, an unseen-knot certificate, causal
embedding evidence, or a guarantee at arbitrary sizes. Q254 historical exposure
limitations remain. Engineering-only augmented views/private optimizer probes
did not modify the evaluated97 checkpoint and are not extra training exposure.

## Hourly update: 2026-09-22 02:55–02:56 UTC, no new committed batch

The unique expected worker17857 under controller98832 is still computing; CPU
time advanced492:40.44 to493:03.38 at approximately99% CPU. There are no new
errors or duplicate workers. Durable coverage remains40/128, six successful
conditions; all five previously replayed shard hashes and the roots hash were
rechecked unchanged. Last manifest/journal update is01:18:21.075536 UTC, journal
2187 bytes. Controller9/13 jobs, random search not started. This is verified live
compute, not additional saved search progress. The unfinished eight-condition
batch has no partial commit; no restart or additional replay was performed.
Next routine inspection: no earlier than03:57 UTC.

## Hourly update: 2026-09-22 03:57–03:58 UTC, prefix48 verified

Controller98832/worker17857 remain the unique expected processes. Worker CPU time
advanced554:54.13 to555:43.59 at97–100% CPU. No errors, source changes, restarts
or duplicate workers. Controller9/13 jobs; random search has not started.
Next routine process inspection: no earlier than04:59 UTC.

Sixth shard `conditions-00040.json` was committed at03:41:44.558758 UTC, SHA256
`ed45ddda25e91576e054a8c49d2af8b3db44e7f4bab6dd6383f1e67ca689a354`.
Recorded batch8601.824134s; matching journal2581 bytes. Total48/128 conditions
(24 roots x two weights),281 episodes,134861 continuation moves,135106 full-path
moves,19649411 action scores and six successful conditions. No new solution in
this batch; the six successes still concern five distinct panel roots.

Incremental replay exited0. All frozen input pins, full roots identity, six shard
hashes and query/RNG identities were checked. The prior40-condition counts remain
bound to the five previously replayed shard hashes. The NEW eight conditions
contain48 episodes,30152 continuation moves,30192 full-path moves and4557257
action scores; all were replayed, and adding their counts exactly reproduces the
manifest totals. Neural-state and committed-time accounting also passed.
These are unsuccessful/censored searches within their budgets, not evidence
that the states are unsolvable or that every move should receive a negative
training label. No old trajectories were replayed again and no TEST data were
imported into training. Full paired evaluation remains pending.

## Hourly update: 2026-09-22 04:59–05:00 UTC, compute active

The expected unique controller98832 and worker17857 are live. Worker CPU time
advanced616:42.98 to617:07.03 at99–100% CPU, with no new errors or duplicate
workers. No new committed shard:48/128 conditions, six successful conditions,
9/13 controller jobs; random search not started. Manifest/journal last updated
03:41:44.558758 UTC, journal2581 bytes. All six audited shard hashes and the
roots hash remain unchanged. This confirms continued computation but does not
add saved progress; no duplicate replay or restart was performed. Next routine
process inspection: no earlier than06:01 UTC.

## Hourly update: 2026-09-22 06:01–06:08 UTC, compute active

The same unique controller98832 and worker17857 remain live with the expected
command lines and parentage. The opening sample recorded worker CPU678:37.40;
after the scheduling-request interruption, fresh samples recorded684:32.08 and
684:48.13 at98–100% CPU. This completes the same due inspection, not a new
between-hour poll. Durable coverage remains48/128, six successful conditions on
five roots; nine controller jobs are complete, with random search still pending.
All six previously replayed shard hashes and the roots hash remain unchanged.
The journal remains2581 bytes, last committed batch03:41:44 UTC; no new errors
or duplicate workers were found. Continued CPU activity is not additional saved
progress. No restart, protocol change, or duplicate trajectory replay performed.
The active heartbeat is confirmed hourly; older GPU/MPS heartbeats remain paused.
Next routine process inspection: no earlier than07:09 UTC.

## Hourly update: 2026-09-22 07:09–07:10 UTC, prefix56 verified

No experiment state was polled between the preceding inspection and07:09 UTC.
The unique expected controller98832 and worker17857 remain active; worker CPU
advanced746:00.74 to747:01.29 at98–100% CPU. Controller9/13 jobs, Q254 search
continues, random search still pending. No duplicate worker or new logged error.
Next routine process inspection: no earlier than08:12 UTC.

Seventh shard `conditions-00048.json` committed at06:49:10.960721 UTC, SHA256
`275577563216767b8b780efb352567b8865adec7b353f4cedd25b92e20e067f1`.
Batch time11246.410962s; matching journal2976 bytes. Saved coverage is56/128
conditions (28 roots at two cost weights),329 episodes,171637 continuation
moves,171922 full-path moves,27210215 policy action scores. Six successful
conditions on five roots remain; this batch adds no solution.

The incremental replay exited0. All1264 frozen input pins, full roots identity,
seven committed shard hashes, and every saved condition's query/RNG identities
were checked. The earlier48-condition totals are bound to the six previously
audited unchanged shard hashes. All NEW eight conditions were replayed:48
episodes,36776 continuation moves,36816 full-path moves,7560804 action scores.
Adding these counts exactly reproduces the saved manifest. Neural-state count
and committed-time accounting also passed. These finite-budget unsuccessful
searches are censored observations, not impossible states or uniformly bad moves.
No old trajectory replay, training modification, TEST import, or restart occurred.
The final paired search/cost audit and scientific promotion decision remain open.

## Hourly update: 2026-09-22 08:12–08:13 UTC, compute active

The same unique controller98832 and worker17857 are live with unchanged expected
commands and parentage. Worker CPU advanced808:53.22 to809:41.87 at99–100% CPU.
No new committed shard or logged error:56/128 conditions, six successful
conditions on five roots; nine controller jobs completed, random search pending.
Manifest/journal last updated06:49:12 UTC, journal2976 bytes. All seven audited
shard hashes and the roots hash were rechecked unchanged. This confirms live
computation, not extra saved progress. No duplicate replay, restart, additional
worker or protocol change. Next routine inspection: no earlier than09:14 UTC.

## Hourly update: 2026-09-22 09:14–09:15 UTC, compute active

The expected unique controller98832 and worker17857 remain live. Worker CPU
advanced870:47.34 to871:28.00; sampled utilization87–100%. Saved coverage remains
56/128 conditions, six successful conditions on five roots. Controller9/13 jobs,
random search pending; no new logged error or duplicate worker. All seven prior
shard hashes and the roots hash are unchanged. Manifest/journal still last
updated06:49:12 UTC, journal2976 bytes. This is verified computation, not a new
committed result; the current batch's elapsed time alone does not imply failure.
No restart, protocol change, or repeated replay. Next routine inspection: no
earlier than10:16 UTC.

## Hourly update: 2026-09-22 10:16–10:17 UTC, prefix64 verified

Unique expected controller98832/worker17857 remain active. Worker CPU advanced
932:29.95 to933:51.48 at99–100% utilization; controller9/13 jobs and random search
still pending. No duplicate workers or new logged errors. No experiment state
was polled between the previous inspection and this hourly window. Next routine
process inspection: no earlier than11:19 UTC.

Eighth shard `conditions-00056.json` committed at09:47:13.584836 UTC, SHA256
`8cd541727a17d5c19b2cd1e711012d10f5931f8f3b536d006a3a32a6ab491a1f`.
Batch time10680.716454s; manifest/journal updated09:47:15 UTC, journal3371 bytes.
Coverage is64/128 conditions (32 roots at two cost weights),372 episodes,
201557 continuation moves,201884 full-path moves and34830552 policy action
scores. Nine successful conditions now cover SEVEN distinct roots.

Incremental audit exited0. All1264 frozen input pins,128 query roots, eight shard
hashes and all saved query/RNG identities were checked. The prior56-condition
counts remain bound to the seven previously replayed unchanged shard hashes.
The NEW eight conditions were fully replayed:43 episodes,29920 continuation
moves,29962 full-path moves,7620337 action scores and three successes. Combined
counts exactly reproduce the manifest; neural-state and committed-time accounting
passed. Each new solution AND its reference additionally passed the per-prefix
semantic cost identity, with fresh independent integer determinant checks. CC
signature lower bounds come from the already audited frozen panel.

### New long-word successes and native-cost bounds

| Condition / root | Search lambda | Input length / strands | Found CC / moves | Cost for this search | Certified optimal cost | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 61 / 30 | 1000 | 392 / 25 | 2 / 210 | 2210 | 2210 | 0 |
| 62 / 31 | 10 | 407 / 28 | 2 / 301 | 321 | 239 | 82 |
| 63 / 31 | 1000 | 407 / 28 | 2 / 251 | 2251 | 2219 | 32 |

Root30: determinant42581, certified lowerCC2, q0=208. The found2CC/210-move
witness has zero excess (184 REDUCE,24 DESTABILIZE,2 CROSSING_CHANGE), matching
the reference cost and attaining the proven native semantic-cost floor. It is
therefore optimal for this representation and cost model, not a newly improved
database bound. The same witness has L10=230, also at its floor, but the paired
lambda10 SEARCH did not succeed; this does not add a second successful condition.

Root31: determinant65661, certified lowerCC2, q0=217. Its reference2CC/219 moves
has zero excess and proves floors239/2219. The lambda10 search witness adds82
BRAID moves, giving excess82. The lambda1000 witness has excess32: one positive
stabilization,10 INSERT and10 BRAID moves. Both attain minimal CC2 but not minimal
native L10/L1000. Their costs differ by50 moves, but a single paired root with
different RNG/resource contexts is not a causal result about lambda conditioning.

### Bounded length AND strand-count extrapolation

Both newly solved raw inputs exceed the actual97 optimization support in BOTH
coordinates. The hash-verified121 census remains
`1541d03cff334a5f0377ea132746e07e986b98f7fc5987a8f735c78ccf86752d`:
metric maximum254 letters/21 strands; positive-policy maximum208/18. All four
complete relative groups were reread from unchanged118 selection SHA256
`3565cdf41970cd791a5b310f309ac314a23ea86611a0fb686e411219a6687271`;
their maxima are128/17,14/5,37/4,25/4. The evaluated Q254-final checkpoint still
has SHA256`c2a7131ed7a232871d6decc3816339f938dab6c192315f379466ef9a33ddb74f`.

Thus392/25 and407/28 are genuine bounded size extrapolation relative to protocol97
training, rather than merely accepting a larger tensor. This is not unseen-knot
certification relative to all Q254 history, arbitrary-size reliability, causal
embedding benefit, or proof the training improved search: the random-final arm,
full paired audits and final analysis remain pending, and this matrix has no
initial-model MCTS or no-embedding control. Engineering-only123/127 inputs did
not train this checkpoint. No TEST trajectories were imported into training or
the proof database; no protocol changes, extra training or promotion performed.

## Hourly update: 2026-09-22 11:19–11:20 UTC, prefix72 verified

Controller98832 and worker17857 remain the unique expected active processes;
worker CPU advanced995:24.67 to996:34.28 at99–100%. Controller9/13 jobs, random
search pending. No new error, duplicate worker, restart or protocol modification.
Next routine process inspection: no earlier than12:22 UTC.

Ninth shard `conditions-00064.json` committed at11:14:18.196175 UTC, SHA256
`64ba14ba73e03625fe3a5c6b86dda8d94e1e7313ff581fabbab0227488c00409`.
Recorded batch5223.031235s; manifest/journal updated11:14:19 UTC, journal3767
bytes. Saved coverage72/128 conditions (36 roots at both cost weights),419
episodes,222816 continuation moves,223183 full-path moves,38378942 policy action
scores. Ten successful conditions cover EIGHT distinct roots.

The incremental audit exited0. All1264 input pins, full128-root identity, nine
shard hashes and every saved query/RNG identity passed. Prior64 counts remain
bound to the eight unchanged previously audited shards; only the NEW eight
conditions were replayed:47 episodes,21259 continuation moves,21299 full-path
moves,3548390 action scores, one success. Adding these counts reproduces the
saved manifest; neural-state and committed-time checks also passed.

New success is condition64, root32, lambda10: raw length64, five strands,
determinant18603 and certified lowerCC1. Both found and reference paths have
1CC/35 moves, q0=34, excess0 (30 REDUCE,4 DESTABILIZE,1 CROSSING_CHANGE).
Found costL10=45 equals the certified floor, hence is optimal under this native
semantic-cost contract. Its L1000=1035 also reaches that floor, but the paired
lambda1000 SEARCH did not succeed; do not count it as another successful query.
Found and reference witnesses independently replayed with the prefix cost
identity and fresh integer determinant check. This is not a database improvement,
a new size-extrapolation claim, or causal evidence that embeddings improved
search. No TEST trajectory imported into training; full paired evaluation pending.

## Hourly update: 2026-09-22 12:22–12:23 UTC, compute active

The unique expected controller98832/worker17857 remain live; worker CPU advanced
1058:16.73 to1058:58.01 at98–100% utilization. No new logged error or duplicate
worker. No new committed batch:72/128 conditions, ten successful conditions on
eight roots, controller9/13 jobs; random search remains pending. All nine audited
shard hashes and the roots hash are unchanged. Manifest/journal last updated
11:14:19 UTC, journal3767 bytes. This establishes continued computation, not
additional saved results. No repeated replay, restart or protocol changes.
Next routine process inspection: no earlier than13:24 UTC.

## Hourly update: 2026-09-22 13:24–13:25 UTC, compute active

The unique expected controller98832/worker17857 remain live; worker CPU advanced
1120:12.53 to1120:51.54 at99% utilization. No new logged error or duplicate worker.
Saved coverage remains72/128 conditions, ten successful conditions on eight
roots; controller9/13 jobs, random search pending. All nine audited shard hashes
and the roots hash remain unchanged. Last manifest/journal update11:14:19 UTC,
journal3767 bytes. There is live computation but no new saved batch to replay.
No restart, source/protocol change or duplicate replay. Next routine inspection:
no earlier than14:26 UTC.

## Hourly update: 2026-09-22 14:26–14:27 UTC, prefix88 verified

Two new batches were committed since the previous hourly inspection. Controller
98832 and worker17857 remain the unique expected live processes; worker CPU
advanced1182:08.36 to1183:26.16 at98–99%. Controller9/13 jobs, random search not
started. No new errors, duplicate workers, restart or protocol changes. Next
routine process inspection: no earlier than15:29 UTC.

New immutable shards:

- `conditions-00072.json`, SHA256
  `9e9e9da9d7a1cb8db1854f12ea53095ef1c1d2ba41cab5f5c9fa17f4cd886387`,
  committed13:41:02.175077 UTC, recorded batch8802.401454s; no new success.
- `conditions-00080.json`, SHA256
  `38618ee2f900df3fff84e49a9edf223747073d086ba15d3bbfc19139f9307f3c`,
  committed14:17:59.951463 UTC, recorded batch2216.001786s; two new successes.

Latest manifest/journal update14:18:01 UTC, journal4561 bytes. Saved coverage is
88/128 conditions (44 roots at two cost weights),514 episodes,268503 continuation
moves,268951 full-path moves,46408891 action scores. Twelve successful conditions
cover TEN distinct roots.

Incremental audit exited0. All1264 input pins, full128-root identity, eleven
shard hashes and every saved query/RNG context passed. Earlier72 counts are bound
to nine unchanged previously audited shard hashes. All NEW16 conditions replayed:
95 episodes,45687 continuation moves,45768 full-path moves,8029949 action scores,
two successes. Combined counts match the saved manifest, including neural-state
and committed-time checks. No old search trajectories replayed again.

| Condition / root | Search lambda | Length / strands | Found CC / moves | Found L1000 | Certified optimal L1000 | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 81 / 40 | 1000 | 165 / 12 | 1 / 91 | 1091 | 1089 | 2 |
| 83 / 41 | 1000 | 196 / 21 | 2 / 112 | 2112 | 2110 | 2 |

Both found witnesses and references passed full prefix cost identities and fresh
independent integer determinant checks. Root40 determinant4115, lowerCC1, q0=88;
reference1CC/89 moves has zero excess. Root41 determinant2589, lowerCC2, q0=108;
reference2CC/110 moves has zero excess. Each found path has exactly one INSERT
and one extra reduction, hence excess2 and certified regret2. This accounting
does NOT authorize deleting two arbitrary actions without replay.

Both lambda10 searches failed within their frozen budgets. These are two new
roots, not four successes obtained by evaluating each witness under both costs.
Neither improves an existing witness/database bound or adds whole-model size
extrapolation (root41's21 strands are above policy-positive18 but within metric21).
No TEST import, checkpoint promotion or training change. Full paired comparison
and final replay/cost audit still pending.

## Hourly update: 2026-09-22 15:29–15:30 UTC, prefix96 verified

The unique expected controller98832/worker17857 remain live. Worker CPU advanced
1245:00.22 to1246:15.00 at99–100%; controller9/13 jobs, random search pending.
No new logged error, duplicate worker, restart or source/protocol changes.
Next routine process inspection: no earlier than16:32 UTC.

Twelfth shard `conditions-00088.json` committed at15:03:17.145555 UTC, SHA256
`c34f3f94a09ffb65faa50452361c14a686d9d491c6842719567d3e43e28b6f21`.
Recorded batch2715.518858s; manifest/journal updated15:03:18 UTC, journal4958
bytes. Saved coverage96/128 conditions (48 roots at both weights),559 episodes,
283366 continuation moves,283856 full-path moves,48145815 action scores.
Fourteen successful conditions cover ELEVEN distinct roots.

Incremental audit exited0. All1264 input pins,128-root identity, twelve shard
hashes and saved query/RNG identities passed. Prior88 counts remain bound to the
eleven unchanged audited hashes. NEW eight conditions fully replayed:45 episodes,
14863 continuation moves,14905 full-path moves,1736924 action scores and two
successes. Combined counts reproduce the manifest; neural-state/committed-time
accounting also passed. No repeated replay of old search trajectories.

Both new successes belong to root46 (length109, eight strands, determinant2823,
certified lowerCC1, q0=58):

| Condition | Search lambda | Found CC / moves | Search cost | Certified optimal cost | Regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| 92 | 10 | 1 / 67 | 77 | 69 | 8 |
| 93 | 1000 | 1 / 61 | 1061 | 1059 | 2 |

The reference1CC/59-move witness has zero excess and proves both cost floors.
Condition92 has eight BRAID moves (excess8); condition93 has one INSERT plus its
extra reduction (excess2). Both found paths and the reference replayed with all
prefix identities and independent integer determinant checks. Neither improves
the reference. These are two successful cost conditions of ONE new representation,
not two new roots. No new size-extrapolation claim, TEST training import, database
import, or checkpoint promotion; full paired search evaluation remains pending.

## Hourly update: 2026-09-22 16:32–16:33 UTC, prefix104 verified

Unique expected controller98832/worker17857 remain live. Worker CPU advanced
1307:54.50 to1309:03.88 at100%; controller9/13 jobs, random search pending.
No new logged error, duplicate worker, restart or protocol/source changes.
Next routine process inspection: no earlier than17:35 UTC.

Thirteenth shard `conditions-00096.json` committed at15:50:23.205503 UTC, SHA256
`7a7417a62bfa4d5e1067e7abe7ee47e0983d9e76a96ac37b0090b6fb82fb6d00`.
Recorded batch2824.481870s; manifest/journal updated15:50:24 UTC, journal5358
bytes. Saved coverage104/128 conditions (52 roots at two cost weights),604
episodes,298846 continuation moves,299378 full-path moves,50018811 action scores.
Seventeen successful conditions cover THIRTEEN distinct roots.

Incremental audit exited0. All1264 input pins,128-root identity, thirteen shard
hashes and all saved query/RNG contexts passed. Prior96 totals remain bound to
twelve unchanged audited hashes. All NEW eight conditions replayed:45 episodes,
15480 continuation moves,15522 full-path moves,1872996 action scores and three
successes. Combined counters reproduce the manifest; neural-state and committed
timing checks passed. No duplicate replay of old search trajectories.

| Condition / root | Search lambda | Length / strands | Found CC / moves | Search cost | Certified optimum | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 / 48 | 10 | 194 / 17 | 1 / 106 | 116 | 116 | 0 |
| 100 / 50 | 10 | 136 / 5 | 1 / 73 | 83 | 81 | 2 |
| 101 / 50 | 1000 | 136 / 5 | 1 / 71 | 1071 | 1071 | 0 |

Both new roots have certified lowerCC1 and zero-excess references. Root48:
determinant5739, q0=105, reference1CC/106 moves. Its found path attains the native
cost floor; paired lambda1000 search failed within its budget. Root50:
determinant54507, q0=70, reference1CC/71 moves. Lambda10 has two extra BRAID moves;
lambda1000 has zero excess and attains its floor. All found/reference witnesses
replayed with per-prefix identities and fresh independent integer determinants.

These are two new representations, with three successful cost conditions and
two new optimal-cost search results. No new database improvement or whole-model
size extrapolation, no TEST training import or checkpoint promotion. Full paired
evaluation and final independent replay/cost analysis remain pending.

## Hourly update: 2026-09-22 17:35–17:36 UTC, prefix112 verified

Expected controller98832 and worker17857 remain uniquely active. Worker CPU
advanced1370:47.83 to1372:00.24 at100%; no logged errors or duplicate workers.
Controller9/13 jobs, random search pending. No restart or source/protocol changes.
Next routine process inspection: no earlier than18:38 UTC.

Fourteenth shard `conditions-00104.json` committed at16:58:50.986564 UTC, SHA256
`401055b7066943bfb2cdb8750346256035d7d251f12d463f74ef555f8e869acd`.
Batch4106.216652s; manifest/journal updated16:58:52 UTC, journal5754 bytes.
Saved coverage112/128 conditions (56 roots at both weights),651 episodes,
318946 continuation moves,319519 full-path moves,52917259 action scores.
Eighteen successful conditions cover FOURTEEN distinct roots.

Incremental audit exited0: all1264 input pins,128-root identity, fourteen shard
hashes and all saved query/RNG contexts passed. Earlier104 totals remain bound
to thirteen unchanged audited hashes. NEW eight conditions replayed in full:
47 episodes,20100 continuation moves,20141 full-path moves,2898448 action scores,
one success. Combined counts reproduce the manifest; neural-state/committed-time
checks passed. Old search trajectories were not replayed again.

New success condition108/root54/lambda10: length164,17 strands, determinant8911,
certified lowerCC1, q0=90. Found1CC/93 moves has L10=103 and excess2 (one INSERT
and extra reduction). The reference1CC/91 moves has zero excess and certifies
the optimumL10=101, hence found regret2. Both paths passed full prefix identities
and independent integer determinant checks. Paired lambda1000 search failed
within budget. No reference improvement, new size extrapolation, TEST training
import or checkpoint promotion. Full paired evaluation remains pending.

## Hourly update: 2026-09-22 18:38–18:39 UTC, Q254 search complete; full audit active

The Q254-final search completed all128 conditions normally at approximately
18:26 UTC. Old worker17857 has exited. The unchanged controller98832 recorded
10/13 jobs complete and automatically launched `search-audit-q254`, PID97409,
PPID98832, using `scripts/audit_raster_log_search.py`. Its exact command and unique
process were checked; CPU advanced12:13.92 to12:39.47 at99–100%. No new error or
duplicate worker; audit log241 bytes contains only the known optional tkinter
warning. This is the intended full path/resource/count/cost audit, NOT a stalled
search or a reason to restart. Random-final search remains the controller's next
stage after that audit. Next routine inspection: no earlier than19:41 UTC.

Final search manifest SHA256:
`a88c3f09f54bdee789670df0cdd37eed3d2c45b139955b45b87d3803227d88a6`, matching the
controller's completed-job record. All16 shard hashes, unchanged roots hash and
exact128-condition coverage were checked; all746 recorded episodes counted.
The two newly completed shards are:

- `conditions-00112.json`, SHA256
  `66cf99fe7df68c48cc52f6ae854e76a70a34570fe92ff2b9339e8ff8b3340504`,
  committed17:40:05.539915 UTC, recorded batch2472.924500s.
- `conditions-00120.json`, SHA256
  `24582e0bde19116a314f58db0621c1526c314337a106a3934da862bb0ded1c82`,
  committed18:26:07.193627 UTC, recorded batch2760.063751s.

Final search manifest/journal updated18:26:09/18:26:08 UTC; search journal6551
bytes. Recorded totals:746 episodes,351448 continuation moves,352103 full-path
moves,55953078 action scores;85743.710965s across committed batches. Neural
measurements:351448 policy-state evaluations/181676 batches and3072 tree
embedding evaluations/1584 batches,5589 tree action scores. These are recorded
measurements, not an independent neural rerun or unique-state/game counts.

Recorded search success is9/64 at lambda10 and11/64 at lambda1000,20 successful
conditions covering16 distinct representations. The last two newly reported
successes are condition120/root60/lambda10 (reportedL10=110) and
condition125/root62/lambda1000 (reportedL1000=1110). Their final legal replay and
cost conclusions are deliberately deferred to the already running full audit;
no duplicate incremental audit was launched. Earlier112 conditions have the
documented incremental checks; do not mislabel the full128 as already replayed
while the full audit is unfinished.

This completes ONE search arm, not the paired experiment or the embedding goal.
No comparison to unmatched older architectures establishes a causal benefit.
Wait for the full audit, random-final128-condition search/audit and final summary;
no checkpoint promotion, protocol change, TEST import or database release.

## Hourly update: 2026-09-22 19:41–19:42 UTC, Q254 audit passed; random search active

Full Q254 replay/cost audit passed, SHA256
`9ccc083ea71f2ae7378d03beea5f4471b8545ac16222ebd41b0ad84416e0291f`.
All1287 audit input pins and independent JSON cost/coverage arithmetic rechecked.
All128 conditions/746 episodes/20 solutions are now fully replay-verified.
Outcome9/64 L10 and11/64 L1000,16 distinct representations; four optimal-cost
search solutions, zero reference improvements. Detailed completed-arm report131
records counts, certified/unknown regrets, size extrapolation and limitations.

Controller98832 completed11/13 jobs and automatically started random-final search,
worker2020, PPID98832, correct unique command/cwd paths. CPU advanced59:01.27 to
60:00.61 at98–99%. Old Q254 search/audit processes exited normally. The random
manifest was created18:42:12 UTC; next_condition0 and no first shard yet. Log241
bytes, last updated18:41:29 UTC, only the known optional tkinter warning.
Its final trained checkpoint hash17433748fac7308cf4c36ef0c40650c7ec83d58dee8625834ce531c2f5bb40c6
and common root hashc1da996e1e6e95251986dd04bdad1467d45ad3f07055fc2b6ca79424131d7967
were verified. No duplicate worker/replay, new training or protocol modification.
Next routine process inspection: no earlier than20:44 UTC.

## Hourly update: 2026-09-22 20:44–20:45 UTC, random search computing first batch

Controller98832 still has11/13 jobs complete and identifies search-random,
worker2020, PPID98832. The live command uses the trained random arm and canonical
pgx-mcts-bench checkout; the anchored process inventory contains exactly this
worker and its controller, without a duplicate. Worker CPU time advanced from
112:42.10 to113:30.19 over approximately49 wall seconds, utilization95–99%.
This verifies computation, not saved search progress or successful solutions.

No condition shard has yet committed: next_condition0, counts empty. Manifest
179955 bytes and roots399247 bytes remain dated18:42:12 UTC. Root SHA256 remains
`c1da996e1e6e95251986dd04bdad1467d45ad3f07055fc2b6ca79424131d7967`.
The search log remains241 bytes dated18:41:29 UTC, containing only the known
optional tkinter warning and no logged failure. No new replay was necessary.
The first eight-condition batch is still computing; no restart, interruption,
protocol modification or additional worker was performed. Next routine process
inspection: no earlier than21:46 UTC.

## Hourly update: 2026-09-22 21:46 UTC, random first16 conditions replay-verified

The controller remains98832,11/13 jobs complete, current search-random worker2020
with matching parent and canonical pgx-mcts-bench command. Only one search worker
was present. CPU advanced173:44.04 to174:09.48 in approximately26 seconds,
utilization97–98%. The manifest now records next_condition16, with two durable
shards. Manifest180667 bytes and log1016 bytes updated21:26:59 UTC; no logged
failure, only the known optional tkinter warning and two batch progress entries.
Next routine inspection: no earlier than22:48 UTC; no intermediate search polling.

New immutable shards (UTC timestamps):

- `conditions-00000.json`,1708261 bytes,20:52:30 UTC,
  SHA256 `474acab8681afc67ae483e66a232fd9e04f77f674e5983b03c48ed2820f9160a`,
  recorded batch7373.048841750249 seconds.
- `conditions-00008.json`,1224748 bytes,21:26:58 UTC,
  SHA256 `545ca51ec49d1cec83b34951d628ef9f782244e3e0117068e0e422d7ecf6f87f`,
  recorded batch2065.432165041566 seconds.

A read-only incremental audit completed with exit0. It checked all1264 manifest
input pins twice, the completed trained random checkpoint and pair-audit lineage,
the unchanged common root SHA, exact128 reconstructed roots/resources, both shard
hashes, exact prefix0..15 coverage, per-condition seeds and every recorded path.
All95 episodes replayed:40594 continuation moves,40674 full-path moves,
6027428 policy action scores, two successful conditions. Recorded neural counters
are40594 policy-state evaluations/20744 batches,375 tree embedding evaluations/
195 batches and698 tree action scores. Recorded committed time9438.481007 seconds.
These neural/timing counters were checked for consistency, not independently
rerun. No duplicate full audit or new search worker was launched.

Both newly found solutions also passed legal witness replay, every-prefix
semantic cost identity, reference replay and independent integer determinant
checks. Signature bounds use the previously audited frozen panel:

- Condition9/root4/lambda1000: raw147 letters/8 strands, determinant13021,
  certified lower CC1, q0=77. Found1CC/80 moves gives L1000=1080, versus
  certified-optimal reference1CC/78 moves,1078. Proven regret2; excess2 consists
  of one INSERT (with the corresponding extra REDUCE).
- Condition10/root5/lambda10: raw165 letters/14 strands, determinant537,
  certified lower CC1, q0=89. Found1CC/94 moves gives L10=104, versus
  certified-optimal reference1CC/90 moves,100. Proven regret4; excess4 consists
  of two INSERT operations (and the corresponding extra REDUCE operations).

Partial outcome is1/8 successes at each lambda across the first eight roots;
14/16 conditions are budget-censored failures, not unsolvability or bad-move
labels. Neither success improves the reference. Do not extrapolate these partial
counts to the full128 or claim a causal embedding/training/initialization benefit.
The paired comparison still awaits all random conditions and its final audit.

## Hourly update: 2026-09-22 22:48 UTC, random next batch still computing

Controller98832 remains at11/13 completed jobs and identifies the unique
search-random worker2020, PPID98832, with the unchanged canonical checkout and
trained-random command. CPU advanced235:13.45 to235:50.54 in37 wall seconds at
99–100%. No duplicate search/audit processes were present. This is a verified
computational wait, not a new saved result or a stalled/dead-worker diagnosis.

The committed prefix remains16 conditions/95 episodes/two solutions. Manifest
180667 bytes and log1016 bytes are still dated21:26:59 UTC; no third shard or
new logged error. The two previously replayed shard hashes and common roots hash
were rechecked unchanged; no trajectories were redundantly replayed. No restart,
new training, protocol mutation, checkpoint promotion or database import.
Next routine inspection: no earlier than23:50 UTC.

## Hourly update: 2026-09-22 23:50–23:51 UTC, random24 conditions verified

Controller98832 still reports11/13 completed jobs and the unique random-final
search worker2020, PPID98832, with the unchanged canonical checkout command.
CPU advanced296:56.07 to297:56.28 over60 wall seconds at97–99%. No duplicate
search/audit worker or new logged error. The manifest advanced to24 conditions;
manifest180842 bytes and log1408 bytes updated23:12:04 UTC.

New shard `conditions-00016.json`,1852051 bytes, committed23:12:02 UTC,
SHA256 `f358a0015ba68937971dccfdf5f0f52e8351507996f4d995770171019ecc030e`,
recorded batch6302.4119262499735 seconds. Earlier two shard hashes remain unchanged.
The read-only incremental audit finished exit0:1264 input pins checked twice,
trained-checkpoint/pair-audit lineage, common roots and reconstructed resources,
all shard hashes, exact0..23 coverage and all query/RNG identities verified.
Only new conditions16..23 were replayed:47 episodes,24281 continuation moves,
24322 full-path moves,3697909 action scores and one successful condition.
Combined verified prefix:24 conditions,142 episodes,64875 continuation moves,
64996 full-path moves,9725337 action scores, three solutions. Recorded neural
counts64875 policy evaluations/33420 batches and577 tree embedding evaluations/
300 batches,1053 tree action scores; committed batch time15740.892933 seconds.
Neural/timing measurements were not independently rerun.

New success condition23/root11/lambda1000 passed legal replay, reference replay,
every-prefix semantic cost identity and independent determinant verification:
raw210 letters/3 strands, determinant187614808503, certified lower CC1, q0=106.
Reference2CC/108 moves has L1000=2108; found2CC/126 moves has L1000=2126.
Excess18 consists of7 INSERT operations and4 BRAID operations. The gap18 is
relative to the reference, NOT certified optimality regret: the certified lower
cost is1107 and the reference itself is not proven optimal (u remains[1,2]).
The necessary move envelope for a1CC solution no worse than reference is1108,
versus current cap456; this resource limitation neither proves such a path
exists nor justifies changing this frozen evaluation.

Partial outcomes:1/12 at lambda10 and2/12 at lambda1000;21/24 conditions remain
budget-censored failures. No reference improvement or causal embedding claim.
The existing controller continues; no duplicate replay, training, restart or
protocol changes. Next routine inspection: no earlier than2026-09-23 00:53 UTC.

## Hourly update: 2026-09-23 00:53 UTC, random search still computing

The controller98832 remains at11/13 completed jobs with the unique random-final
worker2020, PPID98832, and the unchanged canonical checkout command. CPU advanced
359:38.27 to360:14.96 over37 wall seconds at99–100%. This confirms computation;
it is not evidence of an additional saved result. No duplicate worker was found.

The committed prefix remains24 conditions/142 episodes/three verified solutions.
Manifest180842 bytes and log1408 bytes remain dated2026-09-22 23:12:04 UTC, with
no new logged failure. The root hash and all three previously replayed shard
hashes are unchanged. No repeated trajectory audit, restart, training, protocol
mutation, promotion or database import. Next routine inspection: no earlier
than2026-09-23 01:55 UTC.

## Hourly update: 2026-09-23 01:55–01:56 UTC, random32 conditions verified

The unchanged controller98832 has11/13 jobs complete and identifies unique
search-random worker2020, PPID98832, using the canonical pgx-mcts-bench checkout.
CPU advanced421:21.65 to422:10.60 over49 wall seconds at99–100%. No duplicate
search/audit worker. Manifest181012 bytes and log1800 bytes updated01:19:54 UTC,
now recording32 completed conditions; no new logged failure.

New shard `conditions-00024.json`,1783134 bytes, committed01:19:52 UTC,
SHA256 `6b227d5ac387a30257b08eaf05f9ad2e647c5ee594d77a060cb73055b3bcf031`,
recorded batch7668.163366875611 seconds. All earlier shard hashes are unchanged.
Read-only incremental audit finished exit0:1264 manifest input pins checked
twice, trained checkpoint/pair-audit lineage, common roots with exact128-root
reconstruction, all shard hashes, prefix0..31 coverage and query/RNG identities
passed. Only conditions24..31 were replayed:48 episodes,27320 continuation moves,
27360 full-path moves,5346943 policy action scores, no successful conditions.

Combined verified prefix:32 conditions/190 episodes/three solutions,
92195 continuation moves,92356 full-path moves,15072280 action scores. Recorded
neural counters92195 policy evaluations/47080 batches and757 tree embedding
evaluations/394 batches,1397 tree action scores. Committed time23409.056300 seconds.
These compute counters are consistency-checked measurements, not a neural rerun.
Partial search success is1/16 at lambda10 and2/16 at lambda1000;29 conditions are
budget-censored, not proofs of unsolvability or intrinsically bad trajectories.

No restart, duplicate replay, protocol change, training, promotion or database
import. The existing paired evaluation continues; next routine process inspection
no earlier than2026-09-23 02:58 UTC.

## Hourly update: 2026-09-23 02:58–02:59 UTC, random40 conditions verified

Controller98832 continues with11/13 jobs complete and the unique random-final
worker2020, PPID98832, using the unchanged canonical checkout command. CPU
advanced484:17.67 to485:08.04 in approximately51 wall seconds at98–99%. No duplicate
search/audit worker or logged failure. Manifest181184 bytes and log2193 bytes
updated01:59:58 UTC, with committed prefix40.

New shard `conditions-00032.json`,1083141 bytes, committed01:59:56 UTC,
SHA256 `970003ea051f735bc2b8ff20bb9055e21c3200cfa19c2cda68773d59ca8e359d`,
recorded batch2402.4464187501 seconds. The incremental audit completed exit0:
1264 manifest pins checked twice, trained checkpoint and pair-audit lineage,
common roots and exact reconstructed resources, all five shard hashes,0..39
coverage and RNG/query identities passed. Earlier four shards remained unchanged;
only conditions32..39 were replayed. Their48 episodes contain17000 continuation
moves,17040 full-path moves,1403669 action scores and no successful condition.

Combined verified prefix:40 conditions,238 episodes,109195 continuation moves,
109396 full-path moves,16475949 policy action scores, three successful conditions.
Recorded neural counts109195 policy-state evaluations/55580 batches,938 tree
embedding evaluations/487 batches,1743 tree action scores; committed time
25811.502719 seconds. Neural/timing counts are not independently rerun compute.
Partial outcomes remain1/20 at lambda10 and2/20 at lambda1000;37 conditions are
budget-censored failures, not unsolvability or bad-trajectory labels.

No restart, extra training, protocol change, checkpoint promotion or database
import. Next routine process inspection: no earlier than2026-09-23 04:01 UTC.

## Hourly update: 2026-09-23 04:01 UTC, random next batch computing

Controller98832 remains at11/13 completed jobs with unique random-final worker2020,
PPID98832, and the unchanged canonical checkout command. CPU advanced547:08.82
to547:53.16 over44 wall seconds at99%. This confirms active computation, not new
saved progress. No duplicate search/audit worker was found.

Committed prefix remains40 conditions/238 episodes/three verified solutions.
Manifest181184 bytes and log2193 bytes remain dated01:59:58 UTC; no new logged
failure. All five previously replayed shard hashes and common roots hash were
rechecked unchanged. No redundant replay, restart, training, protocol change,
promotion or database import. Next routine process inspection: no earlier
than2026-09-23 05:03 UTC.

## Hourly update: 2026-09-23 05:03–05:04 UTC, random48 conditions verified

Controller98832 remains at11/13 completed jobs, with unique random-final worker2020,
PPID98832, using the unchanged canonical checkout command. CPU advanced609:04.67
to610:05.81 over61 wall seconds at98–100%. No duplicate search/audit worker or
new logged failure. Manifest181357 bytes and log2589 bytes updated04:21:52 UTC,
with committed prefix48.

New shard `conditions-00040.json`,2081240 bytes, committed04:21:50 UTC,
SHA256 `0c9da73e9effc57b2b9ca79defd81e29e660f032a97dd81a966e7772cb778030`,
recorded batch8512.647248750553 seconds. Incremental audit finished exit0:
1264 input pins checked twice, trained checkpoint/pair-audit lineage, common root
hash and exact128 reconstructed roots/resources, all six shard hashes, prefix
coverage and query/RNG identities verified. Earlier five shards were unchanged;
only new conditions40..47 were replayed. Their48 episodes contain29856
continuation moves,29896 full-path moves,4497005 action scores and one solution.

New success condition40/root20/lambda10: raw163 letters/16 strands,
independent determinant36589, certified lower CC1, q0=89. Found1CC/92 moves
gives L10=102 versus certified-optimal reference1CC/90 moves,L10=100. Proven
regret2; excess consists of two BRAID operations. Found/reference legal replay,
every-prefix semantic cost identity and determinant/panel-bound consistency
passed. This is not a reference improvement or a successful lambda1000 search.

Combined verified prefix:48 conditions,286 episodes,139051 continuation moves,
139292 full-path moves,20972954 action scores and four successful conditions.
Recorded neural counts139051 policy-state evaluations/70656 batches and1121 tree
embedding evaluations/582 batches,2093 tree action scores; committed time
34324.149967 seconds. Neural/timing counts are not independently rerun compute.
Partial outcomes2/24 at each lambda;44 budget-censored failures. Full paired
comparison remains pending, with no causal embedding/initialization conclusion.

No restart, new training, protocol mutation, promotion or database import.
Next routine process inspection: no earlier than2026-09-23 06:06 UTC.

## Hourly update: 2026-09-23 06:06 UTC, random search computing next batch

Controller98832 remains at11/13 completed jobs with unique random-final worker2020,
PPID98832, using the unchanged canonical checkout command. CPU advanced672:04.28
to672:45.38 over41 wall seconds at99–100%. Computation is active; this does not
establish new saved progress. No duplicate search/audit worker was found.

The committed prefix remains48 conditions/286 episodes/four verified solutions.
Manifest181357 bytes and log2589 bytes remain dated04:21:52 UTC, with no new logged
failure. All six previously replayed shard hashes and common roots hash were
checked unchanged. No repeated trajectory replay, restart, training, protocol
change, promotion or database import. Next routine inspection: no earlier
than2026-09-23 07:08 UTC.

## Hourly update: 2026-09-23 07:08 UTC, random next batch remains active

Controller98832 still reports11/13 completed jobs with unique random-final
worker2020, PPID98832, and unchanged canonical checkout command. CPU advanced
733:57.84 to734:35.56 over approximately38 wall seconds at98–99%, confirming
computation without claiming additional saved output. No duplicate worker.

Committed prefix remains48 conditions/286 episodes/four verified solutions.
Manifest181357 bytes and log2589 bytes remain dated04:21:52 UTC, without a new
logged failure. All six previously replayed shard hashes and common roots hash
remain unchanged. No redundant replay, restart, new training, protocol change,
promotion or database import. Next routine process inspection: no earlier
than2026-09-23 08:10 UTC.

## Hourly update: 2026-09-23 08:10–08:11 UTC, random batch not yet committed

Controller98832 remains at11/13 completed jobs with the unique random-final
worker2020, PPID98832, using the unchanged canonical checkout command. CPU
advanced795:51.67 to796:40.50 over49 wall seconds at98–99%. The worker is computing;
CPU advancement alone does not establish new conditions or successful solutions.
No duplicate search/audit worker was present.

Committed prefix remains48 conditions/286 episodes/four verified solutions.
Manifest181357 bytes and log2589 bytes are still dated04:21:52 UTC, with no new
logged failure. The next batch has not committed after approximately3h49m;
elapsed time alone is not evidence of a dead worker or authorization to restart.
All six prior shard hashes and the common root hash remain unchanged. No repeated
trajectory audit, restart, new training, protocol mutation, promotion or database
import. Next routine inspection: no earlier than2026-09-23 09:13 UTC.

## Hourly update: 2026-09-23 09:13–09:14 UTC, random56 conditions verified

Controller98832 remains at11/13 completed jobs with unique random-final
worker2020, PPID98832, using the unchanged canonical checkout command. CPU advanced
858:41.71 to859:51.72 over70 wall seconds at99–100%. No duplicate search/audit
worker or new logged failure. Manifest181530 bytes and log2986 bytes updated
08:13:30 UTC, now recording56 conditions. The formerly long-running batch
completed normally; no intervention or restart was needed.

New shard `conditions-00048.json`,2418305 bytes, committed08:13:28 UTC,
SHA256 `a8427d0253e96e855dc9ed9f4a4666ca0afe39a5cd0f60270b0445db1e156d66`,
recorded batch13896.025949832983 seconds. Incremental audit finished exit0:
1264 input pins checked twice, trained checkpoint/pair-audit lineage, common
root hash with exact128-root/resource reconstruction, all seven shard hashes,
prefix coverage and query/RNG identities passed. Earlier six shards remained
unchanged; only new conditions48..55 were replayed. Their48 episodes contain
36776 continuation moves,36816 full-path moves,9744147 action scores and no
successful condition.

Combined verified prefix:56 conditions,334 episodes,175827 continuation moves,
176108 full-path moves,30717101 action scores and four successful conditions.
Recorded neural counts175827 policy-state evaluations/89044 batches and1302
tree embedding evaluations/675 batches,2439 tree action scores; committed time
48220.175917 seconds. Neural/timing counts were not independently rerun.
Partial outcomes2/28 at each lambda;52 budget-censored failures, not proofs of
unsolvability. No full-pair comparison or causal embedding claim yet.

No new training, protocol change, promotion or database import. Next routine
process inspection: no earlier than2026-09-23 10:16 UTC.

## Hourly update: 2026-09-23 10:16 UTC, random next batch computing

Controller98832 remains at11/13 completed jobs and identifies unique random-final
worker2020, PPID98832, with unchanged canonical checkout command. CPU advanced
921:33.97 to922:14.85 over41 wall seconds at95–99%. Computation is active; no
additional saved results are claimed. No duplicate search/audit worker was found.

Committed prefix remains56 conditions/334 episodes/four verified solutions.
Manifest181530 bytes and log2986 bytes remain dated08:13:30 UTC, with no new
logged failure. All seven previously replayed shard hashes and common roots
hash were checked unchanged. No redundant replay, restart, new training,
protocol mutation, promotion or database import. Next routine inspection:
no earlier than2026-09-23 11:18 UTC.

## Hourly update: 2026-09-23 11:18 UTC, current random block still computing

After waiting for the scheduled inspection, controller98832 still identifies
search-random worker2020, PPID98832, with11/13 jobs completed. The process scan
found exactly the controller and its expected search worker, not a duplicate.
Worker CPU advanced983:27.18 to984:07.82 over approximately41 wall seconds,
at98.7–100% CPU; the second sample was11:18:58 UTC and stateR.

The saved prefix is unchanged at56 conditions,334 episodes and four verified
solutions. Manifest181530 bytes and log2986 bytes both retain08:13:30 UTC
mtime. The current block has not committed after about3h05m. This is an active
computation with no new durable result, not evidence of a stopped process.
All seven committed shard hashes and the common root hash remain unchanged;
no repeated trajectory replay or full audit was launched. No new logged error,
restart, source mutation, model change or additional search worker.

Independent public-source and database-readiness work is recorded in132/133;
its artifacts were kept outside current97's inputs. It did not alter the run.
Next routine process inspection: no earlier than2026-09-23 12:20 UTC.

## Hourly update: 2026-09-23 12:20–12:21 UTC, no newly committed random shard

Controller98832 remains at11/13 jobs, with the unique expected random-final
worker2020 and unchanged canonical PGX command. Worker CPU advanced1045:18.49
to1046:07.41 over approximately49 seconds, at99.7–100%; the last sample was
12:21:04 UTC, stateR. The worker remains live and computing. This does not
establish advancement within its unsaved block.

Saved progress is still56 conditions/334 episodes/four verified solutions.
Manifest181530 bytes and log2986 bytes retain08:13:30 UTC mtime, now about4h07m
without a committed block. All seven prior shard hashes and the common root
hash are unchanged. No new logged error or duplicate search/audit process.
For context only, the completed Q254 arm spent10680.716454 seconds (~2h58m)
on conditions56..63; different policy trajectories can take different time,
so this does not supply a random-arm completion deadline or a failure test.

No replays, restart, source/model mutation, extra training or database write.
This is a verified computational wait, not a terminal blocker or completed
paired evaluation. Next routine inspection: no earlier than2026-09-23 13:23 UTC.

## Hourly update: 2026-09-23 13:23 UTC, random64 conditions replayed

The scheduled inspection found the same unique controller98832 and random-final
worker2020 under it, with11/13 controller jobs complete. Worker CPU advanced
1108:11.90 to1109:24.88 between13:23:15 and13:24:28 UTC, at99.8–100%, stateR.
The long-running block completed without intervention; the worker is now
computing beyond the saved prefix. No duplicate search/audit worker or new
search-log error was found.

Manifest181702 bytes and log3384 bytes were updated12:53:19 UTC, recording
64/128 conditions. New shard `conditions-00056.json`,2650228 bytes, committed
12:53:17 UTC, SHA256
`5f05ae088fbe748033cff99349444625a80747e7673a902f692160fe078be873`,
records16786.948101 seconds (~4h40m). Its duration was not a stalled-process
diagnosis; it produced the expected eight complete condition records.

Incremental replay finished exit0:1264 input pins checked twice; audited training
pair and final random-checkpoint lineage; fixed128-root reconstruction and hash;
all eight shard hashes; exact prefix/query/RNG identities; and all48 new episodes
passed. Only new conditions56..63 were trajectory-replayed. They contain39992
continuation moves,40032 full-path moves and12391458 action scores, with no
successful solution. The seven previously audited shards remained unchanged.

Verified prefix totals:64 conditions,382 episodes,215819 continuation moves,
216140 full-path moves,43108559 action scores and four successful conditions.
Recorded compute totals:215819 policy-state evaluations/109040 batches,
1485 tree-embedding evaluations/770 batches,2789 tree action scores; committed
batch time65007.124019 seconds. These compute/timing counts were checked for
consistency, not rerun as neural inference. Partial success remains2/32 at each
lambda, with60 budget-censored outcomes, not proofs of impossibility.

No new training, protocol mutation, model promotion, database import or release.
The independent paired evaluation remains incomplete. Next routine inspection:
no earlier than2026-09-23 14:31 UTC.

## Hourly update: 2026-09-23 14:31 UTC, random72 with two new L1000 solutions

Controller98832 is still at11/13 jobs with its unique expected worker2020 and
unchanged canonical PGX command. Worker CPU advanced1176:04.90 to1177:24.94
between14:31:14 and14:32:34 UTC, at98.8–100%, stateR. No duplicate search/audit
worker or new search-log failure was found. Between hourly inspections only
the clock was polled; no experiment input or process was changed.

Manifest181873 bytes and log3781 bytes updated14:14:39 UTC, now72/128 conditions.
New shard `conditions-00064.json`,1857322 bytes, committed14:14:37 UTC, SHA256
`2cac9d0a9d487d89871a5f4b2f138b738caae6317d8ee06cf4bab340bd30b7e3`,
records4878.347859 seconds. All eight prior shard hashes remain unchanged.

Incremental audit finished exit0. It checked1264 input pins twice, audited
training-pair/checkpoint lineage, exact128-root reconstruction and common hash,
all nine shard hashes, query/RNG identities and compute/timing consistency.
Only new conditions64..71 were episode-replayed:46 episodes,19996 continuation
moves,20037 full-path moves,3293348 action scores and two successful conditions.
Both new solutions and their references additionally passed full witness/prefix
cost-identity checks and independently recomputed integer determinants.

| Condition / root | Objective | Input length / strands | Found CC / moves / cost | Reference cost | Interpretation |
| --- | --- | --- | --- | --- | --- |
|69 /34|L1000|203 /10|1 /131 /1131|1107|Reference is certified optimal; regret24 moves, from12 INSERTs.|
|71 /35|L1000|204 /5|2 /114 /2114|2106|u interval[1,2]; gap8 to reference, not proven regret8.|

Root34 has determinant1988147, CC lower1 andq0=106; reference1CC/107 moves
attains its cost floor1107. Root35 has determinant121471, CC lower1 andq0=104;
reference2CC/106 moves is not certified optimal, with cost floor1105. Its found
path has two INSERTs and four BRAIDs, giving excess8. Neither solution improves
the reference CC upper bound or L1000 cost. These are synthetic TEST roots,
not new named-public-knot bounds or evidence for a database release.

Verified totals:72 conditions,428 episodes,235815 continuation moves,236177
full-path moves,46401907 action scores and six successful conditions on six
distinct roots. Partial success2/36 atL10,4/36 atL1000;66 budget-censored outcomes.
Recorded neural counts235815 policy-state evaluations/119056 batches,
1692 tree-embedding evaluations/876 batches,3141 tree action scores; committed
time69885.471877 seconds. Neural/timing counts were not independently rerun.

No new training, protocol/source mutation, model promotion, database import or
release. No causal embedding or final paired-comparison claim. Next routine
inspection: no earlier than2026-09-23 15:36 UTC.

## Hourly update: 2026-09-23 15:36–15:37 UTC, next random block computing

Controller98832 remains at11/13 jobs with its unique expected worker2020 and
unchanged canonical checkout command. Worker CPU advanced1241:03.83 to1241:59.33
between15:36:19 and15:37:14 UTC, at99.1–100%, stateR. This confirms live
computation, not advancement within the unsaved block. No duplicate search/audit
worker was found.

Saved progress remains72/128 conditions,428 episodes and six verified solutions.
Manifest181873 bytes and log3781 bytes retain14:14:39 UTC mtime; the current
block has not committed after approximately1h22m. No new logged error. All nine
previously replayed shard hashes and the common root hash were checked unchanged.
No repeated replay, restart, training/source/configuration change or database
write. This is a verified computational wait, not a stopped process or final
paired result. Next routine inspection: no earlier than2026-09-23 16:39 UTC.

## Hourly update: 2026-09-23 16:39–16:40 UTC, random block remains active

Controller98832 still identifies the unique expected worker2020, with11/13
jobs completed and unchanged canonical PGX command. Worker CPU advanced
1303:54.69 to1304:50.74 between16:39:16 and16:40:13 UTC, at99.7–100%, stateR.
This is a verified live computation, not proof of within-block advancement.

The saved prefix remains72 conditions/428 episodes/six verified solutions.
Manifest181873 bytes and log3781 bytes retain14:14:39 UTC mtime, approximately
2h25m without a new committed block. No duplicate search/audit worker or new
logged failure. All nine prior shard hashes and the frozen root hash remain
unchanged; no redundant trajectory replay was run. No restart, training change,
configuration mutation, checkpoint promotion or database write. The paired
evaluation is still incomplete. Next routine inspection: no earlier than
2026-09-23 17:42 UTC.

## Hourly update: 2026-09-23 17:42 UTC, random88 conditions verified

Controller98832 remains at11/13 jobs with its unique expected worker2020 and
unchanged canonical PGX command. Worker CPU advanced1366:41.68 to1367:43.70
between17:42:14 and17:43:16 UTC, at98.2–100%, stateR. No duplicate search/audit
worker or new search-log failure. Two new blocks committed; the earlier long
block completed without intervention.

Manifest182216 bytes and log4576 bytes updated17:23:06 UTC, now88/128 conditions.
The new immutable shards are:

| Shard | Bytes | Commit UTC | SHA256 | Recorded seconds |
| --- | ---: | --- | --- | ---: |
|conditions-00072.json|1958065|16:39:45|`3d74225cd0cd9c6b9f738eb43db90b32d6c311ce380e31c619f356da13f8fb58`|8706.807860|
|conditions-00080.json|1092407|17:23:04|`422fe7c934aeb2f9582118a7365692ce04493cd31889eea5fa3008b196ad0277`|2597.400809|

Incremental audit finished exit0:1264 input pins checked twice; audited training
pair and checkpoint lineage; exact128-root reconstruction/common hash; all11
shard hashes; query/RNG identities; and all96 new episodes passed. Only new
conditions72..87 were trajectory-replayed. Their totals are47008 continuation
moves,47088 full-path moves and7959088 action scores, with no new solution.
The first block contributes30056/30096 continuation/full moves and6435451 scores;
the second16952/16992 moves and1523637 scores. Each contains48 episodes.
The nine earlier audited shards remain unchanged.

Verified prefix:88 conditions,524 episodes,282823 continuation moves,283265
full-path moves,54360995 action scores and six successful conditions. Partial
success2/44 atL10 and4/44 atL1000;82 budget-censored outcomes, not impossibility
proofs or automatic negative action labels. Recorded neural counts282823 policy
evaluations/142560 batches,2051 tree-embedding evaluations/1061 batches,
3827 tree action scores; committed time81189.680545 seconds. Compute/timing
counters were checked for consistency, not independently rerun neural inference.

No restart, new training, source/configuration change, model promotion or
database write. The final paired evaluation and embedding-utility decision
remain incomplete. Next routine inspection: no earlier than2026-09-23 18:48 UTC.

## Hourly update: 2026-09-23 18:48 UTC, random96 with a seventh solution

Controller98832 remains at11/13 jobs, with unique expected worker2020 and the
unchanged canonical PGX command. Worker CPU advanced1432:21.94 to1433:40.68
between18:48:15 and18:49:34 UTC, at98.8–100%, stateR. No duplicate search/audit
worker or new search-log failure was found.

Manifest182388 bytes and log4975 bytes updated18:11:10 UTC, recording96/128
conditions. New shard `conditions-00088.json`,1157041 bytes, committed18:11:08 UTC,
SHA256 `175fa4d83704718a98e64013b5bf2c0269e796e0e1d8e0de46bd8780d236c955`,
records2882.448512 seconds. All11 earlier audited shard hashes remain unchanged.

Incremental audit finished exit0:1264 input pins checked twice; audited training
pair and checkpoint lineage; exact128-root reconstruction/common hash; all12
shard hashes; query/RNG identities; and all47 new episodes passed. Only new
conditions88..95 were trajectory-replayed:16609 continuation moves,16649 full-path
moves,1752584 action scores and one solution. Compute/timing accounting passed.

New successful condition91, root45, L1000:103 letters/6 strands,1CC/57 moves,
cost1057. The reference is1CC/55 moves, cost1055. Independent determinant1552881
certifies nontriviality and hence CC lower1; q0=54 makes1055 the semantic cost
floor. The stored pair interval itself is[0,1]; the determinant supplies the
stronger lower bound used by this cost audit. Both reference and found witness
passed full replay and every-prefix cost identity. One INSERT produces excess2,
so the certified regret is2 moves. This does not improve the reference upper
bound or cost, and is a synthetic TEST result, not a new public-knot theorem.

Verified totals:96 conditions,571 episodes,299432 continuation moves,299914
full-path moves,56113579 action scores and seven successful conditions on seven
distinct roots. Partial success2/48 atL10 and5/48 atL1000;89 budget-censored
outcomes. Recorded compute:299432 policy-state evaluations/151084 batches,
2245 tree-embedding evaluations/1162 batches,4179 tree action scores; committed
time84072.129057 seconds. Neural/timing counts were not independently rerun.

No restart, new training, protocol/source mutation, model promotion or database
write. The paired evaluation remains incomplete; no causal embedding conclusion.
Next routine inspection: no earlier than2026-09-23 19:53 UTC.

## Hourly update: 2026-09-23 19:53 UTC, random104 and first certified cost optimum

Controller98832 remains at11/13 jobs with unique expected worker2020 and the
unchanged canonical PGX command. Worker CPU advanced1496:57.04 to1498:14.67
between19:53:18 and19:54:36 UTC, at98.1–99.2%, stateR. No duplicate search/audit
worker or new search-log failure was found.

Manifest182562 bytes and log5376 bytes updated19:04:48 UTC, recording104/128
conditions. New shard `conditions-00096.json`,1288923 bytes, committed19:04:46 UTC,
SHA256 `5e0019419910fc8cf0404b44817751a64b5523aa9f3aec275168f0a97f287a6f`,
records3215.999684 seconds. All12 earlier audited shard hashes remain unchanged.

Incremental audit finished exit0:1264 input pins checked twice; audited training
pair and checkpoint lineage; exact128-root reconstruction/common hash; all13
shard hashes; query/RNG identities; and all47 new episodes passed. Only new
conditions96..103 were trajectory-replayed:18039 continuation moves,18080
full-path moves,1996764 action scores and one solution. Compute/timing accounting
passed; earlier trajectories were not replayed again.

New successful condition103, root51, L1000:142 letters/7 strands,1CC/75 moves,
cost1075. It equals the reference and reaches the certified semantic cost floor.
Independent determinant2076081 supplies CC lower1, although the stored interval
is[0,1]; q0=74 gives floor74+1001=1075. Reference and found witness both passed
full replay and every-prefix cost identity. Each contains one CROSSING_CHANGE,
68 REDUCE and six DESTABILIZE actions, with zero excess and zero certified regret.
This is the first proven cost-optimal found path in the current random-final
prefix. It matches an existing bound, rather than improving it, and concerns
this synthetic representation and semantic action-cost contract. It does not
establish new public u bounds, embedding causality or training benefit.

Verified totals:104 conditions,618 episodes,317471 continuation moves,317994
full-path moves,58110343 action scores and eight successful conditions on eight
distinct roots. Partial success2/52 atL10 and6/52 atL1000;96 budget-censored
outcomes. Recorded neural counts317471 policy evaluations/160520 batches,
2440 tree-embedding evaluations/1263 batches,4525 tree action scores; committed
time87288.128742 seconds. Neural/timing counts were not independently rerun.

No restart, new training, protocol/source mutation, model promotion or database
write. The complete paired evaluation remains pending. Next routine inspection:
no earlier than2026-09-23 20:58 UTC.

## Hourly update: 2026-09-23 20:58 UTC, random120 with five more solutions

Controller98832 remains at11/13 jobs with unique expected worker2020 and the
unchanged canonical PGX command. Worker CPU advanced1561:15.86 to1562:42.27
between20:58:17 and20:59:43 UTC, at100%, stateR. No duplicate search/audit worker
or new search-log failure. The final eight conditions were not yet committed
at the inspected snapshot; the controller's full final audit is also pending.

Manifest182905 bytes and log6178 bytes updated20:49:16 UTC, recording120/128
conditions. Two new shards:

| Shard | Bytes | Commit UTC | SHA256 | Recorded seconds |
| --- | ---: | --- | --- | ---: |
|conditions-00104.json|1657016|20:10:29|`c34733305d05447f29db13fd24e5a64aa3990d2b12a4f20e0ef19afcd0789e27`|3941.473379|
|conditions-00112.json|1220147|20:49:14|`a92058edaa5c55fb4618c71ae659d08332b16a136ab60a15ab8973f0d49cb7d7`|2323.082909|

Incremental audit finished exit0:1264 input pins checked twice; audited training
pair/checkpoint lineage; exact128-root reconstruction/common hash; all15 shard
hashes; query/RNG identities; and all93 new episodes passed. Only conditions
104..119 were trajectory-replayed:33867 continuation moves,33949 full-path moves,
3941061 action scores and five solutions. First new block has47 episodes,
19366/19407 continuation/full moves and2590310 scores; second46 episodes,
14501/14542 moves and1350751 scores. All13 earlier shard hashes remain unchanged.

All five found witnesses and their references passed full replay and every-prefix
cost identities, with independent integer determinants. Each has CC lower1 and
an optimal1CC reference. Results below are therefore certified regrets, not just
gaps to an unproven incumbent:

| Condition / root | Objective | Input length / strands | Found CC / moves / cost | Optimal reference cost | Regret |
| --- | --- | --- | --- | ---: | ---: |
|106 /53|L10|196 /13|1 /109 /119|115|4|
|108 /54|L10|164 /17|1 /93 /103|101|2|
|114 /57|L10|106 /5|1 /58 /68|66|2|
|115 /57|L1000|106 /5|1 /62 /1062|1056|6|
|119 /59|L1000|142 /17|1 /80 /1080|1080|0|

Determinants for roots53/54/57/59 are29543/8911/546953/18469; corresponding
q0 values104/90/55/79. Roots57 and59 have stored intervals[0,1], strengthened
to lower1 by their nonunit determinants in this cost audit. Excess comes from
two INSERTs at106, one INSERT at108/114, and one INSERT plus four COMMUTEs at115.
Condition119 has zero excess and is the second proven cost optimum in the
random-final prefix. No new result improves reference CC or semantic cost.
Two successes on root57 are distinct cost-conditioned searches, not two knots.

Verified totals:120 conditions,711 episodes,351338 continuation moves,351943
full-path moves,62051404 action scores and13 successful conditions on12 distinct
roots. Partial success5/60 atL10 and8/60 atL1000;107 budget-censored outcomes.
Recorded neural counts351338 policy evaluations/178113 batches,2829 tree-embedding
evaluations/1464 batches,5235 tree action scores; committed time93552.685030
seconds. Neural/timing counts were not independently rerun as inference.

No restart, new training, protocol/source mutation, model promotion or database
write. No causal embedding or complete paired-comparison claim. Next routine
inspection: no earlier than2026-09-23 22:04 UTC.

## Hourly update: 2026-09-23 22:04 UTC, random search and full audit complete

Random's final128-condition manifest committed21:37:28 UTC, SHA256
`1018ea18b77338fdc076aedc947c77709aa23ad08de6f834f8291e885b4d0ab7`.
Its full replay/cost audit committed21:54:44 UTC,260735 bytes, SHA256
`5afbf9c6299bc8b8303f9bbf82f76910638c30cfd5d36d78917a0d14f0a756fe`,
statuspassed. All128 conditions/759 episodes/14 solutions were covered, including
the last shard and all64 references. No second full audit was launched manually.

Controller98832 now records13/13 completed children andstatusscored, currentnull,
but is itself performing the combined summary's prescribed re-verification.
The process scan found this controller only, not the old search worker2020.
Controller CPU advanced11:45.49 to13:32.60 between22:04:20 and22:06:08 UTC,
at98.7–100%. The final summary hash was not yet present. This is live finalization,
not a completed controller, stalled search, or reason to launch a duplicate.
The existing unified execution session60520 was successfully reattached and
remains running; finalization can be awaited through that handle.

Final random totals:128 conditions,759 episodes,369159 continuation moves,
369804 full-path moves,63553982 action scores and14 solutions on13 raw roots.
Success5/64 atL10 and9/64 atL1000; three certified L1000 cost optima, no improved
reference CC/cost. The last success is condition123/root61:1CC/85 moves,
L1000=1085, certified optimal for its164-letter/5-strand representation.
Final batch SHA9324b032f57123e7387337cff450bcd57c08cb1ef8491e84cb5d71292d9eb953,
1300028 bytes, committed21:37:25 UTC. Compute totals187165 policy batches,
3006 tree-embedding evaluations/1554 batches,5577 tree action scores and
96441.999324 committed seconds; these are recorded work, not new training.

A separate arithmetic/hash pass checked all13 child artifact hashes and1345
combined input pins twice, exact common roots, shards, certificate costs and
paired success coverage. It did not repeat semantic trajectory replay. Both
searches' verified results and interpretation are recorded in134. No new
training, configuration change, promotion or database write; combined summary
completion and the broader embedding-utility goal remain pending.

## Finalization: 2026-09-23 22:26:50 UTC, controller completed and verified

The original execution session60520 exited0. Summary359096 bytes committed
22:26:49 UTC, SHA256
`83037de0cb75a9ade88173cbb1a65a93d9e8efd2a8dfd9835c5b84db9f5dd50e`.
Controller manifest179781 bytes committedcompleted22:26:50 UTC, SHA256
`e1f5888bc225f9d69f8ffec434267088c44873350e6641c7c50a3a714b58aae5`,
13/13 children, currentnull. A fresh process scan found no remaining controller,
search worker or full-audit worker. No process was killed or restarted.

Post-exit verification finished exit0: final summary/controller hashes and
contract binding, all13 child outputs and model identities,1345 combined pins
twice, all distance aggregates/constant controls/ranking and paired deltas,
paired search and behavior summaries, and equality with both full cost-audit
payloads. Additional direct grouped arithmetic checked interval errors and
constant1 controls. No duplicate semantic replay or neural inference.

The finished analysis is134. Per its existing completion instruction, heartbeat
`rf-knots-hourly-mcts-evaluation-check` was setPAUSED using the app tool; the
persisted status was verified and its prompt, schedule and target were unchanged.
The broader embedding goal is NOT paused or marked complete. No further hourly
inspection of this finished experiment is scheduled. No checkpoint promotion,
database import, release, new training or cloud-resource launch occurred.
