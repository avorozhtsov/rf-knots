# v30: preserve verified labels, evaluate random-source yield on fixed windows

User authorization2026-10-25: repair yield_stop separately, restore continuous
new random labels into AdamW, preserve A/counters and MCTS:F50:50.

## Read-only diagnosis

v29 completed44 batches16..59, then closed batch60. Across45 batches, admissible
solved-view distribution:21x4,16x3,7x2,1x1,mean3.2667. Batch60 consumed
128.798671searchCPU and replayed177episodes,16conditions. All four roots had
formal successful conditions. Roots0/1/2 had respectively4/3/2 successful
incumbents, ALL cyclic; root3 had3cyclefree incumbents. Hence only ONE
admissible view. This is not a disk failure, generation exhaustion, missing
optimizer connection, or absence of formal solutions. Existing cycle
rejection is correct and remains unchanged. Search stopped at16simulation
ceiling, not20CPU on every condition. Do not claim full20CPU spent.

## Separate prospective repair, not retry or retrospective gate change

Keep v29 batch60/status/pins untouched. Same seed, same shallow generator,
same actual verified-auto search/caps/16sim/20CPU per condition. Start at
batch61. Exclude ALL historical rosters including failed, unsolved and partial.

Replace fragile per4root batch availability gate with fixed NONOVERLAPPING
four-batch windows:16distinct generated roots,64conditions. Every batch,
including zero admissible success, counts. Require>=8distinct admissible
solved views and totalsearchCPU/solvedview<=300. This preserves the average
2of4 yield and CPU requirement, but tolerates an isolated poor batch. No
resampling, outcome-dependent extension, discarded bad batch, proof label,
cycle admission or budget inflation. Failure STOP/review; never retry.
READY is published only after the entire window passes every episode replay
and independent full additive-bank audit. All old row bytes and quarantines
remain. STOP/resource/partial errors remain closed, never blind resume.

This is an availability repair, not a claim that cycles disappeared or that
neural utility improved. Fixed windows reduce single-batch statistical stops;
they do not guarantee that any future window or supply can never exhaust.

## State and optimizer preservation

Separate collector/pipeline/supervisor/trainer/checker v30, unchanged balanced
objective v29. Original A1628687/SHAa2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c,
activeorigin206004.2251009539, MAINcount39168 and v3 panel remain untouched.
Balanceorigin1645639 and EMA/full71AdamW/RNG/LR5e-5/clip1 persist. Current
phase50:50 MCTS/F; withinMCTS MAIN/random alternate50:50; zero extra auxiliary.
Graceful oldsupervisor STOP_REQUEST only after new window audited READY;
full-state parent handoff, isolatedCPU diagnostics, ordinarynative2commission,
teacher registration after audit and actual advancing new-v30-ID consumption.
No forced periodic tiny checkpoints or live-bank mutation. Existing hourly
automatic substantial audited intake remains; source collector now runs4batches
per fixed window. Fresh MAIN consecutive cursor preserved via latestREADY.
Runtimeguard12GiB; initial/resume>=16stable. No deletion/DB/cloud/Git/promotion.

New checker must observe at least one v30-introduced random semantic ID in
real native update telemetry, not merely old-to-A random samples. The original
v3 clocks follow saved parent lineage including mixed regimens/downtime exclusion.

## Verified launch outcome

First fixedwindow61..64:4/4/3/2admissibleviews,13total,64conditions,
655episode replays,440.435684searchCPU,27newpaths. Independent cumulative
audit1527rows/1522eligible/424normalizedviews/57groups;all1500parentrows retained,
5quarantines remain. v29 gracefully saved1848621 with exit0/user_stop_request.
Native v30 restored fullstate and advanced normally to1848623; audit passed,
teacher switched only afterwards. Real continuous snapshot1848738 proves
117nativeupdates,58MAIN/59random,allfresh,SEVENdistinct newv30randomIDs actually
consumed; measuredMCTSshare0.4999999747,extraaux/rehearsal0. Source nextwindow65..68
and MAIN are live.40tests passed. Disk~63GiB. Historical receipt is
artifacts/connected-learning-live-v30/launch-verification.json; never use it as
fresh health. Original gate67490.49877575517durable seconds/71215MAINepisodes,
not ready. Monitoring updated under OpenAI Docs workflow, same3h schedule,
quiet for unchanged healthy/pending; real consumption/backlog/source failures
are checked. No utility comparison was run or reset.
