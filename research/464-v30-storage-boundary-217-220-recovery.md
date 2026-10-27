# v30 storage-boundary recovery, user authorization2026-10-26

No deletion/cleanup. Original A1628687, v3origin206004.2251009539 and MAIN39168,
balancedorigin1645639, EMA, all71AdamW and Python/Torch/MPS RNG preserved.

## Reviewed boundary, not yield failure or automatic retry

Supervisor stopped cleanly exit0/storage_gate. Fullstep1967718,
SHAbc3f13c4f483a3f44d3b496d2106ef7e8dc5be01139daf5b17ea8db76442c0c9.
Random collector failed disk_guard before mkdir batch220. Batches217..219
are fully persisted and replayed:3/4/3admissibleviews,167/163/167episodes,
109.953599/150.274509/170.048416searchCPU. Window217..220 contract exists;
no batch220 directory or search shard existed at reviewed boundary. No active
supervisor/learner/main/collector processes. Free~17.14GiB in consecutive
fresh measurements, above frozen16GiBlaunch/12GiBruntime but narrow headroom.
Cause of fluctuations unknown; no cleanup authority or guard override.

Separate scripts/recover_random_window_217_220_v30.py records immutable
storage-recovery-217-220.json with originalmanifest/alloldfileSHA BEFORE work.
It validates fullparent71AdamW/RNG/pins, originalwindowcontract/parentbankSHA,
replays every persisted217..219episode and exactly reconstructs old collection
reports, checks sourceactor/spec/roster/pins and cycles/heldout/ownership/caps.
It runs ONLY never-started220, same seed/index/caps/16conditions/16sim/20CPU,
then replays220 and evaluates original full64conditionwindow>=8views and
<=300CPU/view. No resampling or extension. Everyoldfile must remain byteidentical.
Only fullindependentcumulativebankaudit allows atomicREADY and nextcursor221.
Newfailure/partial recovery is preserved, not rerun blindly.

After window audit and stable>=16GiB, SAME supervisorv30 --resume restores
EXACT savedround013, no smoke/commissioning/parentreset/newA. Archived handled
resourceSTOP markers preserve evidence. Main resumes partial frozenbatch after
allsource/actor/shard/reportSHA checks, completedsearch neverrepeated. Random
continues nextfixedwindow221..224. RecoveredREADY is separatefromACTIVE until
normal>=3600continuouswall auditedimmutablehandoff; no hotswap or tiny checkpoint.
Read-only checker must verify advancingnativeupdates, savedcounts restored,
50:50MAIN/random, actualMCTS/Fnormshare50:50, extraaux/rehearsal0, nofailures.
Original31.66h/112175MAINepisode gate continues, downtimeexcluded, no evaluation.

## Verified outcome

Recovered220 yielded3admissibleviews/173episodes/67.410204searchCPU. Fullwindow
13views/670episodes/497.686728CPU passed original gate. Independentfullbankaudit
2475paths/2470eligible/910normalizedviews/113groups;2453parentpaths retained,
22newpaths;all217..219files byteidentical. nextcursor221, originalv30code/pins
unchanged. SAMEsupervisor --resume toolmanagedsession42931 launched once.
Actualnative1967718->1967825 (107newupdates); counters473/474->527/527, measured
MCTS/F0.5000000158/0.4999999842,zeroaux/rehearsal,balanceorigin unchanged.
MAIN resumes frozenpartialbatch; source nextwindow221..224 collecting.
RecoveredREADY13views awaits NORMALhourly intake, not claimed optimized yet.
Prelaunchfree17.1GiB stable; afterprocessload15.1GiB,above12runtime but narrow.
No further launch below16, no cleanup or guard override. Evidence receipt:
artifacts/connected-learning-live-v30/storage-resume-1967718-verification.json.
