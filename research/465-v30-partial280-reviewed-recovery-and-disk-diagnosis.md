# Reviewed partial280 recovery and resource diagnosis,2026-10-26

User explicitly authorized checking conditions, cause analysis, repair and continuation.
No deletion/systempurge/otherappkill/DB/Git/cloud or threshold override.

## Proven vs unknown causes

Storage guard triggered both learner(exit0/storage_gate) and randomcollector
(ValueError disk_guard12GiB before nextcondition). Fullsaved2017313 SHA
add10057bf0356b0a345232e5ebb43f7133030019555b1ab713384ff839c39b1.
v3 durable129760.75310653995active seconds,128766MAINepisodes; downtimeexcluded.
A1628687 SHAa2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c,
activeorigin206004.2251009539 and MAIN39168 unchanged; balanceorigin1645639.

Fresh initialfree24.7GiB then23.7GiB, stable above16launch/12runtime. Machine
RAM16GiB, swap10.50GiB used of11GiB at diagnosis. Allocateddirs: randomv30
2041820KiB,supervisorv30(mainintakebanks)2539776KiB,mainMCTS5134728KiB,
sum~9.27GiB. Our accumulated immutable copied banks contribute to usage but
these measurements DO NOT establish the source of tens-of-GiB fluctuations.
Swap is a plausible contributor, not a proven sole cause. No historical
simultaneous swap/disk record exists at the dips. resource_snapshot_v30.py
adds read-only fresh free/swap/own-workerRSS measurements for future checks;
no claim that system fluctuation is fixed. Keep guards unchanged.

## Concrete contour defect and repair

Pinned collector cannot blindly resume partial window. New separate reviewed
recovery supports exact missing condition enumeration, validates all existing
shards/sourceactor/spec/roster/caps/heldout/cycles and fully replays episodes,
then executes ONLY seven absent conditions with ORIGINAL frozen280actor
step2012576, NOT latest2017313. Completed277..279 and nine280conditions must
remain byte-identical. No seed/budget/roster/gate change or repeated searches.
Original entire277..280 fixed64conditionwindow yield/audit remains required.

First verifier scripts/recover_random_window_277_280_v30.py stopped in read-only
partial replay because it attempted to hash nonexistent finalcollection.json.
ZERO new searches occurred. Its source/plan remain preserved. Review receipt
storage-recovery-277-280-review.json records this engineering bug; separate
v30a skips the final report ONLY during partialaudit; finalfull16audit still
mandatory. Tests cover missing7,completed16,unknownconditionfailclose and
partial9replaywithoutfinalreport, alongside window/v3gate tests.

On successful fullwindow audit, SAME supervisorv30 --resume fromsavedround017,
no smoke/commissioning/reset, full71AdamW/RNG/EMA and 50:50normbudgets retained.
NewREADY and ACTIVE remain separate until normalhourlyimmutableintake.

## Verified continuation

v30a recovery completed ONLYsevenmissing conditions. Full277..280:13admissible
views,696episode replays,552.254677searchCPU;alloldfiles byteidentical. Audit
2813paths/2808eligible/1093normalizedviews/126groups;2790oldrows retained,
23newpaths,5quarantines unchanged. nextcursor281. SAMEsupervisor --resume
toolmanagedsession47344;fullparentfinite71AdamW/RNG/pins checked before resume,
no commission/smoke/newA. Actual2017313->2017576 proves263realupdates;
counts2370/2369->2501/2501,actualMCTSshare0.4999999997,zeroaux/rehearsal.

New strict check_connected_runtime_v30a.py rejects stale savedmetadata at
LIVE==durable or status LAUNCHED_AWAITING_PROGRESS; old v30checker alone can
pass savedmetadata immediately after resume because registeredstep is earlier.
Regression tests cover this too;23tests passed. CurrentstateVERIFIED_ACTIVE,
MAIN running, supply/intake notfailed. Randomsource waits on intentional42view
backpressure until hourlyhandoff; recovered23paths are READY, NOTyetoptimized.

Read-only resources afterload:free~21.67GiB,swap12945.88MiB used vs10751.44MiB
before,about2.14GiB increase. Correlation supports monitoring swap, not a full
historical causal attribution. Future check resource_snapshot_v30.py records
free/swap/ownworkerRSS; do NOT purge/killotherapps or lower12/16guards. Receipt
artifacts/connected-learning-live-v30/storage-resume-2017313-verification.json.
