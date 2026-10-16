# Compute gap explained by low-battery hibernation

2026-09-27 around14:38 UTC (15:38 London), following explicit user request to
continue. Read201, current manifests and actual process commands before diagnosis.

Read-only macOS power log confirms:

- 13:36:18 +0100: Entering Sleep due to Low Power Sleep, battery1%,7135 seconds.
- 15:35:13 +0100: Wake from Hibernate due to acattach/UserActivity, using AC,
  battery1%.

This118m55s sleep interval explains the roughly119-minute wall-versus-CPU gap
observed in201. It was host hibernation, not evidence of a worker crash or MCTS
deadlock. Do not interpret affected wall-clock block durations as isolated
compute throughput or use them unqualified in algorithm speed comparisons.

Both existing workers survived:149 control PID36622 live100% CPU,276m06.66s
accumulated versus274m28.60s in201;135 units PID47974 live100% CPU,
1240m13.50s versus1238m35.33s. Controllers574/20100 remain alive. Saved prefixes
unchanged24/128 with1 success (149 control),80/128 with9 successful conditions
covering7 roots (135 units). No new block since the check two minutes earlier;
no restart or duplicate. Sources and conditions unchanged, no power settings
modified, no forced wake or sleep-prevention process installed.

Continue existing searches and hourly monitoring. Keep AC power connected for
uninterrupted local computation. Preserve the recorded interruption in timing
analysis; witness/CC results are unaffected by this timing caveat. No cloud,
database publication or Git actions.
