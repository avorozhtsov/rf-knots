# Compact storage v31, explicitly authorized2026-10-27

This is a STORAGE version, NOT a new learner/objective, baseline, clock or study.
Original A1628687 SHAa2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c,
v3active origin206004.2251009539, first counted MAIN39168, balanceorigin1645639
and full saved2056741 SHAe9cf0673d3c615520a42edf88d065c8ed288a0189a1a31396bc7f79071793deb
are protected. Objective remains measured50:50 freshMCTS/F; insideMCTS MAIN/random50:50,
zeroextraaux/rehearsal, full71AdamW/PythonTorchMPSRNG/EMA and LR unchanged.

## Representation

artifacts/compact-storage-v31/store.sqlite3 contains zlib-compressed payloads keyed
by SHA256 and a separate original-path index. Byte-identical bank rows have one
object regardless of number of bank versions. All historical manifests, audit
reports, source and spec pins stay byte-identical; SHA verification decompresses
and hashes the ORIGINAL bytes, not the SQLite representation. Index commits and
object integrity checks precede any raw-file removal. SQLite FULL durability/WAL
and a verified rollback export avoid per-payload archive files.

Original v30 consumers deliberately remain pinned, unchanged. An explicit,
inherited v31 Python launcher enables a narrowly scoped adapter ONLY for indexed
artifact paths. Existing physical files remain authoritative, so their corruption
is never hidden. The adapter supports Path/open/read/glob and seekable torch.load;
new bank copies and atomic row publications enter CAS rather than accumulating
full physical snapshots. Writes to archived immutable destinations fail closed.
It does NOT globally alter the user's Python install, site-packages or system.

IMPORTANT: after migration, old direct Python commands cannot see archived paths.
Use the following launcher for EVERY training/data/audit/gate process that consumes
these paths. Its opt-in environment is inherited by supervisor child processes.

Common environment:
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=src:scripts:/Users/artemvorozhtsov/projects/pgx-mcts-bench/src

Read-only health:
.venv/bin/python scripts/run_compact_v31.py scripts/check_connected_runtime_v30a.py --after-step 2056741

Read-only v3gate:
.venv/bin/python scripts/run_compact_v31.py scripts/paired_learning_comparison_v3.py

Eventual authorized clean resume, ONLY after partial305..308 is separately reviewed:
.venv/bin/python scripts/run_compact_v31.py --write scripts/connected_learning_supervisor_v30.py --resume
No --start, new smoke, re-commissioning, parent reset or new baseline.

## Retention

All full step checkpoints from the last48wallhours remain raw. Older raw checkpoints
are reduced to the latest available in each UTC ISOweek within each original run.
Original A/B copies and current saved state are never candidates. Historical full
states needed for actor/provenance/parent restore remain compressed, recoverable
evidence in CAS, not thousands of loose .pt files. No historical state is destroyed
merely because metadata still refers to it. This is online retention plus archival
exceptions, not a claim that all old actor dependencies have vanished.

MAIN search attempts and random search conditions older48hours are compressed and
indexed; unique evidence/episode counts remain available. Recent, partial and failed
window data are untouched. Supply rosters/holdout exclusions, ALL failed conditions,
panels, audits, manifests, seeds and budgets remain preserved. Never retry failed
windows or substitute labels/reference proofs. No seed/budget/gate relaxation.

Periodic maintenance (plain Python, NOT an already patched launcher):
.venv/bin/python scripts/maintain_compact_storage_v31.py
prints a plan; --apply executes ONLY the same fixed immutable-payload policy,
with archive checksum and unchanged inode/size/mtime checks before unlink.
It requires>=16GiB maintenance headroom and an exclusive maintenance lock.
Never remove store.sqlite3/WAL/SHM files manually or run arbitrary broad cleanup.
The read adapter tolerates a verified immutable source being archived concurrently.

Selective rollback/export (plain Python):
.venv/bin/python scripts/restore_compact_v31.py --path artifacts/EXACT/FILE
restores ONE missing payload after SHA verification, refuses overwrite or outside
paths. A full old layout can be reconstructed from the index, but mass restoration
is intentionally not exposed as a routine command because it recreates bloat.

## Verification and authority boundaries

Migration is two-phase. --prepare stores and independently verifies EVERY unique
object, then runs forced archive reads while originals still exist. It compares
original A/fullrestore/AdamW71/RNG/EMA/64prospective sampler choices, full active-bank
SHA and the EXACT v3 durable-time/independently-audited episode gate to the pre-pack
snapshot. No search, optimizer update, panel audit or evaluation is executed.
Only --delete-verified-copies rechecks every source's inode/device/mtime/size/SHA,
removes physical copies, and repeats semantic equivalence checks. Any error stops
and preserves its evidence; do not repeat the one-shot migration blindly.

At request receipt the v30 supervisor is cleanly STOPPED_LEARNER/storage_gate,
but supply_stopped=true: window305..308 has seven persisted305conditions. Storage
migration does NOT authorize retrying that partial window or claim training is
resumed. Keep its original frozen actor/seed/roster and all results unchanged.
Runtime12GiB and launch/resume16GiB guards remain unchanged. No DB/production/cloud,
Git or further general cleanup is authorized by this storage task.

v3 gate remains AND>=172800durableactive seconds AND>=100000unique MAINepisodes,
with downtime/random/eval/intake replay excluded. Before migration values were
142789.37530868698seconds/142443episodes. No utility claimed or comparison repeated.
Only bothgates plus pins/leakage/disk/no duplicate authorize the one existing v3
--run-if-ready comparison, using the same verifiedauto search for both arms.

Migration reports and runtime pins are under artifacts/compact-storage-v31/.
Future monitoring stays every3h, quiet unchanged healthy/pending/known storage
block; notify meaningful failure/recovery/completion or required human action.
