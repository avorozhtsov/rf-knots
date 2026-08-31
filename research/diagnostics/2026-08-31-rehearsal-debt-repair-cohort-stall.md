# 2026-08-31 — A rehearsal-repair timeout stalled a four-branch cohort for 13.5 hours

Found while auditing a long-running job that was consuming a core on the
development host. It was not doing any work: the run had crashed at 02:35:44 and
written nothing since. Three separate defects, all of them live on
`pgx-mcts-bench@main`, not only on the ablation branch this was noticed from.

## What happened

`launcher-status.json`, cohort created `2026-08-31T01:35:24`:

| branch | state | stage |
|---|---|---|
| `cyclic-graph-dual-v3` | QUEUED | awaiting durable Q104 checkpoint |
| `cyclic-memory-deep-v3` | QUEUED | awaiting durable Q104 checkpoint |
| `q-grown-raster-invariant-combined-dual-12` | **LAUNCHED** | q134-barrier |
| `skm-v2-high-combined-dual` | QUEUED | awaiting durable Q104 checkpoint |

The one LAUNCHED branch raised, was re-`EXEC`ed by the recovery wrapper, and
raised again — **4 crashes and 4 relaunches** inside an hour, then silence. Three
sibling branches were waiting on a checkpoint that only the crashing branch could
produce, so the whole cohort deadlocked. Nothing detected it. The processes
stayed alive at the top of the host's CPU list for **13.5 hours** while producing
no output at all.

## Defect 1 — the error message names a cause that is not the condition

`src/pgx_mcts_bench/sv2_curriculum.py:3203` (identical on `main`):

```python
if event.get("hard_timeout") is not None:
    raise RuntimeError("rehearsal debt repair exhausted its cumulative cap")
```

The condition is a **timeout**. The message reports an **exhausted cap**. These
are different failures with different fixes — one says the work is too slow, the
other says the budget is too small — and the message sends the reader to the
wrong one. The first hour of this investigation was spent looking for a budget
that had been consumed, because that is what the error said had happened.

`docs/lessons.md` already records the general form: an assertion whose text
misdescribes its trigger costs more than no assertion, because it is believed.

## Defect 2 — the raise precedes the append, so the crash destroys its evidence

```python
    raise RuntimeError(...)        # <- 3204
repair_events.append(event)        # <- 3205, never reached
```

`event` carries `repair_index`, `requested_debt_iterations` and the
`hard_timeout` value itself. Because the raise comes first, the one event that
explains the failure is the only one never recorded. The post-mortem therefore
cannot say which repair timed out, at what index, or how long it was given.

Appending before raising costs nothing and would have made this a five-minute
diagnosis.

## Defect 3 — a single branch's crash stalls the cohort silently

The launcher re-`EXEC`s a failed branch, which is right for a transient fault and
wrong for a deterministic one: the same input produced the same exception four
times. Meanwhile the three QUEUED branches had no way to observe that their
dependency had died, and the launcher had no liveness check on a branch that is
LAUNCHED but writing nothing.

The cheapest fix is not smarter retry logic. It is a **staleness check on the
cohort's own output**: if the newest write under the run directory is older than
some multiple of the expected step time, the run is not running. That test would
have fired at roughly 03:00 rather than being noticed by hand at 16:00, and it
needs no knowledge of what the branches are doing.

## Relationship to note 20

[20-single-knot-mastery.md](../20-single-knot-mastery.md) specifies rehearsal as
the mechanism that stops self-training destroying inherited skill — a frozen
panel, a 0.80 retention gate, and a rehearsal fraction that rises in increments
when retention falls. This is that mechanism reaching its own ceiling and taking
the run down with it.

That is worth stating plainly because the protocol is otherwise sound: the
failure is not that rehearsal was wrong to demand repair, it is that the demand
had no defined behaviour when it could not be met in time. A protocol that can
fail needs a specified outcome for failing, not an exception.

## Evidence

Preserved under `pgx-mcts-bench/artifacts/local-q-skm-ablation-20260817/continuation/q4000-v1-population-20260818/q154-slow4-20260822/`,
which survives independently of the worktree the run was launched from:

* `logs/q-grown-raster-invariant-combined-dual-12.log` — 70 lines, 4 tracebacks
* `launcher-status.json` — cohort state, launcher pid 566, `created_at 01:35:24`

The crash site is durable at `sv2_curriculum.py:3204` on both
`codex/local-q-skm-ablation` and `main`.

## Disposition

The crashed `q154-slow4` process tree was stopped (worker first, then its two
parents, to avoid orphaning a child under a killed wrapper). It **respawned
within 19 seconds** and rewrote `protocol/q50-1-updated.json`, so the recovery
wrapper is self-restarting and a kill alone does not stop this family of jobs.

**The worktree was NOT removed, and must not be.** An initial audit saw only the
one crashed job and concluded the directory was idle; it is not. It hosts at
least three concurrent experiment trees — `q154-slow4`,
`run_focused_successor_fast_controls_recovery_v3` (8h) and
`run_focused_successor_proof_distilled_recovery_v2` (6h) — and live sessions are
driving work from it. Removing it would pull the `.venv` and scripts out from
under running experiments.

Two durable notes from that near-miss:

* `~/projects/pgx-mcts-bench-local-ablation` is a **worktree** of
  `pgx-mcts-bench` (its `.git` is a 99-byte file pointing at
  `.git/worktrees/pgx-mastery-v3-proof-distill`), so it holds no unique commits
  and nothing in it needs merging. It reads as an abandoned fork only because it
  sits at the top of `~/projects` while the sibling worktrees live under
  `~/projects/.worktrees/`, and because its directory name no longer matches its
  internal worktree name. Either fact alone is enough to get it forgotten.
* **Counting the jobs one `ps` line at a time is not an audit.** The check that
  would have been right the first time is "which processes hold this directory",
  not "what is the job I already know about doing".

Going forward, new runs launch from `pgx-mcts-bench` itself rather than from this
worktree, so it drains rather than being deleted.

The three defects above are unfixed and are on `main`.
