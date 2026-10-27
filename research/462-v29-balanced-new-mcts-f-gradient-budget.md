# v29: new MCTS experience and F share the actual gradient budget

User approved implementation and launch2026-10-24, preserving original A and B
clocks. All v28/v28a/v28b files/checkpoints/failures remain unchanged.

## Objective and definition of50:50

Only two losses are differentiated: strict label-smoothed pairwise F and actual
MCTS root-action NLL+0.1achieved-cost SmoothL1. No additional F auxiliary head
loss, trajectory rehearsal, or distance loss is added. The cost term is INSIDE
the MCTS budget. Failed searches and generator references are not labels.

Each component is differentiated separately on the same current network.
Let f,m be L2 norms of the COMPLETE parameter-gradient vectors (encoder,
policy, cost head and any participating parameters, not just the encoder).
The common magnitude T=min(EMA_F,f,10m); EMA decay0.99 is durable checkpoint
metadata. Apply scales (1-w)T/f and wT/m, then sum, globally clip1, and call
AdamW exactly once. Measure the scaled tensors independently, log their norm
ratio, their cosine and encoder-only norms. Zero/nonfinite components fail
instead of silently falling back to one-sided learning. F is never amplified;
MCTS amplification is bounded. No AMP/CPU fallback.

This50:50 means equal norms of loss-gradient components before AdamW, NOT
equal directional effects, equal parameter displacement, or equal utility.
Gradient conflicts/cancellation and AdamW preconditioning remain measurable
limitations; no unapproved projection or optimizer reset is introduced.

## Fixed warmup, durable through all bank changes

balance_origin_step is frozen to the migration parent's full saved step.
Offsets0..1999: MCTS/F20/80;2000..3999:35/65;>=4000:50/50. This is a fixed
update-count schedule, never score-selected. Resume and bank handoff retain
this origin and EMA. It does NOT reset the older A-relative active-time clock.

Within MCTS, fresh random and fresh MAIN alternate50:50 at the existing
absolute phase351232; group/view/semantic-path rotation remains uniform.
At the final balance their average budgets are25%random,25%MAIN,50%F.

## Freshness correction

The v28b historical MAIN intake cursor was393, whereas post-A search begins
at task39168=batch1224. Simply reweighting the old bank would not implement
the user's request. v29 therefore retains all old rows byte-for-byte and
records the historical backfill cursor separately. A new prospective MAIN
intake cursor starts1224 and consumes consecutive FULL audited batches, with
full episode replay and a separately checked fresh-success evidence ledger.
Only IDs actually observed in these post-A MCTS results can enter the fresh
MAIN sampler, including identical semantic paths observed again after A.
No completed search is rerun; rereading an episode never increments B's counter.
Historical backlog is retained, not claimed optimized, skipped out of existence,
or silently included as fresh. Random IDs must be absent from the A-active
random bank and retain audited actual-MCTS provenance and legacy quarantine.

The storage-interrupted random batch15 remains preserved and excluded from
new generation. Separate v29 supply starts at batch16 with the same seed,
shallow curriculum, search/CPU/yield gates, and ALL prior rosters as exclusions.
It does not retry any completed or partial condition in batch15.

## Verification and launch

23 pure schedule/norm/freshness tests passed without live optimizer updates.
Before native launch, prepare_balanced_continuation_v29.py requires>=16GiB,
no duplicate learner/supervisor, clean full parent and all pins. It independently
intakes the existing post-A FULL MAIN batches and runs6isolated CPU updates
(both sources at20%,35%,50%). CPU weights are never saved/promoted and no
active-time counter is changed by metadata-offset diagnostics. Native
commissioning executes2ordinary updates from the real full parent, restores
model/objective/head/full71AdamW/Python-Torch-MPS RNG bitwise, checks actual
fresh draws and gradient norms, then registers the teacher and continues.
Use check_connected_learning_live_v29.py to verify real advancing metadata,
all fresh source counts, consumed bank hashes, persistent warmup origin and
actual measured budgets. A PID/READY or synthetic CPU offset is not native
learning progress. Runtime guard12GiB/hourly full saves unchanged.

Original A1628687 SHAa2bf6f7b975a134352176e9d8c36059f9e9584bfd4bf47a0614747bf90541d1c,
active origin206004.2251009539 and counted MAINcursor39168 remain unchanged.
Latest migration parent is1645639 with7235.643656345434active seconds since A
and6175audited MAINepisodes. The original48h/100k gate continues through the
parent lineage, excluding downtime. Since the objective changes part-way
through that interval, eventual v3 A/B measures the COMPLETE mixed training
regimen, not an isolated48-hour causal test of v29. No new evaluation/panel
or reset of A is introduced to hide this limitation.

Initial full15-batch episode replay/basic bank audit completed. Additional
freshness identity validation rejected an existing path whose UID ends in10
versus1000, though all semantic braid/caps fields agree. Original v29 intake
script, replay, bank, READY and failure receipt are preserved. A separate
audit_fresh_main_eligibility_v29.py independently rechecks evidence and excludes
only UID and cc_weight from semantic identity; all braid/caps remain strict.
Future batches use intake_fresh_main_v29a.py. No search/full episode replay is
retried and no numerical/input failure is silently ignored.
