# Resource units: prospective paired joint training

2026-09-23. New experiment, not a continuation or modification of protocol97.
Motivation is TRAIN-only gradient evidence119, not a choice made to maximize
TEST70/74. Those tests remain untouched. No database import or public release.

## Hypothesis and fixed contrast

Test whether explicit resource units make joint metric/policy optimization more
useful than raw resource inputs. This is not an embedding/no-embedding test.
Both arms start with identical protocol94 Q254-local-trunk initialization and
new upper layers, not the trained97 endpoint. All parameters are trained.

Native resource order and fixed units are remaining moves256, remaining CC2,
length room256, strand room32, and CC weight1000. These are declared operating
units, not fitted means or clipping thresholds. The treatment uses x/unit and
log1p(x)/log1p(unit), retaining exhaustion and presence flags. All maps are
monotone, unbounded and size-independent; units do not cap supported instances.
The control retains x and log1p(x). Initial policy functions consequently differ;
initial embedding functions and every learned tensor match exactly. No inverse
weight compensation, residual module, loss reweighting or sampler intervention.

Both arms:2048 AdamW updates, lr1e-4, decay.01, gradient clip1, float32/MPS,
CPU threads1, fallback disabled. New draw seed20261071. Same32 metric pairs and
8 set-positive states per update as97; same whole-group log preference every16
updates, weight.16. Reference log probabilities are frozen at each arm's OWN
initial policy. Preserve shared-prefix cancellation. Failures are censored,
not relabelled as impossible states or negative actions. No scalar value head.
65536 metric draws and16384 positive draws per arm are not unique games.

Run units then raw serially to avoid device contention. Both runs are required;
no winner or early success may be declared after the first arm. Initial/final
DEV predictions are mandatory, no DEV-driven early stopping or tuning here.

## Development and independent evaluation

Use all256 existing `embedding-diverse-short-u-corpus-v1` validation rows as
historical DEV, NOT a fresh independent test. Recheck full record replay and
query fingerprint exclusion against all audited RF optimizer lineage, including
ordinary graph paths/suffixes/sentinel. Save every prediction, interval and
constant1 violation, including exact0/1/2 subgroup summaries. Invariant buckets
are conservative groups, not knot identities; Q254 prior exposure is not fully
audited and must not be called proven absent.

Final training checkpoints are fixed at2048 regardless of DEV results. Before
scoring a new TEST, freeze a score-blind panel from new generation seeds:
short-u20261072 and general-distance20261073. Use the existing constructive and
pure-braid generators with replay-certified CC intervals, exclude every audited
TRAIN query/path group, historical DEV query group, and previous TEST query
group (including70/74). Reserve64 short-u roots and up to128 general pairs with
the original exact0/1/2/interval stratification and declared adequacy checks.
No shrinking the denominator or changing seeds to rescue a model's scores.
If supply is inadequate, report that blocker rather than silently reusing TEST.
The fresh panel builder/audit remains to be implemented before evaluation;
training completion alone does not satisfy the research goal.

Search compares BOTH initial and final policies on the same new short-u roots
at L10/L1000, with equal8-simulation resources/seeds, common native cost contract
and no oracle value. Reuse the verified terminal-cost search/replay machinery;
record actual neural calls and elapsed compute as well as successes and CC/move
costs. Include policy-only versus learned-distance prior as a separately labelled
controlled contrast if claiming causal embedding benefit. The two resource arms
alone cannot establish that benefit. No public-knot upper-bound claim follows
from synthetic witnesses. Compare distances against constant1 as well as each
arm's own initialization; report both group- and record-weighted errors.

## Operational contract

New files only: do not edit hash-pinned old model, training or evaluation code.
Pin configuration, input/source hashes, feature mode, units, initial tensors,
fixed reference probabilities, RNG state, optimizer state and paired draw hash.
Lock each writer; reject existing outputs without explicit resume. Save immutable
step checkpoints and atomically publish their manifest pointer. Resume only the
last committed checkpoint, redoing uncommitted work deterministically. Keep all
old experiments and dirty work. No automatic checkpoint promotion or upload.
