# Operational sequence for protocols42 and44

2026-09-20. Fixed before launching the sequence controller or scoring any final
model. This does not alter training, test selection, budgets or promotion rules.

Controller `scripts/run_diverse_metric_sequence.py`, output
`artifacts/embedding-diverse-metric-sequence-v1`:

1. Adopt the already-running candidate by exact command/configuration and actual
   live process, never start another candidate. Wait for its final4096-step
   completed artifact and process exit.
2. Start the protocol42 old-corpus control only if no conflicting process/output
   exists. On controller resume, adopt its matching live child or completed
   artifact. Lost partial workers remain intact and require explicit audited
   trainer resume; the controller never restarts them from scratch.
3. Verify matched configuration except corpus/output, common initialization,
   shared provenance, final4096-step tensor hashes/finiteness, Python and Torch
   RNG agreement. Require metric and rollout-path independence certificates for
   BOTH the fresh short-u panel and common-block supplement. Missing/failed
   certificates prohibit scoring; they are not replaced with a smaller check.
4. Nine metric reports: three models(parent/candidate/control), each on common-
   block original bounds and short-u original/determinant bounds. Each report
   contains original and lifted views and constant controls.
5. Four128-expansion anytime cost-search reports on ALL64 original short-u
   queries, both λ10/1000, witness caps and layout tie-break: three models plus
   length. Then four replay-audited comparisons: candidate vs parent, control
   vs parent, candidate vs control, candidate vs length.

Thus17 evaluation/comparison jobs, after the single control training job. No
intermediate checkpoint is evaluated, no MCTS matrix or new rollout collection
is silently added, and the failed positive-seed panel is explicitly excluded
from scoring while retained as a failed requirement in the interpretation.

One controller lock, exact completed-job prefix, child PID/command/log, source
snapshots and input/checkpoint hashes. Existing outputs are not overwritten;
only matching recorded work can be adopted. The controller never kills training,
and errors leave existing processes/artifacts available for inspection. Final
completion is execution completion, NOT automatic scientific promotion or goal
completion. A separate report must interpret both search coverage/costs and
metric baselines, preserving the protocol44 amendment and uncertainty.
