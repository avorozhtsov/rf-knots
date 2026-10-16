# Protocol149 search/cache/replay workers: PREPARED

2026-09-24. This goal turn made implementation and verification progress. No
new149 scientific search, cache gate, training or evaluation result is claimed.
The running139 search and147 generator were not interrupted, duplicated or
edited. The next routine combined observation remains no earlier than06:32 UTC.

## Workers

- `scripts/check_long_general_search_cache.py`: load the completely audited149
  pair and test the shared initial/control-final/long-final checkpoints. Reuse
  the unchanged139 non-TEST engineering fixtures, full-network/action-mask/
  greedy-action checks and cached-versus-uncached paired tiny searches. Preserve
  the fixed512-entry cache and1e-5 relative/absolute tolerances. Each model has
  102 forward cases and six prior/seed paired-search comparisons. Require shared
  initial tensor and position-aware function identity, unchanged weights, exact
  checkpoint/source bindings and current Torch version. This bounded fixture
  coverage is not a proof for every possible shape or a scientific speed result.
- `scripts/evaluate_long_general_search.py`: only the frozen151 five-search
  matrix is accepted. All64 original150 roots are evaluated at both L10/L1000,
  with seed20261079,8 simulations, widening4, insertion4, c_puct1.5 and the
  unchanged139 per-root caps. No value head, graph tail, TEST learning, secret
  truncation, or new action semantics. The learned rollout policy remains in
  both tree-prior conditions; uniform tree ordering is not removal of the
  embedding channel from the model.
- `scripts/audit_long_general_search.py`: require a completed128-condition
  worker, recheck its exact checkpoint/panel/cache/training contract, replay
  every success and censored trajectory, verify both native costs and all64
  reference witnesses, and compare independently reconstructed counters. Timing
  and physical cache/tree counters are recorded and checked for consistency,
  not independently rerun neural evaluations. Only equality to an applicable
  certified lower cost proves optimality; a small gap is not such a proof.

## Interruption and evidence preservation

Each worker has an exclusive writer lock. Search commits exactly eight
conditions per shard, then updates its manifest; every batch resets to its
registered seed and preserves all paths. Resume validates root bytes, ordered
shard names/hashes, exact prefix coverage, counters and durations. Completed
re-entry performs verification without changing completed files or rerunning
batches. An orphan shard from an interruption between shard and manifest may
be adopted only if its recomputed rows are identical; conflicting evidence is
preserved and causes failure, never overwritten. Explicit `--resume` is needed.

The cache gate and post-exit audit are finite jobs that require new output
files. Their source checks are bound before expensive work and rechecked before
publication. The future dispatch controller must not run a duplicate worker
when the actual process or completed verified output exists.

## Verification

The new worker tests exercise a full128-condition orchestration fixture versus
an interruption after8 conditions and15 resumed batches. Committed rows, data
seeds and final counters match; completed re-entry leaves every file byte-
identical. Separate tests simulate interruption after shard publication but
before manifest commit, test identical and conflicting orphan handling, and
reject changed roots, configuration, counts, hashes, incomplete shared-initial
evidence and missing cache cases/models.

Replay tests use the actual terminal-cost-search implementation with the small
test network, both distance and uniform tree priors, and both a solved one-letter
query and a resource-exhausted trefoil. They reject invented costs, altered
query/tree RNG identity, malformed work/cache counts and missing paths. They
test mechanics, not performance of the Q254 initialization or149 trained models.

The combined new evaluation/search/replay and historical cache/search suites
passed37 tests. Ruff passed on the new worker/audit/test files. No frozen139
source was changed or monkeypatched. The current generation of actual149 models
does not yet exist, so none of these tests is an actual149 checkpoint promotion.

## Remaining execution

Finish147 and its independent mathematical/support audit; acquire and audit150;
run152 real MPS readiness and149 comparative training, then its independent
training audit. The workers above and153 distance scoring can then populate
the complete frozen evaluation. The dispatch controller and final paired
summary still need implementation. A partial model or convenient root prefix
must not become a scientific promotion result.
