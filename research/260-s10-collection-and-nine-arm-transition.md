# S=10 collection boundary and nine-arm transition

2026-10-03. Operational note; no candidate training or DEV ranking is claimed.

The unchanged v3 teacher manifest has committed `next_task=2560` with 2,560
shards. These are exactly 128 TRAIN representations, two objectives (L10 and
L1000), ten independent seeds and **16 simulations per search**. The source
contract also lists budgets 32/128/512, which have **not** been collected.
The bounded `--stop-after 2304` continuation ended at this S=10 boundary;
it is not completion of the entire four-budget contract. A fresh process
inspection found no v3 worker before any new work was started.

The independent full-shard SHA, replay and TRAIN-ownership audit completed
successfully with `scripts/audit_tournament_teacher_v1.py --through 2560`.
The audit exit code was zero and its output was:

| Item | L10 | L1000 | Combined |
| --- | ---: | ---: | ---: |
| Attempts | 1,280 | 1,280 | 2,560 |
| Raw solved attempts | 1,012 | 971 | 1,983 |
| Attempts with a TRAIN-admissible solution | 891 | 865 | 1,756 |

The audit independently checked 29,056 episodes, of which 21,003 were
TRAIN-admissible and 5,780 were both admissible and solved. Exact-path
deduplication leaves 3,237 admissible paths. There are 207 solved
representation/objective groups, 132 groups with strictly different verified
costs and 289 exact state/resource divergence contexts. The protocol254
minimum of 16 cost-different groups and 64 contexts is therefore met. These
figures are **TRAIN collection yield at 16 simulations**, not DEV search
performance or neural training. A missing solution within 16 simulations is
not a negative action label.

A second, SHA-bound loader (`scripts/tournament_s10_dataset_v1.py`) recovered
the same 3,237 distinct replayed paths and separated the effective support:

| Item | L10 | L1000 |
| --- | ---: | ---: |
| Solved representation/objective groups | 105 | 102 |
| Distinct admissible paths | 1,579 | 1,658 |
| Groups with different objective cost | 65 | 67 |
| Groups with at least two paths within best+2 and a worse path | 39 | 36 |
| Groups with differing crossing-change counts | 9 | 8 |

Thus F has nontrivial near-best support (75 groups total), whereas G's
additional high-CC term has only 17 contributing groups; G must report its
effective sample size and is unlikely to warrant a large budget by itself.

TRAIN-only full-signal profiling at `K=4`, near-best margin `2`, gap scale
`lambda`, and gap cap `4` found the following effective coefficient counts:

| Arm | Groups with a signal | Total terms | Median terms/signal | Maximum |
| --- | ---: | ---: | ---: | ---: |
| B | 132 | 264 | 2 | 2 |
| C | 132 | 445 | 3 | 8 |
| D | 132 | 21,815 | 143 | 449 |
| E | 132 | 710 | 3 | 33 |
| F | 106 | 838 | 4 | 41 |
| G | 132 | 845 | 3 | 33 |
| H | 132 | 710 | 3 | 33 |
| I | 132 | 710 | 3 | 33 |

This exposes a material fairness problem *before* launching the nine-arm
trainer: exact full-suffix D has about 31 times E's action-term work. Equal
optimizer steps alone would not mean comparable training compute. The pilot
runner must therefore predeclare either a compute-normalized stochastic
estimator of each full-group preference (with a gradient check against the
exact objective) or an explicit matched-work schedule; it must log actual
network action scores, CPU/wall time and effective distinct groups. Do not
silently truncate D or present its unequal-step result as an equal-budget
comparison. The profiling script is
`scripts/profile_tournament_s10_signals_v1.py`; this profile used only TRAIN.

A separate bounded-term sampler is now implemented in
`scripts/tournament_term_sampler_v1.py`. With a fixed even term budget it
draws positive and negative cancelled coefficients separately in proportion
to magnitude, retaining each side's total mass and an unbiased estimator of
each **linear coefficient**. Four new tests, including empirical coefficient
unbiasedness, pass; all six original signal tests also pass. This is **not
yet a passed training-loss gate**: the nonlinear full-group preference loss
is not automatically unbiased under subsampling. Before a nine-arm training
launch, compare sampled and exact encoder gradients on fixed TRAIN groups
across B-I, set an acceptable cosine/norm-error threshold, and freeze the
term budget and H mixture. If it fails, use the exact objective with measured
unequal training work or design a new separately versioned estimator; do not
silently call the sampler equivalent to exact D.

The first real-network gradient gate used the pinned 32,768-step actor on a
median-sized D group (`base_root_index=59`, L10, 143 exact terms). Averaging
four independently sampled gradients gave encoder-gradient cosine 0.477 at
16 terms and 0.541 at 32 terms versus the exact full-group gradient; norm
ratios were 2.15 and 1.43. These are **not acceptable approximations** for a
matched pilot. The follow-up on the same group gave cosine 0.828 at 64 draws
and 0.941 at 128 draws, with norm ratios 1.06 and 1.11 respectively, when
four independent sampled gradients were averaged. This is evidence that
many draws can recover the local full-gradient direction; it does not prove
that a single 16-term nonlinear preference update is unbiased or that all
TRAIN groups have similar variance. The next gate must test repeated-update
gradient averaging across several groups and explicitly measure the
action-score/CPU cost before fixing D's stage-1 schedule. No model-training
result is claimed for D or for the nine-arm comparison yet.

The manifest SHA256 during audit was
`358f81da2ff6b6e5c2eee17d3c61355a60db5506559b811b20f06a63a0a418ea`;
the audit script SHA256 was
`9b5679bc1a179e557496cbeed745590081146c607801b5236c8eed702e65eec5`.
All six pinned source/input SHA values matched the manifest again after the
audit. The old worker and audit process had both exited on a fresh process
inspection.
Do not extend collection to 32/128/512 merely because those budgets appear
in the manifest: the user's priority is the nine-arm pilot, and collection
adequacy must be judged on verified distinct roots, costs and divergence
contexts first.

After the audit, freeze a separate TRAIN-only pilot contract for control A
and tournament variants B-I. All nine arms must use the same immutable
starting checkpoint, root-balanced audited S=10 data, metric and known-path
distillation draws, optimizer-step count, initialization policy, and search
evaluation budget. Pin K, F's near-best margin, I's gap scale/cap and H's
gradient-matched mixture coefficient before any DEV results are read. The
control receives the same metric/distillation updates without a tournament
term. Report actual usable groups and term counts separately for each arm,
especially F and the CC-specific auxiliary signal in G; a weak G sample
cannot be hidden by oversampling it. If the independent audit invalidates
the adequacy minima, record the measured deficit and amend collection only
under a new versioned contract.

This S=10 teacher is rollout-search data from a frozen actor. It is not
neural training or an AlphaZero-style value-guided search. The multi-output
value-head work of protocol256 remains a distinct gate. TEST150 remains
excluded from selection; fully non-neural canonical `pgx-mcts-bench`
controls are required before any superiority claim.
