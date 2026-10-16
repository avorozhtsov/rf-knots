# Root-group-disjoint value-head gate: both fixed splits fail

2026-10-04. Protocol268 was run exactly twice, with split salts 20261005 and
20261006. For each arm, 96 paired TRAIN representations (1,920 attempts)
trained a fresh frozen-encoder three-output head for 512 native-MPS AdamW
steps; 32 disjoint paired representations (640 attempts, 320 per lambda)
were held out. All eight checkpoint SHA256 pins per arm, source pins, finite
head weights, optimizer step counts and saved RNG states passed independent
audit. The saved reports and hashes are:

- `artifacts/root-group-holdout-value-gate-salt20261005-v1/report.json`,
  SHA256 `005d4cddd498d568a2258d823c43e1c330ad08a5671de9586742fc5a8d7765fe`;
- `artifacts/root-group-holdout-value-gate-salt20261006-v1/report.json`,
  SHA256 `9e555f616a95f4bc110a80e6753a75035477979993da2c37fd30eba7205a6c4f`.

| Split | Lambda | Solved / 320 | Head / conditional-fit baseline Brier | Head / baseline log loss | Mean predicted / observed solve fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261005 | 10 | 240 | .270 / .151 | 1.172 / .446 | .856 / .750 |
| 20261005 | 1000 | 240 | .234 / .134 | 1.071 / .402 | .772 / .750 |
| 20261006 | 10 | 268 | .205 / .173 | 1.023 / .511 | .895 / .838 |
| 20261006 | 1000 | 260 | .215 / .186 | .769 / .519 | .833 / .813 |

The baseline uses only fitting representations, conditioned on lambda,
CC-cap bin and word-length bin, with Beta(1,1) smoothing. Root-weighted and
attempt-weighted point estimates coincide because every root/lambda has ten
attempts; the effective number of independent heldout representations is
32 per split, not 320. The prespecified requirement of no worse Brier and
log loss in both splits and both lambdas fails decisively. In the first
split's long cap-2 stratum, L10 predicts .840 solve probability versus .450
observed; L1000 predicts .664 versus .400. In the second split the overall
gap is smaller, but score deterioration remains in both lambdas.

Conditional cost predictions show a mixed story. First split CC MAE was
.312 (L10) and .439 (L1000), while the trivial `CC = cap` baseline was
.250 and .400. Second split CC MAE was .291 and .367 versus cap baseline
.299 and .400. Neither establishes robust learned CC value. Move MAE beat
the fit-only conditional move constant in both splits (11.46 vs 13.87 and
12.35 vs 14.60 in the first; 11.25 vs 12.05 and 11.72 vs 13.06 in the
second), but this cannot rescue the failed p_solve gate.

CPU encoder-plus-head evaluation of 32 heldout paired representations and
64 lambda queries took 3.60/3.59 seconds cold/warm for split1 and 3.89/3.89
seconds for split2 with one CPU thread. These timings exclude teacher search.

Decision: the present 128-representation, 512-step head recipe does not
generalize bounded success probability reliably to unseen root
representations. Group-level overfit is supported by two disjoint TRAIN
checks as well as independent DEV265, although this does not isolate whether
the main cause is number/diversity of roots, model capacity, regularization,
or some combination. **Do not launch a large multi-budget internal-node
teacher collection under the same recipe or wire this head into MCTS.**
Next, predeclare and test a small *head-only* controlled ablation using the
same two fixed splits: lower-capacity and/or stronger regularization, and
group-aware sampling (one representation before one seed) at equal update
and CPU budgets. No checkpoint selection on these already inspected
holdouts; use a third predeclared root-group split for any promoted variant,
then a fresh external DEV panel. If none passes, prioritize more distinct
TRAIN states before value-guided MCTS. Preserve all failed artifacts.
