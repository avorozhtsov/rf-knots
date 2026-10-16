# Fixed-b16 root value: small head-only ablation fails the baseline gate

2026-10-04. Protocol270 completed all six predeclared arms: narrow, higher
AdamW decay and root-pair-diverse batches, each at split salts 20261005 and
20261006. Every arm trained only the head for 512 native-MPS steps from the
same frozen actor, with 96 fit and 32 heldout paired TRAIN representations.
Seven relevant tests passed before launch. Each of the 48 checkpoint SHA
pins, source pins, finite head tensors, AdamW step numbers, saved RNG states
and six report SHA pins passed independent audit. No earlier experiment source
or artifact was edited.

Numbers below are `head Brier / fit-only conditional-frequency baseline
Brier`; the log-loss comparison is also shown because high-confidence errors
matter for value-guided search.

| Variant | Split | L10 Brier | L10 log loss | L1000 Brier | L1000 log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| narrow width32 | 20261005 | .260 / .151 | 1.011 / .446 | .228 / .134 | .961 / .402 |
| narrow width32 | 20261006 | .198 / .173 | 1.070 / .511 | .268 / .186 | .992 / .519 |
| decay 0.10 | 20261005 | .287 / .151 | 1.223 / .446 | .252 / .134 | 1.203 / .402 |
| decay 0.10 | 20261006 | .217 / .173 | 1.080 / .511 | .218 / .186 | .783 / .519 |
| grouped batch | 20261005 | .257 / .151 | 1.169 / .446 | .224 / .134 | 1.098 / .402 |
| grouped batch | 20261006 | .218 / .173 | 1.306 / .511 | .278 / .186 | 1.270 / .519 |

The width32 head has 7,523 parameters; the other two have 42,371. Recorded
training CPU was 24.1–24.9 seconds per arm, with equal 512 optimizer steps;
these are CPU accounting numbers from an MPS run, not GPU-equivalent FLOPs.
Conditional paired cost metrics and one-thread inference timing are retained
in each full report. The six report SHA256 values, in table order, are:

1. `7ebb515bea6fac54ea7bf468c49f1c7284fa56fb469e3e2019bb93c9ca21fbe9`
2. `69943a4c6fa3052123c8ede9dc59d1e1d2297893a445e91eb65a1991f490c1b7`
3. `813238707caea4b1b5ac046c8a7546ef7fe5440c32ec016143d0d089454c40d0`
4. `62e35d268ee025fa604d7fe25aa037de81907f3c5cd7c66e69ab6cf2dbac4c01`
5. `538c55f0406e59ba745c32f1af6ee0354dee336240025b0c117e4e361c8005bb`
6. `88b1c5a5262524c8e4052c142cb10ae8b8bf55f4343f117a24ca3c9db7e5d62e`

All three interventions fail the prospective no-worse-than-fit-conditional
baseline requirement on both already-inspected exploratory splits and both
lambda values. No variant qualifies for split 20261007 confirmation. Thus
neither a large multi-budget internal-node teacher collection under these
recipes nor wiring this head into AlphaZero-style MCTS is justified.

Next direction: audit the supply of **distinct TRAIN-owned internal states**
already traversed in the saved teacher runs, including state identity,
remaining resources, length, strands and invariant ownership; do not count
multiple seeds or lambda repeats as new states. Use a newly pinned, small
data-diversity pilot only after that audit, with a separate root-group holdout
and a new external DEV panel. Better cost prediction in some slices does not
compensate for the failed bounded solve-probability calibration. Preserve
all failures and keep TEST150 out of method selection.
