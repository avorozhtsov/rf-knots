# Root b16 multi-output head: independent DEV calibration result

2026-10-03. Protocols 264 and 265 were executed without DEV checkpoint
selection. Frozen 32,768-step actor, final step-512 value head. The 256
independent DEV attempts (32 roots x L10/L1000 x four seeds, 16 simulations
each) completed. All 256 shard SHA256 pins and task identities were checked
independently; every successful witness was fully replayed and checked against
its root, CC/move caps and both objective costs. There were 184 successes:
short L10 60/64, short L1000 57/64, long L10 37/64, long L1000 30/64.

Fixed report:
`artifacts/root-budget16-value-dev-calibration-v1/report.json`, SHA256
`308d6833ebbfc372ab30ee286ad2191402bd20c308aa631280754f1eda2fc5cb`.
Teacher manifest SHA256:
`d4b6fd64064edca831c1a5c9b1dc2c5c0abf6f4e109a38cc79c37f57c46a63c4`.

| DEV slice | Solved | Model / TRAIN-constant Brier | Model / TRAIN-constant log loss | Model / TRAIN-constant CC MAE, successes | Model / TRAIN-constant moves MAE, successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| short L10 | 60/64 | .138 / .080 | .629 / .318 | 1.063 / 2.369 | 14.784 / 24.642 |
| short L1000 | 57/64 | .082 / .115 | .512 / .402 | 1.149 / 2.184 | 12.558 / 18.586 |
| long L10 | 37/64 | .392 / .289 | 1.834 / .795 | .012 / .850 | 10.819 / 11.489 |
| long L1000 | 30/64 | .475 / .333 | 1.966 / .885 | .021 / .868 | 10.814 / 11.260 |

The TRAIN-only constant solve frequencies were 0.790625 (L10) and 0.758594
(L1000). For long L10, all 64 predictions were in the top 0.8–1.0 bin
with mean 0.9765 versus observed 0.5781. For long L1000, 60 predictions
were in that bin with mean 0.9585 versus observed 0.4333. Thus the probability
head is strongly overconfident on long DEV roots and worse than its simple
TRAIN-frequency baseline there. It is **not calibrated for value-guided MCTS**.
The paired conditional cost outputs beat the TRAIN-only constant MAE on
successful attempts, but this does not repair the solve-probability failure;
the very small long-CC error may partly reflect a narrow CC distribution and
needs stratified inspection. Failures supply no CC/move targets.

The teacher spent 1,381.51 CPU seconds and 210,223 neural evaluations across
256 attempts. This is search computation, not value-head training. The report
does not measure value-head inference CPU separately, despite the protocol's
request; that missing measurement needs a separately versioned follow-up.

Decision: do not wire this root-b16 head into AlphaZero-style leaf MCTS or
claim multi-budget/internal-node calibration. Diagnose length/strands/CC and
remaining-resource shift, then predeclare a separate multi-budget,
internal-TRAIN-node collection with disjoint DEV calibration and a bounded
head-only retraining. Do not use TEST150 for selection. Preserve the failed
probability calibration as evidence rather than silently recalibrating on DEV.
