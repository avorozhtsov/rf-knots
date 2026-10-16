# Real-weight149 MPS readiness and saved-state replay

2026-09-24. Engineering gate, not scientific training or promotion. Require
completed147 and its ready post-exit consumer. No150/TEST forward is needed.
Existing139 and147 artifacts/processes are not modified.

Start from149's shared Q254-local raster initialization, raw resource features,
default CC objective, MPSfloat32/no fallback,one CPU thread. Select TRAIN rows
score-blind: largest word length(tie larger strands, then lower attempt index)
for each of three length bands and each exact class0/1/2, plus the globally
largest strand count(tie largest word length, lower index). Ten cases; a
repeated row in the last case is disclosed rather than counted as new data.

For each selected row compare CPU/MPS metric forward/backward on unmodified
initial weights. Reuse125 tolerances: scalars rtol1e-3/atol1e-4; whole gradient
and every parameter block relative L2<=.01 and cosine>=.999 when CPU norm>1e-6,
otherwise absolute L2<=1e-5. Preserve vectors, layouts, timings and selection
hashes. Require unchanged initial state bytes and finite values. These ten
shapes are bounded coverage, not a proof for every possible long word/strand.

Next exercise BOTH149 arms on private copies using real complete joint batches.
Consume the registered two RNG streams through draw15 and draw31(zero-based),
including discarded draws, and perform ONE joint AdamW update at each selected
draw. Each includes its complete scheduled relative group,8 positives,32 metric
pairs and the unchanged loss/clip/hyperparameters. This is deliberately TWO
engineering updates across32 data draws, NOT32 sequential training updates or
a publishable149 checkpoint.

After the first private update, save model, objective, AdamW, both local Python
RNG states, Torch/MPS RNGs and draw digests. Run the second update. Restore the
intermediate state into a fresh model/optimizer and repeat the second selection
and update. Immediate restore must be bitwise exact. Final floating parameters
and AdamW state must be close at rtol1e-5/atol1e-6; steps, RNG and non-tensor
fields must match exactly. Also report whether final state is bitwise equal.
Require optimizer step2 and changed private weights. Three physical optimizer
calls per arm, with six calls total; private outputs cannot be promoted.

Bind corpus/audit, shared initialization, inherited TRAIN sources,149 sampler,
runner/loss,152 gate/tests/protocol and library version. Final report includes
all artifacts' hashes and status. Any mismatch/failure stops this gate and
preserves partial artifacts; no silent CPU fallback or tolerance relaxation.
The149 runner must require this completed passed report, matching its current
sources, data and initial tensor digest. Full training, independent scientific
distance/search evaluation and the user's broader goal remain separate.

## Implementation status

The gate is implemented in `scripts/check_long_general_mps.py`. Four targeted
tests cover score-blind case selection, paired RNG replay, private CPU toy
optimizer save/restore, and fail-closed rejection of absent or failed reports.
The149 runner requires `--execution-gate`; it rechecks saved gradient vectors,
private intermediate/final optimizer states, source/data hashes, configuration,
Torch version, shared initialization and the complete artifact set before
production training. This is PREPARED, not a passed real-raster MPS gate: that
execution still requires the completed and independently audited147 corpus.

The combined supply/consumer/sampler/training/MPS-gate/auditor targeted suite
passed90 tests at05:30 UTC; Ruff and formatting checks passed on the new gate,
runner and audit files/tests. The only warning was unavailable optional Plink
GUI support. This test count is engineering validation, not completed147
acquisition, a real152 gate pass or production149 training.
