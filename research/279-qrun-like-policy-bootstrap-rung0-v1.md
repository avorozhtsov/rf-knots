# Q-run-like verified-solution policy bootstrap: first rung

2026-10-04. Prospective, separate response to the user's request to start
training rather than wait for sparse F comparisons. This is **not** the
original Q-run scientist, nor the previously planned F-dominant R2/R4
24-hour experiment. It is a bounded policy bootstrap that can later supply
a less-random search teacher. Do not relabel it as F-dominant training.

Use the independently audited v3 TRAIN roster SHA256
`268f1c4d2e8731bcfd4cc743bed767c9864fb74b2a2022ed679fbe36af134506`.
Rung 0 contains all 140 distinct stage-0 query groups, each with one
constructive witness that passed complete replay, TRAIN ownership and
heldout exclusion. For each root, retain paired L10/L1000 resource
instances. Initialize the same raster/graph policy network from the
completed F144 pilot checkpoint SHA256
`0deb9418ea71a432644b7370c84879cf887536dcd3757280466e73ed749ff227`.
Its existing metric objective is copied but frozen. This choice continues
the user's preferred F lineage without asserting F144 beat control A.

Run exactly 256 AdamW steps on native MPS, batch four distinct roots per
step, cycling through all 140 roots before reuse. For each row, choose
L10/L1000 and one position of its replayed complete witness using saved RNG.
The policy loss is mean negative log probability of the demonstrated legal
next action among the usual four-insertion candidate pool plus that action.
This is behavioural distillation, not a proof that unchosen legal actions
are bad. There is no solve-probability target, no censored-failure label,
and no CC/moves auxiliary on this first technical rung. The temporary
loss mix differs intentionally from protocol273 while the teacher lacks
F contrasts; only a separately frozen later contract may switch to an
F-dominant gradient. Learning rate 1e-4, AdamW decay .01, clip norm 1,
seed 20261004, checkpoints every 16 steps including model, frozen metric
objective, optimizer and Python/Torch/MPS RNG. No optimizer reset on resume.

Preserve existing b16/b64 search artifacts. Report actual optimizer steps,
distinct root coverage, loss, MPS/CPU/wall cost. After the fixed endpoint,
run a new prespecified same-start search comparison with the F144 start
on TRAIN rehearsal and disjoint DEV roots, reporting L10/L1000 solved roots,
CC, moves and compute. Do not select an intermediate checkpoint by viewed
DEV results. Only if the first rung shows genuine search benefit, advance
to longer roots with short-root rehearsal and verified new solutions, then
bring the audited F tournament signal in as its coverage grows. TEST150 is
not used for selection. This bootstrap does not establish superiority over
heuristics or start the R2/R4 24-hour clock.
