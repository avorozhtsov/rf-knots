# V4 operational checkpoint cadence and v3 retention

2026-10-08. Protocol308 described one checkpoint per 128-step scheduler block.
The inherited trainer also saves at global multiples of 128; because v4
began at step 31,105, each complete 128-step block normally saves at the
global multiple and again at its final step. The actual cadence is therefore
two checkpoints per block, still about 4–5 times less frequent than v3.
Do not change the pinned v4 trainer to conceal this distinction. The
algorithmic update sequence is unchanged.

After both v4 arms advanced to step 31,233, v3 is no longer needed to resume
training. The approved retention set for each v3 arm is checkpoint step 1,
each available multiple of 1024, and the final migration-parent checkpoint
step 31,105. Preserve the original v3 manifest unchanged and write an
explicit, hash-checked retention ledger before deleting intermediate .pt
files. The original v3 manifest will no longer be directly resumable after
pruning; v4 is the sole continuation. Never remove either v4 artifact.
