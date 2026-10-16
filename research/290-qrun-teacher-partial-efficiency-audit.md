# Interim efficiency check of rung-1 on-policy TRAIN teacher

2026-10-05. The frozen 2,000-condition collection remains active. At
964 committed conditions it had finished stages 0 and 1 and entered stage 2.
All committed shard SHA and contract input SHA matched; the worker was live
at about one CPU core. This is search data collection, not optimizer work.

The new independent auditor in
`scripts/audit_qrun_rung1_onpolicy_teacher_v1.py` verifies source and
contract SHA, reconstructs each TRAIN-ownership decision, and fully replays
every admitted solution. Its first 80-condition smoke audit passed: among
the first 40 stage-0 roots under each lambda, admitted solutions occurred
for 39 L10 and 40 L1000 searches, with 485 and 516 distinct admitted paths.
None of those 80 root/objective groups yielded a path cheaper than its
already verified constructive witness: 58 best search solutions tied the
reference and 21 were dearer; one L10 search found none. These equal-cost
alternative paths may still diversify policy examples, but the volume of
solved episodes alone is not evidence of improved cost labels.

Continue the frozen healthy worker. The full audit must report cost
improvement and CPU by stage and lambda before any next training contract.
If later stages also produce few or no improvements, reassess the value of
distilling equal-cost paths versus using more search budget or a different
teacher; do not mislabel bounded failures as negative actions. No DEV or
TEST data enter this TRAIN collection.
