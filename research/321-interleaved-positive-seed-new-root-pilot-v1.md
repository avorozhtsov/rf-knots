# Interleaved positive-seed TRAIN root source pilot v1

2026-10-11. This is a new candidate **source**, not a model-selected set or
F-labelled bank. The conjugate-block source yielded zero new stage-3 F roots
at b64 and at b256 in independently replayed pilots. Before spending bulk
search CPU, test whether replay-complete interleaved walks from certified
positive-braid seeds supply genuinely different TRAIN query groups in the
long bands. Do not tune this generator against viewed DEV or TEST150.

Freeze seed 20261012, SHA-ranked generator indices 0..4095, at most 16
distinct stage-2 roots (word length 97..128) and 16 stage-3 roots
(129..192). For each index choose a positive-braid seed of genus 1..8 on
3..6 strands, then an interleaved walk with k=0/1 extra crossing changes.
The k=0 branch uses 80..95 settle moves; k=1 uses 48..64 pre-change and
20 settle moves for stage 2. Stage 3 uses 92..99 settle moves when k=0,
or 72..80 pre-change and 32 settle moves when k=1. Targets are 112/152
letters and 8..12 strands; hard caps are 192 letters and 20 strands.
Mirror every other generated witness. Store actual lengths, CC and moves;
targets are drift controls, not labels. Require full reference replay,
paired L10/L1000 instances whose caps admit that reference, unique query
groups not in old 1000 or incremental 256 starts, TRAIN owner at root and
every crossing-change successor, and no legacy/viewed heldout group along
that path. Source code and exact inputs are SHA-pinned before output.

This first 32-candidate roster measures supply only. A separate independent
audit must verify SHA, replay, ownership, exclusions, caps and uniqueness.
Only then pre-register a small balanced canonical pgx-mcts-bench F-yield/CPU
search pilot; completed solutions must be fully replayed before F admission.
If supply fails, preserve the failure and version the changed quotas/config.
No capped failure is a negative action. Keep all prior admitted F roots in
future banks; do not hot-swap the active R2 trainer. No cloud, database
publication/mutation, commit, merge or push.
