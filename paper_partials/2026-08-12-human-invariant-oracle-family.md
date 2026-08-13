# Human-invariant oracle family

The first controlled human-knowledge experiment adds cached whole-knot
invariants to the admitted `raster-axial` scientist. This is an explicitly
labelled oracle family, not a silent change to the learned-invariant baseline.
Five candidates separate feature content from fusion architecture:

| scientist | invariant vector | fusion |
|---|---|---|
| `raster-invariant-classical` | signature, determinant, Alexander/cover summaries | late value/global fusion |
| `raster-invariant-alexander` | classical plus Alexander polynomial summary | late value/global fusion |
| `raster-invariant-jones` | classical plus Jones polynomial summary | late value/global fusion |
| `raster-invariant-combined-film` | classical, Alexander, and Jones | FiLM modulation of the visual trunk |
| `raster-invariant-combined-dual` | classical, Alexander, and Jones | separate invariant tower fused into positional policy and value |

Polynomial summaries contain normalized degree/support statistics and signed-log
values of orders zero through three near one. The environment computes the
vector at episode initialization, carries it unchanged through internal actions
and invariant-preserving braid/Markov rewrites, and recomputes it only after a
crossing change. Thus MCTS can evaluate the invariant consequences of a crossing
change without paying the calculation again on every head shift or memory move.

The implementation passed 331 repository tests, including fixed finite feature
shapes, terminal-unknot handling, exact cache retention over shifts, forward and
backward passes for every architecture, and a positional-policy sensitivity test
for the dual tower. Real one-update local training smokes completed for both ends
of the family: classical late fusion and combined dual fusion. These are
engineering checks, not comparative results.

Three-seed neutral foundation pretraining started on the 32-vCPU Nebius host on
2026-08-12. It uses seeds 81--83, the same six foundation stages, equal L10/L1000
sampling, adaptive `F_native` in `5,8,12,16`, simulations in
`64,128,256,512`, and promotion/retention thresholds 0.80/0.80. The artifact root
is `/srv/braid/artifacts/invariant-oracle-pretrain-20260812`. Its manifest SHA-256
is `ac1f6822e6867d6d49b7a113cbd782249cbb180277d942b47188d6e1a239d67a`;
the exact executable-source hash is
`4d4cd886c8a4ce38f6b994bd71001c971a95fd4d20770a4407797b98fcff8524`.
The source was an uncommitted but hash-certified tree based on commit
`4be0bac60c7dec35f995f9a4912cedf9907a12f2`.

A waiting service selects the lowest fully promoted seed for each architecture
only after all jobs finish. If any architecture has no fully promoted seed, R24
stays closed. Otherwise the selected five scientists automatically enter a
static/no-sharing R24 run with SIM64, F10, adaptive rehearsal, EV4, L10/L1000,
and the frozen reception bank.
